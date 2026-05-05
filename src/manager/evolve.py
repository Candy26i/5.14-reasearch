"""Evolve loop: turn manager failures into new SFT data, then SFT-train manager.

Three steps (called separately or together via pipeline.stages):
  1. build_manager_sft_from_failures: read fail_buffer.jsonl, run subagents
     and an optional teacher to construct multi-turn SFT trajectories.
  2. train_manager_sft: do per-turn SFT on the constructed jsonl.
  3. (back to GRPO with the SFT'd model as init — handled at pipeline level)

The teacher's job here is to PICK A TOOL SEQUENCE (0-3 tools) for each failed
example. The teacher does NOT generate the final answer text; we use the
ground truth to construct the final ANSWER_<TOKEN> line.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from ..benchmarks.base import StandardRow
from ..subagents.runtime import FrozenSubagent, SubagentPool
from ..teachers.base import TeacherClient
from ..utils.io import read_jsonl, write_jsonl, write_json
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

from .prompt import build_manager_system_prompt, build_manager_user_message


_ALLOWED_TOOLS = ("extractor_tool", "reasoner_tool", "rule_applier_tool")
_TOOL_NAME_TO_KIND = {
    "extractor_tool": "extractor",
    "reasoner_tool": "reasoner",
    "rule_applier_tool": "rule_applier",
}


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _final_answer_str(gt: str) -> str:
    return f"ANSWER_{_label_to_token(gt)}"


def _teacher_choose_tool_sequence(
    teacher: Optional[TeacherClient],
    question: str,
    context: str,
    choices: Dict[str, str],
    available_kinds: List[str],
    fallback_seq: Optional[List[str]] = None,
) -> List[str]:
    """Ask the teacher which tool sequence (length 0-3) would best help solve this.

    The teacher does NOT see the GT here — we want it to recommend a sequence
    that a confused-on-this-example manager should follow.

    Returns: list of tool names from _ALLOWED_TOOLS, deduplicated, length<=3.
    """
    available_tools = [k + "_tool" for k in available_kinds if (k + "_tool") in _ALLOWED_TOOLS]

    if teacher is None:
        if fallback_seq is not None:
            return [t for t in fallback_seq if t in available_tools][:3]
        # Heuristic: long context -> extractor first; MCQ -> reasoner; default reasoner.
        seq: List[str] = []
        if context and len(context) > 800 and "extractor_tool" in available_tools:
            seq.append("extractor_tool")
        if "reasoner_tool" in available_tools:
            seq.append("reasoner_tool")
        return seq[:3]

    sys_msg = (
        "You design efficient tool-use plans for a manager agent.\n"
        f"Available tools: {available_tools}.\n"
        "Choose a sequence of 0 to 3 tools (no repeats) that would best help a struggling manager solve the question.\n"
        "Return ONLY JSON: {\"tool_sequence\": [\"tool_a\", \"tool_b\"]}\n"
        "Use fewer tools when the question is simple."
    )
    choices_block = ""
    if choices:
        lines = [f"  {k}. {v}" for k, v in choices.items()]
        choices_block = "CHOICES:\n" + "\n".join(lines) + "\n\n"
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"{choices_block}"
        f"CONTEXT:\n{context if context else '(no context)'}\n"
    )
    try:
        resp = teacher.chat(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            temperature=0.2, max_tokens=200,
        )
        text = resp.text
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e <= s:
            raise ValueError("no JSON in teacher response")
        obj = json.loads(text[s:e + 1])
        seq = obj.get("tool_sequence", [])
        if not isinstance(seq, list):
            raise ValueError("tool_sequence not a list")
        out: List[str] = []
        for item in seq:
            t = str(item).strip()
            if t in available_tools and t not in out:
                out.append(t)
            if len(out) >= 3:
                break
        return out
    except Exception:
        return fallback_seq or (
            ["extractor_tool", "reasoner_tool"]
            if context and len(context) > 800
            else ["reasoner_tool"]
        )


def _tool_call_message(tool_name: str, eid: int, call_id: str, binding_mode: str) -> Dict[str, Any]:
    args = {"example_id": int(eid)} if binding_mode == "argument" else {}
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }


@dataclass
class EvolveSFTConfig:
    base_model: str
    extractor_adapter: Optional[str]
    reasoner_adapter: Optional[str]
    rule_applier_adapter: Optional[str]
    rows: List[StandardRow]
    fail_buffer_jsonl: str
    out_dir: str
    teacher: Optional[TeacherClient] = None
    seed: int = 42
    max_fail_samples: int = 1500
    binding_mode: str = "environment"
    task_description: str = ""


def build_manager_sft_from_failures(cfg: EvolveSFTConfig) -> str:
    """Read fail buffer, build per-turn SFT trajectories, write to disk.

    Output is a JSONL where each row is a per-turn (prompt, response) pair:
      - turn 1: user message -> first tool_call (or final answer if seq is empty)
      - turn 2: turn1 + tool output -> second tool_call (or final answer)
      - turn 3+: ... up to 3 tools, then final answer turn
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build pool from available adapters.
    pool = SubagentPool()
    available_kinds: List[str] = []
    if cfg.extractor_adapter:
        pool.register(FrozenSubagent(cfg.base_model, cfg.extractor_adapter, "extractor", device))
        available_kinds.append("extractor")
    if cfg.reasoner_adapter:
        pool.register(FrozenSubagent(cfg.base_model, cfg.reasoner_adapter, "reasoner", device))
        available_kinds.append("reasoner")
    if cfg.rule_applier_adapter:
        pool.register(FrozenSubagent(cfg.base_model, cfg.rule_applier_adapter, "rule_applier", device))
        available_kinds.append("rule_applier")

    row_index = {int(r.example_id): r for r in cfg.rows}

    # Read failures, dedupe by example_id, cap.
    fails: List[int] = []
    seen = set()
    if not os.path.exists(cfg.fail_buffer_jsonl):
        raise FileNotFoundError(f"fail_buffer not found: {cfg.fail_buffer_jsonl}")
    for row in read_jsonl(cfg.fail_buffer_jsonl):
        eid = row.get("example_id")
        if eid is None:
            continue
        try:
            eid = int(eid)
        except Exception:
            continue
        if eid in seen:
            continue
        if eid not in row_index:
            continue
        seen.add(eid)
        fails.append(eid)
        if len(fails) >= cfg.max_fail_samples:
            break

    print(f"[EVOLVE] {len(fails)} unique failed example_ids selected from buffer.")

    sft_rows: List[Dict[str, Any]] = []
    for eid in fails:
        row = row_index[eid]
        sys_prompt = build_manager_system_prompt(
            label_keys=list(row.choices.keys()),
            task_description=cfg.task_description,
        )
        user_msg = build_manager_user_message(
            example_id=eid, question=row.question,
            context=row.context, choices=row.choices,
            binding_mode=cfg.binding_mode,
        )
        base_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        seq = _teacher_choose_tool_sequence(
            teacher=cfg.teacher,
            question=row.question,
            context=row.context,
            choices=row.choices,
            available_kinds=available_kinds,
        )

        # Pre-fetch tool outputs for the chosen sequence
        tool_outputs: Dict[str, str] = {}
        for tname in seq:
            kind = _TOOL_NAME_TO_KIND[tname]
            if not pool.has(kind):
                continue
            tool_outputs[tname] = pool.call(
                agent_kind=kind, example_id=eid,
                question=row.question, context=row.context, choices=row.choices,
                cache_namespace="evolve",
            )

        final_text = _final_answer_str(row.ground_truth)

        if not seq:
            sft_rows.append({
                "example_id": eid,
                "prompt": base_messages,
                "response": [{"role": "assistant", "content": final_text}],
            })
            continue

        # Walk turns
        history = list(base_messages)
        for i, tname in enumerate(seq):
            call_id = f"call_{eid}_{i+1}"
            asst_call = _tool_call_message(tname, eid, call_id, cfg.binding_mode)
            sft_rows.append({
                "example_id": eid,
                "prompt": list(history),
                "response": [asst_call],
            })
            tool_msg = {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tname,
                "content": tool_outputs.get(tname, '{"error":"tool_not_available"}'),
            }
            history = history + [asst_call, tool_msg]

        # Final answer turn
        sft_rows.append({
            "example_id": eid,
            "prompt": list(history),
            "response": [{"role": "assistant", "content": final_text}],
        })

    out_path = os.path.join(cfg.out_dir, "manager_sft_from_failures.jsonl")
    write_jsonl(out_path, sft_rows)
    write_json(os.path.join(cfg.out_dir, "evolve_run_config.json"), {
        "n_failed_examples": len(fails),
        "n_sft_rows": len(sft_rows),
        "available_kinds": available_kinds,
        "binding_mode": cfg.binding_mode,
        "teacher_provider": cfg.teacher.provider if cfg.teacher else "heuristic",
        "teacher_model": cfg.teacher.model if cfg.teacher else "",
    })
    print(f"[EVOLVE] wrote {len(sft_rows)} SFT rows -> {out_path}")
    return out_path


# -------------- Manager SFT --------------

@dataclass
class ManagerSFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True


def _render_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        )


def _tokenize_manager_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int) -> Dataset:
    eos = tok.eos_token or ""

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response_msgs = ex["response"]
        if isinstance(response_msgs, dict):
            response_msgs = [response_msgs]
        elif isinstance(response_msgs, str):
            response_msgs = [{"role": "assistant", "content": response_msgs}]

        prompt_text = _render_chat(tok, prompt_msgs, add_generation_prompt=True)
        full_text = _render_chat(tok, prompt_msgs + response_msgs, add_generation_prompt=False) + eos

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        plen = min(len(prompt_ids), max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(_map, remove_columns=ds.column_names)


def train_manager_sft(cfg: ManagerSFTConfig) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False

    if cfg.use_lora and PEFT_AVAILABLE:
        candidate = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        target = [m for m in candidate if m in present] or ["q_proj", "v_proj"]
        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target,
        )
        model = get_peft_model(model, lconf)
        print(f"[MANAGER_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    rows = read_jsonl(cfg.train_jsonl)
    if not rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    train_ds = _tokenize_manager_sft(rows, tok, cfg.max_seq_len)
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")

    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collator)
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[MANAGER_SFT] saved -> {cfg.out_dir}")