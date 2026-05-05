"""CLI entry point.

Usage examples (see README for full walkthroughs):

  # Synthesize 500 reasoner samples with Claude as teacher
  python -m src.pipeline.cli synth_subagent \\
      --teacher_provider anthropic --teacher_model claude-sonnet-4-5 \\
      --teacher_id claude_sonnet_4_5 \\
      --agent_kind reasoner --n_samples 500

  # Train the reasoner subagent
  python -m src.pipeline.cli train_subagent \\
      --teacher_id claude_sonnet_4_5 --agent_kind reasoner

  # GRPO-train the manager
  python -m src.pipeline.cli train_manager_grpo \\
      --teacher_id claude_sonnet_4_5

  # One full evolve round
  python -m src.pipeline.cli evolve_round \\
      --teacher_id claude_sonnet_4_5
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from ..benchmarks.base import StandardRow
from ..subagents.schemas import AgentKind
from ..utils.io import read_jsonl, write_jsonl
from . import stages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent_routing")
    parser.add_argument("stage", type=str, choices=[
        "load_medqa",
        "synth_subagent",
        "train_subagent",
        "train_manager_grpo",
        "evolve_build_sft",
        "train_manager_sft",
        "evolve_round",
        "eval_subagents",
        "eval_manager",
    ])

    # Context-level flags
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher_id", type=str, default="default",
                        help="Logical id used to namespace outputs (e.g. claude_sonnet_4_5).")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binding_mode", type=str, default="auto",
                        choices=["auto", "environment", "argument"])

    # MedQA loading
    parser.add_argument("--medqa_source", type=str, default="hf", choices=["hf", "local"])
    parser.add_argument("--medqa_hf_dataset", type=str, default="GBaker/MedQA-USMLE-4-options")
    parser.add_argument("--medqa_local_path", type=str, default="")
    parser.add_argument("--medqa_hf_cache", type=str, default="")
    parser.add_argument("--medqa_max", type=int, default=0)
    parser.add_argument("--medqa_normalized_cache", type=str, default="")

    # Split sizes
    parser.add_argument("--train_size", type=int, default=600)
    parser.add_argument("--dev_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=200)

    # Synth
    parser.add_argument("--teacher_provider", type=str, default="",
                        choices=["", "anthropic", "claude", "openai", "gpt", "deepseek"])
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--agent_kind", type=str, default="",
                        choices=["", "extractor", "reasoner", "rule_applier"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--synth_temperature", type=float, default=0.4)
    parser.add_argument("--synth_max_retries", type=int, default=2)
    parser.add_argument("--synth_no_cache", action="store_true")

    # Subagent SFT
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=2e-4)
    parser.add_argument("--sft_max_seq_len", type=int, default=4096)
    parser.add_argument("--sft_bs", type=int, default=1)
    parser.add_argument("--sft_grad_accum", type=int, default=8)
    parser.add_argument("--sft_max_steps", type=int, default=-1)
    parser.add_argument("--sft_no_lora", action="store_true")
    parser.add_argument("--sft_dev_jsonl", type=str, default="")

    # Manager GRPO
    parser.add_argument("--mgr_bs", type=int, default=2)
    parser.add_argument("--mgr_max_completion_length", type=int, default=2048)
    parser.add_argument("--mgr_temperature", type=float, default=0.9)
    parser.add_argument("--mgr_num_generations", type=int, default=6)
    parser.add_argument("--mgr_grpo_beta", type=float, default=0.01)
    parser.add_argument("--mgr_routing_efficiency_bonus", type=float, default=0.0)
    parser.add_argument("--mgr_max_steps", type=int, default=-1)
    parser.add_argument("--mgr_use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="agent_routing")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--task_description", type=str, default="")

    # Evolve
    parser.add_argument("--evolve_max_fail_samples", type=int, default=1500)
    parser.add_argument("--manager_sft_lr", type=float, default=2e-5)
    parser.add_argument("--manager_sft_epochs", type=int, default=1)

    # Eval
    parser.add_argument("--eval_n_samples", type=int, default=100)
    parser.add_argument("--eval_kinds", type=str, default="extractor,reasoner,rule_applier")
    parser.add_argument("--eval_manager_dir", type=str, default="")

    return parser.parse_args()


def _ctx_from(args) -> stages.StageContext:
    return stages.StageContext(
        base_model=args.base_model,
        teacher_id=args.teacher_id,
        output_root=args.output_root,
        seed=args.seed,
        binding_mode=args.binding_mode,
    )


def _load_or_split(args) -> dict:
    """Load MedQA, split into train/dev/test, also serialize splits to disk."""
    cache = args.medqa_normalized_cache or os.path.join(
        args.output_root, "data", "medqa_normalized.jsonl"
    )
    if not os.path.exists(cache):
        rows = stages.run_load_medqa(
            source=args.medqa_source,
            hf_dataset=args.medqa_hf_dataset,
            local_path=(args.medqa_local_path or None),
            hf_cache_dir=(args.medqa_hf_cache or None),
            max_examples=args.medqa_max,
            cache_normalized_path=cache,
        )
    else:
        from ..benchmarks.base import StandardRow
        rows = [StandardRow(**r) for r in read_jsonl(cache)]
        print(f"[LOAD_MEDQA] loaded cached {len(rows)} rows -> {cache}")

    train, dev, test = stages._split_rows(
        rows=rows, train_size=args.train_size, dev_size=args.dev_size,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"[SPLIT] train/dev/test = {len(train)}/{len(dev)}/{len(test)}")
    return {"all": rows, "train": train, "dev": dev, "test": test}


def main() -> None:
    args = _parse_args()
    ctx = _ctx_from(args)

    if args.stage == "load_medqa":
        _load_or_split(args)
        return

    if args.stage == "synth_subagent":
        if not (args.teacher_provider and args.teacher_model and args.agent_kind):
            sys.exit("synth_subagent requires --teacher_provider, --teacher_model, --agent_kind")
        data = _load_or_split(args)
        kind = AgentKind(args.agent_kind)
        # Synthesize on the train pool
        result = stages.run_synthesize_subagent(
            ctx=ctx, rows=data["train"], agent_kind=kind,
            teacher_provider=args.teacher_provider, teacher_model=args.teacher_model,
            n_samples=args.n_samples,
            base_temperature=args.synth_temperature,
            max_retries=args.synth_max_retries,
            use_cache=(not args.synth_no_cache),
        )
        print("[SYNTH]", result)
        return

    if args.stage == "train_subagent":
        if not args.agent_kind:
            sys.exit("train_subagent requires --agent_kind")
        kind = AgentKind(args.agent_kind)
        result = stages.run_train_subagent(
            ctx=ctx, agent_kind=kind,
            dev_jsonl=(args.sft_dev_jsonl or None),
            epochs=args.sft_epochs, lr=args.sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_SUBAGENT]", result)
        return

    if args.stage == "train_manager_grpo":
        data = _load_or_split(args)
        result = stages.run_train_manager_grpo(
            ctx=ctx, train_rows=data["train"],
            per_device_batch_size=args.mgr_bs,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.mgr_grpo_beta,
            routing_efficiency_bonus=args.mgr_routing_efficiency_bonus,
            max_steps=args.mgr_max_steps,
            use_wandb=args.mgr_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            task_description=args.task_description,
        )
        print("[TRAIN_MGR_GRPO]", result)
        return

    if args.stage == "evolve_build_sft":
        data = _load_or_split(args)
        result = stages.run_evolve_build_sft(
            ctx=ctx, rows=data["all"],
            teacher_provider=(args.teacher_provider or None),
            teacher_model=(args.teacher_model or None),
            max_fail_samples=args.evolve_max_fail_samples,
            task_description=args.task_description,
        )
        print("[EVOLVE_BUILD_SFT]", result)
        return

    if args.stage == "train_manager_sft":
        result = stages.run_train_manager_sft(
            ctx=ctx,
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        print("[TRAIN_MGR_SFT]", result)
        return

    if args.stage == "evolve_round":
        data = _load_or_split(args)
        grpo_kwargs = dict(
            per_device_batch_size=args.mgr_bs,
            max_completion_length=args.mgr_max_completion_length,
            temperature=args.mgr_temperature,
            num_generations=args.mgr_num_generations,
            grpo_beta=args.mgr_grpo_beta,
            routing_efficiency_bonus=args.mgr_routing_efficiency_bonus,
            max_steps=args.mgr_max_steps,
            use_wandb=args.mgr_use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            task_description=args.task_description,
        )
        evolve_kwargs = dict(
            teacher_provider=(args.teacher_provider or None),
            teacher_model=(args.teacher_model or None),
            max_fail_samples=args.evolve_max_fail_samples,
            task_description=args.task_description,
        )
        sft_kwargs = dict(
            epochs=args.manager_sft_epochs,
            lr=args.manager_sft_lr,
            max_seq_len=args.sft_max_seq_len,
            per_device_batch_size=args.sft_bs,
            gradient_accumulation_steps=args.sft_grad_accum,
            use_lora=(not args.sft_no_lora),
            max_steps=args.sft_max_steps,
        )
        result = stages.run_evolve_round(
            ctx=ctx, train_rows=data["train"], full_rows=data["all"],
            grpo_kwargs=grpo_kwargs, evolve_kwargs=evolve_kwargs, sft_kwargs=sft_kwargs,
        )
        print("[EVOLVE_ROUND]", result)
        return

    if args.stage == "eval_subagents":
        data = _load_or_split(args)
        kinds = [AgentKind(k.strip()) for k in args.eval_kinds.split(",") if k.strip()]
        result = stages.run_eval_subagents(
            ctx=ctx, rows=data["dev"] or data["test"], agent_kinds=kinds,
            n_samples=args.eval_n_samples,
        )
        print("[EVAL_SUBAGENTS]", result["by_agent"])
        return

    if args.stage == "eval_manager":
        data = _load_or_split(args)
        result = stages.run_eval_manager(
            ctx=ctx, rows=data["test"] or data["dev"],
            manager_dir=(args.eval_manager_dir or None),
            n_samples=args.eval_n_samples,
            task_description=args.task_description,
        )
        print("[EVAL_MANAGER]", result)
        return

    sys.exit(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()