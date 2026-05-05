"""Reward functions for manager GRPO.

Primary reward: binary correctness (1.0 if final ANSWER matches GT, else 0.0).
Format guard: penalize any sample that emits plaintext tool-call artifacts in
              the final assistant message (these would break the manager's
              tool-call discipline).
Optional: routing efficiency bonus (small +alpha for getting it right with
          fewer tool calls), controlled by a flag in build_reward_funcs.

Reward signature compatible with TRL GRPOTrainer's reward_funcs.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from ..utils.io import append_jsonl
from .prompt import parse_final_answer


# Detect plaintext tool-call leakage in final assistant content
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_TOOLS_TAG_RE = re.compile(r"<tools>", re.IGNORECASE)
_TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
_TOOL_NAMES = ("extractor_tool", "reasoner_tool", "rule_applier_tool")


def _has_plaintext_tool_artifacts(text: str) -> bool:
    if not text:
        return False
    if _TOOL_CALL_TAG_RE.search(text):
        return True
    if _TOOLS_TAG_RE.search(text):
        return True
    if _TOOL_CALLS_FIELD_RE.search(text):
        return True
    for name in _TOOL_NAMES:
        # match name as a function-call-ish prefix
        if re.search(rf"\b{re.escape(name)}\s*[\(\{{:]", text, flags=re.IGNORECASE):
            return True
    return False


def _msg_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict) and "text" in blk:
                out.append(str(blk.get("text", "")))
        return "\n".join(out)
    return str(content)


def _extract_completion_stats(completion: Any) -> Dict[str, Any]:
    """Pull useful stats from a completion (TRL message-list format)."""
    if not isinstance(completion, list):
        text = _msg_text(completion)
        return {
            "last_assistant_text": text,
            "tool_calls": 0,
            "tool_msgs": 0,
            "tool_names_called": [],
            "last_msg_has_tool_calls": False,
            "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(text),
        }
    assistant_msgs = [m for m in completion if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs = [m for m in completion if isinstance(m, dict) and m.get("role") == "tool"]

    tool_calls = 0
    tool_names_called = []
    for m in assistant_msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            tool_calls += len(tc)
            for entry in tc:
                fn = (entry.get("function", {}) or {}).get("name", "") if isinstance(entry, dict) else ""
                if fn:
                    tool_names_called.append(str(fn))

    last_text = ""
    last_has_tc = False
    if assistant_msgs:
        last_text = _msg_text(assistant_msgs[-1].get("content"))
        last_has_tc = bool(assistant_msgs[-1].get("tool_calls"))

    return {
        "last_assistant_text": last_text,
        "tool_calls": tool_calls,
        "tool_msgs": len(tool_msgs),
        "tool_names_called": tool_names_called,
        "last_msg_has_tool_calls": last_has_tc,
        "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(last_text),
    }


def _ensure_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if not x:
            return [None] * n
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n


def build_reward_funcs(
    fail_buffer_jsonl: Optional[str] = None,
    raw_trace_jsonl: Optional[str] = None,
    routing_efficiency_bonus: float = 0.0,
    is_main_process: bool = True,
):
    """Construct the reward function passed to GRPOTrainer.

    Args:
        fail_buffer_jsonl: if set, write every wrong/malformed sample here,
                           keyed by example_id, for the evolve loop.
        raw_trace_jsonl: if set, log full per-completion stats here.
        routing_efficiency_bonus: small bonus per saved tool call, e.g. 0.05.
                                  reward = base + alpha * (3 - tool_calls) when correct.
        is_main_process: only rank 0 should write the side-channel files.
    """

    def reward_fn(prompts=None, completions=None, ground_truth=None, example_id=None,
                  choice_keys=None, **kwargs) -> List[float]:
        n = len(completions)
        gts = _ensure_list(ground_truth, n)
        eids = _ensure_list(example_id, n)
        ck_lists = _ensure_list(choice_keys, n)

        rewards: List[float] = []
        fail_rows: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []

        for c, gt, eid, keys in zip(completions, gts, eids, ck_lists):
            stats = _extract_completion_stats(c)
            keys_list = list(keys) if isinstance(keys, (list, tuple)) else []
            pred = parse_final_answer(stats["last_assistant_text"], keys_list)

            valid_format = (pred is not None)
            no_artifacts = not stats["last_msg_has_plaintext_artifacts"]
            no_tc_in_final = not stats["last_msg_has_tool_calls"]

            base_correct = bool(valid_format and no_artifacts and no_tc_in_final and pred == gt)
            base_reward = 1.0 if base_correct else 0.0

            if base_correct and routing_efficiency_bonus > 0.0:
                saved = max(0, 3 - int(stats["tool_calls"]))
                base_reward = base_reward + routing_efficiency_bonus * saved

            rewards.append(float(base_reward))

            if not base_correct and is_main_process and fail_buffer_jsonl:
                fail_rows.append({
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "valid_format": bool(valid_format),
                    "no_artifacts": bool(no_artifacts),
                    "no_tc_in_final": bool(no_tc_in_final),
                    "tool_calls": int(stats["tool_calls"]),
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                    "last_assistant_text": stats["last_assistant_text"][:2000],
                })

            if is_main_process and raw_trace_jsonl:
                trace_rows.append({
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "reward": float(base_reward),
                    "tool_calls": int(stats["tool_calls"]),
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                })

        if fail_rows and fail_buffer_jsonl:
            append_jsonl(fail_buffer_jsonl, fail_rows)
        if trace_rows and raw_trace_jsonl:
            append_jsonl(raw_trace_jsonl, trace_rows)

        return rewards

    # Provide a stable name for TRL logging
    reward_fn.__name__ = "binary_outcome_with_format"
    return [reward_fn]


# Convenience export for external callers that want the bare binary version.
def binary_outcome_reward(prompts=None, completions=None, ground_truth=None,
                          example_id=None, choice_keys=None, **kwargs) -> List[float]:
    fn_list = build_reward_funcs()
    return fn_list[0](prompts=prompts, completions=completions,
                      ground_truth=ground_truth, example_id=example_id,
                      choice_keys=choice_keys, **kwargs)