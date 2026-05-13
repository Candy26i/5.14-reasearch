"""Runtime system prompts (used at inference, AFTER subagent SFT).

These are the same shape as the teacher synthesis prompts, but stripped of
teacher-facing meta-explanation. The trained subagent should be able to follow
these directly.
"""
from __future__ import annotations

from typing import Dict, List


EXTRACTOR_RUNTIME_SYSTEM = """You are the Extractor sub-agent.

Given a question (and optional choices and context), extract decision-relevant signals. Output ONLY a JSON object with this schema:
{
  "key_evidence": [{"text": str, "relevance": float, "polarity": "support"|"oppose"|"neutral"}],
  "extracted_facts": [str],
  "missing_info": [str],
  "context_summary": str,
  "confidence": float
}

Rules:
- Output ONLY valid JSON, no extra text.
- Do NOT state the final answer.
- If context is empty, key_evidence=[] and use extracted_facts for clinical/factual elements pulled from the question stem.
- Treat all answer choices fairly; do not favor any one.
"""


REASONER_RUNTIME_SYSTEM = """You are the Reasoner sub-agent.

Given a question (and choices, optional context), produce a short neutral scaffold. Output ONLY a JSON object with this schema:
{
  "case_facts": [str],
  "task_type": str,
  "decision_factors": [str],
  "knowledge_slots": [str],
  "candidate_considerations": [{"choice_key": str, "relevant_if": [str], "less_relevant_if": [str]}],
  "missing_information": [str],
  "format_confidence": float
}

Rules:
- Output ONLY valid JSON.
- NEVER state the final answer or which choice is correct.
- candidate_considerations must cover ALL choice keys.
- Use conditional relevant_if / less_relevant_if fields, not support/against.
- Keep the response short and neutral.
"""


RULE_APPLIER_RUNTIME_SYSTEM = """You are the RuleApplier sub-agent.

Given a question (and optional context, choices), identify applicable rules/criteria, map facts to their elements, and produce conditional logic. Output ONLY a JSON object with this schema:
{
  "applicable_rules": [{"rule": str, "source": str}],
  "elements": [{"element": str, "satisfied": "yes"|"no"|"unclear", "evidence": str}],
  "conclusion_logic": str,
  "uncertainty_notes": [str],
  "confidence": float
}

Rules:
- Output ONLY valid JSON.
- Do NOT state the final answer.
- conclusion_logic is the conditional chain, not the conclusion.
- Always produce at least one applicable_rule and one element.
"""


def _format_choices_block(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_extractor_runtime_user(question: str, context: str, choices: Dict[str, str]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        "Produce the JSON object."
    )


def build_reasoner_runtime_user(question: str, context: str, choices: Dict[str, str]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        "Produce the JSON object."
    )


def build_rule_applier_runtime_user(question: str, context: str, choices: Dict[str, str]) -> str:
    return (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices_block(choices)}"
        f"CONTEXT:\n{context if context else '(no context)'}\n\n"
        "Produce the JSON object."
    )


def build_runtime_messages(
    agent_kind: str,
    question: str,
    context: str,
    choices: Dict[str, str],
) -> List[Dict[str, str]]:
    if agent_kind == "extractor":
        return [
            {"role": "system", "content": EXTRACTOR_RUNTIME_SYSTEM},
            {"role": "user", "content": build_extractor_runtime_user(question, context, choices)},
        ]
    if agent_kind == "reasoner":
        return [
            {"role": "system", "content": REASONER_RUNTIME_SYSTEM},
            {"role": "user", "content": build_reasoner_runtime_user(question, context, choices)},
        ]
    if agent_kind == "rule_applier":
        return [
            {"role": "system", "content": RULE_APPLIER_RUNTIME_SYSTEM},
            {"role": "user", "content": build_rule_applier_runtime_user(question, context, choices)},
        ]
    raise ValueError(f"Unknown agent_kind: {agent_kind}")
