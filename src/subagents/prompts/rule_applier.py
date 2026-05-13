"""Teacher prompts for synthesizing RuleApplierAgent SFT data.

Design intent:
  - RuleApplier identifies applicable rules, criteria, or frameworks, maps
    facts to elements, and produces conditional logic.
  - The RuleApplier must not name the final answer or reveal answer signals.
  - Prompts are always GT-blind; the teacher sees only the question, choices,
    and context.
"""
from __future__ import annotations

from typing import Dict, List


_RULE_APPLIER_TEACHER_SYSTEM = """You are an expert annotator producing training data for a RuleApplier sub-agent.

The RuleApplier's job is to identify APPLICABLE RULES, FRAMEWORKS, OR DECISION CRITERIA relevant to a question, map the given facts to each rule's elements, and produce conditional logic. The RuleApplier itself MUST NEVER state the final answer.

You will be given:
- A QUESTION, and CHOICES if this is multiple-choice
- A CONTEXT, which may be empty

Return ONLY a valid JSON object with this schema:
{
  "applicable_rules": [
    {"rule": "<name and brief statement of the rule, criterion, or decision framework>", "source": "<short authority cite, e.g. guideline name or domain principle>"}
  ],
  "elements": [
    {"element": "<one specific element/condition required by an applicable rule>", "satisfied": "yes" | "no" | "unclear", "evidence": "<which fact(s) establish or fail this element>"}
  ],
  "conclusion_logic": "<<= 400 chars: a conditional chain, not a final conclusion>",
  "uncertainty_notes": ["<honest uncertainty>"],
  "confidence": <float 0..1>
}

CRITICAL RULES:
1. Output ONLY valid JSON. No prose, no markdown fences.
2. Do NOT identify the final answer. Never write phrases like "the answer is", "correct answer", "best choice", "we conclude", "therefore choose", or equivalent.
3. Do not copy a full answer-choice text into rule names, evidence, or conclusion_logic.
4. If no clear formal rule applies, produce 1-2 informal decision criteria a domain expert would invoke, and keep elements light.
5. `satisfied` values must be honest: include "no" and "unclear" when facts do not establish an element.
6. conclusion_logic is the chain, not the conclusion. It should describe how rule elements would separate categories without naming the answer.
7. Keep strings within their length limits.
"""


def _format_choices(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_rule_applier_synth_prompt(
    question: str,
    context: str,
    choices: Dict[str, str],
) -> List[Dict[str, str]]:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        "Produce the JSON object now. Remember: no answer disclosure anywhere in your output."
    )
    return [
        {"role": "system", "content": _RULE_APPLIER_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]
