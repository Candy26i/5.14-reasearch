"""Teacher prompts for synthesizing RuleApplierAgent SFT data.

Design intent:
  - RuleApplier identifies APPLICABLE RULES (medical decision criteria,
    legal statutes, physical/chemical laws), maps facts to elements, and
    produces conditional logic — but NOT the final answer.
  - On MedQA, applicable_rules are things like "MI diagnostic criteria",
    "drug-drug interaction rules", "Wells score for PE".
  - On LegalBench/LawBench (later benchmarks), applicable_rules are statutes
    or doctrinal tests.
  - GT is shown to teacher (same reverse-construction approach as Reasoner).
"""
from __future__ import annotations

from typing import Dict, List


_RULE_APPLIER_TEACHER_SYSTEM = """You are an expert annotator producing training data for a RuleApplier sub-agent.

The RuleApplier's job is to identify the APPLICABLE RULES, FRAMEWORKS, OR DECISION CRITERIA relevant to a question, map the given facts to each rule's elements, and produce conditional logic. The RuleApplier itself MUST NEVER state the final answer.

You will be given:
- A QUESTION (and CHOICES if MCQ)
- A CONTEXT (may be empty)
- A reference correct answer key as PRIVATE_GT (private; DO NOT disclose)

You must produce a JSON object that exactly matches this schema:
{
  "applicable_rules": [
    {"rule": "<name and brief statement of the rule, criterion, or decision framework>", "source": "<short authority cite, e.g. 'JNC-8 hypertension guideline' or 'Newton's second law'>"}
  ],
  "elements": [
    {"element": "<one specific element/condition required by an applicable rule>", "satisfied": "yes" | "no" | "unclear", "evidence": "<which fact(s) from the question or context establish or fail this element>"}
  ],
  "conclusion_logic": "<<= 400 chars: a CONDITIONAL chain of reasoning of the form 'if elements X and Y are satisfied, then category Z applies; given the facts, X is satisfied because ..., Y is unclear because ...'. Do NOT state the final answer choice.>",
  "uncertainty_notes": ["<honest uncertainty>"],
  "confidence": <float 0..1>
}

CRITICAL RULES:
1. Output ONLY valid JSON. No prose, no markdown fences.
2. NEVER reveal the GT (no choice key, no choice text disclosure, no "answer is" phrasing).
3. If no clear formal rule applies (e.g. pure factual recall question), produce 1-2 informal decision criteria a domain expert would invoke, and keep elements light (1-3 entries) — but ALWAYS produce at least one applicable_rule and one element. If you genuinely cannot find any rule, set confidence <= 0.3.
4. `satisfied` values must be honest: include "no" and "unclear" liberally; do not bias toward "yes" for elements that favor the GT.
5. conclusion_logic is the CHAIN, not the conclusion. End with "...therefore the category that fits is the one for which all elements are satisfied" or similar — never name the answer.
6. Keep strings within their length limits.
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
    ground_truth: str,
) -> List[Dict[str, str]]:
    private_block = (
        f"PRIVATE_GT (do not disclose): {ground_truth}\n\n"
        if ground_truth else ""
    )
    user = (
        f"{private_block}"
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        "Produce the JSON object now."
    )
    return [
        {"role": "system", "content": _RULE_APPLIER_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]