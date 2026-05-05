"""Teacher prompts for synthesizing ReasonerAgent SFT data.

Design intent:
  - Reasoner does multi-step decomposition + per-choice analysis.
  - We DO show the teacher the ground truth (because GPQA/MedQA are hard
    enough that letting the teacher solve from scratch yields wrong samples).
  - But we use a "REVERSE CONSTRUCTION" prompt:
    "Here is the question and the correct answer. Reconstruct the reasoning
    process an expert would use, WITHOUT ever stating the final answer."
  - The leakage auditor will scan the output for GT label/text and reject.
  - candidate_analysis must cover ALL choices fairly; the teacher cannot
    write "this is correct" for the GT choice.
"""
from __future__ import annotations

from typing import Dict, List


_REASONER_TEACHER_SYSTEM = """You are an expert annotator producing training data for a Reasoner sub-agent.

The Reasoner's job is to PRODUCE A REASONING SCAFFOLD that another agent (the manager) will use to make the final decision. The Reasoner itself MUST NEVER state or disclose the final answer.

You will be given:
- A QUESTION
- CHOICES (always present; this is a multiple-choice setting)
- A reference correct answer key (provided to you privately as PRIVATE_GT) — this is so you can produce a HIGH-QUALITY reasoning trace, but you must NOT reveal it.

You must produce a JSON object that exactly matches this schema:
{
  "sub_questions": ["<a sub-question that decomposes the main question>"],
  "required_knowledge": ["<a domain knowledge item the reasoner needs to recall>"],
  "reasoning_chain": ["<step 1>", "<step 2>", ...],
  "candidate_analysis": [
    {"choice_key": "<choice key, e.g. A>", "support": "<why this choice could be right>", "against": "<why this choice could be wrong>"}
  ],
  "uncertainty_notes": ["<honest uncertainty about a step>"],
  "confidence": <float 0..1>
}

CRITICAL RULES (these are non-negotiable):
1. Output ONLY valid JSON. No prose, no markdown fences.
2. NEVER write the GT choice key, the GT choice text, "the answer is", "correct answer", "we conclude", or any equivalent phrase.
3. candidate_analysis MUST contain an entry for EVERY choice key. For each entry, give a balanced support/against pair. Do NOT make the GT entry obviously stronger; the support and against fields should each be one or two sentences regardless of whether that choice is the GT.
4. reasoning_chain must be 3-7 steps. Each step is a SHORT cognitive move (recall a fact, eliminate a category, compare two options on one dimension). Do not use it as a place to state conclusions.
5. uncertainty_notes should genuinely flag where you (or a reasoner) might go wrong; if the question is hard, this list should be non-empty.
6. confidence reflects how confident a CAREFUL REASONER would be after this scaffolding — not your knowledge of the GT. Hard questions deserve confidence around 0.5-0.7 even if you know the answer.
7. Keep each string under 300 characters.

Why these rules: the trained Reasoner will be used at inference time WITHOUT GT. If your training samples leak GT signals, the trained Reasoner will hallucinate fake "obvious" cues that mislead the manager.
"""


def _format_choices(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_reasoner_synth_prompt(
    question: str,
    context: str,
    choices: Dict[str, str],
    ground_truth: str,
) -> List[Dict[str, str]]:
    """Build messages for teacher to synthesize Reasoner SFT data.

    GT is passed via PRIVATE_GT; the system prompt forbids leaking it.
    """
    private_block = (
        f"PRIVATE_GT (do not disclose, used only to ensure your reasoning aligns with the truth): {ground_truth}\n\n"
        if ground_truth else ""
    )
    user = (
        f"{private_block}"
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        "Produce the JSON object now. Remember: no answer disclosure anywhere in your output."
    )
    return [
        {"role": "system", "content": _REASONER_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]