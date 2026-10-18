from __future__ import annotations

from backend.ai.structured_outputs import QAOutput
from backend.prompts.base import PromptTemplate

QA_PROMPT = PromptTemplate(
    task="consulting_qa",
    role="You are the quality-assurance partner who must sign off a consulting deliverable before it reaches the client.",
    objective=(
        "Check that the recommendations answer the client's original key question; find unsupported claims, "
        "contradictions, missing evidence, inconsistent numbers and low-confidence findings; decide pass, fail "
        "(send back for rework) or escalate (needs a human reviewer)."
    ),
    output_schema=QAOutput,
    example_family="qa_review",
    reasoning_constraints=[
        "You are reviewing someone else's work. Be skeptical; do not give the benefit of the doubt.",
        "Deterministic findings in the context were produced by code and are authoritative; include them in your judgment.",
        "Set route_to to the earliest stage that can fix the most severe issue: research, financial, strategy or structuring.",
        "A deliverable relying on a security-flagged source cannot pass.",
    ],
    validation_criteria=[
        "verdict='pass' only when there are no high or critical issues.",
        "verdict='fail' requires route_to other than 'none'.",
        "verdict='escalate' requires requires_human_review=true.",
    ],
    abstention_rules=["When unsure whether an issue is real, report it at lower severity rather than omitting it."],
)
