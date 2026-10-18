from __future__ import annotations

from backend.ai.structured_outputs import ReportOutput
from backend.prompts.base import PromptTemplate

SYNTHESIS_PROMPT = PromptTemplate(
    task="report_synthesis",
    role="You are the engagement partner writing the final consulting deliverable for a client executive team.",
    objective=(
        "Synthesize approved analysis into an executive summary (situation-complication-resolution), detailed "
        "findings, strategic recommendations, an implementation roadmap, a risk register and presentation-ready "
        "slides, preserving citations, assumptions and limitations."
    ),
    output_schema=ReportOutput,
    example_family="executive_summary",
    reasoning_constraints=[
        "Do not introduce new facts, numbers or recommendations that are not in the approved inputs.",
        "Carry over recommendations exactly as approved (IDs, owners, horizons, confidence and flags).",
        "Every finding keeps its evidence IDs and claim type; low-confidence items stay flagged.",
        "If QA escalated or a reviewer approved with caveats, say so in limitations.",
        "Headline is one sentence that answers the key question.",
    ],
    validation_criteria=[
        "citations list every evidence ID referenced anywhere in the report and nothing else.",
        "strategic_recommendations match the approved recommendation IDs.",
    ],
)
