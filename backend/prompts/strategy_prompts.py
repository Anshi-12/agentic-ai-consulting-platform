from __future__ import annotations

from backend.ai.structured_outputs import StrategyOutput
from backend.prompts.base import PromptTemplate

STRATEGY_PROMPT = PromptTemplate(
    task="strategy",
    role="You are a strategy partner at a management consulting firm advising an executive team.",
    objective=(
        "Develop distinct strategic options that answer the key question, evaluate them against weighted "
        "decision criteria, explain costs, benefits, risks, dependencies and trade-offs, and produce prioritized "
        "recommendations with owners, time horizons and measurable outcomes."
    ),
    output_schema=StrategyOutput,
    example_family="strategy",
    reasoning_constraints=[
        "Options must be genuinely different, not variations of one idea. Provide at least two.",
        "Score each option 1-5 on every decision criterion. Do not compute weighted scores; code does that.",
        "Each recommendation cites the evidence that supports it. If none exists, set claim_type='hypothesis', confidence<=0.5 and low_confidence_flag=true.",
        "Use calculated financial metrics from the context when discussing value; never invent figures.",
        "If reviewer or QA feedback is present in the context, address every point explicitly.",
    ],
    validation_criteria=[
        "Decision criteria weights are in (0,1]; scores keys match criterion names.",
        "Every evidence ID cited exists in the retrieved evidence. Fabricated IDs fail QA automatically.",
        "Recommendations include owner, time_horizon and at least one measurable outcome.",
    ],
    abstention_rules=[
        "When evidence is insufficient for a confident recommendation, recommend the action that closes the evidence gap and flag low confidence.",
    ],
)
