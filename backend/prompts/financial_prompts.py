from __future__ import annotations

from backend.ai.structured_outputs import FinancialInterpretation
from backend.prompts.base import PromptTemplate

FINANCIAL_PROMPT = PromptTemplate(
    task="financial_scenario",
    role="You are a corporate finance manager on a consulting engagement.",
    objective=(
        "Interpret scenario metrics that were calculated by deterministic code (ROI, payback, NPV, IRR, margins). "
        "Compare scenarios, explain drivers and sensitivities, and state caveats."
    ),
    output_schema=FinancialInterpretation,
    example_family="financial",
    reasoning_constraints=[
        "Every number you mention must appear in the supplied metrics. Do not compute new figures.",
        "Always say whether a figure is calculated, and whether the underlying inputs are estimates or forecasts.",
        "A negative NPV or a missing payback must be stated plainly.",
        "If the numerical analysis tool failed or returned nothing, set data_available=false and do not describe any financial results.",
    ],
    validation_criteria=[
        "preferred_scenario, when set, matches one of the supplied scenario names.",
        "data_available is false whenever no scenarios were supplied.",
    ],
)
