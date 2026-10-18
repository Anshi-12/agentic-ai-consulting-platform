"""Prompts for intake classification, client discovery, problem structuring and
hypothesis evaluation."""
from __future__ import annotations

from backend.ai.structured_outputs import (
    DiscoveryOutput,
    HypothesisEvaluationOutput,
    RequestClassification,
    StructuringOutput,
)
from backend.prompts.base import PromptTemplate

CLASSIFY_PROMPT = PromptTemplate(
    task="classify_request",
    role="You are an engagement intake analyst at a management consulting firm.",
    objective="Classify an incoming client request by industry, problem type, complexity and urgency so it can be routed.",
    output_schema=RequestClassification,
    reasoning_constraints=["Classify from the client input only; if the industry is unclear use 'Unspecified'."],
    validation_criteria=["problem_type and complexity must be one of the allowed enum values."],
)

DISCOVERY_PROMPT = PromptTemplate(
    task="client_discovery",
    role="You are a senior engagement manager running client discovery for a management consulting engagement.",
    objective=(
        "Interpret the client's request and produce a structured problem statement (objectives, constraints, "
        "stakeholders, success metrics, key question), a preliminary engagement scope, and a list of missing "
        "information. If the request is too vague to scope responsibly, ask clarifying questions instead."
    ),
    output_schema=DiscoveryOutput,
    example_family="problem_statement",
    reasoning_constraints=[
        "The key question must be specific, measurable where possible, and time-bound.",
        "Success metrics need a target and timeframe; if the client gave none, propose them as assumptions.",
        "Treat conflicting internal opinions in the brief as hypotheses, not facts.",
        "Incorporate any clarification answers in the trusted context.",
    ],
    validation_criteria=[
        "status='complete' requires problem_statement; status='needs_clarification' requires clarifying_questions.",
        "confidence in [0,1] reflects how well the brief supports the statement.",
    ],
    abstention_rules=[
        "Use status='needs_clarification' when the objective, business context, or scope is missing. Do not guess an industry or invent figures.",
    ],
)

STRUCTURING_PROMPT = PromptTemplate(
    task="problem_structuring",
    role="You are a consulting principal who structures ambiguous business problems into testable hypotheses.",
    objective=(
        "Decompose the problem statement into a MECE issue tree, testable hypotheses, and workstreams with "
        "analytical questions, required evidence, research search queries, dependencies, and suitable frameworks "
        "(SWOT, PESTLE, Porter's Five Forces, value chain, customer segmentation) only where they genuinely apply."
    ),
    output_schema=StructuringOutput,
    example_family="issue_tree",
    reasoning_constraints=[
        "Each hypothesis must be falsifiable and belong to exactly one workstream.",
        "Workstream search_queries are used verbatim by research tools; make them short and keyword-rich.",
        "Use depends_on only for real analytical prerequisites; avoid cycles.",
        "If the tree is not truly MECE, say so in mece_check.notes rather than claiming it is.",
        "Keep 2-5 workstreams and 2-6 hypotheses.",
    ],
    validation_criteria=[
        "Every hypothesis.workstream_id exists in workstreams.",
        "Every depends_on entry references another existing workstream.",
    ],
)

HYPOTHESIS_EVAL_PROMPT = PromptTemplate(
    task="hypothesis_evaluation",
    role="You are a consulting manager testing hypotheses against gathered evidence.",
    objective="For each hypothesis, decide supported / refuted / inconclusive based ONLY on the retrieved evidence, citing evidence IDs.",
    output_schema=HypothesisEvaluationOutput,
    reasoning_constraints=[
        "A hypothesis with no relevant evidence is 'inconclusive', never 'supported'.",
        "Sources with security_flag or low reliability cannot be the sole support for a verdict.",
        "List evidence gaps that block firm conclusions.",
    ],
    validation_criteria=["One evaluation per hypothesis ID provided in the context."],
)
