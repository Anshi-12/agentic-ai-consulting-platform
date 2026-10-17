"""Pydantic schemas for every agent's structured output, plus parsing/validation.

These models ARE the AI layer's output contract. Report and API models are
built from them; keep field names stable and coordinate changes with Group 3.
"""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

T = TypeVar("T", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ClaimType(str, Enum):
    FACT = "fact"                # supported by cited evidence
    CALCULATED = "calculated"    # produced by a deterministic Python function
    ESTIMATE = "estimate"        # supplied estimate or analyst judgment on data
    FORECAST = "forecast"        # forward-looking projection
    HYPOTHESIS = "hypothesis"    # to be tested
    ASSUMPTION = "assumption"    # taken as given, needs validation
    ILLUSTRATIVE = "illustrative"


class Metric(StrictModel):
    metric: str
    target: str
    timeframe: str = ""


# --------------------------------------------------------------------------
# A. Client discovery
# --------------------------------------------------------------------------
class Stakeholder(StrictModel):
    role: str
    interest: str = ""
    influence: Literal["low", "medium", "high"] = "medium"


class ProblemStatement(StrictModel):
    summary: str = Field(min_length=10)
    key_question: str = Field(min_length=5)
    business_objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    success_metrics: list[Metric] = Field(default_factory=list)


class EngagementScope(StrictModel):
    in_scope: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    estimated_duration_weeks: int | None = Field(default=None, ge=1, le=104)


class MissingInfo(StrictModel):
    item: str
    why_needed: str = ""
    priority: Literal["low", "medium", "high"] = "medium"


class DiscoveryOutput(StrictModel):
    status: Literal["complete", "needs_clarification"]
    problem_statement: ProblemStatement | None = None
    engagement_scope: EngagementScope | None = None
    missing_information: list[MissingInfo] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("clarifying_questions")
    @classmethod
    def _questions_when_needed(cls, v, info):
        if info.data.get("status") == "needs_clarification" and not v:
            raise ValueError("needs_clarification requires at least one clarifying question")
        return v

    @model_validator(mode="after")
    def _complete_has_statement(self):
        if self.status == "complete" and self.problem_statement is None:
            raise ValueError("status 'complete' requires problem_statement")
        return self


class RequestClassification(StrictModel):
    industry: str
    problem_type: Literal[
        "profitability", "growth", "cost_reduction", "market_entry", "operations",
        "digital_transformation", "organization", "m_and_a", "other"
    ]
    complexity: Literal["low", "medium", "high"]
    urgency: Literal["low", "medium", "high"] = "medium"


# --------------------------------------------------------------------------
# B. Problem structuring
# --------------------------------------------------------------------------
class IssueNode(StrictModel):
    id: str
    question: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    children: list["IssueNode"] = Field(default_factory=list)


class Hypothesis(StrictModel):
    id: str
    statement: str
    rationale: str = ""
    workstream_id: str
    required_evidence: list[str] = Field(default_factory=list)


class Workstream(StrictModel):
    id: str
    name: str
    objective: str
    analytical_questions: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list, description="Queries for the research tools")
    depends_on: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)


class FrameworkChoice(StrictModel):
    name: str
    applies_to: str
    rationale: str


class MeceCheck(StrictModel):
    is_mece: bool
    notes: str = ""


class StructuringOutput(StrictModel):
    issue_tree: IssueNode
    hypotheses: list[Hypothesis] = Field(min_length=1)
    workstreams: list[Workstream] = Field(min_length=1)
    frameworks: list[FrameworkChoice] = Field(default_factory=list)
    mece_check: MeceCheck

    @model_validator(mode="after")
    def _check_refs(self):
        ws_ids = {w.id for w in self.workstreams}
        bad = [h.id for h in self.hypotheses if h.workstream_id not in ws_ids]
        if bad:
            raise ValueError(f"hypotheses reference unknown workstreams: {bad}")
        for w in self.workstreams:
            unknown = set(w.depends_on) - ws_ids
            if unknown or w.id in w.depends_on:
                raise ValueError(f"workstream {w.id} has invalid dependencies {sorted(unknown)}")
        return self


class HypothesisVerdict(StrictModel):
    hypothesis_id: str
    verdict: Literal["supported", "refuted", "inconclusive"]
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class HypothesisEvaluationOutput(StrictModel):
    evaluations: list[HypothesisVerdict] = Field(min_length=1)
    evidence_gaps: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# C. Strategy
# --------------------------------------------------------------------------
class Criterion(StrictModel):
    name: str
    weight: float = Field(gt=0.0, le=1.0)


class StrategicOption(StrictModel):
    id: str
    title: str
    description: str
    costs: str = ""
    benefits: str = ""
    risks: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict, description="criterion name -> score 1-5")
    weighted_score: float | None = Field(default=None, description="Computed by code; model value ignored")
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("scores")
    @classmethod
    def _score_range(cls, v):
        for k, s in v.items():
            if not 1 <= s <= 5:
                raise ValueError(f"score for '{k}' must be 1-5")
        return v


class Recommendation(StrictModel):
    id: str
    title: str
    rationale: str
    option_id: str | None = None
    priority: int = Field(ge=1, le=10)
    owner: str
    time_horizon: Literal["0-3 months", "3-12 months", "12+ months"]
    measurable_outcomes: list[Metric] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: ClaimType = ClaimType.FACT
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence_flag: bool = False


class StrategyOutput(StrictModel):
    decision_criteria: list[Criterion] = Field(min_length=1)
    options: list[StrategicOption] = Field(min_length=2)
    recommendations: list[Recommendation] = Field(min_length=1)
    trade_offs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# D. Financial & scenario
# --------------------------------------------------------------------------
class MetricValue(StrictModel):
    name: str
    value: float | None
    unit: str
    value_type: ClaimType
    source: str = "backend.ai.finance"
    note: str = ""


class ScenarioMetrics(StrictModel):
    scenario: str
    metrics: list[MetricValue]
    input_value_type: ClaimType = ClaimType.ESTIMATE
    assumptions: list[str] = Field(default_factory=list)


class FinancialInterpretation(StrictModel):
    """What the LLM produces. It must reference computed numbers, not invent them."""

    summary: str
    preferred_scenario: str | None = None
    scenario_comparison: list[str] = Field(default_factory=list)
    key_drivers: list[str] = Field(default_factory=list)
    sensitivities: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    data_available: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class FinancialOutput(StrictModel):
    data_available: bool
    currency: str = "USD"
    scenarios: list[ScenarioMetrics] = Field(default_factory=list)
    interpretation: FinancialInterpretation
    calculation_log: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# E. QA
# --------------------------------------------------------------------------
RouteTarget = Literal["research", "financial", "strategy", "structuring", "none"]


class QAIssue(StrictModel):
    id: str
    severity: Literal["low", "medium", "high", "critical"]
    category: Literal[
        "unsupported_claim", "contradiction", "missing_evidence", "inconsistent_number",
        "off_scope", "low_confidence", "fabricated_citation", "security", "schema"
    ]
    description: str
    affected_ids: list[str] = Field(default_factory=list)
    route_to: RouteTarget = "none"
    source: Literal["llm", "deterministic"] = "llm"


class ResearchRequest(StrictModel):
    workstream_id: str
    question: str


class QAOutput(StrictModel):
    verdict: Literal["pass", "fail", "escalate"]
    addresses_original_problem: bool
    issues: list[QAIssue] = Field(default_factory=list)
    low_confidence_findings: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    route_to: RouteTarget = "none"
    additional_research_requests: list[ResearchRequest] = Field(default_factory=list)
    summary: str = ""


# --------------------------------------------------------------------------
# F. Report synthesis
# --------------------------------------------------------------------------
class ExecutiveSummary(StrictModel):
    headline: str
    situation: str
    complication: str
    resolution: str
    key_recommendations: list[str] = Field(min_length=1)


class Finding(StrictModel):
    title: str
    finding: str
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: ClaimType = ClaimType.FACT
    confidence: float = Field(ge=0.0, le=1.0)


class RoadmapPhase(StrictModel):
    phase: str
    timeframe: str
    actions: list[str] = Field(min_length=1)
    owner: str
    milestones: list[str] = Field(default_factory=list)


class Risk(StrictModel):
    id: str
    risk: str
    likelihood: Literal["low", "medium", "high"]
    impact: Literal["low", "medium", "high"]
    mitigation: str
    owner: str


class Slide(StrictModel):
    title: str
    bullets: list[str] = Field(min_length=1)
    speaker_notes: str = ""


class Citation(StrictModel):
    evidence_id: str
    source: str
    title: str = ""


class ReportOutput(StrictModel):
    executive_summary: ExecutiveSummary
    detailed_findings: list[Finding] = Field(min_length=1)
    strategic_recommendations: list[Recommendation] = Field(min_length=1)
    implementation_roadmap: list[RoadmapPhase] = Field(min_length=1)
    risk_register: list[Risk] = Field(min_length=1)
    presentation_slides: list[Slide] = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Parsing & validation helpers
# --------------------------------------------------------------------------
class StructuredOutputError(ValueError):
    def __init__(self, message: str, raw: str = "", errors: Any = None) -> None:
        super().__init__(message)
        self.raw = raw[:2000]
        self.errors = errors


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Pull a JSON object out of model text (handles fences and leading prose)."""
    if not text or not text.strip():
        raise StructuredOutputError("empty model output", text)
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            start = cand.find("{")
            end = cand.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    continue
    raise StructuredOutputError("model output is not valid JSON", text)


def validate_output(schema: type[T], payload: str | dict) -> T:
    data = extract_json(payload) if isinstance(payload, str) else payload
    try:
        return schema.model_validate(data)
    except (ValidationError, ValueError) as exc:
        errors = exc.errors() if isinstance(exc, ValidationError) else str(exc)
        raise StructuredOutputError(f"output failed {schema.__name__} validation", json.dumps(data)[:2000], errors) from exc


def schema_for_prompt(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), indent=1)


def format_validation_errors(err: StructuredOutputError) -> str:
    if isinstance(err.errors, list):
        lines = []
        for e in err.errors[:12]:
            loc = ".".join(str(p) for p in e.get("loc", ()))
            lines.append(f"- {loc or '<root>'}: {e.get('msg')}")
        return "\n".join(lines)
    return str(err.errors or err)
