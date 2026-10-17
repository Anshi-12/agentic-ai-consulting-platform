"""Structured outputs for the hackathon multi-agent architecture
(Management Consulting Manager Agent + four domain agents)."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from backend.ai.structured_outputs import StrictModel

AgentName = Literal["technology", "clinical", "sec", "corporate_responsibility"]
IntentCategory = Literal[
    "troubleshooting", "product_recommendation", "financial_risk", "research_question",
    "sustainability", "cross_domain", "unclear", "out_of_scope",
]
ClaimKind = Literal["fact", "calculated", "hypothesis", "assumption"]


class RoutedTask(StrictModel):
    agent: AgentName
    instruction: str = Field(min_length=5)
    search_queries: list[str] = Field(default_factory=list, max_length=4)
    company: str | None = None
    year: int | None = None
    rationale: str = ""


class RoutingPlan(StrictModel):
    intent: str
    intent_category: IntentCategory
    tasks: list[RoutedTask] = Field(default_factory=list, max_length=6)
    is_cross_domain: bool = False
    needs_clarification: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self):
        if self.needs_clarification and not self.clarifying_questions:
            raise ValueError("needs_clarification requires clarifying_questions")
        if not self.needs_clarification and self.intent_category != "out_of_scope" and not self.tasks:
            raise ValueError("a routable request needs at least one task")
        return self


class DomainFinding(StrictModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: ClaimKind = "fact"
    confidence: float = Field(ge=0.0, le=1.0)


class RecommendedItem(StrictModel):
    name: str
    item_type: Literal["product", "component", "instrument", "action", "investment"]
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)


class DomainAnswer(StrictModel):
    answer: str
    findings: list[DomainFinding] = Field(default_factory=list)
    recommended_items: list[RecommendedItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = False
    confidence: float = Field(ge=0.0, le=1.0)


class ProgressAssessment(StrictModel):
    is_resolved: bool
    progress_made: bool = True
    reason: str
    next_tasks: list[RoutedTask] = Field(default_factory=list, max_length=4)
    need_human: bool = False
    question_for_human: str | None = None

    @model_validator(mode="after")
    def _human_question(self):
        if self.need_human and not self.question_for_human:
            raise ValueError("need_human requires question_for_human")
        return self


class NextAction(StrictModel):
    action: str
    why: str
    expected_impact: str
    metric: str
    target: str
    timeframe: str
    owner: str = "Client lead"
    priority: int = Field(ge=1, le=10)
    domains: list[AgentName] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: ClaimKind = "fact"
    confidence: float = Field(ge=0.0, le=1.0)


class ConsultingAnswer(StrictModel):
    headline: str
    summary: str
    domains_consulted: list[AgentName] = Field(default_factory=list)
    next_actions: list[NextAction] = Field(default_factory=list)
    cross_domain_insights: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_note: str = ""
