"""Request/response models for the AI layer's service interface.

Group 3 may own the HTTP router; these models are the contract either way.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.contracts.schemas import EvidenceItem, WorkflowStatus


class StartWorkflowRequest(BaseModel):
    project_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    client_brief: str = Field(min_length=5, max_length=20000)
    client_name: str | None = Field(default=None, max_length=200)
    industry: str | None = Field(default=None, max_length=200)
    objectives: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)
    mock_scenario: Literal["qa_failure_once", "qa_always_fail"] | None = Field(
        default=None, description="Mock mode only: exercise QA rework / escalation paths")


class ClarificationResume(BaseModel):
    type: Literal["clarification"] = "clarification"
    answers: dict[str, str] = Field(min_length=1)


class ApprovalResume(BaseModel):
    type: Literal["approval"] = "approval"
    decision: Literal["approve", "revise", "reject"]
    reviewer: str | None = None
    comments: str = Field(default="", max_length=4000)


class ResumeWorkflowRequest(BaseModel):
    payload: ClarificationResume | ApprovalResume = Field(discriminator="type")


class StageRecord(BaseModel):
    stage: str
    started_at: str
    completed_at: str
    duration_ms: float


class WorkflowStatusResponse(BaseModel):
    execution_id: str
    project_id: str
    status: WorkflowStatus
    current_stage: str | None
    pending_interrupt: dict[str, Any] | None = None
    final_status: str | None = None
    qa_iterations: int = 0
    stage_history: list[StageRecord] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class UsageSummary(BaseModel):
    model_calls: int = 0
    failed_model_calls: int = 0
    fallback_calls: int = 0
    validation_failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: float = 0.0
    tool_calls: int = 0
    failed_tool_calls: int = 0


class AgentOutputsResponse(BaseModel):
    execution_id: str
    status: WorkflowStatus
    classification: dict[str, Any] | None = None
    discovery: dict[str, Any] | None = None
    structuring: dict[str, Any] | None = None
    work_plan: list[dict[str, Any]] = Field(default_factory=list)
    research_findings: dict[str, Any] = Field(default_factory=dict)
    financial: dict[str, Any] | None = None
    hypothesis_evaluation: dict[str, Any] | None = None
    strategy: dict[str, Any] | None = None
    qa: dict[str, Any] | None = None
    qa_history: list[dict[str, Any]] = Field(default_factory=list)
    approval: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    report_markdown: str | None = None
    security_flags: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)


class LinkedRecommendation(BaseModel):
    recommendation: dict[str, Any]
    evidence: list[EvidenceItem]
    missing_evidence_ids: list[str] = Field(default_factory=list)


class RecommendationsResponse(BaseModel):
    execution_id: str
    status: WorkflowStatus
    approved: bool
    qa_verdict: str | None
    recommendations: list[LinkedRecommendation]


# ---------------------------------------------------------------------------
# Hackathon multi-agent consulting (manager + domain agents)
# ---------------------------------------------------------------------------
class ConsultRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    project_id: str = Field(default="hackathon", pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    mock_scenario: Literal["need_human", "fabricate"] | None = Field(
        default=None, description="Mock mode only: exercise the human-participant loop or citation verification")


class HumanAnswerResume(BaseModel):
    type: Literal["clarification", "information"]
    answer: str = Field(min_length=1, max_length=2000)


class ConsultApprovalResume(BaseModel):
    type: Literal["approval"] = "approval"
    decision: Literal["approve", "reject"]
    reviewer: str | None = None
    comments: str = Field(default="", max_length=4000)


class ConsultResumeRequest(BaseModel):
    payload: HumanAnswerResume | ConsultApprovalResume = Field(discriminator="type")


class ConsultResultResponse(BaseModel):
    execution_id: str
    status: WorkflowStatus
    current_stage: str | None
    pending_interrupt: dict[str, Any] | None = None
    question: str
    routing: dict[str, Any] | None = None
    ledger: dict[str, Any] | None = None
    progress_history: list[dict[str, Any]] = Field(default_factory=list)
    agent_results: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    result_markdown: str | None = None
    stage_history: list[StageRecord] = Field(default_factory=list)
    security_flags: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    error: dict[str, Any] | None = None
