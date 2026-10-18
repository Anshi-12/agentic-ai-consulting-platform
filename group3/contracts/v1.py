"""Canonical JSON-schema-compatible contracts for the Group 3 boundary."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = 'v1'


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    administrator = 'administrator'
    engagement_manager = 'engagement_manager'
    consultant = 'consultant'
    client_executive = 'client_executive'


class TaskStatus(str, Enum):
    queued = 'queued'
    running = 'running'
    awaiting_review = 'awaiting_review'
    completed = 'completed'
    failed = 'failed'
    cancelled = 'cancelled'


class ProjectStatus(str, Enum):
    discovery = 'discovery'
    active = 'active'
    awaiting_review = 'awaiting_review'
    approved = 'approved'


class ContractModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class User(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    email: str
    name: str
    role: Role


class AuthPrincipal(ContractModel):
    user_id: UUID
    role: Role
    project_ids: list[UUID] = Field(default_factory=list)


class ProjectCreate(ContractModel):
    name: str = Field(min_length=3, max_length=120)
    client_name: str = Field(min_length=2, max_length=120)
    industry: str = Field(min_length=2, max_length=100)
    objectives: list[str] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)


class Project(ProjectCreate):
    id: UUID = Field(default_factory=uuid4)
    owner_id: UUID
    member_ids: list[UUID] = Field(default_factory=list)
    status: ProjectStatus = ProjectStatus.discovery
    created_at: datetime = Field(default_factory=utcnow)


class ProjectMemberAdd(ContractModel):
    user_id: UUID


class TaskReference(ContractModel):
    task_id: str
    project_id: UUID
    status: TaskStatus


class Evidence(ContractModel):
    evidence_id: str
    project_id: UUID
    source: str
    title: str
    content: str
    source_type: str
    url: str | None = None
    reliability: float | None = Field(default=None, ge=0, le=1)
    injection_suspected: bool = False


class Citation(ContractModel):
    citation_id: str
    evidence_id: str
    claim: str
    locator: str | None = None


class AgentExecutionResult(ContractModel):
    task_id: str
    project_id: UUID
    agent_name: str
    status: TaskStatus
    summary: str | None = None
    structured_output: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime = Field(default_factory=utcnow)


class ReviewApproval(ContractModel):
    review_id: UUID = Field(default_factory=uuid4)
    task_id: str
    project_id: UUID
    reviewer_id: UUID
    decision: str = Field(pattern='^(approve|request_changes|reject)$')
    comments: str = Field(min_length=1, max_length=2000)
    created_at: datetime = Field(default_factory=utcnow)


class WebSocketEvent(ContractModel):
    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    task_id: str | None = None
    project_id: UUID
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ErrorResponse(ContractModel):
    code: str
    message: str
    request_id: UUID = Field(default_factory=uuid4)


class ResearchRequest(ContractModel):
    question: str = Field(min_length=5, max_length=1000)
    domain: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class ResearchTaskRequest(ResearchRequest):
    source_mode: str = Field(default='indexed', pattern='^(indexed|synthetic_fixture|official_urls)$')
    urls: list[str] = Field(default_factory=list, max_length=20)


class FinancialAnalyticsRequest(ContractModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=10000)
    source_name: str = Field(default='user-uploaded', max_length=200)
    currency: str = Field(default='USD', min_length=3, max_length=3)


class Decision(ContractModel):
    decision: str = Field(pattern='^(approve|request_changes|reject)$')
    comment: str = Field(min_length=1, max_length=2000)


class AuditEvent(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    actor_id: UUID
    action: str
    resource_type: str
    resource_id: UUID | None = None
    created_at: datetime = Field(default_factory=utcnow)


class WorkflowStartRequest(ContractModel):
    client_brief: str = Field(min_length=5, max_length=20000)
    client_name: str | None = Field(default=None, max_length=200)
    industry: str | None = Field(default=None, max_length=200)
    objectives: list[str] = Field(default_factory=list, max_length=20)
    constraints: list[str] = Field(default_factory=list, max_length=20)


class WorkflowResumeRequest(ContractModel):
    payload: dict[str, Any]
