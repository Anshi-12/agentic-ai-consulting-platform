"""PLACEHOLDER shared schemas: workflow run records and evidence items.

Replace with imports from the shared repository once it exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    WorkflowStatus.COMPLETED,
    WorkflowStatus.REJECTED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
}


class EvidenceItem(BaseModel):
    """A unit of evidence supplied by the research / data modules (Group 2).

    `content` is UNTRUSTED text. It must never be concatenated into system
    instructions; see backend/ai/guardrails.py.
    """

    evidence_id: str = Field(pattern=r"^[A-Z]{2,5}-[A-Za-z0-9_-]{1,40}$")
    project_id: str
    source: str
    title: str
    content: str
    source_type: str = "document"  # document | research | dataset | interview
    url: str | None = None
    published_at: str | None = None
    workstream_tags: list[str] = Field(default_factory=list)
    reliability: float = Field(default=0.7, ge=0.0, le=1.0)
    injection_suspected: bool = False


class WorkflowRun(BaseModel):
    execution_id: str
    project_id: str
    requested_by: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_stage: str | None = None
    pending_interrupt: dict[str, Any] | None = None
    cancel_requested: bool = False
    error: dict[str, Any] | None = None
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)
    completed_at: str | None = None
    request: dict[str, Any] = Field(default_factory=dict)
