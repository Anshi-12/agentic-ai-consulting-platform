"""Service interface for the consulting workflow.

These are the functions Group 3's API router should call. They take a
Principal (from the platform's auth), validated request models, and return the
response models in backend/api/models.py.
"""
from __future__ import annotations

from typing import Any

from backend.api.models import (
    AgentOutputsResponse,
    RecommendationsResponse,
    ResumeWorkflowRequest,
    StartWorkflowRequest,
    WorkflowStatusResponse,
)
from backend.config import Settings
from backend.contracts.auth import Principal
from backend.contracts.events import WorkflowEvent
from backend.orchestration.execution import WorkflowExecutor


class ConsultingService:
    def __init__(self, executor: WorkflowExecutor | None = None, settings: Settings | None = None,
                 background: bool = True):
        self.executor = executor or WorkflowExecutor(settings)
        self.background = background

    def start_workflow(self, req: StartWorkflowRequest, principal: Principal) -> WorkflowStatusResponse:
        run = self.executor.start(req, principal, background=self.background)
        return self.executor.status(run.execution_id, principal)

    def get_status(self, execution_id: str, principal: Principal) -> WorkflowStatusResponse:
        return self.executor.status(execution_id, principal)

    def get_outputs(self, execution_id: str, principal: Principal) -> AgentOutputsResponse:
        return self.executor.outputs(execution_id, principal)

    def resume_workflow(self, execution_id: str, req: ResumeWorkflowRequest, principal: Principal
                        ) -> WorkflowStatusResponse:
        self.executor.resume(execution_id, req.payload.model_dump(), principal, background=self.background)
        return self.executor.status(execution_id, principal)

    def retry_workflow(self, execution_id: str, principal: Principal) -> WorkflowStatusResponse:
        self.executor.retry(execution_id, principal, background=self.background)
        return self.executor.status(execution_id, principal)

    def cancel_workflow(self, execution_id: str, principal: Principal) -> WorkflowStatusResponse:
        self.executor.cancel(execution_id, principal)
        return self.executor.status(execution_id, principal)

    def get_recommendations(self, execution_id: str, principal: Principal) -> RecommendationsResponse:
        return self.executor.recommendations(execution_id, principal)

    def get_events(self, execution_id: str, principal: Principal, after: int = 0) -> list[dict[str, Any]]:
        return [e.to_ws_message() for e in self.executor.events(execution_id, principal) if e.sequence > after]

    def subscribe(self, execution_id: str, principal: Principal, callback) -> Any:
        self.executor.status(execution_id, principal)  # authorization check
        return self.executor.bus.subscribe(execution_id, lambda e: callback(e.to_ws_message()))
