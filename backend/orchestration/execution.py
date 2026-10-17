"""Workflow execution engine: start, resume, retry, cancel, status, outputs.

Each workflow is a LangGraph thread keyed by its execution ID. The run record
(status/stage/interrupt) lives in the WorkflowStore; graph state lives in the
checkpointer, so a failed or interrupted run can continue from its last
checkpoint.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any

from langgraph.types import Command

from backend.agents.base import AgentRuntime
from backend.ai.guardrails import configure_logging, get_logger, redact_secrets, sanitize_evidence
from backend.ai.model_router import ModelCallFailed, ModelRouter
from backend.ai.provider_factory import ProviderFactory
from backend.ai.tools import ReviewQueue, ToolCallRecord, ToolContext, ToolExecutor
from backend.api.models import (
    AgentOutputsResponse,
    LinkedRecommendation,
    RecommendationsResponse,
    StartWorkflowRequest,
    UsageSummary,
    WorkflowStatusResponse,
)
from backend.config import Settings, get_settings
from backend.contracts.auth import Principal, authorize_project
from backend.contracts.data_module import DataModuleClient, InMemoryDataModule
from backend.contracts.events import EventBus, EventType, InMemoryEventBus, WorkflowEvent
from backend.contracts.persistence import InMemoryWorkflowStore, JsonFileWorkflowStore, WorkflowStore
from backend.contracts.schemas import TERMINAL_STATUSES, WorkflowRun, WorkflowStatus, utcnow_iso
from backend.orchestration.checkpoints import get_checkpointer, thread_config
from backend.orchestration.graph import GraphServices, WorkflowCancelled, build_graph
from backend.orchestration.state import initial_state

log = get_logger("execution")


class WorkflowStateError(RuntimeError):
    """Operation not valid for the workflow's current status."""


FINAL_STATUS_MAP = {
    "completed": WorkflowStatus.COMPLETED,
    "rejected": WorkflowStatus.REJECTED,
    "insufficient_information": WorkflowStatus.FAILED,
}


class WorkflowExecutor:
    def __init__(self, settings: Settings | None = None, *, data_module: DataModuleClient | None = None,
                 store: WorkflowStore | None = None, bus: EventBus | None = None,
                 factory: ProviderFactory | None = None, router: ModelRouter | None = None,
                 checkpointer=None, review_queue: ReviewQueue | None = None, knowledge_base=None):
        self.settings = settings or get_settings()
        configure_logging(self.settings.log_level)
        self.data = data_module or InMemoryDataModule()
        self.store = store or (JsonFileWorkflowStore(self.settings.run_store_path)
                               if self.settings.run_store_path else InMemoryWorkflowStore())
        self.bus = bus or InMemoryEventBus()
        self.reviews = review_queue or ReviewQueue()
        self.router = router or ModelRouter(self.settings, factory or ProviderFactory(self.settings))
        self.tools = ToolExecutor(self.data, self.store, self.reviews,
                                  max_calls_per_agent=self.settings.max_tool_calls_per_agent,
                                  max_calls_per_run=self.settings.max_tool_calls_per_run,
                                  injection_policy=self.settings.injection_policy,
                                  on_call=self._on_tool_call, knowledge_base=knowledge_base)
        runtime = AgentRuntime(self.settings, self.router, self.tools, self.data)
        self.services = GraphServices(self.settings, runtime, self.store, self.bus)
        self.graph = self._build_graph().compile(checkpointer=checkpointer or get_checkpointer(self.settings))
        self._threads: dict[str, threading.Thread] = {}
        self._locks: dict[str, threading.Lock] = {}

    def _build_graph(self):
        return build_graph(self.services)

    # -- events ---------------------------------------------------------------
    def _emit(self, run: WorkflowRun, type_: str, status: str | None = None, message: str | None = None,
              **payload: Any) -> None:
        self.bus.publish(WorkflowEvent(execution_id=run.execution_id, project_id=run.project_id, type=type_,
                                       stage=run.current_stage, status=status, message=message, payload=payload))

    def _on_tool_call(self, rec: ToolCallRecord, ctx: ToolContext) -> None:
        self.bus.publish(WorkflowEvent(execution_id=ctx.execution_id, project_id=ctx.project_id,
                                       type=EventType.TOOL_INVOKED, stage=ctx.agent,
                                       status="ok" if rec.success else "failed",
                                       payload={"tool": rec.tool, "agent": rec.agent, "latency_ms": rec.latency_ms,
                                                "summary": rec.result_summary[:200]}))

    # -- driving the graph ---------------------------------------------------
    def _lock(self, eid: str) -> threading.Lock:
        return self._locks.setdefault(eid, threading.Lock())

    def _drive(self, eid: str, graph_input: Any) -> None:
        with self._lock(eid):
            run = self.store.update(eid, status=WorkflowStatus.RUNNING, pending_interrupt=None, error=None)
            try:
                self.graph.invoke(graph_input, thread_config(eid))
            except WorkflowCancelled:
                run = self.store.update(eid, status=WorkflowStatus.CANCELLED, completed_at=utcnow_iso())
                self._emit(run, EventType.WORKFLOW_CANCELLED, "cancelled")
                return
            except Exception as exc:  # failure recovery: state stays checkpointed; retry() resumes it
                error = self._normalize_error(exc)
                run = self.store.update(eid, status=WorkflowStatus.FAILED, error=error)
                self._emit(run, EventType.WORKFLOW_FAILED, "failed", error["message"], error=error)
                log.error("workflow %s failed: %s", eid, error["message"])
                return
            self._sync(eid)

    @staticmethod
    def _normalize_error(exc: Exception) -> dict[str, Any]:
        err: dict[str, Any] = {"type": type(exc).__name__, "message": redact_secrets(str(exc))[:500],
                               "recoverable": True}
        if isinstance(exc, ModelCallFailed):
            err["task"] = exc.task
            err["model_call"] = {k: v for k, v in exc.record.to_dict().items() if k != "attempts"}
            le = exc.last_error
            err["cause"] = le.to_dict() if hasattr(le, "to_dict") else str(le)[:300]
        return err

    def _sync(self, eid: str) -> WorkflowRun:
        snap = self.graph.get_state(thread_config(eid))
        run = self.store.get(eid)
        if run.cancel_requested:
            run = self.store.update(eid, status=WorkflowStatus.CANCELLED, completed_at=utcnow_iso())
            self._emit(run, EventType.WORKFLOW_CANCELLED, "cancelled")
            return run
        interrupts = list(getattr(snap, "interrupts", ()) or ())
        if not interrupts:
            for task in snap.tasks:
                interrupts.extend(task.interrupts)
        if interrupts:
            payload = interrupts[0].value
            stage = payload.get("stage") or ("clarification" if payload.get("type") == "clarification"
                                             else "human_approval")
            run = self.store.update(eid, status=WorkflowStatus.AWAITING_INPUT, pending_interrupt=payload,
                                    current_stage=stage)
            self._emit(run, EventType.WORKFLOW_AWAITING_INPUT, "awaiting_input",
                       f"waiting for {payload.get('type')}", interrupt_type=payload.get("type"),
                       escalated=payload.get("escalated", False))
            return run
        if not snap.next:
            final = snap.values.get("final_status", "completed")
            status = FINAL_STATUS_MAP.get(final, WorkflowStatus.COMPLETED)
            error = {"type": "InsufficientInformation", "message": "client did not provide enough information to scope",
                     "recoverable": False} if final == "insufficient_information" else None
            run = self.store.update(eid, status=status, completed_at=utcnow_iso(), current_stage="done", error=error)
            etype = {WorkflowStatus.COMPLETED: EventType.WORKFLOW_COMPLETED,
                     WorkflowStatus.REJECTED: EventType.WORKFLOW_REJECTED}.get(status, EventType.WORKFLOW_FAILED)
            self._emit(run, etype, status.value, final)
        return run

    def _launch(self, eid: str, graph_input: Any, background: bool) -> None:
        if background:
            t = threading.Thread(target=self._drive, args=(eid, graph_input), name=f"wf-{eid}", daemon=True)
            self._threads[eid] = t
            t.start()
        else:
            self._drive(eid, graph_input)

    def wait(self, eid: str, timeout: float | None = None) -> None:
        t = self._threads.get(eid)
        if t:
            t.join(timeout)

    # -- service operations --------------------------------------------------
    def start(self, req: StartWorkflowRequest, principal: Principal, background: bool = False) -> WorkflowRun:
        authorize_project(principal, req.project_id, "write")
        if req.mock_scenario and not self.settings.is_mock:
            raise ValueError("mock_scenario is only allowed in mock provider mode")
        eid = f"wf_{uuid.uuid4().hex[:16]}"
        request = req.model_dump(exclude={"mock_scenario"})
        run = self.store.create(WorkflowRun(execution_id=eid, project_id=req.project_id,
                                            requested_by=principal.user_id, request=request))
        self._emit(run, EventType.WORKFLOW_STARTED, "pending", requested_by=principal.user_id)
        state = initial_state(eid, req.project_id, principal.model_dump(), request, req.mock_scenario)
        self._launch(eid, state, background)
        return self.store.get(eid)

    def _authorized_run(self, eid: str, principal: Principal, action: str = "read") -> WorkflowRun:
        run = self.store.get(eid)
        authorize_project(principal, run.project_id, action)
        return run

    def resume(self, eid: str, payload: dict[str, Any], principal: Principal, background: bool = False) -> WorkflowRun:
        run = self._authorized_run(eid, principal, "write")
        if run.status != WorkflowStatus.AWAITING_INPUT or not run.pending_interrupt:
            raise WorkflowStateError(f"workflow is '{run.status.value}', not awaiting input")
        expected = run.pending_interrupt.get("type")
        if payload.get("type") != expected:
            raise WorkflowStateError(f"workflow is waiting for '{expected}', got '{payload.get('type')}'")
        if expected == "approval":
            authorize_project(principal, run.project_id, "approve")
            payload = {**payload, "reviewer": payload.get("reviewer") or principal.user_id}
        self._emit(run, EventType.WORKFLOW_RESUMED, "running", interrupt_type=expected,
                   decision=payload.get("decision"))
        self._launch(eid, Command(resume=payload), background)
        return self.store.get(eid)

    def retry(self, eid: str, principal: Principal, background: bool = False) -> WorkflowRun:
        """Failure recovery: continue a failed run from its last checkpoint."""
        run = self._authorized_run(eid, principal, "write")
        if run.status != WorkflowStatus.FAILED or (run.error or {}).get("recoverable") is False:
            raise WorkflowStateError(f"workflow is '{run.status.value}' and cannot be retried")
        self._emit(run, EventType.WORKFLOW_RESUMED, "running", retry=True)
        self._launch(eid, None, background)
        return self.store.get(eid)

    def cancel(self, eid: str, principal: Principal) -> WorkflowRun:
        run = self._authorized_run(eid, principal, "write")
        if run.status in TERMINAL_STATUSES:
            raise WorkflowStateError(f"workflow already '{run.status.value}'")
        run = self.store.update(eid, cancel_requested=True)
        if run.status in (WorkflowStatus.AWAITING_INPUT, WorkflowStatus.PENDING, WorkflowStatus.FAILED):
            run = self.store.update(eid, status=WorkflowStatus.CANCELLED, completed_at=utcnow_iso(),
                                    pending_interrupt=None)
            self._emit(run, EventType.WORKFLOW_CANCELLED, "cancelled")
        # RUNNING workflows stop at the next stage boundary (checked by every node).
        return run

    def _values(self, eid: str) -> dict[str, Any]:
        return dict(self.graph.get_state(thread_config(eid)).values or {})

    def status(self, eid: str, principal: Principal) -> WorkflowStatusResponse:
        run = self._authorized_run(eid, principal)
        v = self._values(eid)
        return WorkflowStatusResponse(
            execution_id=eid, project_id=run.project_id, status=run.status, current_stage=run.current_stage,
            pending_interrupt=run.pending_interrupt, final_status=v.get("final_status"),
            qa_iterations=v.get("qa_iterations", 0), stage_history=v.get("stage_history", []), error=run.error,
            created_at=run.created_at, updated_at=run.updated_at, completed_at=run.completed_at)

    def outputs(self, eid: str, principal: Principal) -> AgentOutputsResponse:
        run = self._authorized_run(eid, principal)
        v = self._values(eid)
        calls, tools = v.get("model_calls", []), v.get("tool_calls", [])
        usage = UsageSummary(
            model_calls=len(calls), failed_model_calls=sum(not c["success"] for c in calls),
            fallback_calls=sum(c["fallback_used"] for c in calls),
            validation_failures=sum(c["validation_failures"] for c in calls),
            input_tokens=sum(c["input_tokens"] for c in calls), output_tokens=sum(c["output_tokens"] for c in calls),
            model_latency_ms=round(sum(c["latency_ms"] for c in calls), 1),
            tool_calls=len(tools), failed_tool_calls=sum(not t["success"] for t in tools))
        return AgentOutputsResponse(
            execution_id=eid, status=run.status, classification=v.get("classification"), discovery=v.get("discovery"),
            structuring=v.get("structuring"), work_plan=v.get("work_plan", []),
            research_findings=v.get("research_findings", {}), financial=v.get("financial"),
            hypothesis_evaluation=v.get("hypothesis_evaluation"), strategy=v.get("strategy"), qa=v.get("qa"),
            qa_history=v.get("qa_history", []), approval=v.get("approval"), report=v.get("report"),
            report_markdown=v.get("report_markdown"), security_flags=v.get("security_flags", []),
            warnings=v.get("warnings", []), usage=usage)

    def recommendations(self, eid: str, principal: Principal) -> RecommendationsResponse:
        run = self._authorized_run(eid, principal)
        v = self._values(eid)
        recs = (v.get("report") or {}).get("strategic_recommendations") or (v.get("strategy") or {}).get(
            "recommendations", [])
        linked = []
        for r in recs:
            ids = r.get("evidence_ids", [])
            found, _ = sanitize_evidence(self.data.get_evidence(run.project_id, ids) if ids else [],
                                         self.settings.injection_policy)
            got = {e.evidence_id for e in found}
            linked.append(LinkedRecommendation(recommendation=r, evidence=found,
                                               missing_evidence_ids=[i for i in ids if i not in got]))
        approval = v.get("approval") or {}
        return RecommendationsResponse(execution_id=eid, status=run.status,
                                       approved=approval.get("decision") == "approve",
                                       qa_verdict=(v.get("qa") or {}).get("verdict"), recommendations=linked)

    def events(self, eid: str, principal: Principal) -> list[WorkflowEvent]:
        self._authorized_run(eid, principal)
        return self.bus.history(eid)
