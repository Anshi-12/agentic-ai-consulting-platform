"""Allowlisted tools the agents can call.

Every call is: allowlist check -> argument validation (Pydantic) -> project
authorization -> per-agent and per-run budget check -> execution -> result
sanitization -> ToolCallRecord. There are no shell, filesystem or SQL tools,
and unknown tool names are rejected.

A tool result's `success` flag is the only source of truth for whether a
search or calculation happened; prompts receive the execution log and are
instructed never to claim otherwise.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from backend.ai import finance
from backend.ai.guardrails import get_logger, sanitize_evidence
from backend.ai.tracing import traced
from backend.contracts.auth import AuthorizationError, Principal, authorize_project
from backend.contracts.data_module import DataModuleClient
from backend.contracts.persistence import RunNotFoundError, WorkflowStore

log = get_logger("tools")

_ID = r"^[A-Za-z0-9_\-]{1,64}$"


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class _Args(BaseModel):
    model_config = {"extra": "forbid"}


def _clean_query(v: str) -> str:
    v = v.strip()
    if any(ord(c) < 32 and c not in "\t" for c in v):
        raise ValueError("query contains control characters")
    if "../" in v or "..\\" in v:
        raise ValueError("query looks like a path")
    return v


class QueryResearchArgs(_Args):
    project_id: str = Field(pattern=_ID)
    query: str = Field(min_length=2, max_length=300)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        return _clean_query(v)


class RetrieveDocumentsArgs(QueryResearchArgs):
    pass


class NumericalAnalysisArgs(_Args):
    project_id: str = Field(pattern=_ID)
    analysis: Literal["scenario_metrics"] = "scenario_metrics"


class ProjectContextArgs(_Args):
    project_id: str = Field(pattern=_ID)


class WorkflowStateArgs(_Args):
    execution_id: str = Field(pattern=_ID)


class SubmitForReviewArgs(_Args):
    execution_id: str = Field(pattern=_ID)
    artifact_type: Literal["strategy", "report", "financial", "qa_escalation"]
    summary: str = Field(min_length=1, max_length=2000)
    severity: Literal["routine", "high_impact"] = "routine"


class EvidenceByIdArgs(_Args):
    project_id: str = Field(pattern=_ID)
    evidence_ids: list[str] = Field(min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Records & errors
# ---------------------------------------------------------------------------
class ToolDenied(Exception):
    def __init__(self, reason: str, tool: str):
        super().__init__(f"tool '{tool}' denied: {reason}")
        self.reason = reason
        self.tool = tool


@dataclass
class ToolResult:
    call_id: str
    tool: str
    success: bool
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    security_flags: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    call_id: str
    agent: str
    tool: str
    args: dict[str, Any]
    success: bool
    error: str | None
    latency_ms: float
    result_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReviewQueue:
    """PLACEHOLDER: where `submit_for_review` lands until the shared review service exists."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def submit(self, item: dict[str, Any]) -> str:
        review_id = f"RV-{uuid.uuid4().hex[:8]}"
        self.items.append({"review_id": review_id, **item})
        return review_id


# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------
TOOL_SPECS: dict[str, type[_Args]] = {
    "query_research_results": QueryResearchArgs,
    "retrieve_documents": RetrieveDocumentsArgs,
    "request_numerical_analysis": NumericalAnalysisArgs,
    "read_project_context": ProjectContextArgs,
    "check_workflow_state": WorkflowStateArgs,
    "submit_for_review": SubmitForReviewArgs,
    "get_evidence_by_ids": EvidenceByIdArgs,
}

AGENT_TOOL_ALLOWLIST: dict[str, set[str]] = {
    "client_discovery": {"read_project_context", "retrieve_documents"},
    "problem_structuring": {"read_project_context", "query_research_results", "retrieve_documents"},
    "research": {"query_research_results", "retrieve_documents"},
    "financial_scenario": {"request_numerical_analysis", "read_project_context"},
    "strategy": {"query_research_results", "retrieve_documents", "get_evidence_by_ids", "read_project_context"},
    "consulting_qa": {"get_evidence_by_ids", "check_workflow_state", "submit_for_review"},
    "report_synthesis": {"get_evidence_by_ids", "read_project_context", "submit_for_review"},
    "orchestrator": {"check_workflow_state", "submit_for_review"},
}


@dataclass
class ToolContext:
    execution_id: str
    project_id: str
    principal: Principal
    agent: str
    run_calls_so_far: int = 0


class ToolExecutor:
    def __init__(self, data_module: DataModuleClient, store: WorkflowStore, review_queue: ReviewQueue,
                 max_calls_per_agent: int = 12, max_calls_per_run: int = 80, injection_policy: str = "quarantine",
                 on_call: Callable[[ToolCallRecord, ToolContext], None] | None = None):
        self.data = data_module
        self.store = store
        self.reviews = review_queue
        self.max_per_agent = max_calls_per_agent
        self.max_per_run = max_calls_per_run
        self.injection_policy = injection_policy
        self.on_call = on_call

    def session(self, ctx: ToolContext) -> "ToolSession":
        return ToolSession(self, ctx)

    # -- implementations ------------------------------------------------------
    def _run(self, name: str, args: _Args, ctx: ToolContext) -> tuple[Any, list[dict[str, Any]]]:
        if name in {"query_research_results", "retrieve_documents"}:
            fn = self.data.query_research if name == "query_research_results" else self.data.retrieve_documents
            items, flags = sanitize_evidence(fn(args.project_id, args.query, args.top_k), self.injection_policy)
            return [i.model_dump() for i in items], flags
        if name == "get_evidence_by_ids":
            items, flags = sanitize_evidence(self.data.get_evidence(args.project_id, args.evidence_ids), self.injection_policy)
            return [i.model_dump() for i in items], flags
        if name == "read_project_context":
            ctx_data = self.data.get_project_context(args.project_id)
            if ctx_data is None:
                raise LookupError(f"no context for project '{args.project_id}'")
            return ctx_data, []
        if name == "request_numerical_analysis":
            inputs = self.data.get_financial_inputs(args.project_id)
            if not inputs:
                raise LookupError("no financial inputs available from the data module")
            scenarios, calc_log = finance.compute_scenarios(inputs)
            return {"currency": inputs.get("currency", "USD"),
                    "scenarios": [s.model_dump(mode="json") for s in scenarios],
                    "calculation_log": calc_log}, []
        if name == "check_workflow_state":
            if args.execution_id != ctx.execution_id:
                raise ToolDenied("cross-workflow access", name)
            run = self.store.get(args.execution_id)
            return {"status": run.status.value, "current_stage": run.current_stage,
                    "cancel_requested": run.cancel_requested}, []
        if name == "submit_for_review":
            if args.execution_id != ctx.execution_id:
                raise ToolDenied("cross-workflow access", name)
            rid = self.reviews.submit({"execution_id": args.execution_id, "project_id": ctx.project_id,
                                       "artifact_type": args.artifact_type, "summary": args.summary,
                                       "severity": args.severity, "submitted_by": ctx.agent})
            return {"review_id": rid}, []
        raise ToolDenied("not implemented", name)  # pragma: no cover


class ToolSession:
    """Per-agent-invocation wrapper enforcing allowlist and budgets."""

    def __init__(self, executor: ToolExecutor, ctx: ToolContext):
        self.ex = executor
        self.ctx = ctx
        self.records: list[ToolCallRecord] = []
        self.security_flags: list[dict[str, Any]] = []

    @property
    def calls_made(self) -> int:
        return len(self.records)

    @traced("tool.call", run_type="tool")
    def call(self, tool: str, **raw_args: Any) -> ToolResult:
        call_id = uuid.uuid4().hex[:10]
        start = time.perf_counter()
        try:
            if tool not in TOOL_SPECS:
                raise ToolDenied("unknown tool (not on allowlist)", tool)
            if tool not in AGENT_TOOL_ALLOWLIST.get(self.ctx.agent, set()):
                raise ToolDenied(f"agent '{self.ctx.agent}' is not allowed to use it", tool)
            if self.calls_made >= self.ex.max_per_agent:
                raise ToolDenied(f"per-agent tool budget ({self.ex.max_per_agent}) exhausted", tool)
            if self.ctx.run_calls_so_far + self.calls_made >= self.ex.max_per_run:
                raise ToolDenied(f"per-run tool budget ({self.ex.max_per_run}) exhausted", tool)
            try:
                args = TOOL_SPECS[tool].model_validate(raw_args)
            except ValidationError as exc:
                raise ToolDenied(f"invalid arguments: {exc.errors()[0].get('msg')}", tool) from exc
            pid = getattr(args, "project_id", None)
            if pid is not None:
                if pid != self.ctx.project_id:
                    raise ToolDenied("cross-project access", tool)
                authorize_project(self.ctx.principal, pid, "read")
            data, flags = self.ex._run(tool, args, self.ctx)
            result = ToolResult(call_id, tool, True, data, None, (time.perf_counter() - start) * 1000, flags)
        except (ToolDenied, AuthorizationError) as exc:
            result = ToolResult(call_id, tool, False, None, str(exc), (time.perf_counter() - start) * 1000)
        except (LookupError, RunNotFoundError, finance.FinanceInputError) as exc:
            result = ToolResult(call_id, tool, False, None, f"{type(exc).__name__}: {exc}",
                                (time.perf_counter() - start) * 1000)
        except Exception as exc:  # never leak stack traces into prompts
            log.exception("tool %s crashed", tool)
            result = ToolResult(call_id, tool, False, None, f"internal tool error ({type(exc).__name__})",
                                (time.perf_counter() - start) * 1000)

        summary = self._summarize(result)
        rec = ToolCallRecord(call_id, self.ctx.agent, tool, {k: v for k, v in raw_args.items() if k != "summary"},
                             result.success, result.error, round(result.latency_ms, 2), summary)
        self.records.append(rec)
        self.security_flags.extend(result.security_flags)
        if self.ex.on_call:
            self.ex.on_call(rec, self.ctx)
        return result

    @staticmethod
    def _summarize(r: ToolResult) -> str:
        if not r.success:
            return f"FAILED: {r.error}"
        if isinstance(r.data, list):
            ids = [d.get("evidence_id") for d in r.data if isinstance(d, dict)]
            return f"returned {len(r.data)} items {ids[:10]}"
        if isinstance(r.data, dict):
            return f"returned keys {sorted(r.data)[:8]}"
        return "ok"

    def execution_log(self) -> list[dict[str, Any]]:
        """Compact log passed to prompts so models know exactly what ran."""
        return [{"tool": r.tool, "success": r.success, "result": r.result_summary} for r in self.records]
