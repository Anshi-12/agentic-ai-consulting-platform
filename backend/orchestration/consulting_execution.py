"""Executor for the hackathon multi-agent consulting graph.

Reuses the workflow engine (run store, events, checkpoints, cancel/resume/
retry, authorization) and swaps in the manager + domain-agent graph.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from backend.api.models import ConsultRequest, ConsultResultResponse, UsageSummary
from backend.config import Settings, get_settings
from backend.contracts.auth import Principal, authorize_project
from backend.contracts.events import EventType
from backend.contracts.schemas import WorkflowRun
from backend.orchestration.consulting_graph import build_consulting_graph, initial_consult_state
from backend.orchestration.execution import WorkflowExecutor
from backend.rag.index import KnowledgeBase, load_knowledge_base

ROOT = Path(__file__).resolve().parents[2]


def resolve_kb_dir(settings: Settings) -> Path:
    p = Path(settings.kb_dir)
    return p if p.is_absolute() else ROOT / p


class ConsultingExecutor(WorkflowExecutor):
    def __init__(self, settings: Settings | None = None, *, knowledge_base: KnowledgeBase | None = None,
                 load_kb: bool = True, **kw: Any):
        settings = settings or get_settings()
        kb = knowledge_base
        if kb is None and load_kb:
            kb = load_knowledge_base(resolve_kb_dir(settings))
        self.kb = kb
        super().__init__(settings, knowledge_base=kb, **kw)

    def _build_graph(self):
        return build_consulting_graph(self.services)

    def kb_status(self) -> dict[str, Any]:
        if self.kb is None:
            return {"built": False, "hint": "python scripts/setup_hackathon_data.py <Hackathon_export.zip> && "
                                            "python -m backend.rag.ingest"}
        return {"built": True, **self.kb.stats()}

    def ask(self, req: ConsultRequest, principal: Principal, background: bool = False) -> WorkflowRun:
        authorize_project(principal, req.project_id, "write")
        if req.mock_scenario and not self.settings.is_mock:
            raise ValueError("mock_scenario is only allowed in mock provider mode")
        eid = f"cs_{uuid.uuid4().hex[:16]}"
        run = self.store.create(WorkflowRun(execution_id=eid, project_id=req.project_id,
                                            requested_by=principal.user_id, request={"question": req.question}))
        self._emit(run, EventType.WORKFLOW_STARTED, "pending", requested_by=principal.user_id)
        state = initial_consult_state(eid, req.project_id, principal.model_dump(), req.question, req.mock_scenario)
        self._launch(eid, state, background)
        return self.store.get(eid)

    def result(self, eid: str, principal: Principal) -> ConsultResultResponse:
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
        summaries = {tid: {"agent": r["agent"], "instruction": r["instruction"], "answer": r["answer"],
                           "uncertainty": r["uncertainty"],
                           "retrieval": {"queries": r["retrieval"]["queries"],
                                         "product_matches": r["retrieval"]["product_matches"],
                                         "trial_analytics": r["retrieval"]["trial_analytics"]}}
                     for tid, r in (v.get("agent_results") or {}).items()}
        return ConsultResultResponse(
            execution_id=eid, status=run.status, current_stage=run.current_stage,
            pending_interrupt=run.pending_interrupt, question=v.get("question", run.request.get("question", "")),
            routing=v.get("routing"), ledger=v.get("ledger"), progress_history=v.get("progress_history", []),
            agent_results=summaries, result=v.get("result"), result_markdown=v.get("result_markdown"),
            stage_history=v.get("stage_history", []), security_flags=v.get("security_flags", []),
            warnings=v.get("warnings", []), usage=usage, error=run.error)
