"""LangGraph consulting workflow.

    intake ──► clarification (human) ──► intake
       │
       ▼
    problem_framing ──► work_plan ══► research_task × N  ┐ (parallel, dependency waves)
                                  ══► financial_analysis ┘
                                          │
                                     research_join ══► (next wave) ...
                                          │
                                 hypothesis_evaluation ──► strategy ──► qa
                                                             ▲           │
              prepare_rerun ◄── fail: research/financial ────┼───────────┤
              strategy      ◄── fail: strategy ──────────────┘           │
              problem_framing ◄ fail: structuring                         │
                                                                         ▼
                                  human_approval (always a human; also on escalation)
                                     │ approve        │ revise        │ reject
                                     ▼                ▼               ▼
                                 synthesis ─► finalize ─► END    strategy     rejected ─► END
"""
from __future__ import annotations

import contextvars
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from backend.agents.base import AgentContext, AgentResult, AgentRuntime
from backend.agents.client_discovery_agent import ClientDiscoveryAgent
from backend.agents.consulting_qa_agent import ConsultingQAAgent
from backend.agents.financial_scenario_agent import FinancialScenarioAgent
from backend.agents.problem_structuring_agent import ProblemStructuringAgent
from backend.agents.report_synthesis_agent import ReportSynthesisAgent
from backend.agents.research import ResearchTaskRunner
from backend.agents.strategy_agent import StrategyAgent
from backend.ai.guardrails import get_logger
from backend.ai.model_router import ModelCallFailed
from backend.ai.tracing import traced
from backend.config import Settings
from backend.contracts.auth import Principal
from backend.contracts.events import EventBus, EventType, WorkflowEvent
from backend.contracts.persistence import WorkflowStore
from backend.contracts.schemas import utcnow_iso
from backend.orchestration import routing
from backend.orchestration.state import ConsultingState

log = get_logger("graph")


class WorkflowCancelled(Exception):
    pass


class AgentTimeout(Exception):
    def __init__(self, stage: str, seconds: float):
        super().__init__(f"stage '{stage}' exceeded {seconds:.0f}s timeout")
        self.stage = stage


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, AgentTimeout):
        return True
    if isinstance(exc, ModelCallFailed):
        return exc.retryable
    return False


@dataclass
class GraphServices:
    settings: Settings
    runtime: AgentRuntime
    store: WorkflowStore
    bus: EventBus

    def emit(self, state: dict[str, Any], type_: str, stage: str | None = None, status: str | None = None,
             message: str | None = None, **payload: Any) -> None:
        self.bus.publish(WorkflowEvent(execution_id=state["execution_id"], project_id=state["project_id"],
                                       type=type_, stage=stage, status=status, message=message, payload=payload))


# A single shared pool runs agent bodies so timeouts can be enforced.
_POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="agent")


def _run_with_timeout(fn: Callable[[], Any], seconds: float, stage: str) -> Any:
    ctx = contextvars.copy_context()  # keep tracing context inside the worker thread
    future = _POOL.submit(ctx.run, fn)
    try:
        return future.result(timeout=seconds)
    except FutureTimeout as exc:
        future.cancel()  # best effort; a running thread finishes in the background and its result is discarded
        raise AgentTimeout(stage, seconds) from exc


def _ctx(state: dict[str, Any]) -> AgentContext:
    return AgentContext(execution_id=state["execution_id"], project_id=state["project_id"],
                        principal=Principal.model_validate(state["principal"]),
                        run_tool_calls=state.get("tool_call_count", len(state.get("tool_calls", []))),
                        mock_scenario=state.get("mock_scenario"))


def _audit(result: AgentResult) -> dict[str, Any]:
    return {"model_calls": result.model_calls, "tool_calls": result.tool_calls,
            "security_flags": result.security_flags, "warnings": [f"[{result.agent}] {w}" for w in result.warnings]}


def build_graph(svc: GraphServices) -> StateGraph:
    rt = svc.runtime
    s = svc.settings
    discovery = ClientDiscoveryAgent(rt)
    structuring = ProblemStructuringAgent(rt)
    research = ResearchTaskRunner(rt)
    financial = FinancialScenarioAgent(rt)
    strategy = StrategyAgent(rt)
    qa = ConsultingQAAgent(rt)
    synthesis = ReportSynthesisAgent(rt)

    # -- node wrapper: cancellation, timeout, events, audit trail ---------------
    def stage_node(stage: str, agent_name: str, body: Callable[[dict[str, Any]], dict[str, Any]],
                   parallel: bool = False) -> Callable[[dict[str, Any]], dict[str, Any]]:
        @traced(f"node.{stage}")
        def node(state: dict[str, Any]) -> dict[str, Any]:
            eid = state["execution_id"]
            if svc.store.is_cancel_requested(eid):
                raise WorkflowCancelled(eid)
            if not parallel:
                svc.store.update(eid, current_stage=stage)
            started, t0 = utcnow_iso(), time.perf_counter()
            svc.emit(state, EventType.STAGE_STARTED, stage, "running")
            try:
                update = _run_with_timeout(lambda: body(state), s.timeout_for(agent_name), stage)
            except Exception as exc:
                svc.emit(state, EventType.STAGE_FAILED, stage, "failed", str(exc)[:300],
                         error_type=type(exc).__name__, retryable=is_retryable(exc))
                raise
            for call in update.get("model_calls", []):
                svc.emit(state, EventType.MODEL_CALL, stage, "ok" if call["success"] else "failed",
                         task=call["task"], model=call["model_used"], fallback_used=call["fallback_used"],
                         latency_ms=round(call["latency_ms"], 1), tokens=call["input_tokens"] + call["output_tokens"],
                         validation_failures=call["validation_failures"])
            for flag in update.get("security_flags", []):
                svc.emit(state, EventType.SECURITY_FLAG, stage, "flagged", evidence_id=flag.get("evidence_id"),
                         patterns=flag.get("patterns"))
            duration = round((time.perf_counter() - t0) * 1000, 1)
            svc.emit(state, EventType.STAGE_COMPLETED, stage, "completed", duration_ms=duration)
            update.setdefault("stage_history", []).append(
                {"stage": stage, "started_at": started, "completed_at": utcnow_iso(), "duration_ms": duration})
            return update
        return node

    def emit_route(state: dict[str, Any], frm: str, to: Any) -> None:
        target = to if isinstance(to, str) else [x.node for x in to]
        svc.emit(state, EventType.ROUTING_DECISION, frm, None, f"{frm} -> {target}", target=target)

    # -- node bodies -----------------------------------------------------------
    def intake_body(state):
        r = discovery.run(_ctx(state), state["request"], state.get("clarification_answers"))
        return {"classification": r.output["classification"], "discovery": r.output["discovery"],
                "project_context": r.output["project_context"], **_audit(r)}

    def clarification_node(state):
        d = state.get("discovery") or {}
        svc.store.update(state["execution_id"], current_stage="clarification")
        answer = interrupt({"type": "clarification", "questions": d.get("clarifying_questions", []),
                            "missing_information": d.get("missing_information", [])})
        answers = dict(state.get("clarification_answers") or {})
        if isinstance(answer, dict):
            answers.update(answer.get("answers") or {})
        else:
            answers[f"round_{state.get('clarification_rounds', 0) + 1}"] = str(answer)
        return {"clarification_answers": answers, "clarification_rounds": state.get("clarification_rounds", 0) + 1}

    def framing_body(state):
        r = structuring.structure(_ctx(state), state["discovery"], state.get("classification", {}),
                                  state.get("project_context", {}))
        return {"structuring": r.output["structuring"], **_audit(r)}

    def work_plan_node(state):
        plan = routing.build_work_plan(state["structuring"])
        svc.emit(state, EventType.STAGE_COMPLETED, "work_plan", "completed",
                 tasks=[t["task_id"] for t in plan],
                 dependencies={t["task_id"]: t["depends_on"] for t in plan})
        return {"work_plan": plan, "pending_task_ids": [t["task_id"] for t in plan],
                "research_round": state.get("research_round", 0) + 1, "research_requests": {},
                "stage_history": [{"stage": "work_plan", "started_at": utcnow_iso(), "completed_at": utcnow_iso(),
                                   "duration_ms": 0.0}]}

    def research_body(payload):
        r = research.run(_ctx(payload), payload["workstream"], payload.get("extra_questions"))
        task = payload["task"]
        return {"research_findings": {task["workstream_id"]: r.output},
                "completed_tasks": [f"{payload['research_round']}:{task['task_id']}"], **_audit(r)}

    def financial_body(payload):
        r = financial.run(_ctx(payload))
        return {"financial": r.output["financial"],
                "completed_tasks": [f"{payload['research_round']}:{payload['task']['task_id']}"], **_audit(r)}

    def research_join(state):
        return {}

    def hypothesis_body(state):
        r = structuring.evaluate_hypotheses(_ctx(state), state["structuring"], state.get("research_findings", {}))
        return {"hypothesis_evaluation": r.output["hypothesis_evaluation"], **_audit(r)}

    def strategy_body(state):
        r = strategy.run(_ctx(state), state["discovery"]["problem_statement"], state.get("hypothesis_evaluation", {}),
                         state.get("research_findings", {}), state.get("financial"),
                         state.get("revision_feedback"), state.get("revision_round", 0))
        return {"strategy": r.output["strategy"], **_audit(r)}

    def qa_body(state):
        iteration = state.get("qa_iterations", 0) + 1
        r = qa.run(_ctx(state), state["discovery"]["problem_statement"], state["strategy"], state.get("financial"),
                   state.get("hypothesis_evaluation", {}), state.get("research_findings", {}),
                   state.get("security_flags", []), iteration)
        out = r.output["qa"]
        svc.emit(state, EventType.QA_RESULT, "qa", out["verdict"], out["summary"][:300],
                 iteration=iteration, route_to=out["route_to"],
                 issues=[{"severity": i["severity"], "category": i["category"]} for i in out["issues"]])
        update = {"qa": out, "qa_iterations": iteration,
                  "qa_history": [{"iteration": iteration, "verdict": out["verdict"], "route_to": out["route_to"],
                                  "issue_count": len(out["issues"]), "at": utcnow_iso()}], **_audit(r)}
        if out["verdict"] != "pass":
            update["revision_feedback"] = [f"[QA {i['severity']}] {i['description']}" for i in out["issues"]]
            update["revision_round"] = state.get("revision_round", 0) + 1
        return update

    def prepare_rerun(state):
        targets, requests = routing.rerun_plan(state)
        svc.emit(state, EventType.ROUTING_DECISION, "prepare_rerun", None, f"re-running {targets}", tasks=targets)
        return {"pending_task_ids": targets, "research_requests": requests,
                "research_round": state.get("research_round", 0) + 1}

    def human_approval(state):
        q = state.get("qa") or {}
        escalated = q.get("verdict") != "pass"
        svc.store.update(state["execution_id"], current_stage="human_approval")
        decision = interrupt({
            "type": "approval",
            "escalated": escalated,
            "reason": ("QA escalated or exhausted rework attempts; explicit human decision required"
                       if escalated else "Deliverable passed QA; approval required before release"),
            "qa_verdict": q.get("verdict"),
            "qa_summary": q.get("summary"),
            "qa_issues": q.get("issues", []),
            "low_confidence_findings": q.get("low_confidence_findings", []),
            "recommendations": (state.get("strategy") or {}).get("recommendations", []),
            "allowed_decisions": ["approve", "revise", "reject"],
        })
        decision = decision if isinstance(decision, dict) else {"decision": str(decision)}
        choice = decision.get("decision")
        if choice not in {"approve", "revise", "reject"}:
            choice = "reject"  # anything unrecognized is never treated as approval
        rounds = state.get("approval_rounds", 0) + 1
        if choice == "revise" and rounds > routing.MAX_APPROVAL_REVISIONS:
            choice = "reject"
        approval = {"decision": choice, "reviewer": decision.get("reviewer", "unknown"),
                    "comments": decision.get("comments", ""), "escalated": escalated, "decided_at": utcnow_iso()}
        update: dict[str, Any] = {"approval": approval, "approval_rounds": rounds}
        if choice == "revise":
            update["revision_feedback"] = [f"[Reviewer] {approval['comments'] or 'Revise recommendations'}"]
            update["revision_round"] = state.get("revision_round", 0) + 1
            update["qa_iterations"] = 0  # a human-requested revision gets a fresh QA budget
        return update

    def synthesis_body(state):
        r = synthesis.run(_ctx(state), state["discovery"]["problem_statement"], state["strategy"],
                          state.get("financial"), state.get("hypothesis_evaluation", {}),
                          state.get("research_findings", {}), state.get("qa", {}), state["approval"])
        return {"report": r.output["report"], "report_markdown": r.output["report_markdown"], **_audit(r)}

    def finalize(state):
        return {"final_status": "completed"}

    def rejected(state):
        return {"final_status": "rejected"}

    def insufficient(state):
        return {"final_status": "insufficient_information"}

    # -- wiring ---------------------------------------------------------------
    retry = RetryPolicy(max_attempts=max(1, s.node_max_attempts), initial_interval=s.backoff_base_s,
                        backoff_factor=2.0, max_interval=s.backoff_max_s, jitter=True, retry_on=is_retryable)

    g = StateGraph(ConsultingState)
    g.add_node("intake", stage_node("intake", "client_discovery", intake_body), retry_policy=retry)
    g.add_node("clarification", clarification_node)
    g.add_node("problem_framing", stage_node("problem_framing", "problem_structuring", framing_body), retry_policy=retry)
    g.add_node("work_plan", work_plan_node)
    g.add_node("research_task", stage_node("research", "research", research_body, parallel=True), retry_policy=retry)
    g.add_node("financial_analysis", stage_node("financial_analysis", "financial_scenario", financial_body, parallel=True), retry_policy=retry)
    g.add_node("research_join", research_join)
    g.add_node("hypothesis_evaluation", stage_node("hypothesis_evaluation", "problem_structuring", hypothesis_body), retry_policy=retry)
    g.add_node("strategy", stage_node("strategy", "strategy", strategy_body), retry_policy=retry)
    g.add_node("qa", stage_node("qa", "consulting_qa", qa_body), retry_policy=retry)
    g.add_node("prepare_rerun", prepare_rerun)
    g.add_node("human_approval", human_approval)
    g.add_node("synthesis", stage_node("synthesis", "report_synthesis", synthesis_body), retry_policy=retry)
    g.add_node("finalize", finalize)
    g.add_node("rejected", rejected)
    g.add_node("insufficient_information", insufficient)

    def _route(frm: str, fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        def inner(state):
            target = fn(state)
            emit_route(state, frm, target)
            return target
        return inner

    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", _route("intake", routing.route_after_intake),
                            ["problem_framing", "clarification", "insufficient_information"])
    g.add_edge("clarification", "intake")
    g.add_edge("problem_framing", "work_plan")
    dispatch_targets = ["research_task", "financial_analysis", "hypothesis_evaluation"]
    g.add_conditional_edges("work_plan", _route("work_plan", routing.dispatch), dispatch_targets)
    g.add_edge("research_task", "research_join")
    g.add_edge("financial_analysis", "research_join")
    g.add_conditional_edges("research_join", _route("research_join", routing.dispatch), dispatch_targets)
    g.add_edge("hypothesis_evaluation", "strategy")
    g.add_edge("strategy", "qa")
    g.add_conditional_edges("qa", _route("qa", lambda st: routing.route_after_qa(st, s.max_qa_iterations)),
                            ["human_approval", "prepare_rerun", "strategy", "problem_framing"])
    g.add_conditional_edges("prepare_rerun", _route("prepare_rerun", routing.dispatch), dispatch_targets)
    g.add_conditional_edges("human_approval", _route("human_approval", routing.route_after_approval),
                            ["synthesis", "strategy", "rejected"])
    g.add_edge("synthesis", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("rejected", END)
    g.add_edge("insufficient_information", END)
    return g
