"""Hackathon multi-agent architecture (magentic orchestration) in LangGraph.

    Input ─► manager_plan ──(unclear)──► human_input ─► manager_plan
                 │ ledger: facts, plan, tasks
                 ▼
            dispatch ══► technology_agent ┐
                     ══► clinical_agent   ├─ parallel, each searches only its own knowledge base
                     ══► sec_agent        │
                     ══► corporate_responsibility_agent ┘
                                │
                         progress_check  ("is the request resolved?")
                           │ no: new tasks ─► dispatch           (bounded by CONSULT_MAX_ROUNDS)
                           │ need human ─► human_input ─► dispatch
                           ▼ yes
                      consolidate ─► [approval (optional)] ─► finalize ─► Result
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, Send, interrupt

from backend.agents.domain_agents import DOMAIN_AGENTS
from backend.agents.manager_agent import ManagerAgent
from backend.ai.tracing import traced
from backend.contracts.events import EventType
from backend.contracts.schemas import utcnow_iso
from backend.orchestration.graph import (
    GraphServices,
    WorkflowCancelled,
    _audit,
    _ctx,
    _run_with_timeout,
    is_retryable,
)
from backend.orchestration.state import merge_dicts

AGENT_NODES = {d: f"{d}_agent" for d in DOMAIN_AGENTS}


class ConsultState(TypedDict, total=False):
    execution_id: str
    project_id: str
    principal: dict[str, Any]
    mock_scenario: str | None
    question: str
    human_answers: list[str]
    routing: dict[str, Any]
    ledger: dict[str, Any]
    round: int
    agent_results: Annotated[dict[str, Any], merge_dicts]
    evidence: Annotated[dict[str, Any], merge_dicts]
    progress_history: Annotated[list[dict[str, Any]], operator.add]
    pending_human: dict[str, Any] | None
    result: dict[str, Any]
    result_markdown: str
    approval: dict[str, Any]
    final_status: str
    stage_history: Annotated[list[dict[str, Any]], operator.add]
    model_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    security_flags: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]


def initial_consult_state(execution_id: str, project_id: str, principal: dict[str, Any], question: str,
                          mock_scenario: str | None = None) -> ConsultState:
    return ConsultState(execution_id=execution_id, project_id=project_id, principal=principal, question=question,
                        mock_scenario=mock_scenario, human_answers=[], round=0,
                        ledger={"facts": [], "plan": [], "tasks": []}, pending_human=None)


def _task_status(result: dict[str, Any] | None) -> str:
    if result is None:
        return "pending"
    return "insufficient" if result["answer"]["insufficient_evidence"] else "completed"


def refresh_ledger(ledger: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for t in ledger.get("tasks", []):
        r = results.get(t["task_id"])
        t = {**t, "status": _task_status(r) if r or t["status"] == "pending" else t["status"]}
        if r:
            t["confidence"] = r["uncertainty"]["final_confidence"]
            t["summary"] = r["answer"]["answer"][:240]
        tasks.append(t)
    return {**ledger, "tasks": tasks}


def new_tasks(ledger: dict[str, Any], specs: list[dict[str, Any]], round_no: int) -> list[dict[str, Any]]:
    start = len(ledger.get("tasks", []))
    return [{"task_id": f"T{start + i + 1}", "agent": s["agent"], "instruction": s["instruction"],
             "search_queries": s.get("search_queries") or [], "company": s.get("company"), "year": s.get("year"),
             "rationale": s.get("rationale", ""), "status": "pending", "round": round_no}
            for i, s in enumerate(specs)]


def render_markdown(question: str, result: dict[str, Any], ledger: dict[str, Any]) -> str:
    a, q = result["answer"], result["quality"]
    lines = [f"# {a['headline']}", "", f"**Question:** {question}", "",
             f"**Confidence:** {q['overall_confidence']:.0%} ({q['confidence_level']}) · "
             f"**Agents consulted:** {', '.join(a['domains_consulted']) or 'none'}", "", a["summary"], "",
             "## Next actions"]
    for n in sorted(a["next_actions"], key=lambda x: x["priority"]):
        ev = ", ".join(n["evidence_ids"]) or n["claim_type"]
        lines.append(f"{n['priority']}. **{n['action']}** — {n['why']} "
                     f"_Measure:_ {n['metric']} → {n['target']} ({n['timeframe']}). Owner: {n['owner']}. "
                     f"[{ev}; confidence {n['confidence']:.0%}]")
    for title, key in [("Cross-domain insights", "cross_domain_insights"), ("Confirmed facts", "confirmed_facts"),
                       ("Hypotheses (unconfirmed)", "hypotheses"), ("Risks", "risks"),
                       ("Missing information", "missing_information")]:
        if a.get(key):
            lines += ["", f"## {title}"] + [f"- {x}" for x in a[key]]
    if a.get("uncertainty_note"):
        lines += ["", f"_Uncertainty: {a['uncertainty_note']}_"]
    lines += ["", "## Task and progress ledger", "| Task | Agent | Round | Status | Confidence |", "|---|---|---|---|---|"]
    for t in ledger.get("tasks", []):
        conf = f"{t['confidence']:.0%}" if t.get("confidence") is not None else "-"
        lines.append(f"| {t['task_id']} | {t['agent']} | {t['round']} | {t['status']} | {conf} |")
    lines += ["", "## Sources"]
    for c in result.get("citations", []):
        page = f", p.{c['page']}" if c.get("page") else ""
        lines.append(f"- [{c['evidence_id']}] {c['title']}{page} — {c['source']}")
    return "\n".join(lines)


def build_consulting_graph(svc: GraphServices) -> StateGraph:
    rt, s = svc.runtime, svc.settings
    manager = ManagerAgent(rt)
    agents = {d: cls(rt) for d, cls in DOMAIN_AGENTS.items()}

    def stage(name: str, timeout_key: str, body: Callable[[dict[str, Any]], dict[str, Any]], parallel=False):
        @traced(f"consult.{name}")
        def node(state):
            eid = state["execution_id"]
            if svc.store.is_cancel_requested(eid):
                raise WorkflowCancelled(eid)
            if not parallel:
                svc.store.update(eid, current_stage=name)
            started, t0 = utcnow_iso(), time.perf_counter()
            label = name if not parallel else f"{state['task']['agent']}_agent"
            svc.emit(state, EventType.STAGE_STARTED, label, "running",
                     task_id=state.get("task", {}).get("task_id") if parallel else None)
            try:
                update = _run_with_timeout(lambda: body(state), s.timeout_for(timeout_key), label)
            except Exception as exc:
                svc.emit(state, EventType.STAGE_FAILED, label, "failed", str(exc)[:300],
                         error_type=type(exc).__name__, retryable=is_retryable(exc))
                raise
            for call in update.get("model_calls", []):
                svc.emit(state, EventType.MODEL_CALL, label, "ok" if call["success"] else "failed", task=call["task"],
                         model=call["model_used"], fallback_used=call["fallback_used"],
                         latency_ms=round(call["latency_ms"], 1), tokens=call["input_tokens"] + call["output_tokens"])
            for flag in update.get("security_flags", []):
                svc.emit(state, EventType.SECURITY_FLAG, label, "flagged", evidence_id=flag.get("evidence_id"),
                         patterns=flag.get("patterns"))
            ms = round((time.perf_counter() - t0) * 1000, 1)
            svc.emit(state, EventType.STAGE_COMPLETED, label, "completed", duration_ms=ms)
            update.setdefault("stage_history", []).append({"stage": label, "started_at": started,
                                                           "completed_at": utcnow_iso(), "duration_ms": ms})
            return update
        return node

    # -- manager: plan ---------------------------------------------------------
    def plan_body(state):
        r = manager.plan(_ctx(state), state["question"], state.get("human_answers", []))
        routing = r.output["routing"]
        ledger = dict(state.get("ledger") or {"facts": [], "plan": [], "tasks": []})
        ledger["facts"] = list(dict.fromkeys(ledger.get("facts", []) + routing.get("known_facts", [])))
        ledger["plan"] = routing.get("plan", [])
        update: dict[str, Any] = {"routing": routing, **_audit(r)}
        if routing.get("needs_clarification"):
            update["pending_human"] = {"type": "clarification", "questions": routing["clarifying_questions"]}
        else:
            rnd = state.get("round", 0) + 1
            ledger["tasks"] = ledger.get("tasks", []) + new_tasks(ledger, routing["tasks"], rnd)
            update.update(round=rnd, pending_human=None)
        update["ledger"] = ledger
        svc.emit(state, EventType.ROUTING_DECISION, "manager_plan", None,
                 f"intent={routing['intent_category']} → {[t['agent'] for t in routing['tasks']] or 'none'}",
                 intent=routing["intent"], category=routing["intent_category"],
                 agents=[t["agent"] for t in routing["tasks"]], cross_domain=routing.get("is_cross_domain"),
                 needs_clarification=routing.get("needs_clarification"))
        return update

    # -- domain agents -------------------------------------------------------------
    def agent_body(domain):
        def body(payload):
            r = agents[domain].run(_ctx(payload), payload["task"], payload["question"])
            out = r.output
            return {"agent_results": {out["task_id"]: out}, "evidence": out["evidence"], **_audit(r)}
        return body

    def agents_join(state):
        return {}

    # -- manager: progress ledger ----------------------------------------------------
    def progress_body(state):
        ledger = refresh_ledger(state["ledger"], state.get("agent_results", {}))
        rnd = state.get("round", 1)
        r = manager.assess(_ctx(state), state["question"], ledger, state.get("agent_results", {}), rnd,
                           s.consult_max_rounds)
        a = r.output["assessment"]
        update: dict[str, Any] = {"progress_history": [{"round": rnd, "is_resolved": a["is_resolved"],
                                                        "reason": a["reason"], "need_human": a["need_human"],
                                                        "next_tasks": [t["agent"] for t in a["next_tasks"]],
                                                        "at": utcnow_iso()}], **_audit(r)}
        if a["need_human"] and not a["is_resolved"]:
            update["pending_human"] = {"type": "information", "questions": [a["question_for_human"]]}
        elif not a["is_resolved"] and a["next_tasks"] and rnd < s.consult_max_rounds:
            ledger["tasks"] = ledger["tasks"] + new_tasks(ledger, a["next_tasks"], rnd + 1)
            update["round"] = rnd + 1
            update["pending_human"] = None
        else:
            update["pending_human"] = None
        update["ledger"] = ledger
        svc.emit(state, EventType.QA_RESULT, "progress_check", "resolved" if a["is_resolved"] else "open",
                 a["reason"][:300], round=rnd, need_human=a["need_human"])
        return update

    # -- human participant ---------------------------------------------------------
    def human_input(state):
        ask = state.get("pending_human") or {"type": "information", "questions": ["Please add detail."]}
        svc.store.update(state["execution_id"], current_stage="human_input")
        reply = interrupt({"type": ask["type"], "stage": "human_input", "questions": ask["questions"],
                           "ledger": state.get("ledger"), "question": state["question"]})
        text = reply.get("answer", "") if isinstance(reply, dict) else str(reply)
        text = str(text)[:2000]
        ledger = dict(state["ledger"])
        ledger["facts"] = ledger.get("facts", []) + [f"Human participant: {text}"]
        update: dict[str, Any] = {"human_answers": state.get("human_answers", []) + [text], "ledger": ledger,
                                  "pending_human": {"type": ask["type"], "answered": True}}
        if ask["type"] == "information":  # follow up with the agents of the latest round, now with the new facts
            rnd = state.get("round", 1)
            last = [t for t in ledger["tasks"] if t["round"] == rnd] or ledger["tasks"][-1:]
            specs = [{**t, "instruction": f"{state['question']} Additional customer information: {text}"[:600]}
                     for t in last]
            ledger["tasks"] = ledger["tasks"] + new_tasks(ledger, specs, rnd + 1)
            update["round"] = rnd + 1
        return update

    # -- consolidation, approval, finalize ---------------------------------------------
    def consolidate_body(state):
        if not state.get("agent_results"):
            routing = state.get("routing", {})
            result = {"answer": {"headline": "This request is outside the available knowledge bases",
                                 "summary": "None of the specialist agents (technology, clinical, SEC, corporate "
                                            "responsibility) covers this request.",
                                 "domains_consulted": [], "next_actions": [], "cross_domain_insights": [], "risks": [],
                                 "confirmed_facts": [], "hypotheses": [], "missing_information": [],
                                 "overall_confidence": 0.0, "uncertainty_note": routing.get("intent", "")},
                      "citations": [], "quality": {"overall_confidence": 0.0, "confidence_level": "low",
                                                   "citation_validity": 1.0, "actions_with_evidence": 0.0,
                                                   "action_groundedness": 0.0, "actions_measurable": True,
                                                   "domains_consulted": [], "fabricated_citations_removed": 0}}
            return {"result": result, "result_markdown": render_markdown(state["question"], result, state["ledger"])}
        ledger = refresh_ledger(state["ledger"], state["agent_results"])
        r = manager.consolidate(_ctx(state), state["question"], state.get("routing", {}), state["agent_results"],
                                state.get("evidence", {}), state.get("human_answers", []))
        return {"result": r.output, "ledger": ledger,
                "result_markdown": render_markdown(state["question"], r.output, ledger), **_audit(r)}

    def approval(state):
        svc.store.update(state["execution_id"], current_stage="human_approval")
        decision = interrupt({"type": "approval", "stage": "human_approval", "escalated": False,
                              "reason": "Consulting recommendation ready; approval required before release",
                              "result": state.get("result"), "allowed_decisions": ["approve", "reject"]})
        decision = decision if isinstance(decision, dict) else {"decision": str(decision)}
        choice = decision.get("decision") if decision.get("decision") in ("approve", "reject") else "reject"
        return {"approval": {"decision": choice, "reviewer": decision.get("reviewer", "unknown"),
                             "comments": decision.get("comments", ""), "decided_at": utcnow_iso()}}

    def finalize(state):
        if (state.get("approval") or {}).get("decision") == "reject":
            return {"final_status": "rejected"}
        return {"final_status": "completed"}

    # -- routing ------------------------------------------------------------------------
    def dispatch(state):
        pending = [t for t in state["ledger"]["tasks"] if t["status"] == "pending"
                   and t["task_id"] not in state.get("agent_results", {})]
        if not pending:
            return "consolidate"
        return [Send(AGENT_NODES[t["agent"]], {
            "execution_id": state["execution_id"], "project_id": state["project_id"],
            "principal": state["principal"], "mock_scenario": state.get("mock_scenario"),
            "question": state["question"], "task": t, "tool_call_count": len(state.get("tool_calls", []))})
            for t in pending]

    def after_plan(state):
        if (state.get("pending_human") or {}).get("type") == "clarification":
            return "human_input"
        return dispatch(state)

    def after_progress(state):
        if (state.get("pending_human") or {}).get("type") == "information" and not state["pending_human"].get("answered"):
            return "human_input"
        return dispatch(state)

    def after_human(state):
        return "manager_plan" if state["pending_human"]["type"] == "clarification" else dispatch(state)

    def after_consolidate(state):
        return "approval" if s.consult_require_approval else "finalize"

    def logged(frm, fn):
        def inner(state):
            target = fn(state)
            svc.emit(state, EventType.ROUTING_DECISION, frm, None,
                     f"{frm} -> {target if isinstance(target, str) else [x.node for x in target]}",
                     target=target if isinstance(target, str) else [x.node for x in target])
            return target
        return inner

    retry = RetryPolicy(max_attempts=max(1, s.node_max_attempts), initial_interval=s.backoff_base_s,
                        backoff_factor=2.0, max_interval=s.backoff_max_s, jitter=True, retry_on=is_retryable)
    g = StateGraph(ConsultState)
    g.add_node("manager_plan", stage("manager_plan", "manager_agent", plan_body), retry_policy=retry)
    for d, node in AGENT_NODES.items():
        g.add_node(node, stage(node, node, agent_body(d), parallel=True), retry_policy=retry)
    g.add_node("agents_join", agents_join)
    g.add_node("progress_check", stage("progress_check", "manager_agent", progress_body), retry_policy=retry)
    g.add_node("human_input", human_input)
    g.add_node("consolidate", stage("consolidate", "manager_agent", consolidate_body), retry_policy=retry)
    g.add_node("approval", approval)
    g.add_node("finalize", finalize)

    targets = list(AGENT_NODES.values()) + ["consolidate"]
    g.add_edge(START, "manager_plan")
    g.add_conditional_edges("manager_plan", logged("manager_plan", after_plan), targets + ["human_input"])
    for node in AGENT_NODES.values():
        g.add_edge(node, "agents_join")
    g.add_edge("agents_join", "progress_check")
    g.add_conditional_edges("progress_check", logged("progress_check", after_progress), targets + ["human_input"])
    g.add_conditional_edges("human_input", logged("human_input", after_human), targets + ["manager_plan"])
    g.add_conditional_edges("consolidate", after_consolidate, ["approval", "finalize"])
    g.add_edge("approval", "finalize")
    g.add_edge("finalize", END)
    return g
