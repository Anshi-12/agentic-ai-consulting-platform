"""Conditional routing and dependency-aware task dispatch.

Pure functions of state (plus settings), so they are easy to unit-test. The
graph wraps them to emit routing events.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from backend.orchestration.state import ConsultingState, WorkTask

MAX_CLARIFICATION_ROUNDS = 2
MAX_APPROVAL_REVISIONS = 3


def route_after_intake(state: ConsultingState) -> str:
    status = (state.get("discovery") or {}).get("status")
    if status == "complete":
        return "problem_framing"
    if state.get("clarification_rounds", 0) >= MAX_CLARIFICATION_ROUNDS:
        return "insufficient_information"
    return "clarification"


def build_work_plan(structuring: dict[str, Any]) -> list[WorkTask]:
    tasks: list[WorkTask] = []
    for ws in structuring.get("workstreams", []):
        tasks.append(WorkTask(task_id=f"T-{ws['id']}", type="research", workstream_id=ws["id"],
                              depends_on=[f"T-{d}" for d in ws.get("depends_on", [])]))
    tasks.append(WorkTask(task_id="T-FIN", type="financial", workstream_id=None, depends_on=[]))
    return tasks


def ready_tasks(state: ConsultingState) -> list[WorkTask]:
    """Pending tasks whose dependencies (within the pending set) completed in this round."""
    rnd = state.get("research_round", 0)
    done = {c.split(":", 1)[1] for c in state.get("completed_tasks", []) if c.startswith(f"{rnd}:")}
    pending = [t for t in state.get("work_plan", []) if t["task_id"] in set(state.get("pending_task_ids", []))
               and t["task_id"] not in done]
    pending_ids = {t["task_id"] for t in pending}
    ready = [t for t in pending if all(d in done or d not in pending_ids for d in t["depends_on"])]
    if pending and not ready:  # cycle guard: never deadlock
        ready = pending
    return ready


def dispatch_payload(state: ConsultingState, task: WorkTask) -> dict[str, Any]:
    ws = None
    if task["workstream_id"]:
        ws = next((w for w in state["structuring"]["workstreams"] if w["id"] == task["workstream_id"]), None)
    return {
        "execution_id": state["execution_id"],
        "project_id": state["project_id"],
        "principal": state["principal"],
        "mock_scenario": state.get("mock_scenario"),
        "research_round": state.get("research_round", 0),
        "task": task,
        "workstream": ws,
        "extra_questions": (state.get("research_requests") or {}).get(task["workstream_id"] or "", []),
        "tool_call_count": len(state.get("tool_calls", [])),
    }


def dispatch(state: ConsultingState) -> list[Send] | str:
    tasks = ready_tasks(state)
    if not tasks:
        return "hypothesis_evaluation"
    sends = []
    for t in tasks:
        node = "financial_analysis" if t["type"] == "financial" else "research_task"
        sends.append(Send(node, dispatch_payload(state, t)))
    return sends


def route_after_qa(state: ConsultingState, max_qa_iterations: int) -> str:
    qa = state.get("qa") or {}
    verdict = qa.get("verdict")
    if verdict == "pass":
        return "human_approval"
    if verdict == "escalate" or state.get("qa_iterations", 0) > max_qa_iterations:
        return "human_approval"  # escalated: a human decides; the workflow never self-approves
    target = qa.get("route_to", "strategy")
    return {"research": "prepare_rerun", "financial": "prepare_rerun", "structuring": "problem_framing"}.get(
        target, "strategy")


def route_after_approval(state: ConsultingState) -> str:
    decision = (state.get("approval") or {}).get("decision")
    if decision == "approve":
        return "synthesis"
    if decision == "revise":
        return "strategy"
    return "rejected"


def rerun_plan(state: ConsultingState) -> tuple[list[str], dict[str, list[str]]]:
    """Which tasks to re-run after a QA failure, and extra research questions per workstream."""
    qa = state.get("qa") or {}
    plan = state.get("work_plan", [])
    if qa.get("route_to") == "financial":
        return ["T-FIN"], {}
    requests: dict[str, list[str]] = {}
    for r in qa.get("additional_research_requests", []):
        requests.setdefault(r["workstream_id"], []).append(r["question"])
    valid_ws = {t["workstream_id"] for t in plan if t["type"] == "research"}
    targets = [f"T-{ws}" for ws in requests if ws in valid_ws]
    if not targets:
        targets = [t["task_id"] for t in plan if t["type"] == "research"]
    return targets, {k: v for k, v in requests.items() if k in valid_ws}
