from __future__ import annotations

import time

import pytest

from backend.ai.mock_responses import client_discovery as default_discovery, consulting_qa as default_qa
from backend.contracts.auth import AuthorizationError, Principal
from backend.contracts.events import EventType
from backend.contracts.persistence import JsonFileWorkflowStore
from backend.contracts.schemas import WorkflowStatus
from backend.orchestration import routing
from backend.orchestration.execution import WorkflowStateError
from backend.tests.conftest import VAGUE_BRIEF


def event_types(h, eid):
    return [e.type for e in h.ex.bus.history(eid)]


# -- happy path ------------------------------------------------------------------------------
def test_full_workflow_end_to_end(harness, principal):
    run = harness.start(principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT and run.pending_interrupt["type"] == "approval"
    assert run.pending_interrupt["escalated"] is False
    run = harness.approve(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED

    v = harness.values(run.execution_id)
    stages = [s["stage"] for s in v["stage_history"]]
    for expected in ["intake", "problem_framing", "work_plan", "research", "financial_analysis",
                     "hypothesis_evaluation", "strategy", "qa", "synthesis"]:
        assert expected in stages
    assert stages.index("intake") < stages.index("problem_framing") < stages.index("strategy") < stages.index("qa")
    assert v["final_status"] == "completed" and v["report"]["approval"]["decision"] == "approve"
    assert v["report"]["citations"] and v["report_markdown"].startswith("# ")

    out = harness.ex.outputs(run.execution_id, principal)
    assert out.usage.model_calls >= 8 and out.usage.tool_calls > 10 and out.usage.failed_model_calls == 0
    st = harness.ex.status(run.execution_id, principal)
    assert st.completed_at and st.final_status == "completed"


def test_parallel_research_with_dependency_waves(harness, principal):
    run = harness.start(principal)
    v = harness.values(run.execution_id)
    rnd = v["research_round"]
    done = [c.split(":")[1] for c in v["completed_tasks"] if c.startswith(f"{rnd}:")]
    assert set(done) == {"T-WS1", "T-WS2", "T-WS3", "T-FIN"}
    assert done.index("T-WS3") > done.index("T-WS1")  # WS3 depends on WS1 -> second wave
    routes = [e.payload["target"] for e in harness.ex.bus.history(run.execution_id)
              if e.type == EventType.ROUTING_DECISION and e.stage in ("work_plan", "research_join")]
    assert sorted(routes[0]) == ["financial_analysis", "research_task", "research_task"]  # wave 1 in parallel
    assert routes[1] == ["research_task"] and routes[2] == "hypothesis_evaluation"
    assert set(v["research_findings"]) == {"WS1", "WS2", "WS3"}


# -- routing units --------------------------------------------------------------------------
def test_ready_tasks_and_cycle_guard():
    plan = [{"task_id": "T-A", "type": "research", "workstream_id": "A", "depends_on": []},
            {"task_id": "T-B", "type": "research", "workstream_id": "B", "depends_on": ["T-A"]}]
    state = {"work_plan": plan, "pending_task_ids": ["T-A", "T-B"], "research_round": 1, "completed_tasks": []}
    assert [t["task_id"] for t in routing.ready_tasks(state)] == ["T-A"]
    state["completed_tasks"] = ["1:T-A"]
    assert [t["task_id"] for t in routing.ready_tasks(state)] == ["T-B"]
    state["completed_tasks"] = ["0:T-A"]  # previous round does not count
    assert [t["task_id"] for t in routing.ready_tasks(state)] == ["T-A"]
    cyc = [{"task_id": "T-A", "type": "research", "workstream_id": "A", "depends_on": ["T-B"]},
           {"task_id": "T-B", "type": "research", "workstream_id": "B", "depends_on": ["T-A"]}]
    assert len(routing.ready_tasks({"work_plan": cyc, "pending_task_ids": ["T-A", "T-B"],
                                    "research_round": 1, "completed_tasks": []})) == 2


@pytest.mark.parametrize("qa,iters,expected", [
    ({"verdict": "pass"}, 1, "human_approval"),
    ({"verdict": "fail", "route_to": "strategy"}, 1, "strategy"),
    ({"verdict": "fail", "route_to": "research"}, 1, "prepare_rerun"),
    ({"verdict": "fail", "route_to": "financial"}, 2, "prepare_rerun"),
    ({"verdict": "fail", "route_to": "structuring"}, 1, "problem_framing"),
    ({"verdict": "fail", "route_to": "strategy"}, 3, "human_approval"),
    ({"verdict": "escalate", "route_to": "research"}, 1, "human_approval"),
])
def test_route_after_qa(qa, iters, expected):
    assert routing.route_after_qa({"qa": qa, "qa_iterations": iters}, max_qa_iterations=2) == expected


def test_rerun_plan_targets():
    plan = routing.build_work_plan({"workstreams": [{"id": "WS1"}, {"id": "WS2"}]})
    st = {"work_plan": plan, "qa": {"route_to": "research",
                                    "additional_research_requests": [{"workstream_id": "WS2", "question": "q?"}]}}
    assert routing.rerun_plan(st) == (["T-WS2"], {"WS2": ["q?"]})
    st["qa"] = {"route_to": "financial"}
    assert routing.rerun_plan(st)[0] == ["T-FIN"]


# -- clarification gate -----------------------------------------------------------------------
def test_clarification_gate_then_completion(harness, principal):
    run = harness.start(principal, brief=VAGUE_BRIEF)
    assert run.status == WorkflowStatus.AWAITING_INPUT and run.pending_interrupt["type"] == "clarification"
    assert run.pending_interrupt["questions"]
    run = harness.ex.resume(run.execution_id, {"type": "clarification",
                                               "answers": {"q1": "+3 points gross margin in 18 months"}}, principal)
    assert run.pending_interrupt["type"] == "approval"
    v = harness.values(run.execution_id)
    assert v["clarification_answers"]["q1"].startswith("+3") and v["clarification_rounds"] == 1


def test_insufficient_information_after_max_clarifications(make_harness, principal):
    def always_unclear(req):
        req.context["clarification_answers"] = {}
        return default_discovery(req)
    h = make_harness({"client_discovery": always_unclear})
    run = h.start(principal, brief=VAGUE_BRIEF)
    for _ in range(routing.MAX_CLARIFICATION_ROUNDS):
        run = h.ex.resume(run.execution_id, {"type": "clarification", "answers": {"q": "dunno"}}, principal)
    assert run.status == WorkflowStatus.FAILED and run.error["type"] == "InsufficientInformation"
    assert h.values(run.execution_id).get("structuring") is None


# -- QA loops & approval -----------------------------------------------------------------------
def test_qa_failure_routes_back_and_then_passes(harness, principal):
    run = harness.start(principal, scenario="qa_failure_once")
    v = harness.values(run.execution_id)
    assert [q["verdict"] for q in v["qa_history"]] == ["fail", "pass"]
    assert v["revision_round"] == 1 and any("do not exist" in f for f in v["revision_feedback"])
    assert run.pending_interrupt["escalated"] is False


def test_qa_never_self_approves_and_escalates_to_human(harness, principal):
    run = harness.start(principal, scenario="qa_always_fail")
    v = harness.values(run.execution_id)
    assert len(v["qa_history"]) == harness.ex.settings.max_qa_iterations + 1
    assert all(q["verdict"] == "fail" for q in v["qa_history"])
    assert run.status == WorkflowStatus.AWAITING_INPUT  # stops for a human; does not proceed on its own
    assert run.pending_interrupt["escalated"] is True and run.pending_interrupt["qa_issues"]
    assert v.get("report") is None
    run = harness.approve(run.execution_id, principal, comments="accept with caveats")
    rep = harness.values(run.execution_id)["report"]
    assert run.status == WorkflowStatus.COMPLETED
    assert any("QA verdict was 'fail'" in x for x in rep["limitations"])


def test_qa_route_to_research_reruns_tasks(make_harness, principal):
    calls = {"n": 0}

    def qa_once(req):
        calls["n"] += 1
        out = default_qa(req)
        if calls["n"] == 1:
            out.update(verdict="fail", route_to="research", summary="need more evidence",
                       issues=[{"id": "x", "severity": "high", "category": "missing_evidence",
                                "description": "Need carrier benchmarks", "route_to": "research"}],
                       additional_research_requests=[{"workstream_id": "WS2", "question": "carrier rate benchmark"}])
        return out
    h = make_harness({"consulting_qa": qa_once})
    run = h.start(principal)
    v = h.values(run.execution_id)
    assert v["research_round"] == 2 and v["pending_task_ids"] == ["T-WS2"]
    assert "2:T-WS2" in v["completed_tasks"] and "2:T-WS1" not in v["completed_tasks"]
    assert "carrier rate benchmark" in v["research_findings"]["WS2"]["queries"]
    assert [q["verdict"] for q in v["qa_history"]] == ["fail", "pass"]


def test_reject_ends_without_report(harness, principal):
    run = harness.start(principal)
    run = harness.approve(run.execution_id, principal, decision="reject", comments="not convinced")
    assert run.status == WorkflowStatus.REJECTED
    assert harness.values(run.execution_id).get("report") is None
    assert EventType.WORKFLOW_REJECTED in event_types(harness, run.execution_id)


def test_revise_loops_back_to_strategy_with_feedback(harness, principal):
    run = harness.start(principal)
    run = harness.approve(run.execution_id, principal, decision="revise", comments="Add a pricing pilot")
    assert run.status == WorkflowStatus.AWAITING_INPUT
    v = harness.values(run.execution_id)
    assert "Add a pricing pilot" in v["strategy"]["recommendations"][0]["rationale"]
    assert harness.approve(run.execution_id, principal).status == WorkflowStatus.COMPLETED


def test_resume_validation(harness, principal):
    run = harness.start(principal)
    with pytest.raises(WorkflowStateError):
        harness.ex.resume(run.execution_id, {"type": "clarification", "answers": {"a": "b"}}, principal)
    harness.approve(run.execution_id, principal)
    with pytest.raises(WorkflowStateError):
        harness.approve(run.execution_id, principal)


# -- cancellation, timeouts, failure recovery ----------------------------------------------------
def test_cancel_while_awaiting_input(harness, principal):
    run = harness.start(principal)
    run = harness.ex.cancel(run.execution_id, principal)
    assert run.status == WorkflowStatus.CANCELLED
    with pytest.raises(WorkflowStateError):
        harness.approve(run.execution_id, principal)
    with pytest.raises(WorkflowStateError):
        harness.ex.cancel(run.execution_id, principal)


def test_cancel_running_workflow_stops_at_next_stage(harness, principal):
    harness.faults.add("strategy", ["slow:0.6"])
    run = harness.start(principal, background=True)
    deadline = time.time() + 5
    while harness.ex.store.get(run.execution_id).current_stage != "strategy" and time.time() < deadline:
        time.sleep(0.02)
    harness.ex.cancel(run.execution_id, principal)
    harness.ex.wait(run.execution_id, 10)
    final = harness.ex.store.get(run.execution_id)
    assert final.status == WorkflowStatus.CANCELLED
    assert harness.values(run.execution_id).get("qa") is None  # QA never ran
    assert EventType.WORKFLOW_CANCELLED in event_types(harness, run.execution_id)


def test_agent_timeout_retries_then_fails_then_recovers(make_harness, principal):
    h = make_harness(settings_override={"agent_timeouts": {"strategy": 0.2}, "node_max_attempts": 2})
    h.faults.add("strategy", ["slow:0.5", "slow:0.5"])
    run = h.start(principal)
    assert run.status == WorkflowStatus.FAILED and run.error["type"] == "AgentTimeout"
    fails = [e for e in h.ex.bus.history(run.execution_id) if e.type == EventType.STAGE_FAILED]
    assert len(fails) == 2 and fails[0].payload["retryable"]  # bounded node retry
    run = h.ex.retry(run.execution_id, principal)  # resumes from the last checkpoint
    assert run.status == WorkflowStatus.AWAITING_INPUT
    stages = [s["stage"] for s in h.values(run.execution_id)["stage_history"]]
    assert stages.count("intake") == 1  # completed stages were not re-run


def test_provider_outage_uses_fallback_in_workflow(harness, principal):
    harness.faults.add("strategy", ["rate_limit"] * 3, model="primary")
    run = harness.start(principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT
    out = harness.ex.outputs(run.execution_id, principal)
    assert out.usage.fallback_calls == 1
    assert any(e.payload.get("fallback_used") for e in harness.ex.bus.history(run.execution_id)
               if e.type == EventType.MODEL_CALL)


def test_permanent_failure_is_normalized_and_recoverable(harness, principal):
    harness.faults.add("strategy", ["server"] * 20)
    run = harness.start(principal)
    assert run.status == WorkflowStatus.FAILED
    assert run.error["type"] == "ModelCallFailed" and run.error["cause"]["kind"] == "server"
    assert run.error["task"] == "strategy"
    harness.faults.clear()
    assert harness.ex.retry(run.execution_id, principal).status == WorkflowStatus.AWAITING_INPUT


def test_invalid_structured_output_repaired_in_workflow(harness, principal):
    harness.faults.add("problem_structuring", ["invalid_json"])
    run = harness.start(principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT
    assert harness.ex.outputs(run.execution_id, principal).usage.validation_failures == 1


# -- events, persistence, authorization -----------------------------------------------------------
def test_events_are_ordered_and_ws_compatible(harness, principal):
    run = harness.start(principal)
    events = harness.ex.bus.history(run.execution_id)
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert events[0].type == EventType.WORKFLOW_STARTED
    msg = events[-1].to_ws_message()
    assert msg["channel"] == f"workflow:{run.execution_id}" and msg["event"]["execution_id"] == run.execution_id
    types = set(event_types(harness, run.execution_id))
    assert {EventType.STAGE_STARTED, EventType.TOOL_INVOKED, EventType.MODEL_CALL, EventType.ROUTING_DECISION,
            EventType.QA_RESULT, EventType.SECURITY_FLAG, EventType.WORKFLOW_AWAITING_INPUT} <= types


def test_state_persists_across_executor_restart(make_harness, principal, tmp_path):
    kw = dict(settings_override={"checkpoint_backend": "sqlite", "checkpoint_sqlite_path": str(tmp_path / "cp.sqlite")})
    store_dir = str(tmp_path / "runs")
    h1 = make_harness(store=JsonFileWorkflowStore(store_dir), **kw)
    run = h1.start(principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT

    h2 = make_harness(store=JsonFileWorkflowStore(store_dir), **kw)  # simulated process restart
    assert h2.ex.status(run.execution_id, principal).status == WorkflowStatus.AWAITING_INPUT
    assert h2.approve(run.execution_id, principal).status == WorkflowStatus.COMPLETED
    assert h2.values(run.execution_id)["report"]


def test_project_level_authorization(harness, principal):
    run = harness.start(principal)
    outsider = Principal(user_id="eve", project_ids=["other"], roles=["approver"])
    with pytest.raises(AuthorizationError):
        harness.ex.status(run.execution_id, outsider)
    with pytest.raises(AuthorizationError):
        harness.ex.outputs(run.execution_id, outsider)
    consultant = Principal(user_id="bob", project_ids=["*"], roles=["consultant"])
    with pytest.raises(AuthorizationError):
        harness.approve(run.execution_id, consultant)  # only approvers can release deliverables
    with pytest.raises(AuthorizationError):
        harness.start(outsider)


def test_recommendations_are_evidence_linked(harness, principal):
    run = harness.start(principal)
    harness.approve(run.execution_id, principal)
    rec = harness.ex.recommendations(run.execution_id, principal)
    assert rec.approved and rec.qa_verdict == "pass"
    for item in rec.recommendations:
        assert {e.evidence_id for e in item.evidence} == set(item.recommendation["evidence_ids"])
        assert item.missing_evidence_ids == []


def test_mock_scenario_rejected_in_live_mode(make_harness, principal):
    h = make_harness(settings_override={"provider_mode": "live"})
    with pytest.raises(ValueError):
        h.start(principal, scenario="qa_failure_once")
