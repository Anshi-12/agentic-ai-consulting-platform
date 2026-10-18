from __future__ import annotations

import pytest

from backend.agents.base import AgentContext
from backend.agents.client_discovery_agent import ClientDiscoveryAgent
from backend.agents.consulting_qa_agent import ConsultingQAAgent, choose_route
from backend.agents.financial_scenario_agent import FinancialScenarioAgent, parse_money, unverified_amounts
from backend.agents.problem_structuring_agent import ProblemStructuringAgent
from backend.agents.report_synthesis_agent import ReportSynthesisAgent
from backend.agents.research import ResearchTaskRunner
from backend.agents.strategy_agent import StrategyAgent, weighted_score
from backend.ai import finance
from backend.ai.mock_responses import consulting_qa as default_qa, strategy as default_strategy
from backend.ai.structured_outputs import QAIssue
from backend.tests.conftest import GOOD_BRIEF, VAGUE_BRIEF


def ctx(principal, project="proj-northpeak", scenario=None):
    return AgentContext("wf_test", project, principal, 0, scenario)


@pytest.fixture
def rt(harness):
    return harness.ex.services.runtime


# -- A. discovery ----------------------------------------------------------------------
def test_discovery_complete(rt, principal):
    r = ClientDiscoveryAgent(rt).run(ctx(principal), {"client_brief": GOOD_BRIEF})
    d = r.output["discovery"]
    assert d["status"] == "complete" and d["problem_statement"]["key_question"]
    assert d["problem_statement"]["success_metrics"] and d["engagement_scope"]["deliverables"]
    assert r.output["classification"]["problem_type"] == "profitability"
    assert [c["role"] for c in r.model_calls] == ["light", "primary"]
    assert r.tool_calls[0]["tool"] == "read_project_context" and r.tool_calls[0]["success"]


def test_discovery_asks_clarifying_questions_for_vague_brief(rt, principal):
    d = ClientDiscoveryAgent(rt).run(ctx(principal), {"client_brief": VAGUE_BRIEF}).output["discovery"]
    assert d["status"] == "needs_clarification" and d["problem_statement"] is None
    assert len(d["clarifying_questions"]) >= 2 and d["missing_information"]


# -- B. structuring ----------------------------------------------------------------------
def _discovery(rt, principal):
    return ClientDiscoveryAgent(rt).run(ctx(principal), {"client_brief": GOOD_BRIEF}).output


def test_structuring_produces_consistent_tree(rt, principal):
    d = _discovery(rt, principal)
    s = ProblemStructuringAgent(rt).structure(ctx(principal), d["discovery"], d["classification"], {}).output["structuring"]
    ws_ids = {w["id"] for w in s["workstreams"]}
    assert all(h["workstream_id"] in ws_ids for h in s["hypotheses"])
    assert all(w["search_queries"] for w in s["workstreams"])
    assert s["issue_tree"]["children"] and s["frameworks"]


def test_hypothesis_eval_downgrades_unsupported_and_drops_fake_ids(make_harness, principal):
    def responder(req):
        return {"evaluations": [
            {"hypothesis_id": "H1", "verdict": "supported", "evidence_ids": ["EV-NOPE"], "rationale": "x", "confidence": 0.9},
            {"hypothesis_id": "H2", "verdict": "supported", "evidence_ids": ["EV-NP009"], "rationale": "x", "confidence": 0.9}]}
    rt = make_harness({"hypothesis_evaluation": responder}).ex.services.runtime
    structuring = {"hypotheses": [{"id": "H1", "statement": "s", "workstream_id": "WS1"},
                                  {"id": "H2", "statement": "s", "workstream_id": "WS1"}]}
    flagged = {"evidence_id": "EV-NP009", "project_id": "proj-northpeak", "source": "f", "title": "t",
               "content": "[withheld]", "injection_suspected": True}
    out = ProblemStructuringAgent(rt).evaluate_hypotheses(ctx(principal), structuring,
                                                          {"WS1": {"evidence": [flagged]}})
    evals = {e["hypothesis_id"]: e for e in out.output["hypothesis_evaluation"]["evaluations"]}
    assert evals["H1"]["verdict"] == "inconclusive" and evals["H1"]["evidence_ids"] == []
    assert evals["H2"]["verdict"] == "inconclusive"  # flagged source cannot be sole support
    assert any("unknown evidence" in w for w in out.warnings)


def test_research_runner_uses_tools(rt, principal):
    ws = {"id": "WS1", "name": "Pricing", "search_queries": ["pricing promotion margin"]}
    out = ResearchTaskRunner(rt).run(ctx(principal), ws)
    assert out.output["status"] == "ok" and out.output["evidence"]
    assert {t["tool"] for t in out.tool_calls} == {"query_research_results", "retrieve_documents"}


# -- D. financial -------------------------------------------------------------------------
def test_finance_functions_exact():
    assert finance.roi(150, 100) == pytest.approx(0.5)
    assert finance.payback_period(100, [40, 40, 40]) == pytest.approx(2.5)
    assert finance.payback_period(100, [10, 10]) is None
    assert finance.npv(0.1, 100, [110]) == pytest.approx(0.0)
    assert finance.irr(100, [110]) == pytest.approx(0.10, abs=1e-5)
    assert finance.gross_margin(200, 120) == pytest.approx(0.4)
    assert finance.operating_margin(200, 120, 50) == pytest.approx(0.15)
    with pytest.raises(finance.FinanceInputError):
        finance.compute_scenarios({"scenarios": [{"name": "x", "investment": "lots", "annual_cash_flows": []}]})


def test_financial_agent_uses_calculated_values(rt, principal):
    out = FinancialScenarioAgent(rt).run(ctx(principal)).output["financial"]
    assert out["data_available"]
    promo = next(s for s in out["scenarios"] if s["scenario"].startswith("Promotion"))
    m = {x["name"]: x for x in promo["metrics"]}
    assert m["roi"]["value"] == pytest.approx(6.0833, abs=1e-4) and m["roi"]["value_type"] == "calculated"
    assert m["payback_years"]["value"] == pytest.approx(0.6)
    assert m["investment"]["value_type"] == "estimate"  # supplied input, not calculated
    assert out["interpretation"]["preferred_scenario"] in {s["scenario"] for s in out["scenarios"]}


def test_financial_agent_without_data_never_claims_results(rt, principal):
    out = FinancialScenarioAgent(rt).run(ctx(principal, "proj-medisupply"))
    fin = out.output["financial"]
    assert fin["data_available"] is False and fin["scenarios"] == []
    assert fin["interpretation"]["data_available"] is False
    assert out.tool_calls[0]["success"] is False


def test_financial_agent_flags_invented_numbers(make_harness, principal):
    def responder(req):
        return {"summary": "Option saves $42M per year.", "preferred_scenario": "Imaginary", "data_available": True,
                "confidence": 0.9}
    out = FinancialScenarioAgent(make_harness({"financial_scenario": responder}).ex.services.runtime).run(ctx(principal))
    interp = out.output["financial"]["interpretation"]
    assert interp["preferred_scenario"] is None and interp["confidence"] <= 0.5
    assert any("Unverified figures" in c for c in interp["caveats"])


def test_money_parsing():
    assert parse_money("$5.7M and $1,200,000 and $3k") == [5.7e6, 1.2e6, 3e3]
    assert unverified_amounts(["NPV $5.73M"], [5734635.61]) == []
    assert unverified_amounts(["NPV $9M"], [5734635.61]) == [9e6]


# -- C. strategy ---------------------------------------------------------------------------
def _analysis(rt, principal):
    fin = FinancialScenarioAgent(rt).run(ctx(principal)).output["financial"]
    r = ResearchTaskRunner(rt).run(ctx(principal), {"id": "WS1", "name": "Pricing",
                                                     "search_queries": ["pricing promotion margin"]})
    return fin, {"WS1": r.output}


def test_strategy_recomputes_weighted_scores_and_flags(make_harness, principal):
    def responder(req):
        out = default_strategy(req)
        out["recommendations"][1]["confidence"] = 0.3
        out["recommendations"][1]["low_confidence_flag"] = False
        return out
    rt = make_harness({"strategy": responder}).ex.services.runtime
    fin, findings = _analysis(rt, principal)
    out = StrategyAgent(rt).run(ctx(principal), {"key_question": "q"}, {"evaluations": []}, findings, fin)
    s = out.output["strategy"]
    assert all(o["weighted_score"] != 9.9 for o in s["options"])  # model arithmetic discarded
    crit = s["decision_criteria"]
    assert s["options"][0]["weighted_score"] == weighted_score(s["options"][0]["scores"], crit)
    assert s["options"] == sorted(s["options"], key=lambda o: o["weighted_score"], reverse=True)
    assert s["recommendations"][1]["low_confidence_flag"] is True


def test_weighted_score_normalizes():
    assert weighted_score({"a": 5, "b": 1}, [{"name": "a", "weight": 0.5}, {"name": "b", "weight": 0.5}]) == 3.0
    assert weighted_score({}, [{"name": "a", "weight": 1}]) is None


# -- E. QA -----------------------------------------------------------------------------------
def _qa_inputs(rt, principal, scenario=None, strategy_responder=None):
    fin, findings = _analysis(rt, principal)
    strat = StrategyAgent(rt).run(ctx(principal, scenario=scenario), {"key_question": "q"},
                                  {"evaluations": []}, findings, fin).output["strategy"]
    return fin, findings, strat


def test_qa_detects_fabricated_citation_and_overrides_llm_pass(rt, principal):
    fin, findings, strat = _qa_inputs(rt, principal, scenario="qa_failure_once")
    out = ConsultingQAAgent(rt).run(ctx(principal), {"key_question": "q"}, strat, fin, {"evaluations": []},
                                    findings, [], 1).output["qa"]
    assert out["verdict"] == "fail" and out["route_to"] == "strategy"
    assert any(i["category"] == "fabricated_citation" and i["source"] == "deterministic" for i in out["issues"])


def test_qa_detects_unsupported_fact_inconsistent_number_and_flagged_source(make_harness, principal):
    def responder(req):
        out = default_strategy(req)
        out["recommendations"][0]["evidence_ids"] = []
        out["recommendations"][0]["claim_type"] = "fact"
        out["recommendations"][1]["rationale"] = "This will save $99M immediately."
        out["recommendations"][1]["evidence_ids"] = ["EV-NP009"]
        return out
    rt = make_harness({"strategy": responder}).ex.services.runtime
    fin, findings, strat = _qa_inputs(rt, principal)
    out = ConsultingQAAgent(rt).run(ctx(principal), {"key_question": "q"}, strat, fin, {"evaluations": []},
                                    findings, [], 1).output["qa"]
    cats = {i["category"] for i in out["issues"]}
    assert {"unsupported_claim", "inconsistent_number", "security"} <= cats
    assert out["verdict"] == "fail"


def test_qa_escalation_submits_for_review(make_harness, principal):
    def responder(req):
        out = default_qa(req)
        out.update(verdict="escalate", requires_human_review=True, summary="Conflicting evidence on a high-impact decision")
        return out
    h = make_harness({"consulting_qa": responder})
    rt = h.ex.services.runtime
    fin, findings, strat = _qa_inputs(rt, principal)
    out = ConsultingQAAgent(rt).run(ctx(principal), {"key_question": "q"}, strat, fin, {"evaluations": []},
                                    findings, [], 1)
    assert out.output["qa"]["verdict"] == "escalate" and out.output["qa"]["requires_human_review"]
    assert h.ex.reviews.items and h.ex.reviews.items[0]["artifact_type"] == "qa_escalation"


def test_choose_route_prefers_most_severe_then_earliest_stage():
    issues = [QAIssue(id="1", severity="high", category="contradiction", description="d", route_to="strategy"),
              QAIssue(id="2", severity="high", category="missing_evidence", description="d", route_to="research"),
              QAIssue(id="3", severity="low", category="low_confidence", description="d", route_to="structuring")]
    assert choose_route(issues) == "research"
    assert choose_route([issues[2]]) == "none"


# -- F. synthesis ------------------------------------------------------------------------------
def test_synthesis_restores_approved_recs_and_verifies_citations(make_harness, principal):
    from backend.ai.mock_responses import report_synthesis as default_report

    def responder(req):
        out = default_report(req)
        out["strategic_recommendations"][0]["title"] = "Something the reviewer never approved"
        out["strategic_recommendations"] = out["strategic_recommendations"][:1]
        out["detailed_findings"][0]["evidence_ids"].append("EV-GHOST")
        return out
    rt = make_harness({"report_synthesis": responder}).ex.services.runtime
    fin, findings, strat = _qa_inputs(rt, principal)
    evals = {"evaluations": [{"hypothesis_id": "H1", "verdict": "supported", "evidence_ids": ["EV-NP001"],
                              "rationale": "r", "confidence": 0.8}]}
    qa = {"verdict": "escalate", "issues": [], "low_confidence_findings": []}
    approval = {"decision": "approve", "reviewer": "lead", "comments": "ship with caveats"}
    out = ReportSynthesisAgent(rt).run(ctx(principal), {"key_question": "q"}, strat, fin, evals, findings, qa, approval)
    rep = out.output["report"]
    assert [r["id"] for r in rep["strategic_recommendations"]] == [r["id"] for r in strat["recommendations"]]
    cited = {c["evidence_id"] for c in rep["citations"]}
    assert "EV-GHOST" not in cited and "EV-NP001" in cited
    assert any("QA verdict was 'escalate'" in x for x in rep["limitations"])
    assert "# " in out.output["report_markdown"] and "## Risk register" in out.output["report_markdown"]
