from __future__ import annotations

from backend.evaluation.metrics import evidence_coverage, recommendation_relevance, unsupported_claim_rate
from backend.evaluation.run_eval import check_thresholds, load_cases, run_all


def test_metric_functions():
    recs = [{"title": "Cut promotion frequency", "evidence_ids": ["A"], "claim_type": "fact"},
            {"title": "Hire a mascot", "evidence_ids": ["GHOST"], "claim_type": "fact"},
            {"title": "Pilot pricing test", "evidence_ids": [], "claim_type": "hypothesis"}]
    assert recommendation_relevance(recs, ["promotion", "pricing"]) == 2 / 3
    assert evidence_coverage(recs, {"A"}, set()) == 1 / 3
    assert unsupported_claim_rate(recs, {"A"}, set()) == 1 / 3
    assert unsupported_claim_rate(recs[:1], {"A"}, {"A"}) == 1.0  # flagged evidence is not support


def test_evaluation_dataset_runs_end_to_end(harness):
    cases = load_cases()
    assert len(cases) >= 6 and any(c["expected"]["expect_clarification"] for c in cases)
    report = run_all(cases=cases, executor=harness.ex)
    agg = report["aggregate"]
    assert agg["workflow_success"] == 1.0 and agg["schema_validity"] == 1.0
    assert agg["clarification_correct"] == 1.0 and agg["financial_correct"] == 1.0
    assert agg["escalation_correct"] == 1.0
    esc = next(r for r in report["cases"] if r["id"] == "eval-06-qa-escalation")
    assert esc["metrics"]["unsupported_claim_rate"] > 0  # human-approved escalation still shows up in metrics
    assert check_thresholds({**agg, "unsupported_claim_rate": 0.5}) != []
