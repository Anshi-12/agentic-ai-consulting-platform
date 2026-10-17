"""Evaluation metrics for consulting workflow runs.

Metrics are deterministic and cheap so they can run in CI (mock mode) and on
live runs. `recommendation_relevance` is lexical by design; swap in an
LLM-as-judge evaluator in LangSmith for semantic scoring once live models are
configured.
"""
from __future__ import annotations

import re
from statistics import mean
from typing import Any

from pydantic import ValidationError

from backend.ai.structured_outputs import ReportOutput, StrategyOutput

_WORD = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def recommendation_relevance(recs: list[dict[str, Any]], keywords: list[str]) -> float:
    """Share of recommendations mentioning at least one expected topic keyword (prefix match)."""
    if not recs:
        return 0.0
    kws = [k.lower() for k in keywords]
    hits = 0
    for r in recs:
        words = _words(" ".join([r.get("title", ""), r.get("rationale", "")] +
                                [m.get("metric", "") for m in r.get("measurable_outcomes", [])]))
        if any(w.startswith(k[:6]) for k in kws for w in words):
            hits += 1
    return hits / len(recs)


def evidence_coverage(recs: list[dict[str, Any]], valid_ids: set[str], flagged_ids: set[str]) -> float:
    if not recs:
        return 0.0
    ok = sum(1 for r in recs if any(i in valid_ids and i not in flagged_ids for i in r.get("evidence_ids", [])))
    return ok / len(recs)


def unsupported_claim_rate(recs: list[dict[str, Any]], valid_ids: set[str], flagged_ids: set[str]) -> float:
    """Recommendations presented as fact without valid support, or citing non-existent/flagged evidence."""
    if not recs:
        return 0.0
    bad = 0
    for r in recs:
        ids = r.get("evidence_ids", [])
        invalid = [i for i in ids if i not in valid_ids or i in flagged_ids]
        unsupported_fact = r.get("claim_type") == "fact" and not [i for i in ids if i in valid_ids and i not in flagged_ids]
        if invalid or unsupported_fact:
            bad += 1
    return bad / len(recs)


def schema_validity(values: dict[str, Any]) -> float:
    """Share of model calls that validated first time, times final-artifact validity (0 or 1)."""
    calls = values.get("model_calls", [])
    first_time = mean([1.0 if c["validation_failures"] == 0 and c["success"] else 0.0 for c in calls]) if calls else 0.0
    try:
        if values.get("strategy"):
            StrategyOutput.model_validate(values["strategy"])
        if values.get("report"):
            ReportOutput.model_validate({k: v for k, v in values["report"].items() if k not in ("qa_status", "approval")})
        final_ok = 1.0
    except ValidationError:
        final_ok = 0.0
    return round(first_time * final_ok, 4)


def score_run(values: dict[str, Any], status: str, expected: dict[str, Any], valid_ids: set[str],
              clarification_seen: bool, escalation_seen: bool) -> dict[str, Any]:
    recs = (values.get("report") or {}).get("strategic_recommendations") or (values.get("strategy") or {}).get(
        "recommendations", [])
    flagged = {f["evidence_id"] for f in values.get("security_flags", []) if "evidence_id" in f}
    calls = values.get("model_calls", [])
    fin = values.get("financial") or {}
    result = {
        "workflow_success": 1.0 if status == "completed" else 0.0,
        "recommendation_relevance": round(recommendation_relevance(recs, expected.get("relevance_keywords", [])), 4),
        "evidence_coverage": round(evidence_coverage(recs, valid_ids, flagged), 4),
        "unsupported_claim_rate": round(unsupported_claim_rate(recs, valid_ids, flagged), 4),
        "schema_validity": schema_validity(values),
        "clarification_correct": 1.0 if clarification_seen == expected.get("expect_clarification", False) else 0.0,
        "financial_correct": 1.0 if bool(fin.get("data_available")) == expected.get("expect_financial", True) else 0.0,
        "escalation_correct": 1.0 if escalation_seen == expected.get("expect_escalation", False) else 0.0,
        "qa_iterations": values.get("qa_iterations", 0),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "model_latency_ms": round(sum(c["latency_ms"] for c in calls), 1),
        "model_calls": len(calls),
        "fallback_calls": sum(c["fallback_used"] for c in calls),
    }
    result["meets_evidence_threshold"] = result["evidence_coverage"] >= expected.get("min_evidence_coverage", 0.0)
    return result


AGGREGATE_KEYS = ["workflow_success", "recommendation_relevance", "evidence_coverage", "unsupported_claim_rate",
                  "schema_validity", "clarification_correct", "financial_correct", "escalation_correct"]


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    agg = {k: round(mean(r["metrics"][k] for r in results), 4) for k in AGGREGATE_KEYS}
    agg["total_input_tokens"] = sum(r["metrics"]["input_tokens"] for r in results)
    agg["total_output_tokens"] = sum(r["metrics"]["output_tokens"] for r in results)
    agg["mean_wall_time_s"] = round(mean(r["wall_time_s"] for r in results), 3)
    agg["cases"] = len(results)
    return agg
