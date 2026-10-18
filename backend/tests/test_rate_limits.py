"""Free-tier rate limits: quota detection, provider-requested waits, request pacing, eval degrades gracefully."""
from __future__ import annotations

import dataclasses

import httpx

from backend.ai.model_router import ModelCallFailed, ModelRouter
from backend.ai.provider_factory import ProviderError, normalize_http_error
from backend.config import Settings

GEMINI_DAILY = ('{"error":{"code":429,"message":"You exceeded your current quota","details":[{"@type":'
                '"type.googleapis.com/google.rpc.QuotaFailure","violations":[{"quotaId":'
                '"GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},{"@type":'
                '"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"37s"}]}}')
GEMINI_MINUTE = GEMINI_DAILY.replace("PerDay", "PerMinute")


def _err(body: str, headers=None) -> ProviderError:
    return normalize_http_error("openai", "gemini", response=httpx.Response(429, text=body, headers=headers or {}))


def test_daily_quota_is_not_retried_and_minute_limit_is():
    daily, minute = _err(GEMINI_DAILY), _err(GEMINI_MINUTE)
    assert daily.kind == "quota_exhausted" and not daily.retryable
    assert minute.kind == "rate_limit" and minute.retryable
    assert minute.retry_after == 37.0
    assert _err("{}", {"retry-after": "12"}).retry_after == 12.0


class _Seq:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def get(self, ref):
        return self

    def complete(self, request):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        from backend.ai.provider_factory import ChatResponse
        return ChatResponse(text='{"x": 1}', provider="t", model="m")


def test_router_waits_as_long_as_provider_asks_and_stops_on_daily_quota():
    from pydantic import BaseModel

    class X(BaseModel):
        x: int

    s = Settings(provider_mode="live", model_primary="t:m", model_fallback="t:m", max_retries=3)
    slept = []
    fac = _Seq([_err(GEMINI_MINUTE)])
    out, _ = ModelRouter(s, fac, sleep=slept.append).call_structured("t", "sys", [{"role": "user", "content": "q"}], X)
    assert out.x == 1 and slept and slept[0] >= 37

    fac = _Seq([_err(GEMINI_DAILY)] * 5)
    try:
        ModelRouter(s, fac, sleep=slept.append).call_structured("t", "sys", [{"role": "user", "content": "q"}], X)
        raise AssertionError("expected failure")
    except ModelCallFailed as e:
        assert e.last_error.kind == "quota_exhausted"
    assert fac.calls == 1  # no pointless retries


def test_pacing_spaces_requests():
    s = dataclasses.replace(Settings(), min_request_interval_s=2.0)
    slept = []
    r = ModelRouter(s, sleep=slept.append)
    r._pace()
    r._pace()
    assert slept and 1.5 < slept[0] <= 2.0


def test_eval_routing_stops_cleanly_when_quota_runs_out():
    from backend.ai.model_router import CallRecord
    from backend.evaluation.hackathon_eval import routing_metrics, to_markdown

    cases = [{"id": f"r{i}", "question": "q", "domains": ["technology"]} for i in range(5)]
    n = {"i": 0}

    def predict(q):
        n["i"] += 1
        if n["i"] > 2:
            raise ModelCallFailed("manager_plan", _err(GEMINI_DAILY), CallRecord("c", "manager_plan", "primary"))
        return ["technology"], False

    m = routing_metrics(cases, predict)
    assert m["cases"] == 2 and m["exact_match_accuracy"] == 1.0 and "quota" in m["stopped"]
    none = routing_metrics(cases, lambda q: (_ for _ in ()).throw(
        ModelCallFailed("manager_plan", _err(GEMINI_DAILY), CallRecord("c", "manager_plan", "primary"))))
    assert none["cases"] == 0
    md = to_markdown({"provider_mode": "live", "retrieval_mode": "bm25", "routing": {"rule-based": m, "LLM": none},
                      "retrieval": {"full": {"cases": 1, "hit@1": 1, "hit@3": 1, "hit@5": 1, "mrr": 1}},
                      "products": {"cases": 1, "hit@1": 1, "hit@3": 1}})
    assert "n/a" in md and "stopped" in md
