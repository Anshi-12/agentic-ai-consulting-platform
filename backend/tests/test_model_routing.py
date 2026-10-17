from __future__ import annotations

import json
import random
from dataclasses import replace

import httpx
import pytest

from backend.ai.model_router import ModelCallFailed, ModelRouter
from backend.ai.provider_factory import (
    AnthropicProvider,
    ChatRequest,
    FaultPlan,
    OpenAIProvider,
    ProviderError,
    ProviderFactory,
    parse_model_ref,
)
from backend.ai.structured_outputs import RequestClassification, StructuredOutputError, extract_json
from backend.config import load_settings

SYSTEM = "classify"
MSGS = [{"role": "user", "content": "We sell outdoor gear and margin is falling"}]
CTX = {"brief": "margin is falling", "industry_hint": "Retail"}


def make_router(settings, faults, **settings_kw):
    s = replace(settings, **settings_kw)
    sleeps: list[float] = []
    router = ModelRouter(s, ProviderFactory(s, faults=faults), sleep=sleeps.append, rng=random.Random(1))
    return router, sleeps


def call(router):
    return router.call_structured("classify_request", SYSTEM, MSGS, RequestClassification, context=CTX)


# -- configuration ----------------------------------------------------------------
def test_parse_model_ref():
    assert parse_model_ref("anthropic:claude-sonnet-5-5") == ("anthropic", "claude-sonnet-5-5")
    assert parse_model_ref("openai:gpt-x") == ("openai", "gpt-x")
    with pytest.raises(ValueError):
        parse_model_ref("no-provider")
    with pytest.raises(ValueError):
        parse_model_ref("cohere:model")


def test_env_configuration_live_vs_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_MODE", "live")
    monkeypatch.setenv("MODEL_PRIMARY", "anthropic:claude-opus-5-5")
    monkeypatch.setenv("MODEL_LIGHT", "anthropic:claude-haiku-5-5")
    monkeypatch.setenv("MODEL_FALLBACK", "openai:gpt-fallback")
    monkeypatch.setenv("AGENT_TIMEOUT_STRATEGY", "42")
    s = load_settings()
    assert s.model_primary == "anthropic:claude-opus-5-5"
    assert s.model_light == "anthropic:claude-haiku-5-5"
    assert s.model_fallback == "openai:gpt-fallback"
    assert s.model_synthesis == "anthropic:claude-opus-5-5"  # unset role inherits primary, never a mock
    assert s.timeout_for("strategy") == 42 and s.timeout_for("consulting_qa") == s.agent_timeout_s
    monkeypatch.setenv("LLM_PROVIDER_MODE", "mock")
    assert load_settings().model_primary == "mock:primary"
    monkeypatch.setenv("LLM_PROVIDER_MODE", "live")
    monkeypatch.delenv("MODEL_PRIMARY")
    with pytest.raises(ValueError):
        load_settings()


def test_role_routing_uses_light_model_for_classification(settings, faults):
    router, _ = make_router(settings, faults)
    _, rec = call(router)
    assert rec.role == "light" and rec.model_used == "mock:light"


# -- retries, backoff, fallback ----------------------------------------------------
def test_retry_with_exponential_backoff_then_success(settings, faults):
    faults.add("classify_request", ["rate_limit", "server"], model="light")
    router, sleeps = make_router(settings, faults, backoff_base_s=1.0, backoff_max_s=30.0, max_retries=3)
    parsed, rec = call(router)
    assert rec.success and not rec.fallback_used
    assert [a.outcome for a in rec.attempts] == ["provider_error:rate_limit", "provider_error:server", "ok"]
    assert len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 2.0  # equal-jitter exponential backoff


def test_backoff_is_capped(settings, faults):
    router, _ = make_router(settings, faults, backoff_base_s=2.0, backoff_max_s=5.0)
    assert all(router.backoff_delay(n) <= 5.0 for n in range(1, 12))


def test_fallback_after_retries_exhausted(settings, faults):
    faults.add("classify_request", ["timeout"] * 3, model="light")
    router, sleeps = make_router(settings, faults, max_retries=2)
    _, rec = call(router)
    assert rec.success and rec.fallback_used and rec.model_used == "mock:fallback"
    assert sum(a.model == "mock:light" for a in rec.attempts) == 3  # bounded: 1 + max_retries


def test_auth_error_moves_to_fallback_without_retry(settings, faults):
    faults.add("classify_request", ["auth"], model="light")
    router, sleeps = make_router(settings, faults)
    _, rec = call(router)
    assert rec.fallback_used and sleeps == []


def test_invalid_request_is_not_retried_or_fallen_back(settings, faults):
    faults.add("classify_request", ["invalid_request"], model="light")
    router, _ = make_router(settings, faults)
    with pytest.raises(ModelCallFailed) as ei:
        call(router)
    assert isinstance(ei.value.last_error, ProviderError) and not ei.value.retryable
    assert len(ei.value.record.attempts) == 1


def test_all_models_fail_raises_normalized_error(settings, faults):
    faults.add("classify_request", ["server"] * 10)
    router, _ = make_router(settings, faults, max_retries=1)
    with pytest.raises(ModelCallFailed) as ei:
        call(router)
    assert ei.value.retryable and ei.value.last_error.kind == "server"
    assert len(ei.value.record.attempts) == 4  # 2 attempts x 2 models


# -- structured output validation -------------------------------------------------
def test_invalid_json_is_repaired(settings, faults):
    faults.add("classify_request", ["invalid_json"], model="light")
    router, _ = make_router(settings, faults)
    parsed, rec = call(router)
    assert rec.validation_failures == 1 and rec.success and not rec.fallback_used
    provider = router.factory.get("mock:light")
    repair_turn = provider.calls[-1].messages[-1]["content"]
    assert "failed schema validation" in repair_turn


def test_persistent_schema_violation_falls_back_then_fails(settings, faults):
    faults.add("classify_request", ["schema_violation"] * 4)
    router, _ = make_router(settings, faults)
    with pytest.raises(ModelCallFailed) as ei:
        call(router)
    assert isinstance(ei.value.last_error, StructuredOutputError)
    assert ei.value.record.validation_failures == 4  # 2 tries on each of 2 models


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


# -- provider adapters (HTTP mocked) ----------------------------------------------
def _settings_with_keys(settings):
    return replace(settings, anthropic_api_key="sk-ant-test-key-123456", openai_api_key="sk-test-key-1234567890abcd")


def test_anthropic_adapter_success_and_headers(settings):
    seen = {}

    def handler(request: httpx.Request):
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": true}'}],
                                         "usage": {"input_tokens": 11, "output_tokens": 5}})

    p = AnthropicProvider("claude-test", _settings_with_keys(settings), transport=httpx.MockTransport(handler))
    resp = p.complete(ChatRequest(task="t", system="sys", messages=MSGS))
    assert resp.text == '{"ok": true}' and resp.input_tokens == 11
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "sys" and seen["body"]["model"] == "claude-test"


@pytest.mark.parametrize("status,kind,retryable", [(429, "rate_limit", True), (401, "auth", False),
                                                   (529, "server", True), (400, "invalid_request", False)])
def test_anthropic_error_normalization(settings, status, kind, retryable):
    transport = httpx.MockTransport(lambda r: httpx.Response(status, json={"error": {"message": "x"}}))
    p = AnthropicProvider("m", _settings_with_keys(settings), transport=transport)
    with pytest.raises(ProviderError) as ei:
        p.complete(ChatRequest(task="t", system="s", messages=MSGS))
    assert ei.value.kind == kind and ei.value.retryable is retryable and ei.value.status == status


def test_timeout_normalized(settings):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    p = OpenAIProvider("m", _settings_with_keys(settings), transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as ei:
        p.complete(ChatRequest(task="t", system="s", messages=MSGS))
    assert ei.value.kind == "timeout" and ei.value.retryable


def test_openai_adapter_requests_json_mode(settings):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 1}})

    p = OpenAIProvider("gpt-test", _settings_with_keys(settings), transport=httpx.MockTransport(handler))
    p.complete(ChatRequest(task="t", system="sys", messages=MSGS))
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}


def test_missing_api_key_is_auth_error(settings):
    with pytest.raises(ProviderError) as ei:
        AnthropicProvider("m", replace(settings, anthropic_api_key=""))
    assert ei.value.kind == "auth"


def test_live_router_falls_back_across_providers(settings):
    """Primary (anthropic) returns 503s; fallback (openai) answers."""
    def handler(request: httpx.Request):
        if request.url.host == "api.anthropic.com":
            return httpx.Response(503, json={"error": "overloaded"})
        body = {"industry": "Retail", "problem_type": "profitability", "complexity": "medium", "urgency": "high"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}], "usage": {}})

    s = replace(_settings_with_keys(settings), provider_mode="live", model_light="anthropic:claude-haiku-5-5",
                model_fallback="openai:gpt-fallback", max_retries=1)
    router = ModelRouter(s, ProviderFactory(s, transport=httpx.MockTransport(handler)), sleep=lambda _: None)
    parsed, rec = call(router)
    assert parsed.problem_type == "profitability"
    assert rec.fallback_used and rec.model_used == "openai:gpt-fallback"
