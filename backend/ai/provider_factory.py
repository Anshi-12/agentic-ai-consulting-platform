"""Provider adapters with normalized errors.

Adapters: `mock` (offline, deterministic), `anthropic` (Messages API) and
`openai` (Chat Completions; also works with OpenAI-compatible servers via
OPENAI_BASE_URL). Raw HTTP via httpx keeps dependencies small and makes the
adapters easy to test with httpx.MockTransport.
"""
from __future__ import annotations

import json
import re
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from backend.config import Settings


# ---------------------------------------------------------------------------
# Request / response / errors
# ---------------------------------------------------------------------------
@dataclass
class ChatRequest:
    task: str                                   # e.g. "client_discovery"
    system: str
    messages: list[dict[str, str]]              # [{"role": "user"|"assistant", "content": str}]
    context: dict[str, Any] = field(default_factory=dict)  # structured context (mock + tracing)
    max_tokens: int = 4096
    temperature: float = 0.2
    json_mode: bool = True


@dataclass
class ChatResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class ProviderError(Exception):
    """Normalized provider failure.

    kind: timeout | rate_limit | quota_exhausted | server | auth | invalid_request | connection | unknown
    quota_exhausted = a daily/billing quota is used up: retrying minutes later will not help.
    retry_after = seconds the provider asked us to wait (Retry-After header / Gemini RetryInfo).
    """

    RETRYABLE_KINDS = {"timeout", "rate_limit", "server", "connection"}

    def __init__(self, kind: str, message: str, provider: str = "", model: str = "", status: int | None = None,
                 retry_after: float | None = None):
        super().__init__(f"[{provider}:{model}] {kind}: {message}")
        self.kind = kind
        self.provider = provider
        self.model = model
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "provider": self.provider, "model": self.model, "status": self.status,
                "retryable": self.retryable, "message": str(self)[:300]}


# Daily / billing quotas (Gemini "...PerDay..." quota IDs, OpenAI insufficient_quota, Anthropic credit balance).
_QUOTA_EXHAUSTED = re.compile(r"PerDay|insufficient_quota|credit_balance|billing_hard_limit", re.I)


def _retry_after(response: httpx.Response, raw: str) -> float | None:
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', raw)  # Gemini RetryInfo
    return float(m.group(1)) if m else None


def normalize_http_error(provider: str, model: str, exc: Exception | None = None,
                         response: httpx.Response | None = None) -> ProviderError:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError("timeout", "request timed out", provider, model)
    if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
        return ProviderError("connection", str(exc)[:200], provider, model)
    if response is not None:
        status = response.status_code
        raw = response.text or ""
        try:
            body = response.json()
            detail = json.dumps(body.get("error", body) if isinstance(body, dict) else body)[:300]
        except Exception:
            detail = raw[:300]
        retry_after = _retry_after(response, raw)
        if status == 429:
            kind = "quota_exhausted" if _QUOTA_EXHAUSTED.search(raw) else "rate_limit"
        elif status in (401, 403):
            kind = "auth"
        elif status in (400, 404, 413, 422):
            kind = "invalid_request"
        elif status >= 500:  # includes 529 overloaded
            kind = "server"
        else:
            kind = "unknown"
        return ProviderError(kind, detail, provider, model, status, retry_after)
    return ProviderError("unknown", str(exc)[:200] if exc else "unknown error", provider, model)


class BaseProvider(ABC):
    name = "base"

    def __init__(self, model: str, settings: Settings):
        self.model = model
        self.settings = settings

    @abstractmethod
    def complete(self, request: ChatRequest) -> ChatResponse: ...


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------
class AnthropicProvider(BaseProvider):
    name = "anthropic"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, settings: Settings, transport: httpx.BaseTransport | None = None):
        super().__init__(model, settings)
        if not settings.anthropic_api_key:
            raise ProviderError("auth", "ANTHROPIC_API_KEY is not set", self.name, model)
        self._client = httpx.Client(
            base_url=settings.anthropic_base_url,
            timeout=settings.request_timeout_s,
            transport=transport,
            headers={"x-api-key": settings.anthropic_api_key, "anthropic-version": self.API_VERSION,
                     "content-type": "application/json"},
        )

    def complete(self, request: ChatRequest) -> ChatResponse:
        body = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": request.messages,
        }
        start = time.perf_counter()
        try:
            resp = self._client.post("/v1/messages", json=body)
        except httpx.HTTPError as exc:
            raise normalize_http_error(self.name, self.model, exc=exc) from exc
        if resp.status_code >= 400:
            raise normalize_http_error(self.name, self.model, response=resp)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return ChatResponse(text, self.name, self.model, usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                            (time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# OpenAI / OpenAI-compatible Chat Completions
# ---------------------------------------------------------------------------
class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str, settings: Settings, transport: httpx.BaseTransport | None = None):
        super().__init__(model, settings)
        if not settings.openai_api_key:
            raise ProviderError("auth", "OPENAI_API_KEY is not set", self.name, model)
        self._json_mode_ok = os.getenv("OPENAI_JSON_MODE", "true").lower() not in {"0", "false", "no", "off"}
        base = settings.openai_base_url.rstrip("/") + "/"  # so relative paths keep the /v1 or /openai/ prefix
        self._client = httpx.Client(
            base_url=base,
            timeout=settings.request_timeout_s,
            transport=transport,
            headers={"authorization": f"Bearer {settings.openai_api_key}", "content-type": "application/json"},
        )

    def complete(self, request: ChatRequest) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system}, *request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        use_json_mode = request.json_mode and self._json_mode_ok
        if use_json_mode:
            body["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        try:
            resp = self._client.post("chat/completions", json=body)
            if resp.status_code == 400 and use_json_mode:
                # Some OpenAI-compatible servers (Gemini, Groq models, Ollama builds) reject JSON mode.
                # Retry once without it; prompts already demand JSON and the router validates output.
                body.pop("response_format", None)
                retry = self._client.post("chat/completions", json=body)
                if retry.status_code < 400:
                    self._json_mode_ok = False
                resp = retry
        except httpx.HTTPError as exc:
            raise normalize_http_error(self.name, self.model, exc=exc) from exc
        if resp.status_code >= 400:
            raise normalize_http_error(self.name, self.model, response=resp)
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage", {})
        return ChatResponse(text, self.name, self.model, usage.get("prompt_tokens", 0),
                            usage.get("completion_tokens", 0), (time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------------------
# Mock provider (offline; supports fault injection for tests)
# ---------------------------------------------------------------------------
Responder = Callable[[ChatRequest], Any]


class FaultPlan:
    """Scripted failures: plan.add("strategy", ["rate_limit", "rate_limit"], model="primary").

    Actions: timeout | rate_limit | server | auth | invalid_request | invalid_json |
    schema_violation | slow:<seconds>. Each scheduled action is consumed once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plan: dict[tuple[str, str], list[str]] = {}

    def add(self, task: str, actions: list[str], model: str = "*") -> "FaultPlan":
        with self._lock:
            self._plan.setdefault((model, task), []).extend(actions)
        return self

    def next_action(self, model: str, task: str) -> str | None:
        with self._lock:
            for key in ((model, task), ("*", task), (model, "*"), ("*", "*")):
                queue = self._plan.get(key)
                if queue:
                    return queue.pop(0)
        return None

    def clear(self) -> None:
        with self._lock:
            self._plan.clear()


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, model: str, settings: Settings, responders: dict[str, Responder] | None = None,
                 faults: FaultPlan | None = None):
        super().__init__(model, settings)
        if responders is None:
            from backend.ai.mock_responses import DEFAULT_RESPONDERS
            responders = DEFAULT_RESPONDERS
        self.responders = responders
        self.faults = faults or FaultPlan()
        self.calls: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        action = self.faults.next_action(self.model, request.task)
        if action:
            if action.startswith("slow:"):
                time.sleep(float(action.split(":", 1)[1]))
            elif action in ProviderError.RETRYABLE_KINDS | {"auth", "invalid_request", "unknown"}:
                raise ProviderError(action, "injected mock fault", self.name, self.model)
            elif action == "invalid_json":
                return ChatResponse("Sure! Here is my analysis: {not valid json", self.name, self.model, 50, 10, 1.0)
            elif action == "schema_violation":
                return ChatResponse(json.dumps({"unexpected": "shape"}), self.name, self.model, 50, 10, 1.0)
        responder = self.responders.get(request.task)
        if responder is None:
            raise ProviderError("invalid_request", f"no mock responder for task '{request.task}'", self.name, self.model)
        payload = responder(request)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        in_tok = sum(len(m["content"]) for m in request.messages) // 4 + len(request.system) // 4
        return ChatResponse(text, self.name, self.model, in_tok, len(text) // 4, 2.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def parse_model_ref(ref: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"model reference '{ref}' must look like 'provider:model-id'")
    provider, model = ref.split(":", 1)
    provider = provider.strip().lower()
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider '{provider}'. Supported: {sorted(_PROVIDERS)}")
    return provider, model.strip()


class ProviderFactory:
    """Builds and caches one adapter per model reference."""

    def __init__(self, settings: Settings, mock_responders: dict[str, Responder] | None = None,
                 faults: FaultPlan | None = None, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.mock_responders = mock_responders
        self.faults = faults or FaultPlan()
        self.transport = transport
        self._cache: dict[str, BaseProvider] = {}
        self._lock = threading.Lock()

    def get(self, ref: str) -> BaseProvider:
        with self._lock:
            if ref in self._cache:
                return self._cache[ref]
            provider, model = parse_model_ref(ref)
            if provider == "mock":
                adapter: BaseProvider = MockProvider(model, self.settings, self.mock_responders, self.faults)
            else:
                adapter = _PROVIDERS[provider](model, self.settings, transport=self.transport)  # type: ignore[call-arg]
            self._cache[ref] = adapter
            return adapter
