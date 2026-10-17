"""Configurable model router.

Roles (all configured through env, see backend/config.py):
    primary   - complex orchestration and strategic reasoning
    light     - classification and task routing
    synthesis - synthesis and long structured generation
    fallback  - used when a role's model fails with a recoverable error

Every call goes through: provider adapter -> bounded retries with exponential
backoff + jitter on retryable errors -> JSON extraction -> Pydantic validation
-> bounded repair turn on invalid output -> fallback model -> normalized error.
"""
from __future__ import annotations

import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from backend.ai.guardrails import get_logger
from backend.ai.provider_factory import ChatRequest, ChatResponse, ProviderError, ProviderFactory
from backend.ai.structured_outputs import (
    StructuredOutputError,
    format_validation_errors,
    validate_output,
)
from backend.ai.tracing import traced
from backend.config import Settings

T = TypeVar("T", bound=BaseModel)
log = get_logger("model_router")

ROLES = ("primary", "light", "synthesis", "fallback")

# Default task -> role mapping. Override per call if needed.
TASK_ROLES: dict[str, str] = {
    "classify_request": "light",
    "client_discovery": "primary",
    "problem_structuring": "primary",
    "hypothesis_evaluation": "primary",
    "strategy": "primary",
    "financial_scenario": "synthesis",
    "consulting_qa": "primary",
    "report_synthesis": "synthesis",
}


class ModelCallFailed(Exception):
    """All candidate models failed. `last_error` is a ProviderError or StructuredOutputError."""

    def __init__(self, task: str, last_error: Exception, record: "CallRecord"):
        super().__init__(f"model call for '{task}' failed after {len(record.attempts)} attempts: {last_error}")
        self.task = task
        self.last_error = last_error
        self.record = record

    @property
    def retryable(self) -> bool:
        return isinstance(self.last_error, ProviderError) and self.last_error.retryable


@dataclass
class AttemptRecord:
    model: str
    attempt: int
    outcome: str  # ok | provider_error:<kind> | invalid_output
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class CallRecord:
    call_id: str
    task: str
    role: str
    model_used: str | None = None
    fallback_used: bool = False
    success: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    validation_failures: int = 0
    attempts: list[AttemptRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelRouter:
    def __init__(self, settings: Settings, factory: ProviderFactory | None = None,
                 sleep: Callable[[float], None] = time.sleep, rng: random.Random | None = None):
        self.settings = settings
        self.factory = factory or ProviderFactory(settings)
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.listeners: list[Callable[[CallRecord], None]] = []

    # -- configuration -------------------------------------------------------
    def model_for_role(self, role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}'")
        return getattr(self.settings, f"model_{role}")

    def candidates(self, role: str) -> list[str]:
        primary = self.model_for_role(role)
        fallback = self.settings.model_fallback
        return [primary] if fallback == primary else [primary, fallback]

    def backoff_delay(self, attempt: int) -> float:
        base = min(self.settings.backoff_base_s * (2 ** (attempt - 1)), self.settings.backoff_max_s)
        return base * (0.5 + self._rng.random() / 2)  # equal jitter

    # -- calls ---------------------------------------------------------------
    def _complete_with_retries(self, ref: str, request: ChatRequest, record: CallRecord) -> ChatResponse:
        last: ProviderError | None = None
        max_attempts = max(1, self.settings.max_retries + 1)
        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            try:
                provider = self.factory.get(ref)
                resp = provider.complete(request)
                record.attempts.append(AttemptRecord(ref, attempt, "ok", (time.perf_counter() - start) * 1000))
                return resp
            except ProviderError as err:
                last = err
                record.attempts.append(AttemptRecord(ref, attempt, f"provider_error:{err.kind}",
                                                     (time.perf_counter() - start) * 1000, str(err)[:200]))
                log.warning("model call failed task=%s model=%s attempt=%d kind=%s", request.task, ref, attempt, err.kind)
                if not err.retryable or attempt == max_attempts:
                    break
                self._sleep(self.backoff_delay(attempt))
        assert last is not None
        raise last

    @traced("model_router.call_structured", run_type="llm")
    def call_structured(self, task: str, system: str, messages: list[dict[str, str]], schema: type[T],
                        context: dict[str, Any] | None = None, role: str | None = None,
                        max_tokens: int | None = None) -> tuple[T, CallRecord]:
        role = role or TASK_ROLES.get(task, "primary")
        record = CallRecord(call_id=uuid.uuid4().hex[:12], task=task, role=role)
        started = time.perf_counter()
        last_error: Exception | None = None

        for idx, ref in enumerate(self.candidates(role)):
            convo = list(messages)
            request = ChatRequest(task=task, system=system, messages=convo, context=context or {},
                                  max_tokens=max_tokens or self.settings.max_output_tokens,
                                  temperature=self.settings.temperature)
            try:
                for repair in range(self.settings.structured_repair_attempts + 1):
                    resp = self._complete_with_retries(ref, request, record)
                    record.input_tokens += resp.input_tokens
                    record.output_tokens += resp.output_tokens
                    try:
                        parsed = validate_output(schema, resp.text)
                    except StructuredOutputError as verr:
                        last_error = verr
                        record.validation_failures += 1
                        record.attempts.append(AttemptRecord(ref, repair + 1, "invalid_output", 0.0,
                                                             format_validation_errors(verr)[:300]))
                        if repair >= self.settings.structured_repair_attempts:
                            raise
                        convo = convo + [
                            {"role": "assistant", "content": resp.text[:6000]},
                            {"role": "user", "content": (
                                "Your previous reply failed schema validation:\n"
                                f"{format_validation_errors(verr)}\n"
                                "Return ONLY a corrected JSON object that satisfies the output schema. No prose."
                            )},
                        ]
                        request = ChatRequest(task=task, system=system, messages=convo, context=context or {},
                                              max_tokens=request.max_tokens, temperature=request.temperature)
                        continue
                    record.success = True
                    record.model_used = ref
                    record.fallback_used = idx > 0
                    record.latency_ms = (time.perf_counter() - started) * 1000
                    self._notify(record)
                    return parsed, record
            except ProviderError as perr:
                last_error = perr
                if perr.kind == "invalid_request":
                    break  # the request itself is bad; another model will not fix it
                continue  # recoverable -> next candidate (fallback)
            except StructuredOutputError:
                continue  # try fallback model

        record.latency_ms = (time.perf_counter() - started) * 1000
        self._notify(record)
        raise ModelCallFailed(task, last_error or RuntimeError("no candidates"), record)

    def _notify(self, record: CallRecord) -> None:
        for cb in list(self.listeners):
            try:
                cb(record)
            except Exception:
                pass
