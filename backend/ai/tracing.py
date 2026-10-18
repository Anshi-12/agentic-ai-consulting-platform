"""LangSmith observability with redaction.

Tracing is configured only through environment variables:

    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=consulting-ai-group1
    TRACE_ALLOW_CLIENT_DATA=false   # default: client text is replaced by size/hash summaries

LangGraph automatically traces graph/node runs when LANGSMITH_TRACING is set.
`traced` adds explicit spans (model calls, tool calls, routing, validation)
with input/output processors that strip secrets and, unless explicitly
authorized, client content.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable

from backend.ai.guardrails import fingerprint, redact_obj

try:  # langsmith is a dependency, but keep the layer importable without it
    from langsmith import traceable as _ls_traceable
except Exception:  # pragma: no cover
    _ls_traceable = None


_CONTENT_KEYS = {"messages", "system", "content", "prompt", "text", "evidence", "client_brief", "raw", "output"}


def _tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes"} or os.getenv(
        "LANGCHAIN_TRACING_V2", ""
    ).lower() in {"1", "true", "yes"}


def _client_data_allowed() -> bool:
    return os.getenv("TRACE_ALLOW_CLIENT_DATA", "").lower() in {"1", "true", "yes"}


def _summarize(value: Any) -> Any:
    text = value if isinstance(value, str) else repr(value)
    return {"redacted": True, "chars": len(text), "sha256_12": fingerprint(text)}


def scrub_for_trace(data: Any) -> Any:
    """Secrets are always removed; client content only kept when authorized."""
    data = redact_obj(data, pii=not _client_data_allowed())
    if _client_data_allowed() or not isinstance(data, dict):
        return data
    return {k: (_summarize(v) if k in _CONTENT_KEYS else v) for k, v in data.items()}


def traced(name: str, run_type: str = "chain") -> Callable:
    """Decorator: LangSmith span when tracing is on, zero-overhead no-op otherwise."""

    def deco(fn: Callable) -> Callable:
        if _ls_traceable is None:
            return fn
        wrapped = _ls_traceable(
            name=name,
            run_type=run_type,
            process_inputs=lambda inputs: scrub_for_trace(dict(inputs)),
            process_outputs=lambda outputs: scrub_for_trace(outputs if isinstance(outputs, dict) else {"output": outputs}),
        )(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            if _tracing_enabled():
                return wrapped(*args, **kwargs)
            return fn(*args, **kwargs)

        return inner

    return deco


def langsmith_status() -> dict[str, Any]:
    return {
        "enabled": _tracing_enabled(),
        "project": os.getenv("LANGSMITH_PROJECT", "consulting-ai-group1"),
        "api_key_set": bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")),
        "client_data_allowed": _client_data_allowed(),
    }
