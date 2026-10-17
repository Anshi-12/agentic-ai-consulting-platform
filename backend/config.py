"""Environment-based configuration for the Group 1 AI layer.

Every tunable lives here and is read from environment variables (optionally
loaded from a `.env` file). Nothing about model availability is hardcoded:
models are referenced as ``"<provider>:<model-id>"`` strings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # --- provider mode -----------------------------------------------------
    # "mock" runs everything offline with deterministic responses (tests, demos).
    # "live" calls real providers configured below.
    provider_mode: str = "mock"

    # --- model routing (provider:model) ------------------------------------
    model_primary: str = "mock:primary"
    model_light: str = "mock:light"
    model_synthesis: str = "mock:synthesis"
    model_fallback: str = "mock:fallback"

    # --- provider credentials / endpoints ----------------------------------
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # --- request resilience -------------------------------------------------
    request_timeout_s: float = 60.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 20.0
    max_output_tokens: int = 4096
    temperature: float = 0.2
    structured_repair_attempts: int = 1

    # --- agent / workflow limits -------------------------------------------
    agent_timeout_s: float = 180.0
    agent_timeouts: dict[str, float] = field(default_factory=dict)
    node_max_attempts: int = 2
    max_qa_iterations: int = 2
    max_tool_calls_per_agent: int = 12
    max_tool_calls_per_run: int = 80
    low_confidence_threshold: float = 0.55

    # --- persistence --------------------------------------------------------
    checkpoint_backend: str = "memory"  # memory | sqlite
    checkpoint_sqlite_path: str = "data/checkpoints.sqlite"
    run_store_path: str = ""  # empty = in-memory run records; else JSON file dir

    # --- security -----------------------------------------------------------
    injection_policy: str = "quarantine"  # quarantine | flag
    dev_allowed_projects: str = "*"  # DEV ONLY placeholder auth

    # --- observability ------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_project: str = "consulting-ai-group1"
    trace_allow_client_data: bool = False
    log_level: str = "INFO"

    def timeout_for(self, agent_name: str) -> float:
        return self.agent_timeouts.get(agent_name, self.agent_timeout_s)

    @property
    def is_mock(self) -> bool:
        return self.provider_mode == "mock"


AGENT_NAMES = (
    "client_discovery",
    "problem_structuring",
    "strategy",
    "financial_scenario",
    "consulting_qa",
    "report_synthesis",
)


def load_settings() -> Settings:
    _load_dotenv()
    mode = os.getenv("LLM_PROVIDER_MODE", "mock").strip().lower()
    default_model = "mock:{}"
    timeouts = {}
    for name in AGENT_NAMES:
        env_name = f"AGENT_TIMEOUT_{name.upper()}"
        if os.getenv(env_name):
            timeouts[name] = float(os.environ[env_name])

    primary_live = os.getenv("MODEL_PRIMARY", "").strip()
    if mode == "live" and not primary_live:
        raise ValueError("LLM_PROVIDER_MODE=live requires MODEL_PRIMARY (e.g. anthropic:claude-opus-5-5)")

    def model(env: str, role: str) -> str:
        if mode == "mock":
            return default_model.format(role)
        # Unset roles in live mode inherit the primary model rather than silently using a mock.
        return os.getenv(env, "").strip() or primary_live

    tracing = _bool("LANGSMITH_TRACING", False) or _bool("LANGCHAIN_TRACING_V2", False)
    return Settings(
        provider_mode=mode,
        model_primary=model("MODEL_PRIMARY", "primary"),
        model_light=model("MODEL_LIGHT", "light"),
        model_synthesis=model("MODEL_SYNTHESIS", "synthesis"),
        model_fallback=model("MODEL_FALLBACK", "fallback"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        request_timeout_s=_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0),
        max_retries=_int("LLM_MAX_RETRIES", 3),
        backoff_base_s=_float("LLM_BACKOFF_BASE_SECONDS", 1.0),
        backoff_max_s=_float("LLM_BACKOFF_MAX_SECONDS", 20.0),
        max_output_tokens=_int("LLM_MAX_OUTPUT_TOKENS", 4096),
        temperature=_float("LLM_TEMPERATURE", 0.2),
        structured_repair_attempts=_int("LLM_STRUCTURED_REPAIR_ATTEMPTS", 1),
        agent_timeout_s=_float("AGENT_TIMEOUT_SECONDS", 180.0),
        agent_timeouts=timeouts,
        node_max_attempts=_int("NODE_MAX_ATTEMPTS", 2),
        max_qa_iterations=_int("MAX_QA_ITERATIONS", 2),
        max_tool_calls_per_agent=_int("MAX_TOOL_CALLS_PER_AGENT", 12),
        max_tool_calls_per_run=_int("MAX_TOOL_CALLS_PER_RUN", 80),
        low_confidence_threshold=_float("LOW_CONFIDENCE_THRESHOLD", 0.55),
        checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "memory").lower(),
        checkpoint_sqlite_path=os.getenv("CHECKPOINT_SQLITE_PATH", "data/checkpoints.sqlite"),
        run_store_path=os.getenv("RUN_STORE_PATH", ""),
        injection_policy=os.getenv("INJECTION_POLICY", "quarantine").lower(),
        dev_allowed_projects=os.getenv("DEV_ALLOWED_PROJECTS", "*"),
        langsmith_tracing=tracing,
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "consulting-ai-group1"),
        trace_allow_client_data=_bool("TRACE_ALLOW_CLIENT_DATA", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
