"""Shared fixtures. All tests run in mock provider mode with no credentials."""
from __future__ import annotations

import os
import random
from dataclasses import replace
from typing import Any, Callable

import pytest

# Force offline mode before anything reads settings.
os.environ["LLM_PROVIDER_MODE"] = "mock"
os.environ.pop("LANGSMITH_TRACING", None)
os.environ.pop("LANGCHAIN_TRACING_V2", None)

from backend.ai.model_router import ModelRouter  # noqa: E402
from backend.ai.provider_factory import FaultPlan, ProviderFactory  # noqa: E402
from backend.api.models import StartWorkflowRequest  # noqa: E402
from backend.config import Settings, load_settings  # noqa: E402
from backend.contracts.auth import Principal, dev_principal  # noqa: E402
from backend.orchestration.execution import WorkflowExecutor  # noqa: E402

GOOD_BRIEF = (
    "Our gross margin has fallen from 46% to 39% over six quarters. We run frequent sitewide promotions and "
    "fulfillment costs keep rising. The CFO wants a plan to restore margin within 18 months without alienating "
    "loyal enthusiast customers."
)
VAGUE_BRIEF = "We want to fix our margins. Help?"


@pytest.fixture
def settings() -> Settings:
    return replace(load_settings(), provider_mode="mock", backoff_base_s=0.0, backoff_max_s=0.0,
                   max_retries=2, node_max_attempts=2, max_qa_iterations=2, agent_timeout_s=10.0,
                   checkpoint_backend="memory", run_store_path="", log_level="ERROR")


@pytest.fixture
def principal() -> Principal:
    return dev_principal("tester")


@pytest.fixture
def faults() -> FaultPlan:
    return FaultPlan()


class Harness:
    def __init__(self, settings: Settings, faults: FaultPlan, responders: dict[str, Callable] | None = None, **kw):
        from backend.ai.mock_responses import DEFAULT_RESPONDERS
        self.responders = {**DEFAULT_RESPONDERS, **(responders or {})}
        self.factory = ProviderFactory(settings, mock_responders=self.responders, faults=faults)
        self.sleeps: list[float] = []
        self.router = ModelRouter(settings, self.factory, sleep=self.sleeps.append, rng=random.Random(7))
        self.ex = WorkflowExecutor(settings, router=self.router, **kw)
        self.faults = faults

    def start(self, principal: Principal, brief: str = GOOD_BRIEF, project: str = "proj-northpeak",
              scenario: str | None = None, background: bool = False):
        return self.ex.start(StartWorkflowRequest(project_id=project, client_brief=brief, mock_scenario=scenario),
                             principal, background=background)

    def approve(self, eid: str, principal: Principal, decision: str = "approve", comments: str = ""):
        return self.ex.resume(eid, {"type": "approval", "decision": decision, "comments": comments}, principal)

    def values(self, eid: str) -> dict[str, Any]:
        return self.ex._values(eid)


@pytest.fixture
def make_harness(settings, faults):
    def _make(responders: dict[str, Callable] | None = None, settings_override: dict[str, Any] | None = None, **kw):
        s = replace(settings, **(settings_override or {}))
        return Harness(s, faults, responders, **kw)
    return _make


@pytest.fixture
def harness(make_harness) -> Harness:
    return make_harness()
