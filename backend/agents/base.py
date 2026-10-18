"""Shared agent plumbing: runtime dependencies, per-invocation context, results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.ai.model_router import ModelRouter
from backend.ai.tools import ToolContext, ToolExecutor, ToolSession
from backend.config import Settings
from backend.contracts.auth import Principal
from backend.contracts.data_module import DataModuleClient
from backend.contracts.schemas import EvidenceItem
from backend.prompts.base import PromptTemplate

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentRuntime:
    settings: Settings
    router: ModelRouter
    tools: ToolExecutor
    data: DataModuleClient


@dataclass
class AgentContext:
    execution_id: str
    project_id: str
    principal: Principal
    run_tool_calls: int = 0
    mock_scenario: str | None = None


@dataclass
class AgentResult:
    agent: str
    output: dict[str, Any]
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    security_flags: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def evidence_index(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    """Compact, non-content view of evidence for trusted context / mock responders."""
    return [{"evidence_id": e.evidence_id, "title": e.title, "source": e.source, "tags": e.workstream_tags,
             "reliability": e.reliability, "injection_suspected": e.injection_suspected} for e in items]


def dedupe_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    seen: dict[str, EvidenceItem] = {}
    for e in items:
        seen.setdefault(e.evidence_id, e)
    return list(seen.values())


class BaseAgent:
    name: str = "base"
    tool_agent: str | None = None  # key into AGENT_TOOL_ALLOWLIST (defaults to name)

    def __init__(self, runtime: AgentRuntime):
        self.rt = runtime

    def session(self, ctx: AgentContext, agent_key: str | None = None) -> ToolSession:
        return self.rt.tools.session(ToolContext(
            execution_id=ctx.execution_id, project_id=ctx.project_id, principal=ctx.principal,
            agent=agent_key or self.tool_agent or self.name, run_calls_so_far=ctx.run_tool_calls))

    def llm(self, prompt: PromptTemplate, context: dict[str, Any], *, evidence: list[EvidenceItem] | None = None,
            client_text: str | None = None, tool_log: list[dict[str, Any]] | None = None,
            mock_context: dict[str, Any] | None = None, result: AgentResult, role: str | None = None) -> Any:
        system, messages = prompt.build(context, evidence, client_text, tool_log)
        parsed, record = self.rt.router.call_structured(
            task=prompt.task, system=system, messages=messages, schema=prompt.output_schema,
            context=mock_context if mock_context is not None else context, role=role)
        result.model_calls.append(record.to_dict())
        return parsed

    @staticmethod
    def finish(result: AgentResult, session: ToolSession | None) -> AgentResult:
        if session is not None:
            result.tool_calls.extend(r.to_dict() for r in session.records)
            result.security_flags.extend(session.security_flags)
        return result
