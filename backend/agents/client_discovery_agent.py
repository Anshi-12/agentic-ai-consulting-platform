"""A. Client Discovery Agent: classify the request, interpret requirements,
produce a problem statement + scope, or ask clarifying questions."""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent
from backend.ai.structured_outputs import DiscoveryOutput, RequestClassification
from backend.prompts.discovery_prompts import CLASSIFY_PROMPT, DISCOVERY_PROMPT


class ClientDiscoveryAgent(BaseAgent):
    name = "client_discovery"

    def run(self, ctx: AgentContext, request: dict[str, Any],
            clarification_answers: dict[str, Any] | None = None) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)

        pc = session.call("read_project_context", project_id=ctx.project_id)
        project_context = pc.data if pc.success else {}
        brief = request.get("client_brief", "")
        industry_hint = request.get("industry") or project_context.get("industry")

        classification: RequestClassification = self.llm(
            CLASSIFY_PROMPT, {"industry_hint": industry_hint}, client_text=brief, result=result,
            mock_context={"brief": brief, "industry_hint": industry_hint})

        trusted = {
            "client_name": request.get("client_name") or project_context.get("client_name"),
            "industry": industry_hint or classification.industry,
            "classification": classification.model_dump(),
            "project_context": project_context,
            "stated_objectives": request.get("objectives") or [],
            "stated_constraints": request.get("constraints") or [],
            "clarification_answers": clarification_answers or {},
        }
        discovery: DiscoveryOutput = self.llm(
            DISCOVERY_PROMPT, trusted, client_text=brief, tool_log=session.execution_log(), result=result,
            mock_context={"brief": brief, "client_name": trusted["client_name"],
                          "clarification_answers": clarification_answers or {},
                          "objectives": request.get("objectives"), "constraints": request.get("constraints")})

        if discovery.status == "complete" and discovery.confidence < self.rt.settings.low_confidence_threshold:
            result.warnings.append(f"discovery confidence {discovery.confidence:.2f} below threshold")

        result.output = {
            "classification": classification.model_dump(mode="json"),
            "discovery": discovery.model_dump(mode="json"),
            "project_context": project_context,
        }
        return self.finish(result, session)
