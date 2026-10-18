"""Research task execution (tool-driven, no LLM).

Research content is produced by other modules; this step only *retrieves* it
through allowlisted tools, injection-scans it, and attaches it to a workstream.
"""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent


class ResearchTaskRunner(BaseAgent):
    name = "research"
    MAX_QUERIES = 3

    def run(self, ctx: AgentContext, workstream: dict[str, Any], extra_questions: list[str] | None = None,
            top_k: int = 5) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        queries = (list(extra_questions or []) + list(workstream.get("search_queries") or [])
                   + [workstream.get("name", "")])
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if len(q) >= 2][: self.MAX_QUERIES]

        evidence: dict[str, dict[str, Any]] = {}
        failures: list[str] = []
        for q in queries:
            for tool in ("query_research_results", "retrieve_documents"):
                res = session.call(tool, project_id=ctx.project_id, query=q[:300], top_k=top_k)
                if not res.success:
                    failures.append(f"{tool}('{q[:60]}'): {res.error}")
                    continue
                for item in res.data:
                    evidence.setdefault(item["evidence_id"], item)

        result.output = {
            "workstream_id": workstream["id"],
            "queries": queries,
            "evidence": list(evidence.values()),
            "tool_failures": failures,
            "status": "ok" if evidence else ("tool_failure" if failures else "no_evidence"),
        }
        return self.finish(result, session)
