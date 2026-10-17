"""B. Problem Structuring Agent: issue tree, hypotheses, workstreams, frameworks;
and hypothesis evaluation once research has returned evidence."""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent, evidence_index
from backend.ai.structured_outputs import HypothesisEvaluationOutput, StructuringOutput
from backend.contracts.schemas import EvidenceItem
from backend.prompts.discovery_prompts import HYPOTHESIS_EVAL_PROMPT, STRUCTURING_PROMPT


class ProblemStructuringAgent(BaseAgent):
    name = "problem_structuring"

    def structure(self, ctx: AgentContext, discovery: dict[str, Any], classification: dict[str, Any],
                  project_context: dict[str, Any]) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        trusted = {
            "problem_statement": discovery.get("problem_statement"),
            "engagement_scope": discovery.get("engagement_scope"),
            "classification": classification,
            "project_context": project_context,
        }
        out: StructuringOutput = self.llm(STRUCTURING_PROMPT, trusted, tool_log=session.execution_log(), result=result)
        for ws in out.workstreams:
            if not ws.search_queries:  # guarantee research has something concrete to run
                ws.search_queries = [ws.name] + ws.analytical_questions[:1]
                result.warnings.append(f"{ws.id}: no search queries supplied; derived from name/questions")
        result.output = {"structuring": out.model_dump(mode="json")}
        return self.finish(result, session)

    def evaluate_hypotheses(self, ctx: AgentContext, structuring: dict[str, Any],
                            findings: dict[str, dict[str, Any]]) -> AgentResult:
        result = AgentResult(self.name, {})
        evidence_by_ws: dict[str, list[EvidenceItem]] = {
            ws: [EvidenceItem.model_validate(e) for e in f.get("evidence", [])] for ws, f in findings.items()
        }
        all_evidence = {e.evidence_id: e for items in evidence_by_ws.values() for e in items}
        trusted = {
            "hypotheses": structuring["hypotheses"],
            "research_summary": {ws: {"evidence_ids": [e.evidence_id for e in items],
                                      "tool_failures": findings[ws].get("tool_failures", [])}
                                 for ws, items in evidence_by_ws.items()},
        }
        mock_ctx = {"hypotheses": structuring["hypotheses"],
                    "findings": {ws: evidence_index(items) for ws, items in evidence_by_ws.items()}}
        out: HypothesisEvaluationOutput = self.llm(
            HYPOTHESIS_EVAL_PROMPT, trusted, evidence=list(all_evidence.values()), mock_context=mock_ctx, result=result)

        # Deterministic evidence discipline: unknown IDs are dropped; flagged sources cannot be sole support;
        # 'supported' without valid evidence is downgraded.
        known = set(all_evidence)
        for ev in out.evaluations:
            unknown = [i for i in ev.evidence_ids if i not in known]
            if unknown:
                result.warnings.append(f"{ev.hypothesis_id}: removed unknown evidence IDs {unknown}")
            ev.evidence_ids = [i for i in ev.evidence_ids if i in known]
            clean = [i for i in ev.evidence_ids if not all_evidence[i].injection_suspected]
            if ev.verdict != "inconclusive" and not clean:
                result.warnings.append(f"{ev.hypothesis_id}: verdict '{ev.verdict}' had no valid support; set to inconclusive")
                ev.verdict = "inconclusive"
                ev.confidence = min(ev.confidence, 0.35)
        expected = {h["id"] for h in structuring["hypotheses"]}
        missing = expected - {e.hypothesis_id for e in out.evaluations}
        if missing:
            out.evidence_gaps.extend(f"No evaluation returned for {m}" for m in sorted(missing))
        result.output = {"hypothesis_evaluation": out.model_dump(mode="json")}
        return result
