"""C. Strategy Agent: options, weighted evaluation, prioritized recommendations."""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent, dedupe_evidence, evidence_index
from backend.ai.structured_outputs import StrategyOutput
from backend.contracts.schemas import EvidenceItem
from backend.prompts.strategy_prompts import STRATEGY_PROMPT


def weighted_score(scores: dict[str, float], criteria: list[dict[str, Any]]) -> float | None:
    """Deterministic weighted score on a 1-5 scale; weights are normalized to sum to 1."""
    total_w = sum(c["weight"] for c in criteria if c["name"] in scores)
    if total_w <= 0:
        return None
    return round(sum(scores[c["name"]] * c["weight"] for c in criteria if c["name"] in scores) / total_w, 3)


class StrategyAgent(BaseAgent):
    name = "strategy"

    def run(self, ctx: AgentContext, problem_statement: dict[str, Any], hypothesis_evaluation: dict[str, Any],
            findings: dict[str, dict[str, Any]], financial: dict[str, Any] | None,
            revision_feedback: list[str] | None = None, revision_round: int = 0) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        evidence = dedupe_evidence([EvidenceItem.model_validate(e)
                                    for f in findings.values() for e in f.get("evidence", [])])
        fin_scenarios = (financial or {}).get("scenarios", [])
        trusted = {
            "problem_statement": problem_statement,
            "hypothesis_evaluations": hypothesis_evaluation.get("evaluations", []),
            "evidence_gaps": hypothesis_evaluation.get("evidence_gaps", []),
            "financial": {
                "data_available": (financial or {}).get("data_available", False),
                "scenarios": fin_scenarios,
                "interpretation": (financial or {}).get("interpretation"),
            },
            "revision_feedback": revision_feedback or [],
            "revision_round": revision_round,
        }
        mock_ctx = {**trusted, "evidence_index": evidence_index(evidence), "financial_scenarios": fin_scenarios,
                    "mock_scenario": ctx.mock_scenario}
        out: StrategyOutput = self.llm(STRATEGY_PROMPT, trusted, evidence=evidence, tool_log=session.execution_log(),
                                       mock_context=mock_ctx, result=result)

        criteria = [c.model_dump() for c in out.decision_criteria]
        threshold = self.rt.settings.low_confidence_threshold
        for opt in out.options:
            missing = {c["name"] for c in criteria} - set(opt.scores)
            if missing:
                result.warnings.append(f"{opt.id} missing scores for {sorted(missing)}")
            opt.weighted_score = weighted_score(opt.scores, criteria)  # never trust model arithmetic
        for rec in out.recommendations:
            if rec.confidence < threshold or not rec.evidence_ids:
                if not rec.low_confidence_flag:
                    result.warnings.append(f"{rec.id}: low_confidence_flag set by guardrail")
                rec.low_confidence_flag = True
        out.recommendations.sort(key=lambda r: r.priority)
        out.options.sort(key=lambda o: o.weighted_score or 0, reverse=True)

        result.output = {"strategy": out.model_dump(mode="json")}
        return self.finish(result, session)
