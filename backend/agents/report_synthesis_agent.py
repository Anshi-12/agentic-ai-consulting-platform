"""F. Report Synthesis Agent: final deliverable as structured JSON (+ Markdown rendering)."""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent, dedupe_evidence, evidence_index
from backend.ai.structured_outputs import Citation, Recommendation, ReportOutput
from backend.contracts.schemas import EvidenceItem
from backend.prompts.synthesis_prompts import SYNTHESIS_PROMPT


class ReportSynthesisAgent(BaseAgent):
    name = "report_synthesis"

    def run(self, ctx: AgentContext, problem_statement: dict[str, Any], strategy: dict[str, Any],
            financial: dict[str, Any] | None, hypothesis_evaluation: dict[str, Any],
            findings: dict[str, dict[str, Any]], qa: dict[str, Any], approval: dict[str, Any]) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        evidence = dedupe_evidence([EvidenceItem.model_validate(e)
                                    for f in findings.values() for e in f.get("evidence", [])])
        approved_recs = strategy.get("recommendations", [])
        qa_note = qa.get("verdict", "unknown")
        limitations: list[str] = []
        if qa_note != "pass":
            limitations.append(f"QA verdict was '{qa_note}'; deliverable released by human approval "
                               f"({approval.get('reviewer', 'unknown reviewer')}).")
        if approval.get("comments"):
            limitations.append(f"Reviewer comments: {str(approval['comments'])[:300]}")
        if not (financial or {}).get("data_available"):
            limitations.append("No financial inputs were available; financial impact is unquantified.")
        limitations += [f"Low-confidence: {x}" for x in qa.get("low_confidence_findings", [])]

        trusted = {
            "problem_statement": problem_statement,
            "recommendations": approved_recs,
            "options": strategy.get("options", []),
            "trade_offs": strategy.get("trade_offs", []),
            "evaluations": hypothesis_evaluation.get("evaluations", []),
            "financial": {"data_available": (financial or {}).get("data_available", False),
                          "interpretation": (financial or {}).get("interpretation"),
                          "scenarios": (financial or {}).get("scenarios", [])},
            "qa_status": qa_note,
            "qa_issues_open": [i["description"] for i in qa.get("issues", [])],
            "approval": approval,
            "assumptions": strategy.get("assumptions", []),
            "limitations": limitations,
        }
        mock_ctx = {**trusted, "evidence_index": evidence_index(evidence)}
        report: ReportOutput = self.llm(SYNTHESIS_PROMPT, trusted, evidence=evidence, tool_log=session.execution_log(),
                                        mock_context=mock_ctx, result=result)

        # Enforce: recommendations are exactly the approved ones.
        approved_ids = [r["id"] for r in approved_recs]
        if approved_recs and [r.id for r in report.strategic_recommendations] != approved_ids:
            result.warnings.append("synthesis altered recommendations; restored approved set")
            report.strategic_recommendations = [Recommendation.model_validate(r) for r in approved_recs]

        # Rebuild citations from every referenced ID, verified through the tool.
        referenced = sorted({i for r in report.strategic_recommendations for i in r.evidence_ids}
                            | {i for f in report.detailed_findings for i in f.evidence_ids})
        verified: dict[str, dict[str, Any]] = {}
        if referenced:
            res = session.call("get_evidence_by_ids", project_id=ctx.project_id, evidence_ids=referenced[:50])
            if res.success:
                verified = {e["evidence_id"]: e for e in res.data}
        unknown = [i for i in referenced if i not in verified]
        if unknown:
            result.warnings.append(f"report referenced unknown evidence {unknown}; noted in limitations")
            report.limitations.append(f"Unverified evidence references removed from citations: {unknown}")
        report.citations = [Citation(evidence_id=i, source=verified[i]["source"], title=verified[i]["title"])
                            for i in referenced if i in verified]
        for lim in limitations:
            if lim not in report.limitations:
                report.limitations.append(lim)

        out = report.model_dump(mode="json")
        out["qa_status"] = qa_note
        out["approval"] = approval
        result.output = {"report": out, "report_markdown": render_markdown(out, problem_statement)}
        return self.finish(result, session)


def render_markdown(report: dict[str, Any], problem_statement: dict[str, Any] | None = None) -> str:
    es = report["executive_summary"]
    lines = [f"# {es['headline']}", ""]
    if problem_statement:
        lines += [f"**Key question:** {problem_statement.get('key_question', '')}", ""]
    lines += ["## Executive summary", f"**Situation.** {es['situation']}", "",
              f"**Complication.** {es['complication']}", "", f"**Resolution.** {es['resolution']}", ""]
    lines += ["## Recommendations"]
    for r in report["strategic_recommendations"]:
        flag = " ⚠ low confidence" if r.get("low_confidence_flag") else ""
        ev = ", ".join(r.get("evidence_ids", [])) or "no evidence cited"
        lines.append(f"{r['priority']}. **{r['title']}**{flag} — owner: {r['owner']}, horizon: {r['time_horizon']}, "
                     f"confidence {r['confidence']:.0%} ({r['claim_type']}; {ev})")
        for m in r.get("measurable_outcomes", []):
            lines.append(f"   - {m['metric']}: {m['target']} ({m.get('timeframe', '')})")
    lines += ["", "## Findings"]
    for f in report["detailed_findings"]:
        lines.append(f"- **{f['title']}** — {f['finding']} [{', '.join(f['evidence_ids']) or f['claim_type']}]")
    lines += ["", "## Implementation roadmap"]
    for p in report["implementation_roadmap"]:
        lines.append(f"- **{p['phase']}** ({p['timeframe']}, {p['owner']}): {'; '.join(p['actions'])}")
    lines += ["", "## Risk register", "| ID | Risk | Likelihood | Impact | Mitigation | Owner |", "|---|---|---|---|---|---|"]
    for k in report["risk_register"]:
        lines.append(f"| {k['id']} | {k['risk']} | {k['likelihood']} | {k['impact']} | {k['mitigation']} | {k['owner']} |")
    lines += ["", "## Assumptions"] + ([f"- {a}" for a in report.get("assumptions", [])] or ["- None stated"])
    lines += ["", "## Limitations"] + [f"- {x}" for x in report.get("limitations", [])]
    lines += ["", "## Sources"] + [f"- [{c['evidence_id']}] {c['source']} — {c.get('title', '')}" for c in report.get("citations", [])]
    lines += ["", f"_QA status: {report.get('qa_status')}; approved by {report.get('approval', {}).get('reviewer', 'n/a')}_"]
    return "\n".join(lines)
