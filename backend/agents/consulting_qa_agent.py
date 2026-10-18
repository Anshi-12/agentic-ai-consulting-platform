"""E. Consulting QA Agent.

Two layers:
1. Deterministic checks (authoritative): fabricated citations (verified with the
   `get_evidence_by_ids` tool), reliance on security-flagged sources, uncited
   "fact" claims, low confidence, unverified currency figures, missing financials.
2. LLM review for relevance, contradictions and reasoning quality.

The final verdict can be made stricter by layer 1 but never looser: an LLM
"pass" cannot override a deterministic high/critical issue.
"""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent
from backend.agents.financial_scenario_agent import unverified_amounts
from backend.ai.structured_outputs import QAIssue, QAOutput
from backend.prompts.qa_prompts import QA_PROMPT

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
STAGE_ORDER = {"structuring": 0, "research": 1, "financial": 2, "strategy": 3, "none": 9}


def choose_route(issues: list[QAIssue]) -> str:
    routable = [i for i in issues if i.route_to != "none" and SEVERITY_RANK[i.severity] >= 2]
    if not routable:
        return "none"
    top = max(SEVERITY_RANK[i.severity] for i in routable)
    return min((i.route_to for i in routable if SEVERITY_RANK[i.severity] == top), key=lambda r: STAGE_ORDER[r])


class ConsultingQAAgent(BaseAgent):
    name = "consulting_qa"

    def deterministic_checks(self, ctx: AgentContext, session, strategy: dict[str, Any],
                             financial: dict[str, Any] | None, hypothesis_evaluation: dict[str, Any],
                             findings: dict[str, dict[str, Any]]) -> tuple[list[QAIssue], list[str]]:
        issues: list[QAIssue] = []
        low_conf: list[str] = []
        n = 0

        def add(**kw):
            nonlocal n
            n += 1
            issues.append(QAIssue(id=f"DQ{n}", source="deterministic", **kw))

        recs = strategy.get("recommendations", [])
        options = strategy.get("options", [])
        if not recs:
            add(severity="critical", category="off_scope", description="No recommendations were produced.",
                route_to="strategy")

        cited = sorted({i for item in recs + options for i in item.get("evidence_ids", [])})
        gathered = {e["evidence_id"]: e for f in findings.values() for e in f.get("evidence", [])}
        verified: dict[str, dict[str, Any]] = {}
        if cited:
            res = session.call("get_evidence_by_ids", project_id=ctx.project_id, evidence_ids=cited[:50])
            if res.success:
                verified = {e["evidence_id"]: e for e in res.data}
            else:
                add(severity="high", category="missing_evidence", route_to="research",
                    description=f"Citation verification tool failed: {res.error}")
        if verified or not cited:
            for item in recs + options:
                fake = [i for i in item.get("evidence_ids", []) if i not in verified]
                if fake:
                    add(severity="critical", category="fabricated_citation", affected_ids=[item["id"], *fake],
                        route_to="strategy", description=f"{item['id']} cites evidence IDs that do not exist: {fake}")
        for item in recs + options:
            flagged = [i for i in item.get("evidence_ids", [])
                       if verified.get(i, {}).get("injection_suspected") or gathered.get(i, {}).get("injection_suspected")]
            if flagged:
                add(severity="critical", category="security", affected_ids=[item["id"], *flagged], route_to="strategy",
                    description=f"{item['id']} relies on security-flagged source(s) {flagged}")

        threshold = self.rt.settings.low_confidence_threshold
        for r in recs:
            if r.get("claim_type") == "fact" and not r.get("evidence_ids") and gathered:
                add(severity="high", category="unsupported_claim", affected_ids=[r["id"]], route_to="strategy",
                    description=f"{r['id']} is labelled 'fact' but cites no evidence although evidence exists.")
            if r.get("confidence", 1) < threshold:
                low_conf.append(f"{r['id']} ({r.get('confidence'):.2f}): {r.get('title')}")
                if not r.get("low_confidence_flag"):
                    add(severity="medium", category="low_confidence", affected_ids=[r["id"]], route_to="strategy",
                        description=f"{r['id']} has confidence below {threshold} but is not flagged.")

        fin_ok = bool(financial and financial.get("data_available"))
        if not fin_ok:
            for r in recs:
                if r.get("claim_type") == "calculated":
                    add(severity="high", category="unsupported_claim", affected_ids=[r["id"]], route_to="financial",
                        description=f"{r['id']} claims calculated values but no financial calculation succeeded.")
        else:
            known = [m["value"] for s in financial["scenarios"] for m in s["metrics"]
                     if m.get("value") is not None and m.get("unit") == "currency"]
            for r in recs:
                bad = unverified_amounts([r.get("title", ""), r.get("rationale", "")], known)
                if bad:
                    add(severity="high", category="inconsistent_number", affected_ids=[r["id"]], route_to="financial",
                        description=f"{r['id']} states amounts {bad[:3]} that match no calculated metric.")

        evals = hypothesis_evaluation.get("evaluations", [])
        if evals and all(e["verdict"] == "inconclusive" for e in evals):
            add(severity="medium", category="missing_evidence", route_to="research",
                description="All hypotheses are inconclusive; recommendations rest on thin evidence.")
        return issues, low_conf

    def run(self, ctx: AgentContext, problem_statement: dict[str, Any], strategy: dict[str, Any],
            financial: dict[str, Any] | None, hypothesis_evaluation: dict[str, Any],
            findings: dict[str, dict[str, Any]], security_flags: list[dict[str, Any]],
            iteration: int) -> AgentResult:
        result = AgentResult(self.name, {})
        session = self.session(ctx)
        det_issues, low_conf = self.deterministic_checks(ctx, session, strategy, financial,
                                                         hypothesis_evaluation, findings)
        flagged_ids = sorted({f["evidence_id"] for f in security_flags if "evidence_id" in f})
        trusted = {
            "key_question": (problem_statement or {}).get("key_question"),
            "business_objectives": (problem_statement or {}).get("business_objectives", []),
            "recommendations": strategy.get("recommendations", []),
            "options": strategy.get("options", []),
            "available_evidence_ids": sorted({e["evidence_id"] for f in findings.values() for e in f.get("evidence", [])}),
            "flagged_evidence_ids": flagged_ids,
            "financial_metrics": (financial or {}).get("scenarios", []),
            "financial_data_available": bool((financial or {}).get("data_available")),
            "deterministic_findings": [i.description for i in det_issues],
            "qa_iteration": iteration,
        }
        llm_out: QAOutput = self.llm(QA_PROMPT, trusted, tool_log=session.execution_log(), result=result)

        # Merge: deterministic issues are authoritative; verdict can only get stricter.
        issues = det_issues + [i.model_copy(update={"id": f"LQ{k + 1}", "source": "llm"})
                               for k, i in enumerate(llm_out.issues)]
        worst = max((SEVERITY_RANK[i.severity] for i in issues), default=-1)
        verdict = llm_out.verdict
        if worst >= 2 and verdict == "pass":
            verdict = "fail"
        if not llm_out.addresses_original_problem and verdict == "pass":
            verdict = "fail"
        route = choose_route(issues)
        if verdict == "fail" and route == "none":
            route = llm_out.route_to if llm_out.route_to != "none" else "strategy"
        requires_human = llm_out.requires_human_review or verdict == "escalate"
        if verdict == "escalate":
            session.call("submit_for_review", execution_id=ctx.execution_id, artifact_type="qa_escalation",
                         summary=(llm_out.summary or "QA escalation")[:2000], severity="high_impact")

        final = QAOutput(
            verdict=verdict, addresses_original_problem=llm_out.addresses_original_problem, issues=issues,
            low_confidence_findings=sorted(set(low_conf + llm_out.low_confidence_findings)),
            requires_human_review=requires_human, route_to=route if verdict != "pass" else "none",
            additional_research_requests=llm_out.additional_research_requests,
            summary=llm_out.summary if verdict == llm_out.verdict else
            f"Verdict tightened from '{llm_out.verdict}' to '{verdict}' by deterministic checks. Most severe: "
            f"{max(issues, key=lambda i: SEVERITY_RANK[i.severity]).description if issues else 'n/a'}",
        )
        result.output = {"qa": final.model_dump(mode="json")}
        return self.finish(result, session)
