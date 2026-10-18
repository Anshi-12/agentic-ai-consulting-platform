"""Management Consulting Manager Agent: intent detection, routing, the task and
progress ledger, the "is the request resolved?" loop, and consolidation."""
from __future__ import annotations

from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent
from backend.agents.domain_agents import support_ratio
from backend.agents.intent_rules import detect_companies, detect_year, expand_query, route_by_rules
from backend.ai.consulting_schemas import ConsultingAnswer, ProgressAssessment, RoutingPlan
from backend.prompts.manager_prompts import CONSOLIDATE_PROMPT, MANAGER_PLAN_PROMPT, PROGRESS_PROMPT
from backend.rag.documents import DOMAINS


def rule_tasks(question: str) -> list[dict[str, Any]]:
    r = route_by_rules(question)
    tasks = []
    for d in r.domains:
        company = None
        if d == "sec" and r.companies:
            company = r.companies[0]
        tasks.append({"agent": d, "instruction": question, "search_queries": [expand_query(question)[:300]],
                      "company": company, "year": r.year, "rationale": f"keyword route (score {r.scores[d]})"})
    return tasks


class ManagerAgent(BaseAgent):
    name = "manager_agent"

    # -- 1. intent detection + routing + ledger ----------------------------------
    def plan(self, ctx: AgentContext, question: str, human_answers: list[str]) -> AgentResult:
        result = AgentResult(self.name, {})
        rules = route_by_rules(question + " " + " ".join(human_answers))
        trusted = {"question": question, "human_answers": human_answers, "available_agents": list(DOMAINS),
                   "rule_based_hint": {"domains": rules.domains, "category": rules.category}}
        mock_ctx = {**trusted, "mock_scenario": ctx.mock_scenario}
        plan: RoutingPlan = self.llm(MANAGER_PLAN_PROMPT, trusted, client_text=question,
                                     mock_context=mock_ctx, result=result, role="primary")
        tasks = [t.model_dump() for t in plan.tasks]
        if human_answers and plan.needs_clarification and rules.domains:
            result.warnings.append("router still unsure after human input; using rule-based route")
            plan.needs_clarification = False
            tasks = rule_tasks(question + " " + " ".join(human_answers))
        if not tasks and not plan.needs_clarification and plan.intent_category != "out_of_scope" and rules.domains:
            result.warnings.append("router returned no tasks; fell back to rule-based routing")
            tasks = rule_tasks(question)
        # Fill obvious entities the model skipped (company for SEC, year for food-security reports).
        for t in tasks:
            if t["agent"] == "sec" and not t.get("company"):
                comps = detect_companies(t["instruction"] + " " + question)
                t["company"] = comps[0] if len(comps) == 1 else None
            if not t.get("year"):
                t["year"] = detect_year(question)
        seen, unique = set(), []
        for t in tasks:  # one task per agent per round
            if t["agent"] not in seen:
                seen.add(t["agent"])
                unique.append(t)
        out = plan.model_dump(mode="json")
        out["tasks"] = unique
        out["is_cross_domain"] = len(unique) > 1
        out["rule_based_domains"] = rules.domains
        result.output = {"routing": out}
        return result

    # -- 2. progress ledger check -------------------------------------------------
    def assess(self, ctx: AgentContext, question: str, ledger: dict[str, Any], results: dict[str, Any],
               round_no: int, max_rounds: int) -> AgentResult:
        result = AgentResult(self.name, {})
        summaries = {tid: {"agent": r["agent"], "answer": r["answer"]["answer"],
                           "insufficient_evidence": r["answer"]["insufficient_evidence"],
                           "missing_information": r["answer"]["missing_information"],
                           "confidence": r["uncertainty"]["final_confidence"]} for tid, r in results.items()}
        trusted = {"question": question, "ledger": ledger, "agent_results": summaries, "round": round_no,
                   "max_rounds": max_rounds}
        out: ProgressAssessment = self.llm(PROGRESS_PROMPT, trusted, mock_context={**trusted,
                                           "mock_scenario": ctx.mock_scenario}, result=result, role="light")
        done = {(t["agent"], t["instruction"].strip().lower()) for t in ledger.get("tasks", [])}
        next_tasks = []
        for t in out.next_tasks:
            key = (t.agent, t.instruction.strip().lower())
            if key in done:
                result.warnings.append(f"dropped repeated task for {t.agent}")
                continue
            next_tasks.append({**t.model_dump(), "year": t.year or detect_year(question)})
        data = out.model_dump(mode="json")
        data["next_tasks"] = next_tasks
        if not out.is_resolved and not next_tasks and not out.need_human:
            data["reason"] += " (no new tasks proposed; consolidating with stated uncertainty)"
        result.output = {"assessment": data}
        return result

    # -- 3. consolidation into consulting recommendations -------------------------
    def consolidate(self, ctx: AgentContext, question: str, routing: dict[str, Any], results: dict[str, Any],
                    evidence: dict[str, dict[str, Any]], human_answers: list[str]) -> AgentResult:
        result = AgentResult(self.name, {})
        answers = {}
        for r in results.values():
            a = r["answer"]
            answers.setdefault(r["agent"], []).append({
                "answer": a["answer"], "findings": a["findings"], "recommended_items": a["recommended_items"],
                "missing_information": a["missing_information"], "insufficient_evidence": a["insufficient_evidence"],
                "confidence": r["uncertainty"]["final_confidence"]})
        trusted = {"question": question, "intent": routing.get("intent"), "intent_category": routing.get("intent_category"),
                   "human_answers": human_answers, "agent_answers": answers}
        answer: ConsultingAnswer = self.llm(CONSOLIDATE_PROMPT, trusted, client_text=question,
                                            mock_context={**trusted, "mock_scenario": ctx.mock_scenario},
                                            result=result, role="primary")
        out = self.verify(answer, results, evidence, result)
        result.output = out
        return result

    def verify(self, answer: ConsultingAnswer, results: dict[str, Any], evidence: dict[str, dict[str, Any]],
               result: AgentResult) -> dict[str, Any]:
        """Deterministic check of the final answer: citations, claim labels, calibrated confidence."""
        known = set(evidence)
        removed = []
        supports = []
        seen_actions: set[str] = set()
        unique_actions = []
        for a in answer.next_actions:  # drop duplicate actions (e.g. repeated across rounds)
            key = " ".join(a.action.lower().split())
            if key not in seen_actions:
                seen_actions.add(key)
                unique_actions.append(a)
        answer.next_actions = sorted(unique_actions, key=lambda a: a.priority)
        for i, a in enumerate(answer.next_actions, start=1):
            a.priority = min(i, 10)
        for a in answer.next_actions:
            bad = [i for i in a.evidence_ids if i not in known]
            removed += bad
            a.evidence_ids = [i for i in a.evidence_ids if i in known]
            if a.claim_type == "fact" and not a.evidence_ids:
                a.claim_type = "hypothesis"
                a.confidence = min(a.confidence, 0.5)
            if a.evidence_ids:
                supports.append(support_ratio(f"{a.action} {a.why}", [evidence[i]["text"] for i in a.evidence_ids]))
        if removed:
            result.warnings.append(f"final answer cited unknown evidence {sorted(set(removed))}; removed")
        consulted = sorted({r["agent"] for r in results.values()})
        answer.domains_consulted = consulted  # type: ignore[assignment]
        agent_conf = [r["uncertainty"]["final_confidence"] for r in results.values()]
        cap = (sum(agent_conf) / len(agent_conf) + 0.1) if agent_conf else 0.2
        if any(r["answer"]["insufficient_evidence"] for r in results.values()):
            cap = min(cap, 0.55)
        overall = round(min(answer.overall_confidence, cap), 3)
        if overall < answer.overall_confidence:
            answer.uncertainty_note = (answer.uncertainty_note + " Confidence capped by specialist evidence strength."
                                       ).strip()
        answer.overall_confidence = overall
        cited = sorted({i for a in answer.next_actions for i in a.evidence_ids}
                       | {i for r in results.values() for f in r["answer"]["findings"] for i in f["evidence_ids"]})
        citations = [{"evidence_id": i, "title": evidence[i]["title"], "page": evidence[i].get("page"),
                      "source": evidence[i]["source"], "doc_type": evidence[i]["doc_type"]} for i in cited if i in evidence]
        level = "high" if overall >= 0.7 else "medium" if overall >= 0.45 else "low"
        return {
            "answer": answer.model_dump(mode="json"),
            "citations": citations,
            "quality": {
                "overall_confidence": overall, "confidence_level": level,
                "citation_validity": round(1 - len(removed) / max(len(removed) + len(cited), 1), 3),
                "actions_with_evidence": round(sum(1 for a in answer.next_actions if a.evidence_ids)
                                               / max(len(answer.next_actions), 1), 3),
                "action_groundedness": round(sum(supports) / len(supports), 3) if supports else 0.0,
                "actions_measurable": all(a.metric and a.target and a.timeframe for a in answer.next_actions),
                "domains_consulted": consulted,
                "fabricated_citations_removed": len(removed) + sum(
                    r["uncertainty"]["fabricated_citations_removed"] for r in results.values()),
            },
        }
