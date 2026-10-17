"""Offline responders for the multi-agent consulting graph (LLM_PROVIDER_MODE=mock).

Routing uses the deterministic intent rules; domain answers are *extractive*
(the best-matching sentences from the actually retrieved passages, with their
evidence IDs), so offline runs return real, grounded content from the
professor's documents. Live mode replaces these with model-written answers.

mock_scenario values: need_human (manager asks the human on round 1),
fabricate (domain agent cites a non-existent evidence ID).
"""
from __future__ import annotations

from typing import Any

from backend.agents.intent_rules import detect_companies, detect_year, expand_query, route_by_rules
from backend.ai.provider_factory import ChatRequest
from backend.rag.text import split_sentences, tokenize

DOMAIN_METRICS = {
    "technology": ("Network availability on affected sites", "≥ 99.9% monthly", "90 days"),
    "clinical": ("Requirement fit confirmed in vendor demo/pilot", "All must-have requirements met", "60 days"),
    "sec": ("Vendor risk items reviewed in supplier scorecard", "All material risks owned and mitigated", "Next quarter"),
    "corporate_responsibility": ("Programme outcome KPI (e.g. food-insecure households reached)", "Baseline + target set",
                                 "12 months"),
}


def manager_plan(req: ChatRequest) -> dict[str, Any]:
    ctx = req.context
    question = ctx["question"] + " " + " ".join(ctx.get("human_answers", []))
    r = route_by_rules(question)
    if r.needs_clarification and not ctx.get("human_answers"):
        return {"intent": "Unclear request", "intent_category": "unclear", "tasks": [], "needs_clarification": True,
                "clarifying_questions": ["Which area does this concern: network technology, clinical/laboratory "
                                         "equipment, company financials (SEC filings), or food security/sustainability?",
                                         "What decision are you trying to make, and what constraints apply?"],
                "plan": ["Wait for clarification"], "known_facts": [], "confidence": 0.2}
    if not r.domains:
        return {"intent": "Outside the knowledge of the available agents", "intent_category": "out_of_scope",
                "tasks": [], "needs_clarification": False, "clarifying_questions": [], "plan": [],
                "known_facts": [], "confidence": 0.3}
    tasks = []
    for d in r.domains:
        comps = detect_companies(question)
        tasks.append({"agent": d, "instruction": ctx["question"],
                      "search_queries": [expand_query(ctx["question"])[:300]],
                      "company": comps[0] if (d == "sec" and len(comps) == 1) else None,
                      "year": detect_year(question), "rationale": f"keyword route score {r.scores[d]}"})
    return {"intent": ctx["question"][:200], "intent_category": r.category, "tasks": tasks,
            "is_cross_domain": len(tasks) > 1, "needs_clarification": False, "clarifying_questions": [],
            "plan": [f"{d} agent retrieves evidence" for d in r.domains] + ["Manager consolidates recommendations"],
            "known_facts": list(ctx.get("human_answers", [])), "confidence": 0.75 if len(r.domains) == 1 else 0.65}


def domain_answer(req: ChatRequest) -> dict[str, Any]:
    ctx = req.context
    instruction = ctx["instruction"]
    q = set(tokenize(expand_query(instruction)))
    scored = []
    for e in ctx.get("evidence", []):
        if e.get("injection_suspected"):
            continue
        for s in split_sentences(e["text"]):
            if len(s) < 40 or len(s) > 420:
                continue
            overlap = len(q & set(tokenize(s)))
            if overlap >= 2:
                scored.append((overlap, -len(s), s, e["evidence_id"]))
    scored.sort(reverse=True)
    findings, used = [], set()
    for overlap, _, s, eid in scored:
        if eid in used and len(findings) < 2:
            continue
        if any(s[:60] == f["statement"][:60] for f in findings):
            continue
        findings.append({"statement": s, "evidence_ids": [eid], "claim_type": "fact",
                         "confidence": round(min(0.9, 0.45 + 0.08 * overlap), 2)})
        used.add(eid)
        if len(findings) >= 4:
            break
    if ctx.get("mock_scenario") == "fabricate" and findings:
        findings[0]["evidence_ids"].append("SEC-0000000000")
    ta = ctx.get("trial_analytics")
    if ta:
        findings.insert(0, {"statement": (f"{ta['population_size']} trials match the filters {ta['filters_applied']}; "
                                          f"status breakdown {ta['status_counts']}."),
                            "evidence_ids": [], "claim_type": "calculated", "confidence": 0.9})
    items = [{"name": p["name"], "item_type": "instrument" if ctx["agent"] == "clinical" else "product",
              "reason": p["explanation"], "evidence_ids": p["catalog_ids"][:1], "match_score": p["match_score"]}
             for p in ctx.get("product_matches", [])[:3]]
    missing = []
    if ctx["agent"] == "technology":
        missing = ["Exact device model and software version", "Interface error counters and logs around failures"]
        if not ctx.get("catalog_enriched_with_datasheets"):
            missing.append("Full datasheet text (run scripts/fetch_web_sources.py to index Cisco datasheets)")
    insufficient = not [f for f in findings if f["claim_type"] != "calculated"] and not items and not ta
    answer = (" ".join(f["statement"] for f in findings[:2]) if findings else
              "The retrieved evidence does not answer this instruction.")
    return {"answer": answer[:900], "findings": findings, "recommended_items": items, "missing_information": missing,
            "follow_up_questions": [], "insufficient_evidence": insufficient,
            "confidence": 0.25 if insufficient else round(min(0.85, 0.5 + 0.08 * len(findings)), 2)}


def progress_check(req: ChatRequest) -> dict[str, Any]:
    ctx = req.context
    rnd = ctx["round"]
    if ctx.get("mock_scenario") == "need_human" and rnd == 1 and not any(
            "human" in str(f).lower() for f in ctx["ledger"].get("facts", [])):
        return {"is_resolved": False, "progress_made": True, "reason": "Device details are needed",
                "next_tasks": [], "need_human": True,
                "question_for_human": "What exact model and software version is the affected device running?"}
    weak = [r for r in ctx["agent_results"].values() if r["insufficient_evidence"]]
    if weak and rnd < ctx["max_rounds"]:
        r = weak[0]
        return {"is_resolved": False, "progress_made": True, "reason": f"{r['agent']} lacked evidence; broadening",
                "next_tasks": [{"agent": r["agent"], "instruction": f"Overview: {ctx['question']}"[:400],
                                "search_queries": [expand_query(ctx["question"])[:200] + " overview"]}],
                "need_human": False}
    return {"is_resolved": True, "progress_made": True,
            "reason": "All routed agents returned evidence" if not weak else "Stopping: remaining gaps marked",
            "next_tasks": [], "need_human": False}


def consolidate(req: ChatRequest) -> dict[str, Any]:
    ctx = req.context
    actions, facts, hyps, missing, risks = [], [], [], [], []
    prio = 1
    for agent, answers in ctx["agent_answers"].items():
        metric, target, timeframe = DOMAIN_METRICS[agent]
        for a in answers:
            missing += a["missing_information"]
            if agent == "technology" and a["missing_information"]:
                actions.append({"action": "Collect diagnostics before ordering parts: " + "; ".join(a["missing_information"][:2]),
                                "why": "Separates confirmed faults from hypotheses so procurement targets the actual failure",
                                "expected_impact": "Avoids replacing healthy components", "metric": "Root cause confirmed",
                                "target": "Yes", "timeframe": "1 week", "owner": "Network operations", "priority": prio,
                                "domains": [agent], "evidence_ids": [], "claim_type": "assumption", "confidence": 0.6})
                prio += 1
            for it in a["recommended_items"][:2]:
                actions.append({"action": f"Shortlist {it['name']}", "why": it["reason"],
                                "expected_impact": "Fit-for-purpose product selected faster", "metric": metric,
                                "target": target, "timeframe": timeframe, "owner": "Procurement", "priority": prio,
                                "domains": [agent], "evidence_ids": it["evidence_ids"], "claim_type": "fact",
                                "confidence": round(0.4 + 0.4 * (it.get("match_score") or 0.5), 2)})
                prio += 1
            for f in a["findings"][:2]:
                (facts if f["claim_type"] in ("fact", "calculated") else hyps).append(f["statement"][:300])
                if f["claim_type"] == "fact" and f["evidence_ids"]:
                    gist = f["statement"] if len(f["statement"]) <= 150 else f["statement"][:150].rsplit(" ", 1)[0] + "…"
                    actions.append({"action": f"Factor into the decision: {gist}",
                                    "why": f"Documented in the {agent.replace('_', ' ')} knowledge base",
                                    "expected_impact": "Decision grounded in authoritative evidence", "metric": metric,
                                    "target": target, "timeframe": timeframe, "owner": "Engagement lead",
                                    "priority": prio, "domains": [agent], "evidence_ids": f["evidence_ids"],
                                    "claim_type": "fact", "confidence": f["confidence"]})
                    prio += 1
            if a["insufficient_evidence"]:
                risks.append(f"{agent} evidence insufficient")
    if not actions:
        actions = [{"action": "Provide more detail or documents for this request", "why": "No evidence answered it",
                    "expected_impact": "Enables a grounded recommendation", "metric": "Request clarified",
                    "target": "Yes", "timeframe": "Now", "owner": "Client lead", "priority": 1, "domains": [],
                    "evidence_ids": [], "claim_type": "assumption", "confidence": 0.3}]
    agents = list(ctx["agent_answers"])
    conf = [a["confidence"] for v in ctx["agent_answers"].values() for a in v] or [0.3]
    return {
        "headline": f"Recommendations for: {ctx['question'][:120]}",
        "summary": " ".join(a["answer"] for v in ctx["agent_answers"].values() for a in v)[:800] or "No evidence found.",
        "domains_consulted": agents, "next_actions": actions[:6],
        "cross_domain_insights": ([f"Combined evidence from {', '.join(agents)} agents; see actions by domain."]
                                  if len(agents) > 1 else []),
        "risks": risks, "confirmed_facts": facts[:6], "hypotheses": hyps[:4], "missing_information": missing[:6],
        "overall_confidence": round(sum(conf) / len(conf), 2),
        "uncertainty_note": "Offline mode: findings are extracted verbatim from retrieved passages, not model-written.",
    }


CONSULTING_RESPONDERS = {"manager_plan": manager_plan, "domain_answer": domain_answer,
                         "progress_check": progress_check, "consolidate": consolidate}
