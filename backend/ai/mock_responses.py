"""Deterministic mock responders used when LLM_PROVIDER_MODE=mock.

They read the structured `ChatRequest.context` each agent sends (not the prompt
text), so mock runs exercise the real orchestration, tools, validation, QA
checks and routing without external credentials. Responses are intentionally
plain; the point is schema-valid, context-dependent output.

Special `mock_scenario` values (set on the workflow request, mock mode only):
    qa_failure_once  - first strategy draft cites a non-existent evidence ID
    qa_always_fail   - every strategy draft does; exercises QA escalation
"""
from __future__ import annotations

from typing import Any

from backend.ai.provider_factory import ChatRequest


def _ctx(req: ChatRequest) -> dict[str, Any]:
    return req.context or {}


def _clean_evidence(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in ctx.get("evidence_index", []) if not e.get("injection_suspected")]


def _pick(evidence: list[dict[str, Any]], words: set[str], n: int = 2) -> list[str]:
    scored = []
    for e in evidence:
        hay = set(" ".join([e.get("title", ""), " ".join(e.get("tags", []))]).lower().split())
        scored.append((len(words & hay), e.get("reliability", 0.5), e["evidence_id"]))
    scored.sort(reverse=True)
    return [s[2] for s in scored[:n] if s[0] > 0] or [s[2] for s in scored[:1]]


# ---------------------------------------------------------------------------
def classify_request(req: ChatRequest) -> dict[str, Any]:
    brief = _ctx(req).get("brief", "").lower()
    ptype = "other"
    for key, words in {
        "profitability": ("margin", "profit", "profitability"),
        "cost_reduction": ("cost", "savings", "efficiency"),
        "growth": ("growth", "revenue", "expand"),
        "market_entry": ("enter", "entry", "new market"),
        "operations": ("operations", "supply chain", "fulfillment"),
        "digital_transformation": ("digital", "ai", "automation"),
    }.items():
        if any(w in brief for w in words):
            ptype = key
            break
    industry = _ctx(req).get("industry_hint") or "General business"
    complexity = "high" if len(brief.split()) > 80 else "medium" if len(brief.split()) > 25 else "low"
    return {"industry": industry, "problem_type": ptype, "complexity": complexity, "urgency": "medium"}


def client_discovery(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    brief: str = ctx.get("brief", "")
    answers = ctx.get("clarification_answers") or {}
    client = ctx.get("client_name") or "the client"
    if len(brief.split()) < 25 and not answers:
        return {
            "status": "needs_clarification",
            "problem_statement": None,
            "engagement_scope": None,
            "missing_information": [
                {"item": "Specific business objective and target", "why_needed": "Cannot scope without a measurable goal", "priority": "high"},
                {"item": "Time horizon and budget constraints", "why_needed": "Shapes feasible options", "priority": "medium"},
            ],
            "clarifying_questions": [
                "What specific outcome would make this engagement a success (metric and target)?",
                "What time horizon and budget constraints apply?",
                "Which business units, products or regions are in scope?",
            ],
            "assumptions": [],
            "confidence": 0.3,
        }
    answer_text = " ".join(str(v) for v in answers.values()) if answers else ""
    objective = ctx.get("objectives") or ["Restore profitability to target levels", "Protect core customer relationships"]
    return {
        "status": "complete",
        "problem_statement": {
            "summary": f"{client} faces the challenge described in the brief: {brief[:220]}"
                       + (f" Clarifications: {answer_text[:160]}" if answer_text else ""),
            "key_question": f"What should {client} do over the next 12-18 months to address its core performance gap?",
            "business_objectives": objective,
            "constraints": ctx.get("constraints") or ["Limited capital budget", "No headcount reductions in customer-facing teams"],
            "stakeholders": [
                {"role": "Engagement sponsor", "interest": "Financial performance", "influence": "high"},
                {"role": "Operations leadership", "interest": "Feasibility of changes", "influence": "medium"},
                {"role": "Commercial leadership", "interest": "Revenue and customer impact", "influence": "high"},
            ],
            "success_metrics": [
                {"metric": "Gross margin", "target": "+3 percentage points", "timeframe": "18 months"},
                {"metric": "Core customer retention", "target": "No decline", "timeframe": "12 months"},
            ],
        },
        "engagement_scope": {
            "in_scope": ["Diagnostic of performance drivers", "Strategic options and business case", "Implementation roadmap"],
            "out_of_scope": ["Systems implementation", "Organizational redesign"],
            "deliverables": ["Diagnostic findings", "Option evaluation", "Recommendations and roadmap"],
            "estimated_duration_weeks": 8,
        },
        "missing_information": [
            {"item": "SKU-level margin data", "why_needed": "Precise driver attribution", "priority": "medium"},
        ],
        "clarifying_questions": [],
        "assumptions": ["Financial data supplied by the data module is accurate"],
        "confidence": 0.8,
    }


def problem_structuring(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    key_q = (ctx.get("problem_statement") or {}).get("key_question", "How can the client close its performance gap?")
    return {
        "issue_tree": {
            "id": "Q0", "question": key_q, "hypothesis_ids": [],
            "children": [
                {"id": "Q1", "question": "Is pricing and promotion policy eroding margin?", "hypothesis_ids": ["H1"], "children": []},
                {"id": "Q2", "question": "Are operating and fulfillment costs above benchmark?", "hypothesis_ids": ["H2"], "children": []},
                {"id": "Q3", "question": "Can channel or customer mix shift improve economics?", "hypothesis_ids": ["H3"], "children": []},
            ],
        },
        "hypotheses": [
            {"id": "H1", "statement": "Excessive promotion frequency and depth is the largest driver of margin decline.",
             "rationale": "Brief cites margin pressure", "workstream_id": "WS1", "required_evidence": ["Promotion calendar", "Margin bridge"]},
            {"id": "H2", "statement": "Fulfillment network inefficiency adds avoidable cost per order.",
             "rationale": "Cost growth outpacing volume", "workstream_id": "WS2", "required_evidence": ["Warehouse utilization", "Carrier rates"]},
            {"id": "H3", "statement": "Channel mix changes can add profitable growth without deepening discounts.",
             "rationale": "Untested channels exist", "workstream_id": "WS3", "required_evidence": ["Channel contribution", "Peer benchmarks"]},
        ],
        "workstreams": [
            {"id": "WS1", "name": "Pricing and promotions", "objective": "Quantify promotion impact on margin",
             "analytical_questions": ["How have promotion depth and frequency changed?", "How price-sensitive is each segment?"],
             "required_evidence": ["Margin trend", "Promotion calendar", "Customer survey"],
             "search_queries": ["pricing promotion markdown margin", "customer segment price sensitivity"],
             "depends_on": [], "frameworks": ["Customer segmentation"]},
            {"id": "WS2", "name": "Cost to serve", "objective": "Identify fulfillment and logistics savings",
             "analytical_questions": ["Where is fulfillment capacity under-used?", "Are carrier rates competitive?"],
             "required_evidence": ["Warehouse utilization", "Carrier contracts"],
             "search_queries": ["fulfillment warehouse cost operations", "carrier freight logistics cost"],
             "depends_on": [], "frameworks": ["Value chain analysis"]},
            {"id": "WS3", "name": "Channel and growth options", "objective": "Assess channel mix opportunities",
             "analytical_questions": ["What is contribution by channel?", "What have peers achieved on marketplaces?"],
             "required_evidence": ["Channel dashboard", "Benchmarks"],
             "search_queries": ["channel marketplace wholesale contribution", "marketplace growth benchmark"],
             "depends_on": ["WS1"], "frameworks": ["Porter's Five Forces"]},
        ],
        "frameworks": [
            {"name": "Customer segmentation", "applies_to": "WS1", "rationale": "Price sensitivity differs by segment"},
            {"name": "Value chain analysis", "applies_to": "WS2", "rationale": "Locate cost leakage across fulfillment"},
            {"name": "Porter's Five Forces", "applies_to": "WS3", "rationale": "Marketplace power affects channel economics"},
        ],
        "mece_check": {"is_mece": True, "notes": "Price, cost and mix levers are mutually exclusive and cover the margin equation."},
    }


def hypothesis_evaluation(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    findings: dict[str, list[dict[str, Any]]] = ctx.get("findings", {})
    evals, gaps = [], []
    for h in ctx.get("hypotheses", []):
        ev = [e for e in findings.get(h["workstream_id"], []) if not e.get("injection_suspected")]
        ids = [e["evidence_id"] for e in ev][:3]
        if ids:
            evals.append({"hypothesis_id": h["id"], "verdict": "supported", "evidence_ids": ids,
                          "rationale": f"Evidence from {len(ids)} source(s) is consistent with the hypothesis.",
                          "confidence": 0.75 if len(ids) > 1 else 0.6})
        else:
            evals.append({"hypothesis_id": h["id"], "verdict": "inconclusive", "evidence_ids": [],
                          "rationale": "No supporting evidence was returned by the research tools.", "confidence": 0.3})
            gaps.append(f"No evidence for {h['id']}")
    if not evals:
        evals = [{"hypothesis_id": "H0", "verdict": "inconclusive", "evidence_ids": [], "rationale": "No hypotheses supplied", "confidence": 0.2}]
    return {"evaluations": evals, "evidence_gaps": gaps}


def strategy(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    evidence = _clean_evidence(ctx)
    scenarios = [s for s in ctx.get("financial_scenarios", []) if s.get("scenario") != "Baseline"]
    scenario_flag = ctx.get("mock_scenario")
    rnd = ctx.get("revision_round", 0)
    fake = scenario_flag == "qa_always_fail" or (scenario_flag == "qa_failure_once" and rnd == 0)

    titles = [s["scenario"] for s in scenarios] or ["Commercial discipline program", "Cost-to-serve program", "Growth channel pilot"]
    keywords = [{"pricing", "promotion", "margin", "discount"}, {"fulfillment", "cost", "warehouse", "carrier"},
                {"channel", "marketplace", "growth", "benchmark"}]
    criteria = [{"name": "Financial impact", "weight": 0.4}, {"name": "Feasibility", "weight": 0.3},
                {"name": "Risk", "weight": 0.2}, {"name": "Strategic fit", "weight": 0.1}]
    score_sets = [{"Financial impact": 5, "Feasibility": 4, "Risk": 4, "Strategic fit": 4},
                  {"Financial impact": 4, "Feasibility": 3, "Risk": 3, "Strategic fit": 4},
                  {"Financial impact": 3, "Feasibility": 3, "Risk": 2, "Strategic fit": 3}]
    options, recs = [], []
    for i, title in enumerate(titles[:3]):
        ev = _pick(evidence, keywords[i % 3]) if evidence else []
        options.append({
            "id": f"OPT{i + 1}", "title": title, "description": f"Pursue {title.lower()}.",
            "costs": "One-off investment per financial analysis; management attention.",
            "benefits": "Improves margin per calculated scenario metrics.",
            "risks": ["Execution capacity", "Customer reaction"] if i == 0 else ["Transition disruption"] if i == 1 else ["Brand control", "Marketplace fees"],
            "dependencies": [] if i == 0 else ["OPT1"],
            "scores": score_sets[i % 3], "weighted_score": 9.9, "evidence_ids": ev,
        })
        if i < 2:
            cite = ev + (["EV-FAKE01"] if fake and i == 0 else [])
            recs.append({
                "id": f"R{i + 1}", "title": f"Implement {title.lower()}",
                "rationale": "Supported by cited evidence and calculated scenario metrics.",
                "option_id": f"OPT{i + 1}", "priority": i + 1,
                "owner": "Chief Commercial Officer" if i == 0 else "Chief Operating Officer",
                "time_horizon": "0-3 months" if i == 0 else "3-12 months",
                "measurable_outcomes": [{"metric": "Gross margin", "target": "+1.5 to +2.5 points" if i == 0 else "+1 point", "timeframe": "12 months"}],
                "evidence_ids": cite, "claim_type": "fact" if cite else "hypothesis",
                "confidence": 0.78 if cite else 0.4, "low_confidence_flag": not cite,
            })
    if ctx.get("revision_feedback"):
        recs[0]["rationale"] += f" Revised per reviewer feedback: {str(ctx['revision_feedback'])[:120]}"
    return {
        "decision_criteria": criteria,
        "options": options if len(options) >= 2 else options + [dict(options[0], id="OPT2", title="Status quo")],
        "recommendations": recs,
        "trade_offs": ["Faster margin recovery versus short-term volume loss", "Capital spend on network versus growth investment"],
        "assumptions": ["Scenario cash flows supplied by the data module are reasonable estimates"],
    }


def financial_scenario(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    scenarios = [s for s in ctx.get("scenarios", []) if s.get("scenario") != "Baseline"]
    if not ctx.get("data_available") or not scenarios:
        return {"summary": "Financial inputs were not available from the data module, so no metrics were calculated.",
                "preferred_scenario": None, "scenario_comparison": [], "key_drivers": [],
                "sensitivities": [], "caveats": ["Request a financial model from the client before quantifying options."],
                "data_available": False, "confidence": 0.2}

    def npv(s):
        return next((m["value"] for m in s["metrics"] if m["name"] == "npv"), None) or 0

    best = max(scenarios, key=npv)
    return {
        "summary": f"'{best['scenario']}' has the highest calculated NPV among {len(scenarios)} scenarios; all inputs are estimates or forecasts.",
        "preferred_scenario": best["scenario"],
        "scenario_comparison": [f"{s['scenario']}: NPV {npv(s):,.0f} (calculated)" for s in scenarios],
        "key_drivers": ["Size and timing of annual cash flows", "Upfront investment"],
        "sensitivities": ["Results are sensitive to the discount rate and to forecast cash flows in later years"],
        "caveats": ["Cash flows are client-supplied estimates, not audited actuals"],
        "data_available": True, "confidence": 0.75,
    }


def consulting_qa(req: ChatRequest) -> dict[str, Any]:
    return {"verdict": "pass", "addresses_original_problem": True, "issues": [], "low_confidence_findings": [],
            "requires_human_review": False, "route_to": "none", "additional_research_requests": [],
            "summary": "Recommendations address the key question and cite evidence."}


def report_synthesis(req: ChatRequest) -> dict[str, Any]:
    ctx = _ctx(req)
    recs = ctx.get("recommendations", [])
    ps = ctx.get("problem_statement") or {}
    evals = ctx.get("evaluations", [])
    findings = [{"title": f"Hypothesis {e['hypothesis_id']}: {e['verdict']}", "finding": e["rationale"],
                 "evidence_ids": e["evidence_ids"], "claim_type": "fact" if e["evidence_ids"] else "hypothesis",
                 "confidence": e["confidence"]} for e in evals] or [
        {"title": "Limited evidence", "finding": "Evidence was insufficient for firm findings.", "evidence_ids": [],
         "claim_type": "hypothesis", "confidence": 0.3}]
    return {
        "executive_summary": {
            "headline": recs[0]["title"] if recs else "Further analysis required",
            "situation": ps.get("summary", "The client engaged us to address a performance challenge.")[:400],
            "complication": "Evidence indicates several compounding drivers behind the gap.",
            "resolution": "A sequenced program of the prioritized recommendations below.",
            "key_recommendations": [r["title"] for r in recs] or ["Gather missing data before deciding"],
        },
        "detailed_findings": findings,
        "strategic_recommendations": recs or [{
            "id": "R0", "title": "Collect missing evidence", "rationale": "Insufficient evidence", "priority": 1,
            "owner": "Engagement lead", "time_horizon": "0-3 months",
            "measurable_outcomes": [{"metric": "Data gaps closed", "target": "100%", "timeframe": "4 weeks"}],
            "evidence_ids": [], "claim_type": "assumption", "confidence": 0.3, "low_confidence_flag": True}],
        "implementation_roadmap": [
            {"phase": "Mobilize", "timeframe": "Weeks 0-4", "actions": ["Set up program office", "Confirm baselines"], "owner": "Engagement sponsor", "milestones": ["Baseline signed off"]},
            {"phase": "Deliver quick wins", "timeframe": "Months 1-3", "actions": [recs[0]["title"] if recs else "Close data gaps"], "owner": recs[0]["owner"] if recs else "Engagement lead", "milestones": ["First results reported"]},
            {"phase": "Scale", "timeframe": "Months 3-12", "actions": [r["title"] for r in recs[1:]] or ["Re-assess options"], "owner": "Chief Operating Officer", "milestones": ["Run-rate benefits achieved"]},
        ],
        "risk_register": [
            {"id": "RK1", "risk": "Customer volume loss exceeds plan", "likelihood": "medium", "impact": "high", "mitigation": "Pilot by region with stop-loss triggers", "owner": "Chief Commercial Officer"},
            {"id": "RK2", "risk": "Operational disruption during transition", "likelihood": "medium", "impact": "medium", "mitigation": "Phased cutover with peak-season freeze", "owner": "Chief Operating Officer"},
        ],
        "presentation_slides": [
            {"title": "Executive summary", "bullets": [r["title"] for r in recs] or ["Further analysis required"], "speaker_notes": "Lead with the answer."},
            {"title": "What we found", "bullets": [f["title"] for f in findings], "speaker_notes": ""},
            {"title": "Roadmap", "bullets": ["Mobilize", "Quick wins", "Scale"], "speaker_notes": ""},
        ],
        "citations": [{"evidence_id": e["evidence_id"], "source": e.get("source", ""), "title": e.get("title", "")}
                      for e in _clean_evidence(ctx)],
        "assumptions": ctx.get("assumptions", []),
        "limitations": ctx.get("limitations", []) or ["Analysis relies on synthetic data supplied by the data module"],
    }


DEFAULT_RESPONDERS = {
    "classify_request": classify_request,
    "client_discovery": client_discovery,
    "problem_structuring": problem_structuring,
    "hypothesis_evaluation": hypothesis_evaluation,
    "strategy": strategy,
    "financial_scenario": financial_scenario,
    "consulting_qa": consulting_qa,
    "report_synthesis": report_synthesis,
}

from backend.ai.mock_consulting import CONSULTING_RESPONDERS  # noqa: E402

DEFAULT_RESPONDERS.update(CONSULTING_RESPONDERS)
