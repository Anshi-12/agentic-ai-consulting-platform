"""Deterministic intent detection and routing rules.

Used three ways:
1. as the router in offline/mock mode,
2. as a baseline the LLM router is evaluated against,
3. as a safety net: if the LLM router returns no valid tasks for a clearly
   in-domain question, the rule-based route is used and the event is logged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "technology": (
        "cisco router", "router", "routers", "routing", "switch", "catalyst", "isr", "asr", "sd-wan", "sdwan", "wan",
        "network", "connectivity", "outage", "ipsec", "vpn", "edge platform", "firmware", "interface", "ibm",
        "google cloud", "gcp", "mcp", "model context protocol", "rag", "retrieval augmented", "cloud", "datasheet",
        "throughput", "packet", "bandwidth", "latency",
    ),
    "clinical": (
        "clinical", "trial", "trials", "patient", "patients", "diagnostic", "diagnosis", "laboratory", "lab",
        "lab requirements", "pathology", "slide scanner", "whole-slide", "whole slide", "stain", "stains",
        "seahorse", "cell metabolic", "metabolism", "mitochondria", "instrument", "covid", "diabetes", "diabetic",
        "heart attack", "cardiac", "myocardial", "knee", "exoskeleton", "disease", "medical", "hospital", "glycemic",
        "wearable", "health",
    ),
    "sec": (
        "sec", "sec filing", "10-k", "10k", "annual report", "filing", "filings", "revenue", "earnings", "financial",
        "finance", "risk factor", "risk factors", "competitive risk", "competition", "operational risk", "margin",
        "profit", "cash flow", "debt", "investor", "shareholder", "alphabet", "fiscal", "guidance", "acquisition",
        "segment", "backlog", "remaining performance obligation", "dividend", "share repurchase",
    ),
    "corporate_responsibility": (
        "food security", "food insecurity", "hunger", "nutrition", "malnutrition", "sofi", "sustainability",
        "sustainable", "esg", "csr", "corporate responsibility", "social responsibility", "climate", "agriculture",
        "agrifood", "famine", "poverty", "healthy diet", "diets", "war zone", "war zones", "food price",
        "food prices", "stunting", "wasting", "undernourishment",
    ),
}
COMPANY_WORDS = {"cisco": "Cisco", "agilent": "Agilent", "alphabet": "Alphabet", "google": "Alphabet"}
FINANCE_HINTS = ("revenue", "risk", "financial", "10-k", "annual report", "sec", "earnings", "competitive",
                 "margin", "profit", "invest", "fiscal", "filing", "operational")


@dataclass
class RuleRoute:
    domains: list[str]
    scores: dict[str, int]
    companies: list[str]
    year: int | None
    category: str
    needs_clarification: bool


_PATTERNS = {d: [re.compile(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])") for w in words]
             for d, words in DOMAIN_KEYWORDS.items()}


def _score(text: str) -> dict[str, int]:
    t = text.lower()
    return {d: sum(1 for p in pats if p.search(t)) for d, pats in _PATTERNS.items()}


def detect_companies(text: str) -> list[str]:
    t = text.lower()
    found = []
    for word, name in COMPANY_WORDS.items():
        if re.search(rf"\b{word}\b", t) and name not in found:
            found.append(name)
    return found


def detect_year(text: str) -> int | None:
    m = re.findall(r"\b(20[12]\d)\b", text)
    return int(m[-1]) if m else None


def route_by_rules(question: str) -> RuleRoute:
    scores = _score(question)
    t = question.lower()
    companies = detect_companies(question)
    if "agilent" in t:  # Agilent is a clinical vendor AND an SEC filer: disambiguate by intent
        if any(re.search(rf"\b{re.escape(h)}", t) for h in FINANCE_HINTS):
            scores["sec"] += 2
        else:
            scores["clinical"] += 2
    if "cisco" in t and not any(h in t for h in FINANCE_HINTS):
        scores["technology"] += 2
    if ("cisco" in t or "alphabet" in t) and any(h in t for h in FINANCE_HINTS):
        scores["sec"] += 1
    domains = [d for d, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0]
    top = scores[domains[0]] if domains else 0
    domains = [d for d in domains if scores[d] >= max(1, top // 3)]  # keep strong secondary domains only
    needs_clar = not domains and len(question.split()) < 12
    if not domains:
        category = "unclear" if needs_clar else "out_of_scope"
    elif len(domains) > 1:
        category = "cross_domain"
    else:
        d = domains[0]
        if d == "technology":
            category = "troubleshooting" if any(w in t for w in ("fail", "failure", "outage", "intermittent", "error",
                                                                  "down", "crash", "drop")) else "product_recommendation"
        elif d == "clinical":
            category = "product_recommendation" if any(w in t for w in ("instrument", "which", "replace", "scanner",
                                                                        "equipment", "meets")) else "research_question"
        elif d == "sec":
            category = "financial_risk"
        else:
            category = "sustainability"
    return RuleRoute(domains, scores, companies, detect_year(question), category, needs_clar)


# Query expansion so keyword retrieval also finds paraphrases (dense embeddings do this natively).
EXPANSIONS = {
    "war": "conflict", "war zones": "conflict-affected countries", "prices": "inflation cost", "price": "inflation cost",
    "outage": "failure downtime", "intermittent": "instability failure", "replacement": "hardware module spare",
    "risks": "risk factors", "competitive": "competition", "esg": "sustainability", "hunger": "undernourishment",
    "reason": "drivers", "reasons": "drivers causes", "lab": "laboratory", "investments": "investment financing",
}


def expand_query(q: str) -> str:
    t = q.lower()
    extra = [v for k, v in EXPANSIONS.items() if re.search(rf"\b{re.escape(k)}\b", t)]
    return q + (" " + " ".join(extra) if extra else "")
