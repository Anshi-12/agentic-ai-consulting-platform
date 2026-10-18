"""The four specialist domain agents from the hackathon architecture.

Each agent: builds search queries -> calls only its own knowledge-base tools ->
answers with citations via the LLM -> deterministic post-checks (unknown IDs
removed, uncited "facts" downgraded) -> uncertainty indicators.
"""
from __future__ import annotations

import re
from typing import Any

from backend.agents.base import AgentContext, AgentResult, BaseAgent
from backend.agents.intent_rules import detect_companies, expand_query
from backend.ai.consulting_schemas import DomainAnswer
from backend.contracts.schemas import EvidenceItem
from backend.prompts.manager_prompts import domain_prompt
from backend.rag.text import tokenize

RELIABILITY = {"sec": 0.95, "corporate_responsibility": 0.95, "trial": 0.9, "pdf": 0.85, "web": 0.8, "catalog": 0.6}
SOURCE_TYPE = {"pdf": "document", "trial": "dataset", "web": "document", "catalog": "document"}
PRODUCT_INTENT = re.compile(r"\b(recommend|replace|replacement|which|order|buy|purchase|instrument|component|"
                            r"scanner|part|parts|upgrade|equipment|model|product|platform|meets|requirement)", re.I)
TRIAL_INTENT = re.compile(r"\b(trial|trials|study|studies|recruiting|phase|sponsor|enrol|enrollment|how many|"
                          r"number of)", re.I)


def support_ratio(statement: str, texts: list[str]) -> float:
    """Share of a statement's content words found in its cited evidence (lexical groundedness)."""
    words = set(tokenize(statement))
    if not words:
        return 0.0
    pool: set[str] = set()
    for t in texts:
        pool |= set(tokenize(t))
    return round(len(words & pool) / len(words), 3)


class DomainAgent(BaseAgent):
    domain: str = ""
    max_queries = 3
    per_query_k = 5
    max_evidence = 10

    @property
    def tool_key(self) -> str:
        return f"{self.domain}_agent"

    # -- query planning ---------------------------------------------------------
    def plan_queries(self, task: dict[str, Any]) -> list[str]:
        raw = list(task.get("search_queries") or []) + [task["instruction"]]
        queries = []
        for q in raw:
            q = expand_query(q.strip())[:300]
            if len(q) >= 2 and q not in queries:
                queries.append(q)
        return queries[: self.max_queries + 1]

    def filters(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        year = task.get("year")
        return [{"year": year}] if year else [{}]

    # -- tool use ---------------------------------------------------------------
    def gather(self, ctx: AgentContext, task: dict[str, Any], session) -> tuple[dict[str, dict], dict[str, Any]]:
        evidence: dict[str, dict[str, Any]] = {}
        extras: dict[str, Any] = {"queries": [], "failures": []}
        for f in self.filters(task):
            for q in self.plan_queries(task):
                res = session.call("search_knowledge_base", domain=self.domain, query=q, top_k=self.per_query_k,
                                   **{k: v for k, v in f.items() if v})
                extras["queries"].append({"query": q, **f, "success": res.success,
                                          "hits": len(res.data) if res.success else 0})
                if not res.success:
                    extras["failures"].append(res.error)
                    continue
                for rank, item in enumerate(res.data):
                    prev = evidence.get(item["evidence_id"])
                    item["_rank"] = min(rank, prev["_rank"]) if prev else rank
                    item["_query_terms"] = len(set(tokenize(q)))
                    if not prev or item["score"] > prev["score"]:
                        evidence[item["evidence_id"]] = item
        return evidence, extras

    def extra_tools(self, ctx: AgentContext, task: dict[str, Any], session, evidence: dict[str, dict],
                    extras: dict[str, Any]) -> None:
        """Hook for domain-specific tools (product matching, trial analytics)."""

    def _product_matches(self, task, session, evidence, extras) -> None:
        text = f"{task['instruction']} {' '.join(task.get('search_queries') or [])}"
        if not PRODUCT_INTENT.search(text):
            return
        res = session.call("recommend_products", domain=self.domain, need=text[:500], top_k=4)
        if not res.success:
            extras["failures"].append(res.error)
            return
        extras["product_matches"] = res.data["items"]
        extras["catalog_enriched"] = res.data["catalog_enriched"]
        idx = self.rt.tools.kb.indexes.get("catalog") if self.rt.tools.kb else None
        for item in res.data["items"]:
            for cid in item["catalog_ids"][:1]:
                c = idx.get(cid) if idx else None
                if c and cid not in evidence:
                    evidence[cid] = {"evidence_id": cid, "title": c.title, "source": c.source, "page": None,
                                     "doc_type": "catalog", "company": c.company, "text": c.text[:900],
                                     "score": 0.0, "matched_terms": item["matched_terms"], "_rank": 99,
                                     "_query_terms": 1, "metadata": {"url": c.source}}

    # -- main -----------------------------------------------------------------
    def run(self, ctx: AgentContext, task: dict[str, Any], question: str) -> AgentResult:
        result = AgentResult(self.tool_key, {})
        session = self.session(ctx, self.tool_key)
        evidence, extras = self.gather(ctx, task, session)
        self.extra_tools(ctx, task, session, evidence, extras)

        ranked = sorted(evidence.values(), key=lambda e: (e["_rank"], -e["score"]))[: self.max_evidence]
        items = [EvidenceItem(
            evidence_id=e["evidence_id"], project_id=ctx.project_id,
            source=f"{e['title']}" + (f", p.{e['page']}" if e.get("page") else ""), title=e["title"][:200],
            content=e["text"], source_type=SOURCE_TYPE.get(e["doc_type"], "document"),
            reliability=RELIABILITY.get(self.domain if e["doc_type"] == "pdf" and self.domain in RELIABILITY
                                        else e["doc_type"], 0.7),
            injection_suspected=bool(e.get("injection_suspected"))) for e in ranked]
        trusted = {
            "agent": self.domain,
            "customer_question": question,
            "instruction": task["instruction"],
            "company": task.get("company"),
            "year": task.get("year"),
            "product_matches": [{k: v for k, v in p.items() if k != "sources"} for p in extras.get("product_matches", [])],
            "catalog_enriched_with_datasheets": extras.get("catalog_enriched"),
            "trial_analytics": extras.get("trial_analytics"),
            "retrieval_failures": extras["failures"],
        }
        mock_ctx = {**trusted, "evidence": [{"evidence_id": e["evidence_id"], "title": e["title"], "text": e["text"],
                                             "page": e.get("page"), "doc_type": e["doc_type"],
                                             "injection_suspected": bool(e.get("injection_suspected"))}
                                            for e in ranked],
                    "mock_scenario": ctx.mock_scenario}
        answer: DomainAnswer = self.llm(domain_prompt(self.domain), trusted, evidence=items,
                                        tool_log=session.execution_log(), mock_context=mock_ctx, result=result)
        out = self.postcheck(answer, {e["evidence_id"]: e for e in ranked}, extras, result)
        result.output = {
            "task_id": task["task_id"], "agent": self.domain, "instruction": task["instruction"],
            "answer": out["answer"], "evidence": {k: _public(v) for k, v in out["evidence"].items()},
            "retrieval": {"queries": extras["queries"], "evidence_count": len(ranked),
                          "product_matches": extras.get("product_matches", []),
                          "trial_analytics": extras.get("trial_analytics")},
            "uncertainty": out["uncertainty"],
        }
        return self.finish(result, session)

    def postcheck(self, answer: DomainAnswer, evidence: dict[str, dict], extras: dict[str, Any],
                  result: AgentResult) -> dict[str, Any]:
        known = set(evidence)
        flagged = {k for k, v in evidence.items() if v.get("injection_suspected")}
        fabricated = 0
        supports = []
        for f in answer.findings:
            bad = [i for i in f.evidence_ids if i not in known]
            if bad:
                fabricated += len(bad)
                result.warnings.append(f"removed unknown evidence IDs {bad}")
            f.evidence_ids = [i for i in f.evidence_ids if i in known and i not in flagged]
            if f.claim_type == "fact" and not f.evidence_ids:
                f.claim_type = "hypothesis"
                f.confidence = min(f.confidence, 0.5)
                result.warnings.append("uncited 'fact' downgraded to hypothesis")
            ratio = support_ratio(f.statement, [evidence[i]["text"] for i in f.evidence_ids])
            if f.evidence_ids:
                supports.append(ratio)
        for it in answer.recommended_items:
            bad = [i for i in it.evidence_ids if i not in known]
            fabricated += len(bad)
            it.evidence_ids = [i for i in it.evidence_ids if i in known and i not in flagged]
        cited = {i for f in answer.findings for i in f.evidence_ids} | {
            i for it in answer.recommended_items for i in it.evidence_ids}
        partial = False
        cited_facts = [f for f in answer.findings if f.claim_type == "fact" and f.evidence_ids]
        if answer.insufficient_evidence and len(cited_facts) >= 2:
            # The model answered its domain's part with cited facts but flagged the overall question as
            # insufficient (usually because other parts belong to other agents). Treat it as a partial answer.
            answer.insufficient_evidence = False
            partial = True
            result.warnings.append("agent flagged insufficient despite >=2 cited facts; treated as partial answer")

        top = sorted(evidence.values(), key=lambda e: e["_rank"])[:3]
        strength = 0.0
        if top:
            strength = sum(min(1.0, len(e.get("matched_terms", [])) / max(e.get("_query_terms", 1), 1))
                           for e in top) / len(top)
        groundedness = round(sum(supports) / len(supports), 3) if supports else 0.0
        model_conf = answer.confidence
        final = min(model_conf, 0.3 + 0.7 * strength) if evidence else min(model_conf, 0.2)
        if answer.insufficient_evidence:
            final = min(final, 0.3)
        if partial:
            final = min(final, 0.6)
        if supports and groundedness < 0.3:
            final = min(final, 0.45)
        final = round(final, 3)
        level = "high" if final >= 0.7 else "medium" if final >= 0.45 else "low"
        return {
            "answer": answer.model_dump(mode="json"),
            "evidence": {k: v for k, v in evidence.items() if k in cited or v["_rank"] < 3},
            "uncertainty": {"model_confidence": model_conf, "retrieval_strength": round(strength, 3),
                            "groundedness": groundedness, "final_confidence": final, "level": level,
                            "evidence_retrieved": len(evidence), "evidence_cited": len(cited),
                            "fabricated_citations_removed": fabricated,
                            "partial_answer": partial,
                            "insufficient_evidence": answer.insufficient_evidence},
        }


def _public(e: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in e.items() if not k.startswith("_")}


class TechnologyAgent(DomainAgent):
    name = "technology_agent"
    domain = "technology"

    def extra_tools(self, ctx, task, session, evidence, extras):
        self._product_matches(task, session, evidence, extras)


class ClinicalAgent(DomainAgent):
    name = "clinical_agent"
    domain = "clinical"

    def extra_tools(self, ctx, task, session, evidence, extras):
        self._product_matches(task, session, evidence, extras)
        text = f"{task['instruction']} {' '.join(task.get('search_queries') or [])}"
        if TRIAL_INTENT.search(text):
            res = session.call("query_clinical_trials", query=task["instruction"][:300], top_k=5)
            if res.success:
                trials = res.data.pop("matching_trials")
                extras["trial_analytics"] = res.data
                for rank, t in enumerate(trials):
                    t["_rank"], t["_query_terms"] = rank, max(len(set(tokenize(task["instruction"]))), 1)
                    evidence.setdefault(t["evidence_id"], t)
            else:
                extras["failures"].append(res.error)


class SECAgent(DomainAgent):
    name = "sec_agent"
    domain = "sec"
    per_query_k = 4

    def filters(self, task):
        companies = [task["company"]] if task.get("company") else []
        companies += [c for c in detect_companies(task["instruction"]) if c not in companies]
        year = task.get("year")
        if not companies:
            return [{"year": year}] if year else [{}]
        return [{"company": c, **({"year": year} if year else {})} for c in companies[:3]]


class CorporateResponsibilityAgent(DomainAgent):
    name = "corporate_responsibility_agent"
    domain = "corporate_responsibility"

    def filters(self, task):
        year = task.get("year")
        return [{"year": year}] if year in (2023, 2024) else [{}]


DOMAIN_AGENTS = {"technology": TechnologyAgent, "clinical": ClinicalAgent, "sec": SECAgent,
                 "corporate_responsibility": CorporateResponsibilityAgent}
