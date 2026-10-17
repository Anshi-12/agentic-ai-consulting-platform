"""Hackathon evaluation: routing accuracy, retrieval accuracy, recommendation
relevance and response quality, measured on the professor's data.

    python -m backend.evaluation.hackathon_eval            # uses data/index and the Retail CSV
    python -m backend.evaluation.hackathon_eval --skip-consult

Writes backend/evaluation/results/hackathon_report.json and hackathon_report.md
(the markdown is ready to paste into the presentation).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from backend.agents.intent_rules import expand_query, route_by_rules
from backend.agents.base import AgentContext
from backend.agents.manager_agent import ManagerAgent
from backend.api.models import ConsultRequest
from backend.config import get_settings
from backend.contracts.auth import dev_principal
from backend.contracts.schemas import WorkflowStatus
from backend.orchestration.consulting_execution import ConsultingExecutor
from backend.rag.catalog_recommender import recommend_products
from backend.rag.index import KnowledgeBase
from backend.recommender.evaluate import evaluate as eval_basket, signal_test, split
from backend.recommender.model import DEFAULT_CSV, Params, load_transactions

HERE = Path(__file__).resolve().parent
DATA = HERE / "datasets"
DOMAINS = ("technology", "clinical", "sec", "corporate_responsibility")


def load(name: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in (DATA / name).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
def routing_metrics(cases: list[dict[str, Any]], predict) -> dict[str, Any]:
    exact = clar_ok = clar_n = cross_ok = cross_n = 0
    tp = {d: 0 for d in DOMAINS}
    fp = dict(tp)
    fn = dict(tp)
    rows = []
    for c in cases:
        pred, clar = predict(c["question"])
        gold = set(c["domains"])
        if c.get("expect_clarification"):
            clar_n += 1
            clar_ok += int(clar)
            ok = clar
        else:
            ok = (set(pred) == gold) and not clar
            if len(gold) > 1:
                cross_n += 1
                cross_ok += int(ok)
        exact += int(ok)
        for d in DOMAINS:
            tp[d] += int(d in pred and d in gold)
            fp[d] += int(d in pred and d not in gold)
            fn[d] += int(d not in pred and d in gold)
        rows.append({"id": c["id"], "gold": sorted(gold), "pred": sorted(pred), "clarify": clar, "correct": ok})
    per = {}
    for d in DOMAINS:
        p = tp[d] / (tp[d] + fp[d]) if tp[d] + fp[d] else 0.0
        r = tp[d] / (tp[d] + fn[d]) if tp[d] + fn[d] else 0.0
        per[d] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    return {"cases": len(cases), "exact_match_accuracy": round(exact / len(cases), 3),
            "cross_domain_accuracy": round(cross_ok / cross_n, 3) if cross_n else None,
            "clarification_accuracy": round(clar_ok / clar_n, 3) if clar_n else None,
            "per_domain": per, "errors": [r for r in rows if not r["correct"]]}


def rule_predict(q: str) -> tuple[list[str], bool]:
    r = route_by_rules(q)
    return r.domains, r.needs_clarification


def llm_predict_factory(ex: ConsultingExecutor):
    mgr = ManagerAgent(ex.services.runtime)
    ctx = AgentContext("eval", "hackathon", dev_principal("evaluator"))

    def predict(q: str) -> tuple[list[str], bool]:
        r = mgr.plan(ctx, q, []).output["routing"]
        return [t["agent"] for t in r["tasks"]], bool(r["needs_clarification"])
    return predict


# ---------------------------------------------------------------------------
def _matches(hit, expect: dict[str, Any]) -> bool:
    c = hit.chunk
    if expect.get("company") and c.company != expect["company"]:
        return False
    if expect.get("year") and c.year != expect["year"]:
        return False
    if expect.get("source_contains") and expect["source_contains"].lower() not in c.source.lower():
        return False
    if expect.get("text_regex") and not re.search(expect["text_regex"], f"{c.title} {c.text}", re.I):
        return False
    return True


def retrieval_metrics(kb: KnowledgeBase, cases: list[dict[str, Any]], use_filters: bool = True,
                      expand: bool = True, k: int = 5) -> dict[str, Any]:
    h1 = h3 = hk = 0
    mrr = 0.0
    misses = []
    for c in cases:
        q = expand_query(c["query"]) if expand else c["query"]
        hits = kb.search_domain(c["domain"], q, k, c.get("filters") if use_filters else None)
        rank = next((i for i, h in enumerate(hits, start=1) if _matches(h, c["expect"])), None)
        h1 += int(rank == 1)
        h3 += int(rank is not None and rank <= 3)
        hk += int(rank is not None)
        mrr += 1 / rank if rank else 0.0
        if rank is None:
            misses.append({"id": c["id"], "query": c["query"], "top": [h.chunk.title[:60] for h in hits[:3]]})
    n = len(cases)
    return {"cases": n, "hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), f"hit@{k}": round(hk / n, 3),
            "mrr": round(mrr / n, 3), "filters": use_filters, "query_expansion": expand, "misses": misses}


def product_metrics(kb: KnowledgeBase, cases: list[dict[str, Any]]) -> dict[str, Any]:
    h1 = h3 = 0
    rows = []
    for c in cases:
        items = recommend_products(kb, c["need"], c["domain"], 5)["items"]
        names = [i["name"] for i in items]
        rank = next((i for i, n in enumerate(names, start=1) if re.search(c["expected"], n, re.I)), None)
        h1 += int(rank == 1)
        h3 += int(rank is not None and rank <= 3)
        rows.append({"id": c["id"], "expected": c["expected"], "rank": rank, "top": names[:3]})
    n = len(cases)
    return {"cases": n, "hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "rows": rows}


# ---------------------------------------------------------------------------
def consult_quality(ex: ConsultingExecutor, cases: list[dict[str, Any]]) -> dict[str, Any]:
    p = dev_principal("evaluator")
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        run = ex.ask(ConsultRequest(question=c["question"]), p)
        guard = 0
        while run.status == WorkflowStatus.AWAITING_INPUT and guard < 3:
            guard += 1
            it = run.pending_interrupt
            payload = ({"type": "approval", "decision": "approve"} if it["type"] == "approval"
                       else {"type": it["type"], "answer": "No additional information is available."})
            run = ex.resume(run.execution_id, payload, p)
        r = ex.result(run.execution_id, p)
        wall = time.perf_counter() - t0
        if not r.result:
            rows.append({"id": c["id"], "status": r.status.value, "success": False})
            continue
        a, q = r.result["answer"], r.result["quality"]
        text = " ".join([a["summary"], *(n["action"] + " " + n["why"] for n in a["next_actions"]),
                         *(f["statement"] for ar in r.agent_results.values() for f in ar["answer"]["findings"])]).lower()
        terms = c.get("expected_terms", [])
        groundedness = [ar["uncertainty"]["groundedness"] for ar in r.agent_results.values()]
        rows.append({
            "id": c["id"], "status": r.status.value, "success": r.status == WorkflowStatus.COMPLETED,
            "routed": sorted({ar["agent"] for ar in r.agent_results.values()}),
            "routing_correct": sorted({ar["agent"] for ar in r.agent_results.values()}) == sorted(c["domains"]),
            "citation_validity": q["citation_validity"], "actions_with_evidence": q["actions_with_evidence"],
            "actions_measurable": q["actions_measurable"],
            "finding_groundedness": round(sum(groundedness) / len(groundedness), 3) if groundedness else 0.0,
            "answer_term_coverage": round(sum(t in text for t in terms) / len(terms), 3) if terms else None,
            "overall_confidence": q["overall_confidence"], "citations": len(r.result["citations"]),
            "rounds": len(r.progress_history), "tokens": r.usage.input_tokens + r.usage.output_tokens,
            "wall_time_s": round(wall, 2)})
    ok = [x for x in rows if x.get("success")]

    def avg(key):
        vals = [x[key] for x in ok if x.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"cases": len(rows), "workflow_success_rate": round(len(ok) / len(rows), 3) if rows else 0.0,
            "routing_accuracy_end_to_end": avg("routing_correct"), "citation_validity": avg("citation_validity"),
            "actions_with_evidence": avg("actions_with_evidence"), "actions_measurable": avg("actions_measurable"),
            "finding_groundedness": avg("finding_groundedness"), "answer_term_coverage": avg("answer_term_coverage"),
            "mean_confidence": avg("overall_confidence"), "mean_wall_time_s": avg("wall_time_s"),
            "mean_tokens": avg("tokens"), "rows": rows}


# ---------------------------------------------------------------------------
def to_markdown(rep: dict[str, Any]) -> str:
    L = [f"# Hackathon evaluation ({rep['provider_mode']} mode, retrieval: {rep['retrieval_mode']})", ""]
    r = rep["routing"]
    L += ["## Routing accuracy (manager agent)", "", "| Router | Exact match | Cross-domain | Clarification |",
          "|---|---|---|---|"]
    for name, m in r.items():
        L.append(f"| {name} | {m['exact_match_accuracy']:.0%} | {m['cross_domain_accuracy']:.0%} | "
                 f"{m['clarification_accuracy']:.0%} |")
    L += ["", f"{next(iter(r.values()))['cases']} labelled questions, including the brief's and professor's examples.", ""]
    L += ["## Retrieval accuracy", "", "| Setting | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for name, m in rep["retrieval"].items():
        L.append(f"| {name} | {m['hit@1']:.0%} | {m['hit@3']:.0%} | {m['hit@5']:.0%} | {m['mrr']:.2f} |")
    L += ["", f"{rep['retrieval']['full']['cases']} paraphrased questions; ground truth verified to exist in the source documents.", ""]
    p = rep["products"]
    L += ["## Recommendation relevance", "", f"- Product/instrument matching: hit@1 {p['hit@1']:.0%}, hit@3 {p['hit@3']:.0%} "
          f"({p['cases']} needs → expected catalog item)."]
    if rep.get("market_basket"):
        mb = rep["market_basket"]
        lo, cs, sig = mb["leave_one_out"], mb["checkout_simulation"], mb["data_signal_test"]
        L += [f"- Market basket, checkout simulation: next-item hit@5 {cs['next_item_hit_rate@5']:.1%} vs popularity "
              f"baseline {cs['baseline_hit_rate@5']:.1%} (coverage {cs['coverage']:.0%}).",
              f"- Leave-one-out hit@5 {lo['hit_rate@5']:.1%} vs baseline {lo['baseline_hit_rate@5']:.1%}.",
              f"- Data signal test: strongest pair co-occurs {sig['real_max_pair_count']}× vs "
              f"{sig['null_max_pair_count_mean']}× in shuffled baskets (p={sig['p_value_max_pair']}) → {sig['interpretation']}."]
    if rep.get("consult"):
        c = rep["consult"]
        L += ["", "## Response quality (end-to-end consultations)", "",
              f"- Workflow success: {c['workflow_success_rate']:.0%} of {c['cases']}",
              f"- End-to-end routing accuracy: {c['routing_accuracy_end_to_end']:.0%}",
              f"- Citation validity: {c['citation_validity']:.0%} · actions citing evidence: {c['actions_with_evidence']:.0%}"
              f" · measurable actions: {c['actions_measurable']:.0%}",
              f"- Finding groundedness (lexical support in cited passage): {c['finding_groundedness']:.0%}",
              f"- Expected-term coverage in answers: {c['answer_term_coverage']:.0%}",
              f"- Mean confidence {c['mean_confidence']:.0%} · mean latency {c['mean_wall_time_s']}s · mean tokens {c['mean_tokens']:.0f}"]
        if rep["provider_mode"] == "mock":
            L += ["", "_Offline mode: agent answers are extracted verbatim from retrieved passages, so groundedness and "
                  "citation validity are trivially high. Run with LLM_PROVIDER_MODE=live to evaluate model-written "
                  "answers; the same checks then catch invented or unsupported claims._"]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-consult", action="store_true")
    ap.add_argument("--skip-llm-routing", action="store_true")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    settings = get_settings()
    ex = ConsultingExecutor(settings)
    if ex.kb is None:
        print("Knowledge base not built. Run: python -m backend.rag.ingest")
        return 1
    rep: dict[str, Any] = {"provider_mode": settings.provider_mode, "retrieval_mode": ex.kb.stats()["retrieval_mode"]}
    routing = load("routing_cases.jsonl")
    rep["routing"] = {"rule-based": routing_metrics(routing, rule_predict)}
    if not settings.is_mock and not args.skip_llm_routing:
        rep["routing"][f"LLM ({settings.model_primary})"] = routing_metrics(routing, llm_predict_factory(ex))
    rc = load("retrieval_cases.jsonl")
    rep["retrieval"] = {"full": retrieval_metrics(ex.kb, rc),
                        "no entity filters": retrieval_metrics(ex.kb, rc, use_filters=False),
                        "no query expansion": retrieval_metrics(ex.kb, rc, expand=False)}
    rep["products"] = product_metrics(ex.kb, load("product_cases.jsonl"))
    if Path(args.csv).exists():
        tx = load_transactions(args.csv)
        train, test = split(tx)
        rep["market_basket"] = eval_basket(train, test, Params(), 5)
        rep["market_basket"]["data_signal_test"] = signal_test(tx)
    if not args.skip_consult:
        rep["consult"] = consult_quality(ex, load("consult_quality_cases.jsonl"))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "hackathon_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = to_markdown(rep)
    (out / "hackathon_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
