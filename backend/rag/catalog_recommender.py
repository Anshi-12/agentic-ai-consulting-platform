"""Content-based product/component/instrument recommender.

Similarity matching (BM25, plus dense vectors when available) between the
customer's stated need and catalog entries built from the professor's link
lists (Cisco routers, Agilent instruments, Google GCP), enriched with fetched
datasheet text when `scripts/fetch_web_sources.py` has been run. Results are
ranked, scored 0-1 relative to the best match, and explained by matched terms.
"""
from __future__ import annotations

from typing import Any

from backend.rag.index import KnowledgeBase


def recommend_products(kb: KnowledgeBase, need: str, domain: str | None = None, top_k: int = 5) -> dict[str, Any]:
    idx = kb.indexes.get("catalog")
    if idx is None or not idx.chunks:
        raise LookupError("product catalog not built")
    filters = {"domain": domain} if domain else None
    hits = idx.search(need, top_k * 3, None, kb.embedder)
    if domain:
        hits = [h for h in hits if h.chunk.domain == domain]
    grouped: dict[str, dict[str, Any]] = {}
    for h in hits:
        name = h.chunk.title
        g = grouped.setdefault(name, {"name": name, "vendor": h.chunk.company, "score": 0.0, "matched_terms": set(),
                                      "sources": [], "evidence_ids": [], "enriched": False})
        g["score"] = max(g["score"], h.score)
        g["matched_terms"].update(h.matched_terms)
        g["sources"].append(h.chunk.source)
        g["evidence_ids"].append(h.chunk.chunk_id)
        g["enriched"] = g["enriched"] or bool(h.chunk.metadata.get("enriched"))
    ranked = sorted(grouped.values(), key=lambda g: (len(g["matched_terms"]), g["score"]), reverse=True)[:top_k]
    top = max((len(g["matched_terms"]) + g["score"] * 10 for g in ranked), default=1.0) or 1.0
    items = []
    for rank, g in enumerate(ranked, start=1):
        raw = len(g["matched_terms"]) + g["score"] * 10
        items.append({
            "rank": rank, "name": g["name"], "vendor": g["vendor"], "match_score": round(raw / top, 3),
            "matched_terms": sorted(g["matched_terms"]), "sources": g["sources"][:3],
            "catalog_ids": g["evidence_ids"][:3],
            "explanation": (f"Matches need on: {', '.join(sorted(g['matched_terms'])) or 'semantic similarity only'}"
                            + ("" if g["enriched"] else " (catalog entry only; datasheet text not fetched)")),
        })
    return {"need": need, "domain": domain, "filters": filters, "items": items, "value_type": "calculated",
            "method": "BM25" + (" + dense embeddings" if kb.embedder else "") + " similarity over catalog entries",
            "catalog_enriched": any(g["enriched"] for g in grouped.values())}
