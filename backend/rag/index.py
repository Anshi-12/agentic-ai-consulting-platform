"""Per-domain retrieval indexes (BM25 + optional dense vectors, fused with RRF)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from backend.rag.bm25 import BM25Index
from backend.rag.documents import DOMAIN_INDEXES, INDEX_NAMES, Chunk
from backend.rag.embeddings import Embedder

RRF_K = 60


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    matched_terms: list[str] = field(default_factory=list)

    def to_evidence(self, max_chars: int = 900) -> dict[str, Any]:
        c = self.chunk
        return {
            "evidence_id": c.chunk_id, "title": c.title, "source": c.source, "page": c.page,
            "doc_type": c.doc_type, "company": c.company, "year": c.year, "index": c.index,
            "text": c.text[:max_chars], "score": round(self.score, 4), "matched_terms": self.matched_terms,
            "metadata": {k: v for k, v in c.metadata.items() if k in ("status", "phases", "topic", "sponsor",
                                                                       "conditions", "enrollment", "url")},
        }


class DomainIndex:
    def __init__(self, name: str, chunks: list[Chunk], bm25: BM25Index, dense: np.ndarray | None = None,
                 embedder_name: str | None = None):
        self.name = name
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.embedder_name = embedder_name
        self.by_id = {c.chunk_id: i for i, c in enumerate(chunks)}

    @classmethod
    def build(cls, name: str, chunks: list[Chunk], embedder: Embedder | None = None,
              embed: bool = True) -> "DomainIndex":
        bm25 = BM25Index().fit([f"{c.title}\n{c.text}" for c in chunks])
        dense = None
        if embedder is not None and embed and chunks:
            dense = embedder.embed_documents([f"{c.title}. {c.text}"[:2000] for c in chunks])
        return cls(name, chunks, bm25, dense, embedder.name if (embedder and dense is not None) else None)

    def _allowed(self, filters: dict[str, Any] | None) -> set[int] | None:
        if not filters:
            return None
        out = set()
        for i, c in enumerate(self.chunks):
            ok = True
            for key, want in filters.items():
                if want in (None, "", []):
                    continue
                val = getattr(c, key, None) if hasattr(c, key) else c.metadata.get(key)
                if isinstance(want, (list, tuple, set)):
                    ok = val in want
                elif isinstance(val, str) and isinstance(want, str):
                    ok = want.lower() in val.lower()
                else:
                    ok = val == want
                if not ok:
                    break
            if ok:
                out.add(i)
        return out

    def search(self, query: str, top_k: int = 6, filters: dict[str, Any] | None = None,
               embedder: Embedder | None = None) -> list[SearchHit]:
        allowed = self._allowed(filters)
        if allowed is not None and not allowed:
            return []
        pool = max(top_k * 6, 30)
        bm = self.bm25.search(query, pool, allowed)
        bm_rank = {doc: r for r, (doc, _) in enumerate(bm, start=1)}
        dense_rank: dict[int, int] = {}
        if self.dense is not None and embedder is not None and embedder.name == self.embedder_name:
            q = embedder.embed_query(query)
            sims = self.dense @ q
            if allowed is not None:
                mask = np.full(len(sims), -np.inf)
                idx = np.fromiter(allowed, dtype=int)
                mask[idx] = sims[idx]
                sims = mask
            top = np.argsort(-sims)[:pool]
            dense_rank = {int(d): r for r, d in enumerate(top, start=1) if np.isfinite(sims[d])}
        candidates = set(bm_rank) | set(dense_rank)
        fused = []
        for doc in candidates:
            s = 0.0
            if doc in bm_rank:
                s += 1.0 / (RRF_K + bm_rank[doc])
            if doc in dense_rank:
                s += 1.0 / (RRF_K + dense_rank[doc])
            fused.append((s, doc))
        fused.sort(reverse=True)
        hits = []
        for s, doc in fused[:top_k]:
            c = self.chunks[doc]
            hits.append(SearchHit(c, s, bm_rank.get(doc), dense_rank.get(doc),
                                  self.bm25.matched_terms(query, f"{c.title} {c.text}")))
        return hits

    def get(self, chunk_id: str) -> Chunk | None:
        i = self.by_id.get(chunk_id)
        return self.chunks[i] if i is not None else None


class KnowledgeBase:
    """All domain indexes plus the embedder used to query them."""

    def __init__(self, indexes: dict[str, DomainIndex], embedder: Embedder | None = None,
                 manifest: dict[str, Any] | None = None):
        self.indexes = indexes
        self.embedder = embedder
        self.manifest = manifest or {}

    # -- persistence -----------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for name, idx in self.indexes.items():
            with open(d / f"{name}.pkl", "wb") as f:
                pickle.dump({"name": name, "chunks": [c.model_dump() for c in idx.chunks], "bm25": idx.bm25,
                             "embedder_name": idx.embedder_name}, f, protocol=pickle.HIGHEST_PROTOCOL)
            if idx.dense is not None:
                np.save(d / f"{name}.npy", idx.dense)
            elif (d / f"{name}.npy").exists():
                (d / f"{name}.npy").unlink()
        (d / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "KnowledgeBase":
        d = Path(directory)
        if not (d / "manifest.json").exists():
            raise FileNotFoundError(f"no knowledge base at {d}; run `python -m backend.rag.ingest`")
        indexes = {}
        for name in INDEX_NAMES:
            p = d / f"{name}.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:  # local file written by our own ingest step
                raw = pickle.load(f)
            dense = np.load(d / f"{name}.npy") if (d / f"{name}.npy").exists() else None
            indexes[name] = DomainIndex(name, [Chunk.model_validate(c) for c in raw["chunks"]], raw["bm25"], dense,
                                        raw.get("embedder_name"))
        return cls(indexes, embedder, json.loads((d / "manifest.json").read_text(encoding="utf-8")))

    # -- queries ---------------------------------------------------------------
    def search(self, index: str, query: str, top_k: int = 6, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        idx = self.indexes.get(index)
        return idx.search(query, top_k, filters, self.embedder) if idx else []

    def search_domain(self, domain: str, query: str, top_k: int = 6,
                      filters: dict[str, Any] | None = None) -> list[SearchHit]:
        """Search every index a domain agent owns (except structured trial rows)."""
        hits: list[SearchHit] = []
        for name in DOMAIN_INDEXES.get(domain, ()):
            if name == "clinical_trials":
                continue
            hits.extend(self.search(name, query, top_k, filters))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def get(self, chunk_id: str) -> Chunk | None:
        for idx in self.indexes.values():
            c = idx.get(chunk_id)
            if c is not None:
                return c
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "indexes": {n: {"chunks": len(i.chunks), "dense": i.dense is not None} for n, i in self.indexes.items()},
            "embedder": self.embedder.name if self.embedder else None,
            "retrieval_mode": "hybrid (BM25 + dense)" if self.embedder and any(
                i.dense is not None for i in self.indexes.values()) else "BM25 keyword",
            "built_at": self.manifest.get("built_at"),
            "sources": self.manifest.get("sources", {}),
        }


def load_knowledge_base(directory: str | Path) -> KnowledgeBase | None:
    """Load the KB if it has been built; attach the embedder only if dense vectors exist."""
    d = Path(directory)
    if not (d / "manifest.json").exists():
        return None
    kb = KnowledgeBase.load(d)
    if any(i.dense is not None for i in kb.indexes.values()):
        from backend.rag.embeddings import load_embedder

        emb = load_embedder()
        built_with = {i.embedder_name for i in kb.indexes.values() if i.dense is not None}
        kb.embedder = emb if emb is not None and emb.name in built_with else None
    return kb
