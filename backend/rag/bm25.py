"""Okapi BM25 with an inverted index (pure Python, picklable)."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from backend.rag.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.4, b: float = 0.72):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, texts: list[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                postings[term].append((i, tf))
        self.postings = dict(postings)
        n = max(len(texts), 1)
        self.avgdl = (sum(self.doc_len) / n) if self.doc_len else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}
        return self

    def scores(self, query: str, allowed: set[int] | None = None) -> dict[int, float]:
        terms = tokenize(query)
        acc: dict[int, float] = defaultdict(float)
        for term in set(terms):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc, tf in plist:
                if allowed is not None and doc not in allowed:
                    continue
                dl = self.doc_len[doc]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                acc[doc] += idf * tf * (self.k1 + 1) / denom
        return acc

    def search(self, query: str, top_k: int = 10, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        sc = self.scores(query, allowed)
        return sorted(sc.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def matched_terms(self, query: str, text: str) -> list[str]:
        doc_terms = set(tokenize(text))
        return sorted({t for t in tokenize(query) if t in doc_terms})
