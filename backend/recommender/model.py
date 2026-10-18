"""Market basket recommender for online checkout.

Each time an item is added, recommendations are recomputed for the *whole
current cart*: candidates are items that appeared in past orders containing
every cart item, ranked by confidence and lift. If that exact combination never
occurred often enough, nothing is shown (no irrelevant fallbacks), as the brief
requires. Support counting uses per-item transaction bitsets, so each request
is a few bitwise ANDs.
"""
from __future__ import annotations

import csv
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "data" / "hackathon" / "Retail" / "CMPE256_Hackathon_market_basket_analysis_Release.csv"


def load_transactions(path: str | Path = DEFAULT_CSV) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    body = rows[1:] if rows[0] and rows[0][0].lower().startswith("transaction") else rows
    out = []
    for r in body:
        items = [c.strip() for c in r[1:] if c and c.strip()]
        if items:
            out.append(list(dict.fromkeys(items)))  # drop duplicates, keep order
    return out


@dataclass
class Params:
    min_support_count: int = 3      # cart combination (and cart+item) must appear at least this often
    min_confidence: float = 0.10
    min_lift: float = 1.0           # only items bought together MORE than chance
    top_k: int = 5
    lift_weight: float = 0.5        # score = confidence * lift ** lift_weight (chosen on a validation split)
    backoff: bool = True            # if the full cart is unseen, use the largest sub-combination that has history


@dataclass
class Recommendation:
    item: str
    score: float
    confidence: float
    lift: float
    support_count: int
    explanation: str
    p_value: float = 1.0
    basis: list[str] = field(default_factory=list)

    @property
    def strength(self) -> str:
        return "strong" if self.p_value < 0.01 else "moderate" if self.p_value < 0.05 else "weak"

    def to_dict(self) -> dict[str, Any]:
        return {"item": self.item, "score": round(self.score, 4), "confidence": round(self.confidence, 4),
                "lift": round(self.lift, 3), "support_count": self.support_count, "p_value": round(self.p_value, 4),
                "evidence_strength": self.strength, "based_on": self.basis, "explanation": self.explanation}


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N, K, n): chance of seeing >= k co-purchases if items were independent."""
    if k <= 0:
        return 1.0
    hi = min(K, n)
    if k > hi:
        return 0.0
    denom = math.comb(N, n)
    return min(1.0, sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, hi + 1)) / denom)


@dataclass
class Monitoring:
    requests: int = 0
    empty_responses: int = 0
    unknown_item_requests: int = 0
    accepted_recommendations: int = 0
    shown_recommendations: int = 0
    total_latency_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"requests": self.requests, "empty_responses": self.empty_responses,
                    "empty_rate": round(self.empty_responses / self.requests, 4) if self.requests else 0.0,
                    "unknown_item_requests": self.unknown_item_requests,
                    "recommendations_shown": self.shown_recommendations,
                    "recommendations_accepted": self.accepted_recommendations,
                    "acceptance_rate": round(self.accepted_recommendations / self.shown_recommendations, 4)
                    if self.shown_recommendations else 0.0,
                    "avg_latency_ms": round(self.total_latency_ms / self.requests, 3) if self.requests else 0.0}


class MarketBasketModel:
    def __init__(self, params: Params | None = None):
        self.params = params or Params()
        self.items: list[str] = []
        self.masks: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        self.n = 0
        self.version = 0
        self.trained_at: str | None = None
        self.source: str | None = None
        self.monitor = Monitoring()

    # -- training -------------------------------------------------------------
    def fit(self, transactions: Iterable[Iterable[str]], source: str | None = None) -> "MarketBasketModel":
        masks: dict[str, int] = {}
        n = 0
        for tid, tx in enumerate(transactions):
            for item in set(tx):
                masks[item] = masks.get(item, 0) | (1 << tid)
            n = tid + 1
        self.masks = masks
        self.counts = {i: m.bit_count() for i, m in masks.items()}
        self.items = sorted(masks)
        self.n = n
        self.version += 1
        self.trained_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.source = source
        return self

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_CSV, params: Params | None = None) -> "MarketBasketModel":
        return cls(params).fit(load_transactions(path), source=str(path))

    # -- inference -------------------------------------------------------------
    def _support_mask(self, items: Iterable[str]) -> int:
        mask = (1 << self.n) - 1
        for c in items:
            mask &= self.masks[c]
        return mask

    def _candidates(self, basis: tuple[str, ...], exclude: set[str]) -> list[Recommendation]:
        p = self.params
        mask = self._support_mask(basis)
        base = mask.bit_count()
        if base < p.min_support_count:
            return []
        out = []
        for item in self.items:
            if item in exclude:
                continue
            together = (mask & self.masks[item]).bit_count()
            if together < p.min_support_count:
                continue
            conf = together / base
            lift = conf / (self.counts[item] / self.n)
            if conf < p.min_confidence or lift < p.min_lift:
                continue
            pv = hypergeom_sf(together, self.n, self.counts[item], base)
            out.append(Recommendation(
                item, conf * (lift ** p.lift_weight), conf, lift, together,
                f"In {together} of {base} past orders containing {' + '.join(_short(b) for b in basis)} "
                f"({conf:.0%}); {lift:.2f}x the average rate", pv, list(basis)))
        return out

    def recommend(self, cart: list[str], top_k: int | None = None, track: bool = True) -> dict[str, Any]:
        t0 = time.perf_counter()
        p = self.params
        k = top_k or p.top_k
        cart = list(dict.fromkeys(c for c in cart if c))
        unknown = [c for c in cart if c not in self.masks]
        status, recs, basis_used = "ok", [], []
        cart_count = 0
        if not cart:
            status = "empty_cart"
        elif unknown:
            status = "unknown_items"
        else:
            exclude = set(cart)
            cart_count = self._support_mask(cart).bit_count()
            sizes = range(len(cart), 0, -1) if p.backoff else [len(cart)]
            for size in sizes:
                merged: dict[str, Recommendation] = {}
                for basis in itertools.combinations(cart, size):
                    for r in self._candidates(basis, exclude):
                        prev = merged.get(r.item)
                        if prev is None or (r.score, r.support_count) > (prev.score, prev.support_count):
                            merged[r.item] = r
                if merged:
                    recs = sorted(merged.values(), key=lambda r: (r.score, r.support_count, -r.p_value), reverse=True)[:k]
                    basis_used = sorted({b for r in recs for b in r.basis})
                    status = "ok" if size == len(cart) else "ok_partial_match"
                    break
            if not recs:
                status = ("combination_not_in_history" if cart_count < p.min_support_count
                          else "no_strong_associations")
        ms = (time.perf_counter() - t0) * 1000
        if track:
            with self.monitor.lock:
                self.monitor.requests += 1
                self.monitor.total_latency_ms += ms
                self.monitor.empty_responses += int(not recs)
                self.monitor.unknown_item_requests += int(bool(unknown))
                self.monitor.shown_recommendations += len(recs)
        messages = {
            "ok": None,
            "ok_partial_match": "Your full cart has not been bought together before; showing items bought with "
                                "part of it.",
            "empty_cart": "Add an item to see recommendations.",
            "unknown_items": f"Not in the catalog: {unknown}",
            "combination_not_in_history": "No part of this cart has been bought with other items often enough to make "
                                          "a reliable recommendation.",
            "no_strong_associations": "No item is bought with this cart more often than chance.",
        }
        return {"cart": cart, "status": status, "message": messages[status], "cart_support_count": cart_count,
                "matched_on": basis_used, "recommendations": [r.to_dict() for r in recs],
                "model_version": self.version, "latency_ms": round(ms, 3)}

    def popular(self, top_k: int = 5) -> list[dict[str, Any]]:
        top = sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{"item": i, "support_count": c, "support": round(c / self.n, 4)} for i, c in top]

    def record_acceptance(self) -> None:
        with self.monitor.lock:
            self.monitor.accepted_recommendations += 1

    # -- association rules (for reporting) ---------------------------------------
    def rules(self, max_antecedent: int = 2, min_support_count: int | None = None, top_n: int = 50) -> list[dict]:
        ms = min_support_count or max(self.params.min_support_count, 5)
        out = []
        for size in range(1, max_antecedent + 1):
            for ante in itertools.combinations(self.items, size):
                mask = (1 << self.n) - 1
                for a in ante:
                    mask &= self.masks[a]
                cnt = mask.bit_count()
                if cnt < ms:
                    continue
                for item in self.items:
                    if item in ante:
                        continue
                    together = (mask & self.masks[item]).bit_count()
                    if together < ms:
                        continue
                    conf = together / cnt
                    lift = conf / (self.counts[item] / self.n)
                    out.append({"antecedent": list(ante), "consequent": item, "support": round(together / self.n, 4),
                                "support_count": together, "confidence": round(conf, 4), "lift": round(lift, 3)})
        out.sort(key=lambda r: (r["lift"], r["support_count"]), reverse=True)
        return out[:top_n]

    def info(self) -> dict[str, Any]:
        return {"version": self.version, "trained_at": self.trained_at, "transactions": self.n,
                "items": len(self.items), "source": self.source, "params": self.params.__dict__,
                "monitoring": self.monitor.snapshot()}


def _short(item: str) -> str:
    return item.split(" (SKU")[0]
