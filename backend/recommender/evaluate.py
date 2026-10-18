"""Offline evaluation of the market basket recommender.

    python -m backend.recommender.evaluate

Protocol: shuffle orders (fixed seed), train on 80%, test on 20%.
* Leave-one-out: for each test order with >= 2 items, hide one item and
  recommend from the rest -> HitRate@K, Precision@K, MRR, coverage.
* Checkout simulation (matches the brief): add the order's items one at a time;
  after each add, is the next item in the recommendations?
Both are compared with a most-popular-items baseline, so "better than chance"
is measured, not assumed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import replace
from typing import Any

from backend.recommender.model import DEFAULT_CSV, MarketBasketModel, Params, load_transactions


def split(transactions: list[list[str]], test_frac: float = 0.2, seed: int = 42):
    tx = list(transactions)
    random.Random(seed).shuffle(tx)
    cut = int(len(tx) * (1 - test_frac))
    return tx[:cut], tx[cut:]


def _score(ranked: list[str], target: str, k: int) -> tuple[int, float]:
    top = ranked[:k]
    if target in top:
        return 1, 1.0 / (top.index(target) + 1)
    return 0, 0.0


def evaluate(train: list[list[str]], test: list[list[str]], params: Params, k: int = 5) -> dict[str, Any]:
    model = MarketBasketModel(params).fit(train)
    popular = [p["item"] for p in model.popular(len(model.items))]
    loo = {"cases": 0, "hits": 0, "mrr": 0.0, "covered": 0, "base_hits": 0, "base_mrr": 0.0}
    sim = {"steps": 0, "hits": 0, "covered": 0, "base_hits": 0}
    for order in test:
        if len(order) < 2:
            continue
        for i, target in enumerate(order):
            cart = order[:i] + order[i + 1:]
            recs = [r["item"] for r in model.recommend(cart, k, track=False)["recommendations"]]
            h, rr = _score(recs, target, k)
            base = [p for p in popular if p not in cart]
            bh, brr = _score(base, target, k)
            loo["cases"] += 1
            loo["hits"] += h
            loo["mrr"] += rr
            loo["covered"] += int(bool(recs))
            loo["base_hits"] += bh
            loo["base_mrr"] += brr
        for i in range(1, len(order)):
            cart, nxt = order[:i], order[i]
            recs = [r["item"] for r in model.recommend(cart, k, track=False)["recommendations"]]
            base = [p for p in popular if p not in cart][:k]
            sim["steps"] += 1
            sim["hits"] += int(nxt in recs)
            sim["covered"] += int(bool(recs))
            sim["base_hits"] += int(nxt in base)
    c, s = max(loo["cases"], 1), max(sim["steps"], 1)
    hit = loo["hits"] / c
    return {
        "k": k, "params": params.__dict__, "train_orders": len(train), "test_orders": len(test),
        "leave_one_out": {"cases": loo["cases"], f"hit_rate@{k}": round(hit, 4), f"precision@{k}": round(hit / k, 4),
                          "mrr": round(loo["mrr"] / c, 4), "coverage": round(loo["covered"] / c, 4),
                          f"baseline_hit_rate@{k}": round(loo["base_hits"] / c, 4),
                          "baseline_mrr": round(loo["base_mrr"] / c, 4),
                          "lift_over_baseline": round(hit / (loo["base_hits"] / c), 3) if loo["base_hits"] else None},
        "checkout_simulation": {"steps": sim["steps"], f"next_item_hit_rate@{k}": round(sim["hits"] / s, 4),
                                "coverage": round(sim["covered"] / s, 4),
                                f"baseline_hit_rate@{k}": round(sim["base_hits"] / s, 4)},
    }


def tune(train: list[list[str]], k: int = 5, seed: int = 7) -> tuple[Params, list[dict[str, Any]]]:
    """Pick parameters on a validation split carved from the training data (never the test set)."""
    fit, val = split(train, 0.2, seed)
    grid = []
    for ms in (2, 3):
        for lw in (0.0, 0.5, 1.0):
            for bo in (True, False):
                p = replace(Params(), min_support_count=ms, lift_weight=lw, backoff=bo)
                r = evaluate(fit, val, p, k)["leave_one_out"]
                grid.append({"params": p.__dict__, "val_hit_rate": r[f"hit_rate@{k}"], "val_coverage": r["coverage"]})
    best = max(grid, key=lambda g: (g["val_hit_rate"], g["val_coverage"]))
    return Params(**best["params"]), grid


def signal_test(transactions: list[list[str]], permutations: int = 30, seed: int = 0) -> dict[str, Any]:
    """Is there real co-purchase structure? Compare pair co-occurrence with shuffled (independent) baskets."""
    from collections import Counter

    def pair_counts(tx):
        c = Counter()
        for t in tx:
            for a, b in itertools.combinations(sorted(set(t)), 2):
                c[(a, b)] += 1
        return c

    real = pair_counts(transactions)
    pool = [i for t in transactions for i in t]
    null_max, null_strong = [], []
    for s in range(permutations):
        rnd = random.Random(seed + s)
        shuffled = pool[:]
        rnd.shuffle(shuffled)
        tx, k = [], 0
        for t in transactions:
            tx.append(shuffled[k:k + len(t)])
            k += len(t)
        pc = pair_counts(tx)
        null_max.append(max(pc.values()))
        null_strong.append(sum(v >= 12 for v in pc.values()))
    real_max = max(real.values())
    pct = sum(m >= real_max for m in null_max) / permutations
    return {"real_max_pair_count": real_max, "null_max_pair_count_mean": round(sum(null_max) / permutations, 2),
            "null_max_pair_count_range": [min(null_max), max(null_max)],
            "real_pairs_ge_12": sum(v >= 12 for v in real.values()),
            "null_pairs_ge_12_mean": round(sum(null_strong) / permutations, 2),
            "p_value_max_pair": round(pct, 3),
            "interpretation": ("co-purchase structure is indistinguishable from random baskets"
                               if pct > 0.05 else "co-purchase structure is stronger than chance")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="grid-search params on a validation split first")
    args = ap.parse_args()
    train, test = split(load_transactions(args.csv))
    params = tune(train, args.k)[0] if args.tune else Params()
    report = evaluate(train, test, params, args.k)
    report["data_signal_test"] = signal_test(load_transactions(args.csv))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
