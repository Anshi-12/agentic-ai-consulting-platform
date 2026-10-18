"""Curated synthetic multi-shot examples, one JSON file per task family."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent

FAMILIES = {
    "problem_statement": "problem_statement.json",
    "issue_tree": "issue_tree.json",
    "strategy": "strategy.json",
    "financial": "financial.json",
    "qa_review": "qa_review.json",
    "executive_summary": "executive_summary.json",
    "manager_routing": "manager_routing.json",
    "domain_answer": "domain_answer.json",
    "consolidation": "consolidation.json",
}


@lru_cache(maxsize=None)
def load_family(family: str) -> dict[str, Any]:
    if family not in FAMILIES:
        raise KeyError(f"unknown example family '{family}'")
    return json.loads((EXAMPLES_DIR / FAMILIES[family]).read_text(encoding="utf-8"))


def load_examples(family: str) -> list[dict[str, Any]]:
    return list(load_family(family)["examples"])
