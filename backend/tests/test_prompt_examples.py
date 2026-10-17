from __future__ import annotations

import json
import re

import pytest

from backend.ai import structured_outputs as so
from backend.prompts.discovery_prompts import DISCOVERY_PROMPT, STRUCTURING_PROMPT
from backend.prompts.examples import FAMILIES, load_family
from backend.prompts.financial_prompts import FINANCIAL_PROMPT
from backend.prompts.qa_prompts import QA_PROMPT
from backend.prompts.strategy_prompts import STRATEGY_PROMPT
from backend.prompts.synthesis_prompts import SYNTHESIS_PROMPT

FAMILY_PROMPTS = {
    "problem_statement": DISCOVERY_PROMPT,
    "issue_tree": STRUCTURING_PROMPT,
    "strategy": STRATEGY_PROMPT,
    "financial": FINANCIAL_PROMPT,
    "qa_review": QA_PROMPT,
    "executive_summary": SYNTHESIS_PROMPT,
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_has_three_valid_diverse_examples(family):
    fam = load_family(family)
    schema = getattr(so, fam["schema"])
    examples = fam["examples"]
    assert len(examples) >= 3
    for ex in examples:
        schema.model_validate(ex["output"])  # every example is a valid target output
    assert len({ex["industry"] for ex in examples}) == len(examples)  # different industries
    assert len({ex["difficulty"] for ex in examples}) >= 3  # different difficulty levels


def test_every_family_includes_an_abstention_or_insufficient_evidence_case():
    checks = {
        "problem_statement": lambda o: o["status"] == "needs_clarification",
        "issue_tree": lambda o: o["mece_check"]["is_mece"] is False,
        "strategy": lambda o: any(r["low_confidence_flag"] and not r["evidence_ids"] for r in o["recommendations"]),
        "financial": lambda o: o["data_available"] is False,
        "qa_review": lambda o: o["verdict"] == "escalate",
        "executive_summary": lambda o: any("No financial inputs" in x for x in o["limitations"]),
    }
    for family, check in checks.items():
        assert any(check(ex["output"]) for ex in load_family(family)["examples"]), family


def test_clarification_example_requests_info_instead_of_inventing():
    ex = next(e for e in load_family("problem_statement")["examples"] if e["difficulty"] == "ambiguous")
    assert ex["output"]["problem_statement"] is None and len(ex["output"]["clarifying_questions"]) >= 3


@pytest.mark.parametrize("family,prompt", list(FAMILY_PROMPTS.items()))
def test_examples_are_used_in_actual_prompt_construction(family, prompt):
    system, messages = prompt.build({"key_question": "real task"})
    examples = load_family(family)["examples"]
    assert len(messages) == 2 * len(examples) + 1
    for i, ex in enumerate(examples):
        assert messages[2 * i]["role"] == "user" and messages[2 * i + 1]["role"] == "assistant"
        assert json.loads(messages[2 * i + 1]["content"]) == ex["output"]
    assert '"real task"' in messages[-1]["content"]
    for section in ["ROLE", "OBJECTIVE", "SECURITY RULES", "REASONING CONSTRAINTS", "VALIDATION CRITERIA",
                    "ABSTENTION AND ESCALATION RULES", "OUTPUT FORMAT"]:
        assert section in system
    assert prompt.output_schema.__name__ in system


def test_live_model_calls_receive_multi_shot_examples(harness, principal):
    harness.start(principal)
    provider = harness.router.factory.get("mock:primary")
    strategy_call = next(c for c in provider.calls if c.task == "strategy")
    first_example = load_family("strategy")["examples"][0]["output"]
    assert json.loads(strategy_call.messages[1]["content"]) == first_example
    assert "<untrusted_evidence" in strategy_call.messages[-1]["content"]


_SECRETS = re.compile(r"sk-[A-Za-z0-9]{10,}|lsv2_|AKIA[0-9A-Z]{16}|api[_-]?key|password|BEGIN (RSA|PRIVATE)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_examples_contain_no_secrets_or_personal_data(family):
    raw = json.dumps(load_family(family))
    assert not _SECRETS.search(raw) and not _EMAIL.search(raw) and not _PHONE.search(raw)
    assert "synthetic" in raw.lower()
