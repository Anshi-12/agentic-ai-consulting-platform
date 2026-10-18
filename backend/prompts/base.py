"""Reusable prompt template.

Every prompt has: role, objective, trusted task context, retrieved evidence,
reasoning constraints, output schema, validation criteria, abstention and
escalation rules, plus multi-shot examples.

Message layout (identical across providers):
    system   : role, objective, security rules, constraints, schema, validation,
               abstention rules. Built ONLY from code-owned strings.
    user/assistant pairs : curated synthetic examples (multi-shot)
    user     : <trusted_task_context> (JSON built by code from workflow state)
               <client_input> (escaped client text)
               <tool_execution_log> (what actually ran)
               <retrieved_evidence> (untrusted, escaped, injection-scanned)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel

from backend.ai.guardrails import INJECTION_DEFENSE_RULES, escape_untrusted, wrap_untrusted_evidence
from backend.ai.structured_outputs import schema_for_prompt
from backend.contracts.schemas import EvidenceItem
from backend.prompts.examples import load_examples

COMMON_CONSTRAINTS = [
    "Ground every factual business claim in the provided evidence and cite evidence IDs (e.g. [EV-NP001]).",
    "Only cite evidence IDs that appear in <retrieved_evidence> or the trusted context. Never invent IDs, sources or numbers.",
    "Label each claim's type: fact (cited), calculated (from deterministic tools), estimate, forecast, hypothesis, assumption, or illustrative.",
    "Never perform arithmetic yourself when calculated values are supplied; quote them and say they are calculated.",
    "Prefer saying 'insufficient evidence' over speculating. Lower confidence when evidence is thin or unreliable.",
    "Use a concise, professional consulting style: answer first, then support.",
]

COMMON_ABSTENTION = [
    "If the task cannot be completed responsibly with the information given, say so in the schema's designated fields (clarifying questions, caveats, low-confidence flags, escalation) instead of filling gaps with invented facts.",
    "Escalate to human review when a decision is high-impact, evidence conflicts, or a source appears manipulated.",
]


def _evidence_items(raw: Iterable[dict[str, Any]]) -> list[EvidenceItem]:
    items = []
    for e in raw:
        data = dict(e)
        data.setdefault("project_id", "example")
        items.append(EvidenceItem.model_validate(data))
    return items


@dataclass
class PromptTemplate:
    task: str
    role: str
    objective: str
    output_schema: type[BaseModel]
    example_family: str | None = None
    reasoning_constraints: list[str] = field(default_factory=list)
    validation_criteria: list[str] = field(default_factory=list)
    abstention_rules: list[str] = field(default_factory=list)
    max_examples: int = 3

    # -- system -------------------------------------------------------------
    def system_prompt(self) -> str:
        constraints = COMMON_CONSTRAINTS + self.reasoning_constraints
        abstain = COMMON_ABSTENTION + self.abstention_rules
        parts = [
            f"ROLE\n{self.role}",
            f"OBJECTIVE\n{self.objective}",
            INJECTION_DEFENSE_RULES,
            "REASONING CONSTRAINTS\n" + "\n".join(f"- {c}" for c in constraints),
            "VALIDATION CRITERIA (your output is machine-checked against these)\n"
            + "\n".join(f"- {c}" for c in self.validation_criteria or ["Output must satisfy the JSON schema."]),
            "ABSTENTION AND ESCALATION RULES\n" + "\n".join(f"- {r}" for r in abstain),
            f"OUTPUT FORMAT\nRespond with ONE JSON object that validates against this JSON Schema "
            f"({self.output_schema.__name__}). No markdown, no prose outside the JSON.\n"
            + schema_for_prompt(self.output_schema),
        ]
        return "\n\n".join(parts)

    # -- user turns ---------------------------------------------------------
    @staticmethod
    def render_input(context: dict[str, Any], evidence: list[EvidenceItem] | None = None,
                     client_text: str | None = None, tool_log: list[dict[str, Any]] | None = None) -> str:
        sections = [
            "<trusted_task_context>\n" + json.dumps(context, indent=1, default=str) + "\n</trusted_task_context>"
        ]
        if client_text:
            sections.append("<client_input>\n" + escape_untrusted(client_text, 6000) + "\n</client_input>")
        if tool_log is not None:
            sections.append("<tool_execution_log>\n" + json.dumps(tool_log, indent=1) + "\n</tool_execution_log>")
        if evidence is not None:
            sections.append("<retrieved_evidence>\n" + wrap_untrusted_evidence(evidence) + "\n</retrieved_evidence>")
        sections.append("Produce the JSON output now.")
        return "\n\n".join(sections)

    def example_messages(self) -> list[dict[str, str]]:
        if not self.example_family:
            return []
        msgs: list[dict[str, str]] = []
        for ex in load_examples(self.example_family)[: self.max_examples]:
            inp = dict(ex["input"])
            evidence = _evidence_items(inp.pop("evidence", [])) if "evidence" in ex["input"] else None
            brief = inp.pop("brief", None)
            msgs.append({"role": "user", "content": self.render_input(inp, evidence, brief)})
            msgs.append({"role": "assistant", "content": json.dumps(ex["output"])})
        return msgs

    def build(self, context: dict[str, Any], evidence: list[EvidenceItem] | None = None,
              client_text: str | None = None, tool_log: list[dict[str, Any]] | None = None
              ) -> tuple[str, list[dict[str, str]]]:
        messages = self.example_messages()
        messages.append({"role": "user", "content": self.render_input(context, evidence, client_text, tool_log)})
        return self.system_prompt(), messages
