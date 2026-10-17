"""Typed shared graph state.

Fields written by parallel branches (research tasks, financial analysis) use
reducers so concurrent updates merge instead of conflicting. Everything is
plain JSON-compatible data so any LangGraph checkpointer can persist it.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class WorkTask(TypedDict):
    task_id: str
    type: str               # "research" | "financial"
    workstream_id: str | None
    depends_on: list[str]   # task ids


class ConsultingState(TypedDict, total=False):
    # identity
    execution_id: str
    project_id: str
    principal: dict[str, Any]
    mock_scenario: str | None
    request: dict[str, Any]

    # discovery
    classification: dict[str, Any]
    project_context: dict[str, Any]
    discovery: dict[str, Any]
    clarification_answers: dict[str, Any]
    clarification_rounds: int

    # structuring & work plan
    structuring: dict[str, Any]
    work_plan: list[WorkTask]
    pending_task_ids: list[str]
    research_round: int
    research_requests: dict[str, list[str]]
    completed_tasks: Annotated[list[str], operator.add]          # "<round>:<task_id>"

    # analysis
    research_findings: Annotated[dict[str, Any], merge_dicts]   # workstream_id -> finding
    financial: dict[str, Any]
    hypothesis_evaluation: dict[str, Any]

    # strategy, QA, approval
    strategy: dict[str, Any]
    revision_feedback: list[str]
    revision_round: int
    qa: dict[str, Any]
    qa_iterations: int
    qa_history: Annotated[list[dict[str, Any]], operator.add]
    approval: dict[str, Any]
    approval_rounds: int

    # deliverables
    report: dict[str, Any]
    report_markdown: str
    final_status: str

    # audit / observability (append-only)
    stage_history: Annotated[list[dict[str, Any]], operator.add]
    model_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    security_flags: Annotated[list[dict[str, Any]], operator.add]
    warnings: Annotated[list[str], operator.add]


def initial_state(execution_id: str, project_id: str, principal: dict[str, Any], request: dict[str, Any],
                  mock_scenario: str | None = None) -> ConsultingState:
    return ConsultingState(
        execution_id=execution_id, project_id=project_id, principal=principal, request=request,
        mock_scenario=mock_scenario, clarification_answers={}, clarification_rounds=0, research_round=0,
        pending_task_ids=[], research_requests={}, revision_feedback=[], revision_round=0, qa_iterations=0,
        approval_rounds=0,
    )
