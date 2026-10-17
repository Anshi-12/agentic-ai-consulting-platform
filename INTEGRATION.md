# Integration notes (Group 1 → shared repository)

Branch: `group-1-ai` → PR against `main` once the shared repo exists.

## What other groups call

`backend/api/service.py` → `ConsultingService`. Every method takes a `Principal` from platform auth and returns a Pydantic model from `backend/api/models.py`.

| Operation | Method | Response |
|---|---|---|
| Start workflow | `start_workflow(StartWorkflowRequest, principal)` | `WorkflowStatusResponse` |
| Status | `get_status(execution_id, principal)` | `WorkflowStatusResponse` |
| Agent outputs | `get_outputs(execution_id, principal)` | `AgentOutputsResponse` |
| Resume after clarification/approval | `resume_workflow(execution_id, ResumeWorkflowRequest, principal)` | `WorkflowStatusResponse` |
| Retry a failed run from checkpoint | `retry_workflow(execution_id, principal)` | `WorkflowStatusResponse` |
| Cancel | `cancel_workflow(execution_id, principal)` | `WorkflowStatusResponse` |
| Evidence-linked recommendations | `get_recommendations(execution_id, principal)` | `RecommendationsResponse` |
| Event backlog / live subscription | `get_events(...)`, `subscribe(...)` | WebSocket envelopes |

**Endpoint ownership:** if Group 3 owns the router, delete or ignore `backend/api/app.py` and mount these service calls in their router. The dev app's paths are a suggestion, not a contract.

**Events:** `WorkflowEvent.to_ws_message()` returns `{"channel": "workflow:<id>", "event": {...}}`. Event types are in `backend/contracts/events.py::EventType`. `sequence` is monotonic per workflow so clients can dedupe and resume (`get_events(after=n)`).

## Placeholders to replace (`backend/contracts/`)

| File | Replace with | Interface the AI layer depends on |
|---|---|---|
| `schemas.py` | shared `EvidenceItem`, `WorkflowRun`, `WorkflowStatus` | field names used by agents/tools |
| `persistence.py` | shared persistence layer | `WorkflowStore` (create/get/update/list/is_cancel_requested) |
| `events.py` | shared event interface / WebSocket hub | `EventBus` (publish/subscribe/history) |
| `auth.py` | platform auth | `Principal`, `authorize_project(principal, project_id, action)` |
| `data_module.py` | Group 2 research/data client | `DataModuleClient` (context, research, documents, financial inputs, evidence by ID) |
| `orchestration/checkpoints.py` | shared DB checkpointer (e.g. Postgres) | any LangGraph `BaseCheckpointSaver` |

Inject replacements through `WorkflowExecutor(data_module=..., store=..., bus=..., checkpointer=...)`. Nothing in `agents/` or `orchestration/` needs to change.

### Data module contract (what Group 2 must supply)

- `query_research(project_id, query, top_k)` / `retrieve_documents(...)` → `list[EvidenceItem]` with stable `evidence_id` values (pattern `^[A-Z]{2,5}-[A-Za-z0-9_-]{1,40}$`).
- `get_evidence(project_id, ids)` → used by QA and synthesis to verify citations.
- `get_financial_inputs(project_id)` → `{"currency", "discount_rate", "baseline": {revenue, cogs, opex}, "scenarios": [{name, investment, annual_cash_flows[], revenue?, cogs?, opex?, value_type: estimate|forecast, assumptions[]}]}` or `None`.

### Output schemas

Agent outputs are defined in `backend/ai/structured_outputs.py` (`DiscoveryOutput`, `StructuringOutput`, `HypothesisEvaluationOutput`, `StrategyOutput`, `FinancialOutput`, `QAOutput`, `ReportOutput`). If the shared repo defines report/recommendation schemas, map them here and keep field names aligned; tests will flag drift.

## Integration steps

1. Create branch `group-1-ai` in the shared repo; copy `backend/agents`, `backend/orchestration`, `backend/prompts`, `backend/ai`, `backend/evaluation`, `backend/tests`, `backend/api/service.py`, `backend/api/models.py`.
2. Implement adapters for the placeholders above and pass them into `WorkflowExecutor`.
3. Merge `requirements.txt` into the shared dependency file.
4. Agree endpoint ownership with Group 3; wire `ConsultingService` into their router and event hub.
5. Run `pytest` (mock mode) in CI; run `python -m backend.evaluation.run_eval` with live models before release.

## Changed-file inventory (all new)

```
backend/config.py
backend/agents/{base,client_discovery_agent,problem_structuring_agent,strategy_agent,
                financial_scenario_agent,consulting_qa_agent,report_synthesis_agent,research}.py
backend/orchestration/{graph,state,routing,checkpoints,execution}.py
backend/prompts/{base,discovery_prompts,strategy_prompts,financial_prompts,qa_prompts,synthesis_prompts}.py
backend/prompts/examples/{problem_statement,issue_tree,strategy,financial,qa_review,executive_summary}.json
backend/ai/{model_router,provider_factory,tools,structured_outputs,guardrails,tracing,finance,mock_responses}.py
backend/api/{service,models,app}.py
backend/contracts/{schemas,events,persistence,auth,data_module}.py      ← placeholders
backend/evaluation/{metrics,run_eval}.py, datasets/consulting_eval_cases.jsonl
backend/fixtures/synthetic_projects.json
backend/tests/{conftest,test_agents,test_orchestration,test_prompt_examples,test_model_routing,
               test_guardrails,test_api,test_evaluation}.py
scripts/run_demo.py
requirements.txt, pyproject.toml, .env.example, .gitignore, .vscode/
```

Filename deviations from the spec: none for required files. Added `agents/base.py`, `agents/research.py`, `ai/guardrails.py`, `ai/tracing.py`, `ai/finance.py`, `ai/mock_responses.py`, `prompts/base.py` and `contracts/`.
