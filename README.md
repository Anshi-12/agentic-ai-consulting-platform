# Consulting AI — Group 1: Agentic AI & Consulting Intelligence

Multi-agent consulting engine built on **LangGraph**. Six specialized agents take a client brief through intake, problem framing, parallel research, hypothesis testing, strategy, QA and human approval, and produce an evidence-linked consulting deliverable as structured JSON (plus Markdown).

Runs fully offline in **mock mode** (no API keys), so you can open it in VS Code and run it immediately.

## Quick start (VS Code)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (optional) copy the env template — defaults already run in mock mode
cp .env.example .env               # Windows: copy .env.example .env

# 4. Run a full synthetic engagement end to end
python scripts/run_demo.py

# 5. Run the tests
pytest
```

In VS Code, open the folder, pick the `.venv` interpreter, and use the **Run and Debug** panel: *Demo: full engagement*, *Demo: interactive*, *Demo: QA rework loop*, *API server*, *Evaluation suite*, *Pytest: all*.

Python 3.11+ required (tested on 3.13).

### Demo variations

| Command | What it shows |
|---|---|
| `python scripts/run_demo.py` | Happy path: parallel research, injection quarantine, QA pass, approval, report |
| `python scripts/run_demo.py --vague` | Clarification gate (agent asks questions instead of inventing facts) |
| `python scripts/run_demo.py --scenario qa_failure_once` | QA catches a fabricated citation and routes back to strategy |
| `python scripts/run_demo.py --scenario qa_always_fail` | QA exhausts rework budget and escalates to a human (never self-approves) |
| `python scripts/run_demo.py --project proj-medisupply` | Thin evidence, no financial data — nothing is calculated or claimed |
| `python scripts/run_demo.py --interactive` | You answer clarifications and make the approval decision |
| `python scripts/run_demo.py --reject` | Rejection at the approval gate |

Outputs land in `outputs/<execution_id>/` (`report.md`, `report.json`, `outputs.json`, `events.json`).

### Dev API

```bash
uvicorn backend.api.app:app --reload
# open http://127.0.0.1:8000/docs
```

Endpoints: `POST /api/v1/workflows`, `GET /{id}`, `GET /{id}/outputs`, `POST /{id}/resume`, `POST /{id}/retry`, `POST /{id}/cancel`, `GET /{id}/recommendations`, `GET /{id}/events`, `WS /ws/workflows/{id}`. This app is for standalone runs only — Group 3 owns the platform router and should call `backend/api/service.py` (see `INTEGRATION.md`).

### Live models

Set in `.env`:

```
LLM_PROVIDER_MODE=live
MODEL_PRIMARY=anthropic:claude-opus-5-5
MODEL_LIGHT=anthropic:claude-haiku-5-5
MODEL_SYNTHESIS=anthropic:claude-sonnet-5-5
MODEL_FALLBACK=openai:<model-id>
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

Any role left unset inherits `MODEL_PRIMARY`. `OPENAI_BASE_URL` lets the OpenAI adapter talk to any OpenAI-compatible server.

## Architecture

```
intake ─► clarification (human) ─► intake
  │
problem_framing ─► work_plan ═► research_task × N ┐  parallel, dependency-ordered waves
                             ═► financial_analysis ┘
                                      │
                      hypothesis_evaluation ─► strategy ─► qa
                                                  ▲         │ fail → back to research / financial / strategy / framing
                                                  └─────────┤
                                                            ▼
                              human_approval (always a human; escalation if QA keeps failing)
                                 approve → synthesis → finalize   revise → strategy   reject → end
```

| Agent | File | Model role | Key guardrail |
|---|---|---|---|
| Client Discovery | `agents/client_discovery_agent.py` | light (classify) + primary | Asks clarifying questions when the brief can't be scoped |
| Problem Structuring | `agents/problem_structuring_agent.py` | primary | Hypotheses validated against workstreams; "supported" without valid evidence is downgraded |
| Strategy | `agents/strategy_agent.py` | primary | Weighted scores recomputed in code; uncited / low-confidence recs flagged |
| Financial & Scenario | `agents/financial_scenario_agent.py` | synthesis | All numbers from `ai/finance.py`; narrative figures checked against calculated values |
| Consulting QA | `agents/consulting_qa_agent.py` | primary | Deterministic checks (fabricated citations via tool, flagged sources, unsupported facts, number mismatches) can only tighten the LLM verdict |
| Report Synthesis | `agents/report_synthesis_agent.py` | synthesis | Approved recommendations restored verbatim; citations rebuilt from verified evidence |

Research tasks (`agents/research.py`) are tool-driven retrieval from the data module — no LLM.

### Repository layout

```
backend/
  agents/          six agents + research runner
  orchestration/   graph.py, state.py, routing.py, checkpoints.py, execution.py
  prompts/         prompt templates per family + examples/ (6 families × 3 synthetic examples)
  ai/              model_router.py, provider_factory.py, tools.py, structured_outputs.py,
                   guardrails.py, tracing.py, finance.py, mock_responses.py
  api/             service.py (for Group 3), models.py, app.py (dev only)
  contracts/       PLACEHOLDER shared schemas, events, persistence, auth, data module
  evaluation/      datasets/, metrics.py, run_eval.py
  fixtures/        synthetic projects (NorthPeak Outdoor, MediSupply Regional)
  tests/           116 tests, all offline
scripts/run_demo.py
```

## How the requirements are met

**Orchestration** — typed `ConsultingState` with reducers for parallel writes; conditional routing; `Send`-based parallel fan-out with dependency waves; LangGraph `RetryPolicy` (bounded, exponential, jitter) per node; per-agent timeouts; cancellation checked at every stage boundary; `interrupt()` human gates for clarification and approval; checkpointed state (`memory` or `sqlite`) so failed runs resume from the last checkpoint via `retry`; execution IDs, timestamps, and structured events for every stage, tool call, model call, routing decision, QA result and security flag.

**Prompting** — every prompt has role, objective, security rules, constraints, schema, validation criteria and abstention rules (`prompts/base.py`). Examples are injected as real user/assistant turns; tests assert they reach the model.

**Model routing** — primary / light / synthesis / fallback from env; Anthropic and OpenAI adapters over httpx; retries with capped equal-jitter backoff on retryable errors; one schema-repair turn on invalid output; fallback on recoverable failures; normalized `ProviderError` kinds.

**Evidence discipline** — tools are the only way to get research, documents, numbers, context or workflow state. Prompts receive the tool execution log. QA verifies every cited ID through `get_evidence_by_ids`.

**Security** — untrusted text is escaped and wrapped in `<untrusted_evidence>` in the user turn, never the system turn; injection heuristics quarantine suspicious sources; per-agent tool allowlists, Pydantic argument validation, cross-project / cross-workflow denial, per-agent and per-run tool budgets; no shell, filesystem or SQL tools exist; secrets redacted from logs and traces; approver role required to release deliverables.

**Observability** — LangSmith via `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT`. LangGraph traces nodes automatically; explicit spans cover model calls, tool calls and nodes, with input/output processors that strip secrets and (unless `TRACE_ALLOW_CLIENT_DATA=true`) replace client content with size/hash summaries.

**Evaluation** — `python -m backend.evaluation.run_eval` runs 6 synthetic cases and reports recommendation relevance, evidence coverage, schema validity, unsupported-claim rate, workflow success, clarification/financial/escalation correctness, tokens and latency. `--upload-langsmith` pushes the dataset to LangSmith.

## Known limitations

- `backend/contracts/` is a placeholder until the shared repo exists. See `INTEGRATION.md`.
- Mock-mode outputs are template-based; they exercise the pipeline, not consulting quality. Judge quality in live mode.
- Tool calls are orchestrated by agent code (deterministic, allowlisted), not chosen freely by the model. This is deliberate for safety; model-selected tool calling can be added behind the same `ToolSession`.
- Timeouts stop waiting on a slow agent but cannot kill a running Python thread; the late result is discarded.
- The per-run tool budget is approximate across parallel branches (each branch sees the count at dispatch time).
- `recommendation_relevance` is lexical; add an LLM-as-judge evaluator in LangSmith for semantic scoring.
