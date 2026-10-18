"""Run the evaluation dataset through the full workflow and report metrics.

    python -m backend.evaluation.run_eval                 # mock mode (CI-safe)
    LLM_PROVIDER_MODE=live python -m backend.evaluation.run_eval
    python -m backend.evaluation.run_eval --upload-langsmith   # also push the dataset to LangSmith

Results are written to backend/evaluation/results/latest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from backend.api.models import StartWorkflowRequest
from backend.config import Settings, get_settings
from backend.contracts.auth import dev_principal
from backend.contracts.schemas import WorkflowStatus
from backend.evaluation.metrics import aggregate, score_run
from backend.orchestration.execution import WorkflowExecutor

HERE = Path(__file__).resolve().parent
DATASET = HERE / "datasets" / "consulting_eval_cases.jsonl"
THRESHOLDS = {"workflow_success": 1.0, "schema_validity": 0.8, "evidence_coverage": 0.8,
              "unsupported_claim_rate_max": 0.1, "clarification_correct": 1.0}


def load_cases(path: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(ex: WorkflowExecutor, case: dict[str, Any]) -> dict[str, Any]:
    principal = dev_principal("evaluator")
    t0 = time.perf_counter()
    scenario = case.get("mock_scenario") if ex.settings.is_mock else None
    run = ex.start(StartWorkflowRequest(project_id=case["project_id"], client_brief=case["brief"],
                                        mock_scenario=scenario), principal)
    clarification_seen = escalation_seen = False
    guard = 0
    while run.status == WorkflowStatus.AWAITING_INPUT and guard < 6:
        guard += 1
        intr = run.pending_interrupt or {}
        if intr.get("type") == "clarification":
            clarification_seen = True
            payload = {"type": "clarification", "answers": case.get("clarification_answers") or {"q": "No further detail"}}
        else:
            escalation_seen = escalation_seen or bool(intr.get("escalated"))
            payload = {"type": "approval", "decision": "approve", "comments": "evaluation auto-approval"}
        run = ex.resume(run.execution_id, payload, principal)
    values = ex._values(run.execution_id)
    metrics = score_run(values, run.status.value, case["expected"], ex.data.all_evidence_ids(case["project_id"]),
                        clarification_seen, escalation_seen)
    return {"id": case["id"], "execution_id": run.execution_id, "status": run.status.value,
            "wall_time_s": round(time.perf_counter() - t0, 3), "metrics": metrics}


def check_thresholds(agg: dict[str, Any]) -> list[str]:
    failures = []
    for key, floor in THRESHOLDS.items():
        if key.endswith("_max"):
            name = key[:-4]
            if agg.get(name, 1) > floor:
                failures.append(f"{name}={agg.get(name)} > {floor}")
        elif agg.get(key, 0) < floor:
            failures.append(f"{key}={agg.get(key)} < {floor}")
    return failures


def run_all(settings: Settings | None = None, cases: list[dict[str, Any]] | None = None,
            executor: WorkflowExecutor | None = None) -> dict[str, Any]:
    ex = executor or WorkflowExecutor(settings or get_settings())
    results = [run_case(ex, c) for c in (cases or load_cases())]
    agg = aggregate(results)
    return {"provider_mode": ex.settings.provider_mode, "aggregate": agg, "threshold_failures": check_thresholds(agg),
            "cases": results}


def upload_to_langsmith(cases: list[dict[str, Any]], name: str = "consulting-ai-group1-eval") -> str:
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise RuntimeError("LANGSMITH_API_KEY is not set")
    from langsmith import Client

    client = Client()
    existing = [d for d in client.list_datasets(dataset_name=name)]
    dataset = existing[0] if existing else client.create_dataset(name, description="Synthetic consulting workflow cases")
    client.create_examples(
        inputs=[{"project_id": c["project_id"], "brief": c["brief"]} for c in cases],
        outputs=[c["expected"] for c in cases],
        metadata=[{"case_id": c["id"], "mock_scenario": c.get("mock_scenario")} for c in cases],
        dataset_id=dataset.id,
    )
    return str(dataset.id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-langsmith", action="store_true")
    args = ap.parse_args()
    report = run_all()
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Provider mode: {report['provider_mode']}")
    print(f"{'case':<38}{'status':<12}{'relev':>7}{'evid':>7}{'unsup':>7}{'schema':>8}{'qa_it':>7}{'tokens':>9}")
    for r in report["cases"]:
        m = r["metrics"]
        print(f"{r['id']:<38}{r['status']:<12}{m['recommendation_relevance']:>7.2f}{m['evidence_coverage']:>7.2f}"
              f"{m['unsupported_claim_rate']:>7.2f}{m['schema_validity']:>8.2f}{m['qa_iterations']:>7}"
              f"{m['input_tokens'] + m['output_tokens']:>9}")
    print("\nAggregate:", json.dumps(report["aggregate"], indent=1))
    if report["threshold_failures"]:
        print("THRESHOLD FAILURES:", report["threshold_failures"])
    if args.upload_langsmith:
        print("Uploaded LangSmith dataset:", upload_to_langsmith(load_cases()))
    return 1 if report["threshold_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
