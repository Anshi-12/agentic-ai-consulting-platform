"""Run a complete synthetic consulting engagement end to end.

    python scripts/run_demo.py                         # mock mode, auto-approves
    python scripts/run_demo.py --interactive           # answer clarifications / approve yourself
    python scripts/run_demo.py --scenario qa_failure_once
    python scripts/run_demo.py --vague                 # triggers the clarification gate

Outputs are written to ./outputs/<execution_id>/ (report.md, report.json, outputs.json, events.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.api.models import StartWorkflowRequest  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.contracts.auth import dev_principal  # noqa: E402
from backend.contracts.events import WorkflowEvent  # noqa: E402
from backend.contracts.schemas import WorkflowStatus  # noqa: E402
from backend.orchestration.execution import WorkflowExecutor  # noqa: E402

DEFAULT_BRIEF = (
    "Our gross margin has fallen from 46% to 39% over six quarters. We run frequent sitewide promotions and "
    "fulfillment costs keep rising. The CFO wants a plan to restore margin within 18 months without alienating "
    "our loyal enthusiast customers, and without major new headcount."
)
VAGUE_BRIEF = "We want to fix our margins. What should we do?"

ICONS = {"stage.started": "▶", "stage.completed": "✔", "stage.failed": "✖", "routing.decision": "↪",
         "qa.result": "🔎", "security.flag": "⚠", "workflow.awaiting_input": "⏸", "workflow.completed": "🏁",
         "workflow.failed": "✖", "workflow.rejected": "⛔", "workflow.cancelled": "⛔"}
QUIET = {"tool.invoked", "model.call"}


def print_event(e: WorkflowEvent, verbose: bool) -> None:
    if e.type in QUIET and not verbose:
        return
    icon = ICONS.get(e.type, "·")
    detail = e.message or ""
    if e.type == "stage.completed" and "duration_ms" in e.payload:
        detail = f"{e.payload['duration_ms']} ms"
    if e.type == "security.flag":
        detail = f"{e.payload.get('evidence_id')} quarantined ({', '.join(e.payload.get('patterns', []))})"
    print(f"  {icon} {e.type:<24} {e.stage or '':<22} {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="proj-northpeak")
    ap.add_argument("--brief", default=None)
    ap.add_argument("--vague", action="store_true", help="use a vague brief to trigger clarification")
    ap.add_argument("--scenario", choices=["qa_failure_once", "qa_always_fail"], default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--reject", action="store_true", help="auto-reject at the approval gate")
    ap.add_argument("--verbose", action="store_true", help="also print tool and model call events")
    args = ap.parse_args()

    settings = get_settings()
    print(f"Provider mode: {settings.provider_mode} | primary={settings.model_primary} | "
          f"fallback={settings.model_fallback}\n")
    ex = WorkflowExecutor(settings)
    ex.bus.subscribe("*", lambda e: print_event(e, args.verbose))
    principal = dev_principal("demo-user")

    brief = args.brief or (VAGUE_BRIEF if args.vague else DEFAULT_BRIEF)
    run = ex.start(StartWorkflowRequest(project_id=args.project, client_brief=brief,
                                        mock_scenario=args.scenario if settings.is_mock else None), principal)
    eid = run.execution_id

    while run.status == WorkflowStatus.AWAITING_INPUT:
        intr = run.pending_interrupt or {}
        print()
        if intr.get("type") == "clarification":
            print("Clarification needed:")
            answers = {}
            for i, q in enumerate(intr.get("questions", []), 1):
                print(f"  Q{i}. {q}")
                answers[f"q{i}"] = (input("     > ") if args.interactive else
                                    "Improve gross margin by 3 points within 18 months; budget under $5M; "
                                    "all US online and wholesale channels are in scope.")
            payload = {"type": "clarification", "answers": answers}
        else:
            print(f"Approval gate ({'ESCALATED' if intr.get('escalated') else 'QA passed'}): {intr.get('reason')}")
            for r in intr.get("recommendations", []):
                flag = " [LOW CONFIDENCE]" if r.get("low_confidence_flag") else ""
                print(f"  - {r['id']}: {r['title']}{flag}  evidence={r.get('evidence_ids')}")
            for i in intr.get("qa_issues", []):
                print(f"  ! {i['severity']}/{i['category']}: {i['description']}")
            if args.interactive:
                decision = input("Decision [approve/revise/reject]: ").strip() or "approve"
                comments = input("Comments: ").strip()
            else:
                decision = "reject" if args.reject else "approve"
                comments = "Auto-decision by demo script"
            payload = {"type": "approval", "decision": decision, "comments": comments}
        print()
        run = ex.resume(eid, payload, principal)

    print(f"\nFinal status: {run.status.value}")
    out_dir = ROOT / "outputs" / eid
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = ex.outputs(eid, principal)
    (out_dir / "outputs.json").write_text(outputs.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps([e.model_dump(mode="json") for e in ex.bus.history(eid)], indent=2),
                                         encoding="utf-8")
    if outputs.report:
        (out_dir / "report.json").write_text(json.dumps(outputs.report, indent=2), encoding="utf-8")
        (out_dir / "report.md").write_text(outputs.report_markdown or "", encoding="utf-8")
    u = outputs.usage
    print(f"Model calls: {u.model_calls} (fallbacks {u.fallback_calls}, validation failures {u.validation_failures}) | "
          f"tokens in/out: {u.input_tokens}/{u.output_tokens} | tool calls: {u.tool_calls}")
    if run.error:
        print(f"Error: {run.error.get('message')}")
    print(f"Outputs written to {out_dir}")
    return 0 if run.status in (WorkflowStatus.COMPLETED, WorkflowStatus.REJECTED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
