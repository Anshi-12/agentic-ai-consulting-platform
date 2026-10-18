"""Ask the multi-agent consulting coworker from the terminal.

    python scripts/ask.py "Our Cisco router has intermittent connectivity failures. What should we order?"
    python scripts/ask.py --demo-human "..."      # mock mode: manager asks you a follow-up question

Prints the routing decision, the live task ledger, and the final recommendation
(also saved to outputs/<id>/answer.md and answer.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.api.models import ConsultRequest  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.contracts.auth import dev_principal  # noqa: E402
from backend.contracts.schemas import WorkflowStatus  # noqa: E402
from backend.orchestration.consulting_execution import ConsultingExecutor  # noqa: E402

SHOW = {"routing.decision", "stage.started", "stage.completed", "qa.result", "security.flag", "workflow.awaiting_input"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--demo-human", action="store_true", help="mock mode: exercise the human-participant loop")
    args = ap.parse_args()
    s = get_settings()
    ex = ConsultingExecutor(s)
    st = ex.kb_status()
    if not st["built"]:
        print("Knowledge base not built.\n  " + st["hint"])
        return 1
    print(f"Models: {s.model_primary if not s.is_mock else 'offline (extractive answers)'} | retrieval: {st['retrieval_mode']}\n")
    ex.bus.subscribe("*", lambda e: e.type in SHOW and print(f"  {e.type:<24} {e.stage or '':<32} {e.message or ''}"))
    p = dev_principal("cli-user")
    run = ex.ask(ConsultRequest(question=" ".join(args.question),
                                mock_scenario="need_human" if args.demo_human and s.is_mock else None), p)
    while run.status == WorkflowStatus.AWAITING_INPUT:
        it = run.pending_interrupt
        print()
        for q in it.get("questions", []):
            print(f"Manager asks: {q}")
        if it["type"] == "approval":
            ans = input("Approve? [y/N]: ").strip().lower()
            payload = {"type": "approval", "decision": "approve" if ans == "y" else "reject"}
        else:
            payload = {"type": it["type"], "answer": input("> ").strip() or "Not known"}
        run = ex.resume(run.execution_id, payload, p)
    r = ex.result(run.execution_id, p)
    print("\n" + (r.result_markdown or f"Workflow {r.status.value}: {r.error}"))
    out = ROOT / "outputs" / run.execution_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer.md").write_text(r.result_markdown or "", encoding="utf-8")
    (out / "answer.json").write_text(r.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0 if r.status == WorkflowStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
