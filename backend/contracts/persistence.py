"""PLACEHOLDER shared persistence layer for workflow run records.

Graph state itself is persisted by the LangGraph checkpointer
(backend/orchestration/checkpoints.py). This store keeps the lightweight run
record (status, stage, pending interrupt, cancel flag) that the API reads.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from backend.contracts.schemas import WorkflowRun, utcnow_iso


class RunNotFoundError(KeyError):
    pass


class WorkflowStore(ABC):
    @abstractmethod
    def create(self, run: WorkflowRun) -> WorkflowRun: ...

    @abstractmethod
    def get(self, execution_id: str) -> WorkflowRun: ...

    @abstractmethod
    def update(self, execution_id: str, **fields) -> WorkflowRun: ...

    @abstractmethod
    def list_for_project(self, project_id: str) -> list[WorkflowRun]: ...

    def is_cancel_requested(self, execution_id: str) -> bool:
        try:
            return self.get(execution_id).cancel_requested
        except RunNotFoundError:
            return False


class InMemoryWorkflowStore(WorkflowStore):
    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}
        self._lock = threading.RLock()

    def create(self, run: WorkflowRun) -> WorkflowRun:
        with self._lock:
            self._runs[run.execution_id] = run
            return run.model_copy(deep=True)

    def get(self, execution_id: str) -> WorkflowRun:
        with self._lock:
            if execution_id not in self._runs:
                raise RunNotFoundError(execution_id)
            return self._runs[execution_id].model_copy(deep=True)

    def update(self, execution_id: str, **fields) -> WorkflowRun:
        with self._lock:
            run = self.get(execution_id)
            updated = run.model_copy(update={**fields, "updated_at": utcnow_iso()})
            self._runs[execution_id] = updated
            return updated.model_copy(deep=True)

    def list_for_project(self, project_id: str) -> list[WorkflowRun]:
        with self._lock:
            return [r.model_copy(deep=True) for r in self._runs.values() if r.project_id == project_id]


class JsonFileWorkflowStore(InMemoryWorkflowStore):
    """Writes each run record to `<dir>/<execution_id>.json` (dev persistence)."""

    def __init__(self, directory: str) -> None:
        super().__init__()
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        for f in self._dir.glob("*.json"):
            run = WorkflowRun.model_validate_json(f.read_text(encoding="utf-8"))
            self._runs[run.execution_id] = run

    def _flush(self, run: WorkflowRun) -> None:
        (self._dir / f"{run.execution_id}.json").write_text(
            json.dumps(run.model_dump(mode="json"), indent=2), encoding="utf-8"
        )

    def create(self, run: WorkflowRun) -> WorkflowRun:
        out = super().create(run)
        self._flush(out)
        return out

    def update(self, execution_id: str, **fields) -> WorkflowRun:
        out = super().update(execution_id, **fields)
        self._flush(out)
        return out
