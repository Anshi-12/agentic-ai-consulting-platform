"""Checkpointer factory (persistent workflow state).

`memory` (default) keeps checkpoints in-process. `sqlite` persists them to disk
so workflows survive restarts (requires langgraph-checkpoint-sqlite).
PLACEHOLDER for the shared persistence layer: when it exists, return a
checkpointer backed by it here (e.g. langgraph-checkpoint-postgres).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from backend.config import Settings


def get_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    backend = settings.checkpoint_backend
    if backend == "memory":
        return InMemorySaver()
    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CHECKPOINT_BACKEND=sqlite requires `pip install langgraph-checkpoint-sqlite`") from exc
        path = Path(settings.checkpoint_sqlite_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    raise ValueError(f"unknown CHECKPOINT_BACKEND '{backend}' (use memory or sqlite)")


def thread_config(execution_id: str) -> dict:
    return {"configurable": {"thread_id": execution_id}}
