"""PLACEHOLDER shared event interface (WebSocket-compatible status events)."""
from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable

from pydantic import BaseModel, Field

from backend.contracts.schemas import utcnow_iso


class EventType:
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_AWAITING_INPUT = "workflow.awaiting_input"
    WORKFLOW_RESUMED = "workflow.resumed"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_REJECTED = "workflow.rejected"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    ROUTING_DECISION = "routing.decision"
    TOOL_INVOKED = "tool.invoked"
    MODEL_CALL = "model.call"
    QA_RESULT = "qa.result"
    SECURITY_FLAG = "security.flag"


class WorkflowEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    execution_id: str
    project_id: str
    type: str
    stage: str | None = None
    status: str | None = None
    message: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utcnow_iso)
    sequence: int = 0

    def to_ws_message(self) -> dict[str, Any]:
        """Envelope a WebSocket gateway can forward verbatim."""
        return {"channel": f"workflow:{self.execution_id}", "event": self.model_dump(mode="json")}


class EventBus(ABC):
    @abstractmethod
    def publish(self, event: WorkflowEvent) -> None: ...

    @abstractmethod
    def subscribe(self, execution_id: str, callback: Callable[[WorkflowEvent], None]) -> Callable[[], None]: ...

    @abstractmethod
    def history(self, execution_id: str) -> list[WorkflowEvent]: ...


class InMemoryEventBus(EventBus):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, list[WorkflowEvent]] = defaultdict(list)
        self._subs: dict[str, list[Callable[[WorkflowEvent], None]]] = defaultdict(list)
        self._global: list[Callable[[WorkflowEvent], None]] = []

    def publish(self, event: WorkflowEvent) -> None:
        with self._lock:
            event.sequence = len(self._history[event.execution_id]) + 1
            self._history[event.execution_id].append(event)
            subs = list(self._subs[event.execution_id]) + list(self._global)
        for cb in subs:
            try:
                cb(event)
            except Exception:  # subscriber failures must never break the workflow
                pass

    def subscribe(self, execution_id: str, callback: Callable[[WorkflowEvent], None]) -> Callable[[], None]:
        with self._lock:
            target = self._global if execution_id == "*" else self._subs[execution_id]
            target.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in target:
                    target.remove(callback)

        return unsubscribe

    def history(self, execution_id: str) -> list[WorkflowEvent]:
        with self._lock:
            return list(self._history.get(execution_id, []))
