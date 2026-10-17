"""HTTP + WebSocket tests for the dev app (which wraps the same service functions Group 3 will call)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app as app_module
from backend.api.service import ConsultingService
from backend.tests.conftest import GOOD_BRIEF


@pytest.fixture
def client(harness, monkeypatch):
    svc = ConsultingService(executor=harness.ex, background=False)
    monkeypatch.setattr(app_module, "_service", svc)
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["provider_mode"] == "mock"


def test_full_http_flow(client):
    r = client.post("/api/v1/workflows", json={"project_id": "proj-northpeak", "client_brief": GOOD_BRIEF})
    assert r.status_code == 202
    eid = r.json()["execution_id"]
    assert r.json()["status"] == "awaiting_input" and r.json()["pending_interrupt"]["type"] == "approval"

    bad = client.post(f"/api/v1/workflows/{eid}/resume", json={"payload": {"type": "approval", "decision": "yolo"}})
    assert bad.status_code == 422

    r = client.post(f"/api/v1/workflows/{eid}/resume", json={"payload": {"type": "approval", "decision": "approve"}})
    assert r.status_code == 200 and r.json()["status"] == "completed"

    outputs = client.get(f"/api/v1/workflows/{eid}/outputs").json()
    assert outputs["report"]["executive_summary"]["headline"] and outputs["usage"]["model_calls"] >= 8

    recs = client.get(f"/api/v1/workflows/{eid}/recommendations").json()
    assert recs["approved"] and recs["recommendations"][0]["evidence"]

    events = client.get(f"/api/v1/workflows/{eid}/events?after=5").json()
    assert events and all(e["event"]["sequence"] > 5 for e in events)

    assert client.post(f"/api/v1/workflows/{eid}/cancel").status_code == 409


def test_errors_map_to_http_codes(client):
    assert client.get("/api/v1/workflows/wf_missing").status_code == 404
    assert client.post("/api/v1/workflows", json={"project_id": "../x", "client_brief": "hello there"}).status_code == 422


def test_websocket_streams_event_backlog(client):
    eid = client.post("/api/v1/workflows", json={"project_id": "proj-northpeak",
                                                 "client_brief": GOOD_BRIEF}).json()["execution_id"]
    with client.websocket_connect(f"/ws/workflows/{eid}") as ws:
        first = ws.receive_json()
        assert first["channel"] == f"workflow:{eid}" and first["event"]["type"] == "workflow.started"
