"""DEV-ONLY standalone HTTP/WebSocket app for the AI layer.

Group 3 owns the platform API. This app exists so Group 1 can run, demo and
integration-test the engine on its own. It calls the same ConsultingService
functions Group 3 should call, and uses placeholder dev auth.

Run:  uvicorn backend.api.app:app --reload     then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect

from backend.ai.tracing import langsmith_status
from backend.api.models import (
    AgentOutputsResponse,
    RecommendationsResponse,
    ResumeWorkflowRequest,
    StartWorkflowRequest,
    WorkflowStatusResponse,
)
from backend.api.service import ConsultingService
from backend.config import get_settings
from backend.contracts.auth import AuthorizationError, Principal, dev_principal
from backend.contracts.persistence import RunNotFoundError
from backend.orchestration.execution import WorkflowStateError

_service: ConsultingService | None = None


def get_service() -> ConsultingService:
    global _service
    if _service is None:
        _service = ConsultingService(settings=get_settings(), background=True)
    return _service


def get_principal(x_user_id: str = Header(default="dev-user")) -> Principal:
    """PLACEHOLDER auth: every caller is a dev principal scoped by DEV_ALLOWED_PROJECTS."""
    return dev_principal(x_user_id, get_settings().dev_allowed_projects)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_service()
    yield


app = FastAPI(title="Consulting AI – Group 1 engine (dev)", version="0.1.0", lifespan=lifespan)


def _call(fn, *args):
    try:
        return fn(*args)
    except RunNotFoundError:
        raise HTTPException(404, "workflow not found")
    except AuthorizationError as e:
        raise HTTPException(403, str(e))
    except WorkflowStateError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "provider_mode": s.provider_mode,
            "models": {"primary": s.model_primary, "light": s.model_light, "synthesis": s.model_synthesis,
                       "fallback": s.model_fallback},
            "checkpoint_backend": s.checkpoint_backend, "langsmith": langsmith_status()}


@app.post("/api/v1/workflows", response_model=WorkflowStatusResponse, status_code=202)
def start_workflow(req: StartWorkflowRequest, p: Principal = Depends(get_principal),
                   svc: ConsultingService = Depends(get_service)):
    return _call(svc.start_workflow, req, p)


@app.get("/api/v1/workflows/{execution_id}", response_model=WorkflowStatusResponse)
def workflow_status(execution_id: str, p: Principal = Depends(get_principal), svc: ConsultingService = Depends(get_service)):
    return _call(svc.get_status, execution_id, p)


@app.get("/api/v1/workflows/{execution_id}/outputs", response_model=AgentOutputsResponse)
def workflow_outputs(execution_id: str, p: Principal = Depends(get_principal), svc: ConsultingService = Depends(get_service)):
    return _call(svc.get_outputs, execution_id, p)


@app.post("/api/v1/workflows/{execution_id}/resume", response_model=WorkflowStatusResponse)
def resume_workflow(execution_id: str, req: ResumeWorkflowRequest, p: Principal = Depends(get_principal),
                    svc: ConsultingService = Depends(get_service)):
    return _call(svc.resume_workflow, execution_id, req, p)


@app.post("/api/v1/workflows/{execution_id}/retry", response_model=WorkflowStatusResponse)
def retry_workflow(execution_id: str, p: Principal = Depends(get_principal), svc: ConsultingService = Depends(get_service)):
    return _call(svc.retry_workflow, execution_id, p)


@app.post("/api/v1/workflows/{execution_id}/cancel", response_model=WorkflowStatusResponse)
def cancel_workflow(execution_id: str, p: Principal = Depends(get_principal), svc: ConsultingService = Depends(get_service)):
    return _call(svc.cancel_workflow, execution_id, p)


@app.get("/api/v1/workflows/{execution_id}/recommendations", response_model=RecommendationsResponse)
def workflow_recommendations(execution_id: str, p: Principal = Depends(get_principal),
                             svc: ConsultingService = Depends(get_service)):
    return _call(svc.get_recommendations, execution_id, p)


@app.get("/api/v1/workflows/{execution_id}/events")
def workflow_events(execution_id: str, after: int = 0, p: Principal = Depends(get_principal),
                    svc: ConsultingService = Depends(get_service)):
    return _call(svc.get_events, execution_id, p, after)


@app.websocket("/ws/workflows/{execution_id}")
async def workflow_ws(ws: WebSocket, execution_id: str):
    svc = get_service()
    principal = dev_principal(ws.headers.get("x-user-id", "dev-user"), get_settings().dev_allowed_projects)
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    try:
        backlog = svc.get_events(execution_id, principal)
        unsubscribe = svc.subscribe(execution_id, principal,
                                    lambda msg: loop.call_soon_threadsafe(queue.put_nowait, msg))
    except (RunNotFoundError, AuthorizationError) as e:
        await ws.send_json({"error": str(e)})
        await ws.close(code=4404)
        return
    try:
        for msg in backlog:
            await ws.send_json(msg)
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
