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


# ===========================================================================
# Hackathon: multi-agent consulting (manager + domain agents) and checkout recommender
# ===========================================================================
import os  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from backend.api.models import ConsultRequest, ConsultResultResponse, ConsultResumeRequest  # noqa: E402
from backend.orchestration.consulting_execution import ConsultingExecutor  # noqa: E402
from backend.recommender.model import DEFAULT_CSV, MarketBasketModel  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
_consulting: ConsultingExecutor | None = None
_recommender: MarketBasketModel | None = None
_lazy_lock = threading.Lock()


def get_consulting() -> ConsultingExecutor:
    global _consulting
    with _lazy_lock:
        if _consulting is None:
            _consulting = ConsultingExecutor(get_settings())
    return _consulting


def retail_csv() -> Path:
    return Path(os.getenv("RETAIL_CSV", str(DEFAULT_CSV)))


def get_recommender() -> MarketBasketModel:
    global _recommender
    with _lazy_lock:
        if _recommender is None:
            path = retail_csv()
            if not path.exists():
                raise HTTPException(503, f"retail data not found at {path}. Run scripts/setup_hackathon_data.py first.")
            _recommender = MarketBasketModel.from_csv(path)
    return _recommender


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((WEB_DIR / name).read_text(encoding="utf-8"))


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return _page("index.html")


@app.get("/consult", response_class=HTMLResponse, include_in_schema=False)
def consult_page():
    return _page("consult.html")


@app.get("/checkout", response_class=HTMLResponse, include_in_schema=False)
def checkout_page():
    return _page("checkout.html")


# -- consulting ---------------------------------------------------------------
@app.get("/api/v1/consult/status")
def consult_status():
    ex = get_consulting()
    s = get_settings()
    return {"knowledge_base": ex.kb_status(), "provider_mode": s.provider_mode,
            "models": {"primary": s.model_primary, "light": s.model_light, "synthesis": s.model_synthesis,
                       "fallback": s.model_fallback}, "max_rounds": s.consult_max_rounds,
            "require_approval": s.consult_require_approval}


@app.post("/api/v1/consult", response_model=ConsultResultResponse, status_code=202)
def consult(req: ConsultRequest, p: Principal = Depends(get_principal)):
    ex = get_consulting()
    run = _call(ex.ask, req, p, True)
    return _call(ex.result, run.execution_id, p)


@app.get("/api/v1/consult/{execution_id}", response_model=ConsultResultResponse)
def consult_result(execution_id: str, p: Principal = Depends(get_principal)):
    return _call(get_consulting().result, execution_id, p)


@app.post("/api/v1/consult/{execution_id}/resume", response_model=ConsultResultResponse)
def consult_resume(execution_id: str, req: ConsultResumeRequest, p: Principal = Depends(get_principal)):
    ex = get_consulting()
    _call(ex.resume, execution_id, req.payload.model_dump(), p, True)
    return _call(ex.result, execution_id, p)


@app.post("/api/v1/consult/{execution_id}/cancel", response_model=ConsultResultResponse)
def consult_cancel(execution_id: str, p: Principal = Depends(get_principal)):
    ex = get_consulting()
    _call(ex.cancel, execution_id, p)
    return _call(ex.result, execution_id, p)


@app.get("/api/v1/consult/{execution_id}/events")
def consult_events(execution_id: str, after: int = 0, p: Principal = Depends(get_principal)):
    ex = get_consulting()
    return [e.to_ws_message() for e in _call(ex.events, execution_id, p) if e.sequence > after]


# -- market basket recommender --------------------------------------------------
class RecommendRequest(BaseModel):
    cart: list[str] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=5, ge=1, le=20)


class FeedbackRequest(BaseModel):
    item: str
    accepted: bool = True


class RetrainRequest(BaseModel):
    csv_path: str | None = Field(default=None, description="Defaults to RETAIL_CSV / the hackathon Retail CSV")


@app.get("/api/v1/recommender/items")
def recommender_items():
    m = get_recommender()
    return {"items": [{"item": i, "support_count": m.counts[i]} for i in m.items]}


@app.post("/api/v1/recommender/recommend")
def recommender_recommend(req: RecommendRequest):
    return get_recommender().recommend(req.cart, req.top_k)


@app.get("/api/v1/recommender/popular")
def recommender_popular(top_k: int = 5):
    return {"items": get_recommender().popular(top_k)}


@app.get("/api/v1/recommender/rules")
def recommender_rules(top_n: int = 25):
    return {"rules": get_recommender().rules(top_n=top_n)}


@app.get("/api/v1/recommender/model")
def recommender_model():
    return get_recommender().info()


@app.post("/api/v1/recommender/feedback")
def recommender_feedback(req: FeedbackRequest):
    if req.accepted:
        get_recommender().record_acceptance()
    return get_recommender().monitor.snapshot()


@app.post("/api/v1/recommender/retrain")
def recommender_retrain(req: RetrainRequest):
    global _recommender
    path = (Path(req.csv_path) if req.csv_path else retail_csv()).resolve()
    data_root = (Path(__file__).resolve().parents[2] / "data").resolve()
    if req.csv_path and data_root not in path.parents:
        raise HTTPException(422, "csv_path must be inside the project's data/ folder")
    if not path.exists() or path.suffix.lower() != ".csv":
        raise HTTPException(422, f"CSV not found: {path.name}")
    old = get_recommender()
    new = MarketBasketModel.from_csv(path, old.params)
    new.version = old.version + 1
    with _lazy_lock:
        _recommender = new
    return new.info()
