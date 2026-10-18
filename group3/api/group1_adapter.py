"""Authenticated Group 3 adapter for the merged Group 1 workflow engine.

Imports are intentionally lazy: the central API can start in demo mode even
when optional LangGraph/model packages are not installed yet.
"""
from .contracts import Role, User, WorkflowResumeRequest, WorkflowStartRequest

_service = None

def _principal(user: User, project_id: str):
    from backend.contracts.auth import Principal
    roles = ["consultant"]
    if user.role in (Role.administrator, Role.engagement_manager):
        roles.append("approver")
    return Principal(user_id=str(user.id), project_ids=[project_id], roles=roles)

def _get_service():
    global _service
    if _service is None:
        from backend.api.service import ConsultingService
        from backend.config import get_settings
        from .group2_data_module import Group2DataModule
        _service = ConsultingService(
            settings=get_settings(),
            background=True,
            executor=None,
        )
        # Replace the default in-memory source at the integration boundary.
        # This keeps Group 1 unchanged while routing its tools through Group 2.
        _service.executor.data = Group2DataModule()
        _service.executor.tools.data = _service.executor.data
        _service.executor.runtime.data = _service.executor.data
    return _service

def _execution_principal(user: User, execution_id: str):
    service = _get_service()
    run = service.executor.store.get(execution_id)
    return service, _principal(user, run.project_id)

def start(user: User, project_id: str, payload: WorkflowStartRequest):
    from backend.api.models import StartWorkflowRequest
    request = StartWorkflowRequest(project_id=project_id, **payload.model_dump())
    return _get_service().start_workflow(request, _principal(user, project_id))

def status(user: User, execution_id: str):
    service, principal = _execution_principal(user, execution_id)
    return service.get_status(execution_id, principal)

def outputs(user: User, execution_id: str):
    service, principal = _execution_principal(user, execution_id)
    return service.get_outputs(execution_id, principal)

def resume(user: User, execution_id: str, payload: WorkflowResumeRequest):
    from backend.api.models import ResumeWorkflowRequest
    request = ResumeWorkflowRequest(payload=payload.payload)
    service, principal = _execution_principal(user, execution_id)
    return service.resume_workflow(execution_id, request, principal)
