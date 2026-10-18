from uuid import UUID
import asyncio
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .contracts import *
from .security import current_user, allow, decode_token, issue
from .store import users, projects, audits
from .group2_adapter import create_research_task, financial_metrics, get_research_task, knowledge_search, list_documents, upload_document

app=FastAPI(title="Agentic Consulting Platform — Central API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:3000"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
def audit(actor,action,kind,rid=None): audits.append(AuditEvent(actor_id=actor.id,action=action,resource_type=kind,resource_id=rid))
def scoped(pid,user):
    p=projects.get(pid)
    if not p: raise HTTPException(404,detail={"code":"NOT_FOUND","message":"Project not found"})
    if user.role in (Role.administrator,Role.engagement_manager): return p
    if user.id != p.owner_id and user.id not in p.member_ids:
        raise HTTPException(403,detail={"code":"FORBIDDEN","message":"Project membership required"})
    if user.role == Role.client_executive and p.status != ProjectStatus.approved:
        raise HTTPException(403,detail={"code":"FORBIDDEN","message":"Client access requires an approved project"})
    return p

@app.get("/health")
def health(): return {"status":"ok","service":"central-api","version":"1.0.0"}
@app.post("/api/v1/auth/demo-login")
def login(role: Role=Role.engagement_manager):
    names={Role.administrator:"Ada Admin",Role.engagement_manager:"Morgan Manager",Role.consultant:"Casey Consultant",Role.client_executive:"Riley Executive"}; u=User(email=f"{role.value}@demo.local",name=names[role],role=role); users[u.id]=u; return {"access_token":issue(u),"user":u}
@app.get("/api/v1/me",response_model=User)
def me(user:User=Depends(current_user)): return user
@app.post("/api/v1/projects",response_model=Project)
def create(payload:ProjectCreate,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    p=Project(**payload.model_dump(),owner_id=user.id,status=ProjectStatus.active); projects[p.id]=p; audit(user,"project.created","project",p.id); return p
@app.get("/api/v1/projects",response_model=list[Project])
def list_projects(user:User=Depends(current_user)):
    if user.role in (Role.administrator,Role.engagement_manager): return list(projects.values())
    return [p for p in projects.values() if (p.owner_id==user.id or user.id in p.member_ids) and (user.role != Role.client_executive or p.status == ProjectStatus.approved)]
@app.get("/api/v1/projects/{project_id}",response_model=Project)
def get(project_id:UUID,user:User=Depends(current_user)): return scoped(project_id,user)
@app.post("/api/v1/projects/{project_id}/research")
async def research(project_id:UUID,payload:ResearchRequest,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user); result=await knowledge_search(project_id,payload); audit(user,"research.requested","project",project_id); return result
@app.get("/api/v1/projects/{project_id}/documents")
async def documents(project_id:UUID,domain:str|None=Query(default=None,max_length=30),user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user); return await list_documents(str(project_id),domain)
@app.post("/api/v1/projects/{project_id}/documents")
async def document_upload(project_id:UUID,domain:str=Query(min_length=2,max_length=30),file:UploadFile=File(...),user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user)
    if not file.filename: raise HTTPException(422,detail={"code":"INVALID_FILE","message":"A filename is required"})
    content=await file.read(10*1024*1024+1)
    if len(content)>10*1024*1024: raise HTTPException(413,detail={"code":"FILE_TOO_LARGE","message":"Files must be 10 MB or smaller"})
    result=await upload_document(str(project_id),domain,file.filename,content,file.content_type); audit(user,"document.uploaded","project",project_id); return result
@app.post("/api/v1/projects/{project_id}/research/tasks")
async def research_task(project_id:UUID,payload:ResearchTaskRequest,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user); result=await create_research_task(str(project_id),payload); audit(user,"research.task.created","project",project_id); return result
@app.get("/api/v1/projects/{project_id}/research/tasks/{task_id}")
async def research_task_status(project_id:UUID,task_id:str,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user); return await get_research_task(str(project_id),task_id)
@app.post("/api/v1/projects/{project_id}/analytics/financial")
async def financial_analysis(project_id:UUID,payload:FinancialAnalyticsRequest,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user); result=await financial_metrics(str(project_id),payload); audit(user,"financial.analysis.created","project",project_id); return result

@app.websocket("/ws/projects/{project_id}/workflows/{execution_id}")
async def workflow_updates(websocket:WebSocket, project_id:UUID, execution_id:str):
    token=websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401); return
    try:
        user=decode_token(token)
        project=scoped(project_id,user)
        from .group1_adapter import _execution_principal, _get_service
        service, principal=_execution_principal(user,execution_id)
        status=service.get_status(execution_id,principal)
        if str(status.project_id)!=str(project.id):
            await websocket.close(code=4403); return
    except HTTPException:
        await websocket.close(code=4403); return
    except (ImportError, KeyError, AttributeError):
        await websocket.close(code=4404); return
    await websocket.accept()
    loop=asyncio.get_running_loop(); queue:asyncio.Queue=asyncio.Queue()
    unsubscribe=service.subscribe(execution_id,principal,lambda message: loop.call_soon_threadsafe(queue.put_nowait,message))
    try:
        for message in service.get_events(execution_id,principal):
            await websocket.send_json(message)
        while True:
            try:
                message=await asyncio.wait_for(queue.get(),timeout=30)
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                await websocket.send_json({"channel":f"workflow:{execution_id}","event":{"type":"heartbeat","execution_id":execution_id}})
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
@app.post("/api/v1/projects/{project_id}/status",response_model=Project)
def status(project_id:UUID,payload:Decision,user:User=Depends(allow(Role.administrator,Role.engagement_manager))):
    p=scoped(project_id,user)
    if payload.decision=="approve": p.status=ProjectStatus.approved
    elif payload.decision=="reject": p.status=ProjectStatus.discovery
    else: p.status=ProjectStatus.awaiting_review
    projects[project_id]=p
    audit(user,f"project.{payload.decision}","project",project_id); return p

@app.post("/api/v1/projects/{project_id}/members",response_model=Project)
def add_project_member(project_id:UUID,payload:ProjectMemberAdd,user:User=Depends(allow(Role.administrator,Role.engagement_manager))):
    p=scoped(project_id,user)
    if not users.get(payload.user_id): raise HTTPException(404,detail={"code":"NOT_FOUND","message":"User not found"})
    if payload.user_id not in p.member_ids: p.member_ids.append(payload.user_id)
    projects[project_id]=p
    audit(user,"project.member_added","project",project_id)
    return p

@app.post("/api/v1/projects/{project_id}/agent-workflows")
def start_agent_workflow(project_id:UUID,payload:WorkflowStartRequest,user:User=Depends(allow(Role.administrator,Role.engagement_manager,Role.consultant))):
    scoped(project_id,user)
    try:
        from .group1_adapter import start
        result=start(user,str(project_id),payload); audit(user,"agent_workflow.started","project",project_id); return result
    except ImportError as exc:
        raise HTTPException(503,detail={"code":"AI_ENGINE_DEPENDENCIES_MISSING","message":"Group 1 engine dependencies are not installed"}) from exc

@app.get("/api/v1/agent-workflows/{execution_id}")
def agent_workflow_status(execution_id:str,user:User=Depends(current_user)):
    try:
        from .group1_adapter import status
        return status(user,execution_id)
    except ImportError as exc:
        raise HTTPException(503,detail={"code":"AI_ENGINE_DEPENDENCIES_MISSING","message":"Group 1 engine dependencies are not installed"}) from exc

@app.get("/api/v1/agent-workflows/{execution_id}/outputs")
def agent_workflow_outputs(execution_id:str,user:User=Depends(current_user)):
    try:
        from .group1_adapter import outputs
        return outputs(user,execution_id)
    except ImportError as exc:
        raise HTTPException(503,detail={"code":"AI_ENGINE_DEPENDENCIES_MISSING","message":"Group 1 engine dependencies are not installed"}) from exc

@app.post("/api/v1/agent-workflows/{execution_id}/resume")
def resume_agent_workflow(execution_id:str,payload:WorkflowResumeRequest,user:User=Depends(current_user)):
    try:
        from .group1_adapter import resume
        return resume(user,execution_id,payload)
    except ImportError as exc:
        raise HTTPException(503,detail={"code":"AI_ENGINE_DEPENDENCIES_MISSING","message":"Group 1 engine dependencies are not installed"}) from exc
@app.get("/api/v1/audit",response_model=list[AuditEvent])
def audit_log(user:User=Depends(allow(Role.administrator,Role.engagement_manager))): return list(audits)
