import os
import httpx
from fastapi import HTTPException
from .contracts import FinancialAnalyticsRequest, ResearchRequest, ResearchTaskRequest

BASE=os.getenv("GROUP2_API_URL","http://127.0.0.1:8001")
async def knowledge_search(project_id, request: ResearchRequest):
    try:
        async with httpx.AsyncClient(base_url=BASE,timeout=20) as client:
            response=await client.post(f"/api/v1/projects/{project_id}/knowledge/search",json=request.model_dump(exclude_none=True))
        if response.status_code >= 400: raise HTTPException(response.status_code,detail={"code":"GROUP2_ERROR","message":"Group 2 returned an error"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503,detail={"code":"GROUP2_UNAVAILABLE","message":"The evidence service is unavailable; no research result was fabricated"}) from e

async def list_documents(project_id: str, domain: str | None = None):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=20) as client:
            response = await client.get(f"/api/v1/projects/{project_id}/documents", params={"domain": domain} if domain else None)
        if response.status_code >= 400: raise HTTPException(response.status_code, detail={"code":"GROUP2_ERROR","message":"Group 2 returned an error"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503, detail={"code":"GROUP2_UNAVAILABLE","message":"The document service is unavailable; no document result was fabricated"}) from e

async def upload_document(project_id: str, domain: str, filename: str, content: bytes, content_type: str | None):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
            response = await client.post(f"/api/v1/projects/{project_id}/documents", params={"domain": domain}, files={"file": (filename, content, content_type or "application/octet-stream")})
        if response.status_code >= 400: raise HTTPException(response.status_code, detail={"code":"GROUP2_ERROR","message":"Group 2 rejected the document"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503, detail={"code":"GROUP2_UNAVAILABLE","message":"The document service is unavailable; the upload was not indexed"}) from e

async def create_research_task(project_id: str, request: ResearchTaskRequest):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
            response = await client.post(f"/api/v1/projects/{project_id}/research/tasks", json=request.model_dump(exclude_none=True))
        if response.status_code >= 400: raise HTTPException(response.status_code, detail={"code":"GROUP2_ERROR","message":"Group 2 could not create the research task"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503, detail={"code":"GROUP2_UNAVAILABLE","message":"The research service is unavailable; no research result was fabricated"}) from e

async def get_research_task(project_id: str, task_id: str):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=20) as client:
            response = await client.get(f"/api/v1/projects/{project_id}/research/tasks/{task_id}")
        if response.status_code >= 400: raise HTTPException(response.status_code, detail={"code":"GROUP2_ERROR","message":"Group 2 could not retrieve the research task"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503, detail={"code":"GROUP2_UNAVAILABLE","message":"The research service is unavailable; no task status was fabricated"}) from e

async def financial_metrics(project_id: str, request: FinancialAnalyticsRequest):
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=60) as client:
            response = await client.post(f"/api/v1/projects/{project_id}/analytics/financial", json=request.model_dump())
        if response.status_code >= 400: raise HTTPException(response.status_code, detail={"code":"GROUP2_ERROR","message":"Group 2 could not calculate financial metrics"})
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(503, detail={"code":"GROUP2_UNAVAILABLE","message":"The analytics service is unavailable; no financial result was fabricated"}) from e
