"""Group 3 adapter that lets Group 1 retrieve evidence from Group 2.

This module only owns the integration boundary. Group 2's implementation and
notebooks remain unchanged. If Group 2 is unavailable, calls fail explicitly
so the workflow cannot claim that evidence was retrieved.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from backend.contracts.data_module import DataModuleClient
from backend.contracts.schemas import EvidenceItem


class Group2DataModule(DataModuleClient):
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("GROUP2_API_URL", "http://127.0.0.1:8001")).rstrip("/")
        self._cache: dict[str, list[EvidenceItem]] = {}

    def _search(self, project_id: str, query: str, top_k: int) -> list[EvidenceItem]:
        with httpx.Client(base_url=self.base_url, timeout=30) as client:
            response = client.post(
                f"/api/v1/projects/{project_id}/knowledge/search",
                json={"question": query, "top_k": top_k},
            )
            response.raise_for_status()
        raw = response.json().get("evidence", [])
        items: list[EvidenceItem] = []
        for item in raw:
            try:
                # Group 2 returns EvidenceHit fields; normalize them into
                # Group 1's shared EvidenceItem contract at the boundary.
                chunk_id = item.get("chunk_id") or item.get("evidence_id") or item.get("id")
                if chunk_id is None:
                    continue
                score = float(item.get("score", 0.7))
                evidence = EvidenceItem(
                    evidence_id=f"G2-{str(chunk_id).replace('_', '-')}",
                    project_id=project_id,
                    source=str(item.get("citation") or item.get("filename") or "Group 2 retrieval"),
                    title=str(item.get("filename") or item.get("title") or "Retrieved evidence"),
                    content=str(item.get("excerpt") or item.get("content") or item.get("text") or ""),
                    source_type=str(item.get("source_type") or item.get("domain") or "document"),
                    url=item.get("source_url") or item.get("url"),
                    reliability=max(0.0, min(1.0, score)),
                )
                items.append(evidence)
            except (TypeError, ValueError):
                continue
        self._cache.setdefault(project_id, [])
        known = {item.evidence_id for item in self._cache[project_id]}
        self._cache[project_id].extend(item for item in items if item.evidence_id not in known)
        return items

    def get_project_context(self, project_id: str) -> dict[str, Any] | None:
        return {"project_id": project_id, "evidence_source": "group2"}

    def query_research(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]:
        return self._search(project_id, query, top_k)

    def retrieve_documents(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]:
        return self._search(project_id, query, top_k)

    def get_financial_inputs(self, project_id: str) -> dict[str, Any] | None:
        # Group 2 exposes financial analytics as a calculation endpoint, not a
        # raw-input endpoint. Do not invent inputs for Group 1.
        return None

    def get_evidence(self, project_id: str, evidence_ids: list[str]) -> list[EvidenceItem]:
        wanted = set(evidence_ids)
        return [item.model_copy() for item in self._cache.get(project_id, []) if item.evidence_id in wanted]

    def all_evidence_ids(self, project_id: str) -> set[str]:
        return {item.evidence_id for item in self._cache.get(project_id, [])}
