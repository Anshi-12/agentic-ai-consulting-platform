"""PLACEHOLDER client for the research / data modules owned by other groups.

The AI layer never fetches the web or databases directly. It asks this client
(through the allowlisted tools in backend/ai/tools.py) for research results,
indexed documents, numerical inputs and project context. The in-memory
implementation serves synthetic fixtures so the workflow runs offline.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from backend.contracts.schemas import EvidenceItem

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "synthetic_projects.json"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "is", "are", "what", "how", "why", "by", "with", "do", "does", "we", "our"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2}


class DataModuleClient(ABC):
    @abstractmethod
    def get_project_context(self, project_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def query_research(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]: ...

    @abstractmethod
    def retrieve_documents(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]: ...

    @abstractmethod
    def get_financial_inputs(self, project_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_evidence(self, project_id: str, evidence_ids: list[str]) -> list[EvidenceItem]: ...

    @abstractmethod
    def all_evidence_ids(self, project_id: str) -> set[str]: ...


class InMemoryDataModule(DataModuleClient):
    def __init__(self, projects: dict[str, Any] | None = None) -> None:
        if projects is None:
            projects = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["projects"]
        self._projects = projects
        self._evidence: dict[str, list[EvidenceItem]] = {
            pid: [EvidenceItem(project_id=pid, **e) for e in p.get("evidence", [])]
            for pid, p in projects.items()
        }

    def register_project(self, project_id: str, context: dict[str, Any], evidence: list[dict] | None = None,
                         financials: dict[str, Any] | None = None) -> None:
        self._projects[project_id] = {"context": context, "evidence": evidence or [], "financials": financials}
        self._evidence[project_id] = [EvidenceItem(project_id=project_id, **e) for e in evidence or []]

    def get_project_context(self, project_id: str) -> dict[str, Any] | None:
        p = self._projects.get(project_id)
        return dict(p["context"]) if p else None

    def _search(self, project_id: str, query: str, top_k: int, types: set[str] | None) -> list[EvidenceItem]:
        q = _tokens(query)
        scored = []
        for item in self._evidence.get(project_id, []):
            if types and item.source_type not in types:
                continue
            hay = _tokens(" ".join([item.title, item.content, " ".join(item.workstream_tags)]))
            tag_hits = len(q & set(item.workstream_tags)) * 2
            score = len(q & hay) + tag_hits
            if score > 0:
                scored.append((score, item.reliability, item))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        return [s[2].model_copy() for s in scored[:top_k]]

    def query_research(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]:
        return self._search(project_id, query, top_k, {"research", "interview", "dataset"})

    def retrieve_documents(self, project_id: str, query: str, top_k: int = 5) -> list[EvidenceItem]:
        return self._search(project_id, query, top_k, {"document", "dataset"})

    def get_financial_inputs(self, project_id: str) -> dict[str, Any] | None:
        p = self._projects.get(project_id)
        fin = p.get("financials") if p else None
        return json.loads(json.dumps(fin)) if fin else None

    def get_evidence(self, project_id: str, evidence_ids: list[str]) -> list[EvidenceItem]:
        wanted = set(evidence_ids)
        return [e.model_copy() for e in self._evidence.get(project_id, []) if e.evidence_id in wanted]

    def all_evidence_ids(self, project_id: str) -> set[str]:
        return {e.evidence_id for e in self._evidence.get(project_id, [])}
