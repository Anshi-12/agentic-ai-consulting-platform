"""Knowledge-base document model and domain registry."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

# Domain agents and the knowledge each one is allowed to search.
DOMAINS = ("technology", "clinical", "sec", "corporate_responsibility")
INDEX_NAMES = ("technology", "clinical", "clinical_trials", "sec", "corporate_responsibility", "catalog")

DOMAIN_INDEXES: dict[str, tuple[str, ...]] = {
    "technology": ("technology",),
    "clinical": ("clinical", "clinical_trials"),
    "sec": ("sec",),
    "corporate_responsibility": ("corporate_responsibility",),
}

PREFIX = {
    "technology": "TECH",
    "clinical": "CLIN",
    "clinical_trials": "CT",
    "sec": "SEC",
    "corporate_responsibility": "CSR",
    "catalog": "PROD",
}

DOMAIN_LABELS = {
    "technology": "Technology (Cisco, IBM, Google)",
    "clinical": "Clinical (Agilent, clinical research PDFs, ClinicalTrials.gov)",
    "sec": "SEC filings (Cisco, Agilent, Alphabet)",
    "corporate_responsibility": "Corporate Responsibility (UN FAO food security reports)",
}


class Chunk(BaseModel):
    """One retrievable passage. `chunk_id` doubles as the citation / evidence ID."""

    chunk_id: str = Field(pattern=r"^[A-Z]{2,5}-[A-Za-z0-9_-]{1,40}$")
    index: str
    domain: str
    title: str
    source: str            # relative file path or URL
    text: str
    page: int | None = None
    doc_type: str = "pdf"   # pdf | trial | web | catalog
    company: str | None = None
    year: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        loc = f", p.{self.page}" if self.page else ""
        return f"{self.title}{loc}"


def make_chunk_id(index: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:10]
    return f"{PREFIX[index]}-{digest}"


def trial_chunk_id(nct: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", nct)[:38] or "UNKNOWN"
    return f"CT-{clean}"


def pretty_title(filename: str) -> str:
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", filename)
    stem = re.sub(r"[_]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()
