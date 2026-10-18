"""Build the knowledge base from the professor's hackathon data.

    python scripts/setup_hackathon_data.py path/to/Hackathon_export.zip   # once: unpacks into data/hackathon
    python -m backend.rag.ingest                                          # builds data/index
    python -m backend.rag.ingest --no-embed                               # BM25 only (no model download)

Folder → domain mapping (matches the hackathon architecture slide):
    Technology               DataSource-Cisco.txt, Google GCP.txt (+ web/technology/ if fetched)
    Clinical                 Clinical/**.pdf, Agilent.txt (+ web/clinical/), Clinical/ctg-studies_*.csv
    SEC                      UnitedStatesSecuritiesandExchangeCommission(SEC)/<Company>/*.pdf
    Corporate Responsibility FoodSecurity/*.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from backend.rag.documents import Chunk, make_chunk_id, pretty_title, trial_chunk_id
from backend.rag.embeddings import load_embedder
from backend.rag.index import DomainIndex, KnowledgeBase
from backend.rag.text import chunk_text, clean_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "hackathon"
DEFAULT_INDEX = ROOT / "data" / "index"
SEC_DIR = "UnitedStatesSecuritiesandExchangeCommission(SEC)"
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def log(msg: str) -> None:
    print(f"[ingest] {msg}", flush=True)


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
def read_pdf_pages(path: Path) -> list[str]:
    import logging

    import pypdf

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(clean_text(page.extract_text() or ""))
        except Exception:
            pages.append("")
    return pages


_FY = re.compile(r"fiscal\s+year\s+ended\s+([A-Z][a-z]+\s+\d{1,2},\s+(\d{4}))", re.I)


def detect_fiscal_year(pages: list[str]) -> int | None:
    for p in pages[:4]:
        m = _FY.search(p)
        if m:
            return int(m.group(2))
    return None


def pdf_chunks(path: Path, index: str, domain: str, rel: str, title: str, company: str | None = None,
               year: int | None = None, metadata: dict[str, Any] | None = None,
               pages: list[str] | None = None) -> list[Chunk]:
    pages = pages if pages is not None else read_pdf_pages(path)
    out = []
    for pno, text in enumerate(pages, start=1):
        if len(text) < 120:
            continue
        for ci, piece in enumerate(chunk_text(text)):
            out.append(Chunk(chunk_id=make_chunk_id(index, rel, pno, ci), index=index, domain=domain, title=title,
                             source=rel, text=piece, page=pno, doc_type="pdf", company=company, year=year,
                             metadata=dict(metadata or {})))
    return out


def sec_chunks(data: Path) -> list[Chunk]:
    out = []
    for pdf in sorted((data / SEC_DIR).glob("*/*.pdf")):
        company = pdf.parent.name
        pages = read_pdf_pages(pdf)
        fy = detect_fiscal_year(pages)
        title = f"{company} Form 10-K" + (f" (fiscal year {fy})" if fy else "")
        rel = str(pdf.relative_to(data))
        out += pdf_chunks(pdf, "sec", "sec", rel, title, company=company, year=fy,
                          metadata={"filing": "10-K"}, pages=pages)
        log(f"SEC  {title}: {len(pages)} pages")
    return out


def food_security_chunks(data: Path) -> list[Chunk]:
    out = []
    for pdf in sorted((data / "FoodSecurity").glob("*.pdf")):
        m = re.search(r"(20\d\d)", pdf.name)
        year = int(m.group(1)) if m else None
        title = f"The State of Food Security and Nutrition in the World {year} (FAO, IFAD, UNICEF, WFP, WHO)"
        rel = str(pdf.relative_to(data))
        chunks = pdf_chunks(pdf, "corporate_responsibility", "corporate_responsibility", rel, title,
                            company="FAO", year=year, metadata={"report": f"SOFI {year}"})
        out += chunks
        log(f"CSR  {pdf.name}: {len(chunks)} chunks")
    return out


def clinical_pdf_chunks(data: Path) -> list[Chunk]:
    out = []
    for pdf in sorted((data / "Clinical").glob("*/*.pdf")):
        topic = pdf.parent.name.replace("_", " ")
        rel = str(pdf.relative_to(data))
        out += pdf_chunks(pdf, "clinical", "clinical", rel, pretty_title(pdf.name), metadata={"topic": topic})
    log(f"CLIN {len(list((data / 'Clinical').glob('*/*.pdf')))} research PDFs → {len(out)} chunks")
    return out


# ---------------------------------------------------------------------------
# ClinicalTrials.gov CSVs
# ---------------------------------------------------------------------------
TRIAL_TOPICS = {"covid": "COVID-19", "diabetes": "Diabetes", "hearattack": "Heart attack", "kneeinjuries": "Knee injuries"}


def trial_chunks(data: Path, max_summary: int = 700) -> list[Chunk]:
    by_nct: dict[str, Chunk] = {}
    for f in sorted((data / "Clinical").glob("ctg-studies_*.csv")):
        key = f.stem.split("_", 1)[1].lower()
        topic = TRIAL_TOPICS.get(key, key)
        n = 0
        with open(f, encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                nct = (row.get("NCT Number") or "").strip()
                if not nct:
                    continue
                n += 1
                if nct in by_nct:
                    topics = by_nct[nct].metadata.setdefault("topics", [])
                    if topic not in topics:
                        topics.append(topic)
                    continue
                status = row.get("Study Status", "")
                phases = row.get("Phases", "")
                summary = (row.get("Brief Summary") or "").strip()
                text = (f"{row.get('Study Title', '').strip()}. Status: {status}. Phase: {phases or 'NA'}. "
                        f"Conditions: {row.get('Conditions', '')}. Interventions: {row.get('Interventions', '')}. "
                        f"Sponsor: {row.get('Sponsor', '')}. Enrollment: {row.get('Enrollment', '')}. "
                        f"Study type: {row.get('Study Type', '')}. Start: {row.get('Start Date', '')}. "
                        f"Primary outcome: {(row.get('Primary Outcome Measures') or '')[:300]}. "
                        f"Summary: {summary[:max_summary]}")
                start_year = re.match(r"(\d{4})", row.get("Start Date", "") or "")
                by_nct[nct] = Chunk(
                    chunk_id=trial_chunk_id(nct), index="clinical_trials", domain="clinical",
                    title=f"{nct}: {row.get('Study Title', '').strip()[:160]}",
                    source=row.get("Study URL") or f"https://clinicaltrials.gov/study/{nct}", text=text,
                    doc_type="trial", company=(row.get("Sponsor") or None),
                    year=int(start_year.group(1)) if start_year else None,
                    metadata={"nct": nct, "status": status, "phases": phases, "topic": topic, "topics": [topic],
                              "sponsor": row.get("Sponsor", ""), "conditions": row.get("Conditions", ""),
                              "interventions": row.get("Interventions", ""), "enrollment": row.get("Enrollment", ""),
                              "study_type": row.get("Study Type", ""), "funder_type": row.get("Funder Type", ""),
                              "start_date": row.get("Start Date", ""), "url": row.get("Study URL", "")})
        log(f"CT   {f.name}: {n} studies")
    return list(by_nct.values())


# ---------------------------------------------------------------------------
# Link catalogs (DataSource-Cisco.txt, Agilent.txt, Google GCP.txt) and fetched web pages
# ---------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+")


def parse_link_file(path: Path) -> list[dict[str, str]]:
    """Turn the professor's link lists into catalog entries {name, url, description}."""
    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    entries: dict[str, dict[str, str]] = {}
    last_name = ""
    pending_desc_for: str | None = None
    for line in lines:
        if not line or line.lower() in {"cisco", "agilent"}:
            continue
        url = _URL.search(line)
        if url:
            u = url.group(0).rstrip(").,")
            name = re.sub(r"^\d+\.\s*", "", last_name).strip() or u
            entry = entries.setdefault(u, {"name": name, "url": u, "description": ""})
            pending_desc_for = u
            continue
        if pending_desc_for and (line.lower().startswith(("covers", "20-page", "19-page")) or "datasheet" in line.lower()):
            entries[pending_desc_for]["description"] = (entries[pending_desc_for]["description"] + " " + line).strip()
            continue
        last_name = line
        pending_desc_for = None
    return list(entries.values())


LINK_FILES = {"DataSource-Cisco.txt": ("technology", "Cisco"), "Google GCP.txt": ("technology", "Google"),
              "Agilent.txt": ("clinical", "Agilent")}


def catalog_entries(data: Path) -> list[dict[str, Any]]:
    out = []
    for fname, (domain, vendor) in LINK_FILES.items():
        p = data / fname
        if not p.exists():
            continue
        for e in parse_link_file(p):
            kind = "datasheet" if e["url"].lower().endswith(".pdf") else "product page"
            out.append({**e, "domain": domain, "vendor": vendor, "kind": kind, "source_file": fname})
    return out


def load_web_manifest(data: Path) -> list[dict[str, Any]]:
    p = data / "web" / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def web_chunks(data: Path, manifest: list[dict[str, Any]]) -> list[Chunk]:
    """Pages downloaded by scripts/fetch_web_sources.py."""
    out = []
    for m in manifest:
        if m.get("status") != "ok":
            continue
        f = data / m["path"]
        if not f.exists():
            continue
        domain = m["domain"]
        title = f"{m.get('vendor', '')} {m.get('name', '')}".strip()
        if f.suffix.lower() == ".pdf":
            pages = read_pdf_pages(f)
        else:
            pages = [clean_text(f.read_text(encoding="utf-8", errors="replace"))]
        for pno, text in enumerate(pages, start=1):
            for ci, piece in enumerate(chunk_text(text)):
                out.append(Chunk(chunk_id=make_chunk_id(domain, m["url"], pno, ci), index=domain, domain=domain,
                                 title=title, source=m["url"], text=piece, page=pno if len(pages) > 1 else None,
                                 doc_type="web", company=m.get("vendor"), metadata={"url": m["url"]}))
    if out:
        log(f"WEB  {len(out)} chunks from {sum(1 for m in manifest if m.get('status') == 'ok')} fetched pages")
    return out


def catalog_chunks(entries: list[dict[str, Any]], web: list[Chunk]) -> tuple[list[Chunk], list[Chunk]]:
    """Catalog index (for product recommendation) + the same entries as citable domain chunks."""
    web_text: dict[str, str] = {}
    for c in web:
        url = c.metadata.get("url", "")
        if len(web_text.get(url, "")) < 4000:
            web_text[url] = (web_text.get(url, "") + " " + c.text)[:4000]
    catalog, citable = [], []
    for e in entries:
        desc = e["description"] or f"{e['vendor']} {e['kind']} for {e['name']}."
        enriched = web_text.get(e["url"], "")
        text = f"{e['name']}. {e['vendor']} {e['kind']}. {desc}" + (f" {enriched}" if enriched else "")
        meta = {"url": e["url"], "vendor": e["vendor"], "kind": e["kind"], "enriched": bool(enriched),
                "source_file": e["source_file"], "domain": e["domain"]}
        catalog.append(Chunk(chunk_id=make_chunk_id("catalog", e["url"]), index="catalog", domain=e["domain"],
                             title=e["name"], source=e["url"], text=text, doc_type="catalog", company=e["vendor"],
                             metadata=meta))
        citable.append(Chunk(chunk_id=make_chunk_id(e["domain"], "catalog", e["url"]), index=e["domain"],
                             domain=e["domain"], title=f"{e['vendor']} {e['name']} ({e['kind']})", source=e["url"],
                             text=text, doc_type="catalog", company=e["vendor"], metadata=meta))
    return catalog, citable


# ---------------------------------------------------------------------------
def build(data: Path = DEFAULT_DATA, out: Path = DEFAULT_INDEX, embed: bool = True, embed_trials: bool = False,
          only: Iterable[str] | None = None) -> KnowledgeBase:
    if not data.exists():
        raise FileNotFoundError(f"{data} not found. Run: python scripts/setup_hackathon_data.py <Hackathon_export.zip>")
    t0 = time.time()
    manifest = load_web_manifest(data)
    web = web_chunks(data, manifest)
    catalog, citable = catalog_chunks(catalog_entries(data), web)
    groups: dict[str, list[Chunk]] = {
        "technology": [c for c in citable if c.domain == "technology"] + [c for c in web if c.domain == "technology"],
        "clinical": clinical_pdf_chunks(data) + [c for c in citable if c.domain == "clinical"]
                    + [c for c in web if c.domain == "clinical"],
        "clinical_trials": trial_chunks(data),
        "sec": sec_chunks(data),
        "corporate_responsibility": food_security_chunks(data),
        "catalog": catalog,
    }
    if only:
        groups = {k: v for k, v in groups.items() if k in set(only)}
    embedder = load_embedder() if embed else None
    indexes = {}
    for name, chunks in groups.items():
        t = time.time()
        do_embed = embed and (name != "clinical_trials" or embed_trials)
        indexes[name] = DomainIndex.build(name, chunks, embedder, embed=do_embed)
        log(f"index {name:<25} {len(chunks):>6} chunks  dense={'yes' if indexes[name].dense is not None else 'no'}"
            f"  {time.time() - t:.1f}s")
    kb = KnowledgeBase(indexes, embedder, {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_dir": str(data),
        "embedder": embedder.name if embedder else None,
        "web_pages_fetched": sum(1 for m in manifest if m.get("status") == "ok"),
        "sources": {name: sorted({c.source for c in ch})[:200] for name, ch in groups.items() if name != "clinical_trials"},
        "counts": {name: len(ch) for name, ch in groups.items()},
    })
    kb.save(out)
    log(f"done in {time.time() - t0:.0f}s → {out}")
    return kb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=str(DEFAULT_INDEX))
    ap.add_argument("--no-embed", action="store_true", help="BM25 only; skips the embedding model download")
    ap.add_argument("--embed-trials", action="store_true", help="also embed the ~32k trial rows (slow on CPU)")
    args = ap.parse_args()
    build(Path(args.data), Path(args.out), embed=not args.no_embed, embed_trials=args.embed_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
