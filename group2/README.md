# Group 2: Notebook-first Domain Knowledge, RAG & Recommender Engine

Self-contained, **Jupyter notebook-first** Group 2 MVP for the SJSU multi-agent management consulting hackathon. Supports the **Technology (Cisco/IBM/Google), Clinical (Agilent), SEC (Cisco/Agilent filings), and Corporate Responsibility (food security)** knowledge domains.

## Why notebooks?
All business logic, experiments, retrieval, evaluation, data processing, and the integration API live in `.ipynb` notebooks. **One small `api_entrypoint.py` is included solely because Uvicorn needs an importable ASGI module.** Do not run that bridge in production; Group 3 will replace dev authorization with real auth.

## Run

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/06_end_to_end_demo.ipynb` and **Run All**. It runs prerequisite notebooks via `%run` and ends with evaluated citations, recommended candidates, cross-domain retrieval, finance metrics, and testable outcomes. `notebooks/08_integration_tests.ipynb` runs security + API + integration checks. `notebooks/09_course_document_import.ipynb` imports the assigned source files once you place them in `data/course_sources/`.

### Research task execution

Notebook 04 has `run_research_task(project_id, domain, question, source_mode, urls, top_k)`. Use `source_mode="synthetic_fixture"` only for explicitly labeled demo tests; `indexed` uses real ingested project documents but excludes synthetic; `official_urls` fetches validated, allowlisted official HTTPS sources. Per-source fetch failures are returned explicitly. This demo is synchronous; a shared worker is still required for production workloads.

### Run the local notebook-backed REST API

```bash
GROUP2_DEV_MODE=true uvicorn api_entrypoint:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs. **Development-only** and NOT authenticated. Never bind to a public interface. If GROUP2_DEV_MODE=false the entrypoint refuses startup.

Sample project UUID: `77777777-7777-4777-8777-777777777777`.

## Notebook sequence

| Notebook | Purpose |
|---|---|
| 00_foundation | SQLite schema + Pydantic contracts + project scope |
| 01_document_ingestion | PDF/DOCX/CSV/XLSX/TXT/JSON/images, OCR, ZIP |
| 02_rag_vector_search | BM25 + TF-IDF offline hybrid; optional OpenAI SDK + native pgvector |
| 03_evidence_recommendations | Evidence verification and constrained ranking |
| 04_research_and_analytics | HTTPS official source fetcher, SEC EDGAR, financial analytics |
| 05_evaluation | Recall@K, Precision@K, MRR, nDCG, citation integrity, recommendation tests |
| 06_end_to_end_demo | Four-domain reproducible demo with multi-domain retrieval |
| 07_fastapi_integration | Group 1 adapter tools and Group 3-compatible dev REST API, research + analytics |
| 08_integration_tests | API/security/project isolation/finance/recommendation checks |
| 09_course_document_import | Import real instructor TXT/ZIP archival sources |

## Grounding and limits

- Original course TXT/ZIP sources are now included. The archive contains actual clinical research and food-security reports. `06_end_to_end_demo.ipynb` continues to use explicit synthetic examples. Cisco datasheets and SEC proxies are included in `data/course_sources/official_pdfs/` and indexed by notebook 17. Detailed hardware compatibility and Form 10-K financial statements still require correct source evidence.
- Local offline TF-IDF + BM25 is lexical/hybrid **not a pretrained semantic embedding system**. A real configured OpenAI-compatible embeddings + PostgreSQL/pgvector path exists in notebook 02. It needs a running PostgreSQL server with pgvector extension and valid embedding credentials; this optional path was not exercised in the offline environment.
- URLs are fetched only with explicit approval and an official-domain allowlist. Browser search/large-scale market-crawling is intentionally excluded from the core hackathon MVP.
- Images and scanned PDFs require Tesseract available on the host. Conversion errors surface explicitly rather than pretending extraction occurred.
- Citation IDs and project scoping are enforced in retrieval and ranking. Source grounding is textual evidence, NOT independent validation of hardware faults, diagnostics, medical suitability, or legal compliance.
- SQLite is a standalone demo store, not multi-tenant enterprise identity infrastructure. Integrate Group 3 auth and PostgreSQL migrations before public deployment.
- `06_end_to_end_demo.ipynb` exercises local paths; no claim of production-readiness.

## Integration

**Group 1**: In notebook 07 call `agent_knowledge_tool(project_id,domain,query,top_k)` or `agent_recommendation_tool(payload)`; outputs include structured evidence and citation data. Domain strings: `technology`, `clinical`, `sec`, `csr`.

**Group 3**: Register the Group 2 FastAPI routes in the central application (see notebook 07), use the shared project context and replace `dev_access`. `POST /api/v1/projects/{project_id}/knowledge/search` expects `question`, optional `domain`, `top_k`. `POST /api/v1/projects/{project_id}/recommendations/rank` expects `query`, `domain`, `device_model`, `candidates`. `POST /api/v1/projects/{project_id}/documents?domain=technology` accepts multipart `file`. Research: `POST /api/v1/projects/{project_id}/research/tasks` and `GET /api/v1/projects/{project_id}/research/tasks/{task_id}`. Financial metrics: `POST /api/v1/projects/{project_id}/analytics/financial`.

## Validation

Run notebook `08_integration_tests.ipynb` (Run All). Evaluation is based on a tiny labeled synthetic set and MUST NOT be represented as a deployment accuracy claim.


## Actual course-source package (updated)

This version **includes the uploaded original** `Agilent.txt`, `DataSource-Cisco.txt`, `Google GCP.txt`, `Clinical.zip`, and `FoodSecurity.zip` under `data/course_sources/`. The complete ZIP is approximately 68 MB; it contains the original archives, not unapproved altered documents.

**In VS Code on Mac:** Open the outer `group2-course-rag-engine` folder. Activate `.venv`, install `requirements.txt`, and use that kernel for all notebooks.

Run in this order:

1. `notebooks/09_course_document_import.ipynb` → **Run All** (indexes research PDFs from both archives; builds source catalog and CSV inventory).
2. `notebooks/10_course_rag_validation.ipynb` → **Run All** (runs five actual-document retrieval smoke tests and shows citations).
3. `notebooks/11_clinical_dataset_preview.ipynb` → **Run All** (optional; previews oversized clinical trials CSV files safely).
4. `notebooks/12_optional_official_source_download.ipynb` (optional; set `DOWNLOAD_OFFICIAL_SOURCES=True` to fetch the official links over HTTPS; internet required).

Your existing `06_end_to_end_demo.ipynb` and `08_integration_tests.ipynb` still use the synthetic baseline and remain runnable.

**Important source limitations:** The original Agilent/Cisco/Google TXT files are link catalogs rather than source text. **Four independently uploaded Cisco PDF datasheets are now bundled** and can be indexed with notebook 17. **Two Agilent and one KLA DEF 14A proxy filings are now bundled**, but these are **not** 2025 Form 10-K annual reports. The Clinical archive contains scientific research, not Agilent equipment service manuals. The local notebook backend uses SQLite + TF-IDF/BM25; native pgvector requires configuration and has not been exercised here.

5. `notebooks/13_course_evidence_recommendations.ipynb` → **Run All** (ranks evidence-supported CSR actions from the actual FAO PDFs and withholds unsupported options).


## Market Basket Analysis (NEW — pages 7–11 retail example)

This independent recommendation module complements **but does not replace** domain-specific RAG and consulting recommendation ranking. It uses **Apriori association rules** over transaction baskets with support, confidence, and lift. Recommendations update when a shopper adds an item; if the exact cart combination is absent from transaction history, the engine abstains rather than guessing. **Co-purchases are not hardware compatibility evidence.**

The instructor's `Retail.zip` was not provided. `data/synthetic/market_basket_transactions_SYNTHETIC.csv` contains deterministic **illustrative fictional purchases**, and `data/synthetic/market_basket_product_catalog.csv` maps SKU to item name. Don't present this dataset as historical consumer behavior.

**Inside VS Code** (open `group2-course-rag-engine`, use your `.venv` kernel):

1. Run All in `notebooks/15_market_basket_demo.ipynb` for dynamic cart recommendations. This runs `14_market_basket_engine.ipynb` automatically, seeds a project-scoped test dataset, and renders optional interactive SKU selection widgets.
2. Run All in `notebooks/16_market_basket_tests.ipynb` for Apriori quality, no-pattern abstention, data isolation, and FastAPI checks.
3. Optional: `notebooks/14_market_basket_engine.ipynb` for implementation details and independent functions.

The existing `notebooks/07_fastapi_integration.ipynb` now exposes:
- `GET /api/v1/projects/{project_id}/market-basket/products`
- `POST /api/v1/projects/{project_id}/market-basket/recommend` (JSON: `{ "cart": ["BOSCH-D7050"], "top_k": 5 }`)
- `POST /api/v1/projects/{project_id}/market-basket/transactions/import` (JSON: `{ "rows": [{"transaction_id":"1","sku":"ITEM-1","product_name":"Product 1"}], "replace": false }`)

The Group 1 bridge is `agent_market_basket_tool(project_id,cart,top_k)`; Group 3 can call those REST routes. Imports use `USER_DATASET` provenance and must be scoped to a project. The FastAPI launcher remains localhost / dev-only without production auth.

**Interpretation:** `confidence` means the proportion of historical carts with the antecedent that also included the suggested item; `lift` compares this to the candidate's overall prevalence. These measure purchase associations rather than causal business impact or product suitability. Apply compatibility and other domain rules separately.


## NEW — Official Cisco datasheets & SEC proxy PDFs (notebooks 17–18)

**Copy-only upgrade:** to avoid redoing your existing project, copy `notebooks/17_official_pdf_import.ipynb`, `notebooks/18_official_pdf_rag_checks.ipynb`, and the folder `data/course_sources/official_pdfs/` from the small official-docs add-on into your existing project folder. Keep notebooks 00–16, your `.venv`, and `.runtime/group2.sqlite3`.

In VS Code, select the same `.venv` kernel and run **17 → 18**. Notebook 17 adds four genuine Cisco datasheets to `technology` and three genuine SEC proxy documents to `sec`. Notebook 18 verifies their project-scoped retrieval and citation IDs. Indexing is repeatable and deduplicates identical documents.

- `agilent.pdf`: **Agilent 2025 DEF 14A**, not 10-K.
- `agient2.pdf`: **Agilent 2026 DEF 14A**, not 10-K.
- `agilent3.pdf`: **KLA 2024 DEF 14A**, not Agilent, not 10-K.
- `datasheet-c78-744088.pdf`: **Cisco Catalyst 8300**.
- `data_sheet-c78-732542.pdf`: **Cisco 4000 ISR**.
- `datasheet-c78-731632.pdf`: **Cisco ASR 1000**.
- `catalyst-8000v-edge-software-ds.pdf`: **Cisco Catalyst 8000V**.

The database indexes page-level passages via existing `ingest_blob`, `hybrid_search`, and `citation_exists` interfaces. The **optional native embeddings + pgvector path is not activated** by these notebooks; the offline TF-IDF/BM25 baseline is used.

**Caution:** The earlier note that no SEC filings existed is now obsolete: three **proxy** filings have been added. However the specific 2025 **Form 10-K** requested in the assignment remains unavailable. A datasheet should not be interpreted as a diagnosis or purchase compatibility guarantee.
