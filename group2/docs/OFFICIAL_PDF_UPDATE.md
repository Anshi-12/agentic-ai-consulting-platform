# Official PDF Upgrade — Group 2

## Files
- 17_official_pdf_import.ipynb: checksum-verified ingestion of seven uploaded PDFs.
- 18_official_pdf_rag_checks.ipynb: title retrieval, citation, tenant isolation and SEC document-type checks.
- data/course_sources/official_pdfs/: original PDF files and metadata manifest.

## In VS Code
Keep the current project and `.venv`. Extract the add-on into the existing project folder (or copy `notebooks/17*`, `notebooks/18*`, and `data/course_sources/official_pdfs/` manually). Run notebook **17** and then notebook **18**. Do not delete `.runtime/group2.sqlite3` — it contains the existing 22 document indexes.

## Facts you may claim
Four Cisco product datasheets are indexed, with page-level citations. Two Agilent and one KLA **DEF 14A proxy statements** are indexed as `sec-def14a-proxy`. Do not claim 2025 Form 10-K annual reports are indexed; these PDFs do not include them. The three proxy statements are not Agilent lab equipment documentation.

## Group 1 / Group 3 contract
- Domains: `technology`, `clinical`, `sec`, `csr`.
- `POST /api/v1/projects/{project_id}/knowledge/search`: `{"question":"Cisco Catalyst 8300 supported power supplies","domain":"technology","top_k":5}`. Returns `evidence` array with chunk IDs, document IDs, filename, page_number, excerpt, citation, score, and source_type.
- `POST /api/v1/projects/{project_id}/recommendations/rank`: action/part candidates; exact-model matching enforced for replacement parts.
- The DEV backend uses `GROUP2_DEV_MODE=true` localhost-only and does NOT replace production auth.
- Your existing market-basket add-on remains separate from RAG document evidence.

## Limits
The 7-question check confirms target-title retrieval and source integrity only; it is not an expert adjudication of technical compatibility, SEC financial risk analysis, or a production recall benchmark. No background LLM or cross-agent routing is performed in Group 2.
