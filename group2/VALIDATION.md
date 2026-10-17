# Group 2 Course Corpus — Validation Results

Tested: October 17, 2026. Source files: user-provided `Clinical.zip`, `FoodSecurity.zip`, `Agilent.txt`, `DataSource-Cisco.txt`, and `Google GCP.txt`.

## Data intake

- 22 PDF documents successfully indexed from supplied ZIP archives: 20 clinical papers plus UN SOFI-2023 and SOFI-2024.
- 3,294 text chunks extracted with page-level citations (709 clinical, 2,585 CSR).
- Four large ClinicalTrials.gov-derived CSV files cataloged as structured datasets; these were not inappropriately converted into prose evidence.
- 19 official-source links cataloged from TXT files; no linked pages are counted as downloaded technical evidence.
- No PDF import failures in this source batch.

## Real-document retrieval smoke test

Five title-focused queries across clinical and CSR: expected source title retrieved in top five in 5/5 cases. Source title rank = 1 for all five in this particular test. All returned citation chunk identifiers were checked against indexed document records. Mean local retrieval latency measured around 1.9 seconds per query in this environment (varies widely by hardware and environment). This is **not** a representative relevance evaluation, full recommendation accuracy measure, or pretrained semantic embedding test.

## Recommendation smoke test

- Two CSR candidate action topics matched real citations from SOFI-2023 and SOFI-2024.
- A deliberately unsupported candidate was withheld.
- Candidate scores are heuristics, not measured likelihoods of effectiveness.

## Existing synthetic baseline

- `06_end_to_end_demo` code path rechecked, including evidence-based recommendation and synthetic research fixture assertions.
- Original `08_integration_tests.ipynb` was previously observed running successfully in the user's VS Code on the older notebook package; **not re-executed here** against the updated course package.

## Boundaries

- The environment used SQLite + TF-IDF/BM25. The optional pgvector adapter is defined but not exercised.
- Agilent, Cisco and Google TXT files are directories of URLs, not the contents of those URLs. Optional HTTPS fetch tool is **disabled by default**.
- Cisco/Agilent 2025 SEC filings and IBM documentation were not supplied.
- Clinical papers are disease/ML research and do not establish Agilent equipment part compatibility or patient-specific clinical appropriateness.
- This package is a hackathon prototype, not a production clinical or financial decision engine.


## Market basket (new)

Run `15_market_basket_demo.ipynb` and `16_market_basket_tests.ipynb`. The retail input is **synthetic**, not instructor-provided purchase history. Apriori computes interpretable support, confidence, lift and project-scoped dynamic-cart recommendations. No complete cart pattern → no suggestion. Only mark tests as passing when notebook executions have been validated.


## Official Cisco/SEC PDF update (notebooks 17–18)

Executed on a clean local SQLite project copy with the seven user-uploaded PDFs:
- 17: 7 PDFs indexed (4 Cisco datasheets, 3 SEC DEF 14A proxies); 1,346 new searchable chunks; 0 failures.
- 18: 7/7 expected-document title targets retrieved in top 5; 100% returned citation IDs resolve; provenance type and project-isolation assertions pass.
- 08: API, document security, ZIP safety, project isolation, and analytics tests passed after adding PDFs.
- 16: Apriori market-basket and FastAPI contract checks passed after adding PDFs.

These are development smoke tests. The Agilent 2025 and 2026 proxy statements sometimes appear together in the top results; the agent must disambiguate by year/document metadata. Technical replacement-part safety, external LLM routing, production auth, and native pgvector remain unverified.
