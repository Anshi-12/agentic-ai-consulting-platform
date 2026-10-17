# Group 2 — Market Basket Analysis add-on

This complements the domain-specific RAG recommendation engine. The module implements **Apriori** frequent-itemset mining, single-consequent association rules, support/confidence/lift statistics, and **dynamic cart-based recommendations** aligned with the retail example on pages 7–11 of the course PDF.

## In VS Code (Mac)

1. Open the **outer folder** `group2-course-rag-engine` (the folder containing `requirements.txt`). Select your existing `.venv` notebook kernel.
2. Install new optional interactive-widget dependency using `python -m pip install -r requirements.txt` in the terminal after activating `.venv`.
3. Open `notebooks/15_market_basket_demo.ipynb` and **Run All**. It automatically loads `14_market_basket_engine.ipynb` and seeds 248 clearly labeled *synthetic* electronics orders. Recommendations change after cart additions. Interactive multi-select cart works when `ipywidgets` is installed.
4. Open `notebooks/16_market_basket_tests.ipynb` and **Run All** to verify Apriori, rule metrics, safe abstention, project isolation, input validation and FastAPI routes.

**Do not run the entire notebook list again.** Existing notebooks `06`, `08`, `09`, `10`, `11`, `12`, and `13` remain useful and compatible, but are not needed for the basket demo.

## Where the data live

- `data/synthetic/market_basket_transactions_SYNTHETIC.csv` — 248 illustrative orders in long format: `transaction_id,sku,product_name,category,quantity,source_type`.
- `data/synthetic/market_basket_product_catalog.csv` — SKU lookup.

The original `Retail.zip` was not provided, so **no real historical transaction data were used**. To import real data, first normalize to rows with `transaction_id`, `sku`, `product_name`, optional `category`, and optional `quantity`; load the rows into `import_basket_rows(project_id,rows,replace=True,source_type='USER_DATASET')`. `source_type` in this call is a separate argument, not a per-row field.

## The dynamic recommendation rule

- Use the **full current cart as the rule antecedent**, not partial matches that could produce irrelevant advice.
- Apply minimum support (default 5%), confidence (40%), and lift (1.05).
- Rank candidates by an explicit reproducible score weighting confidence, normalized lift, and support.
- Never suggest items already in the cart.
- If an exact basket combination is unseen, or no association passes thresholds, return an explanation and an empty recommendation list.

Co-purchase statistics **never establish actual hardware compatibility**, regulatory acceptability, or medical suitability. These rules are separate from evidence-grounded RAG recommendations.

## FastAPI (existing minimal `.py` launcher only)

Run from the project root with development mode enabled:

```bash
GROUP2_DEV_MODE=true uvicorn api_entrypoint:app --host 127.0.0.1 --port 8000
```

- `GET /api/v1/projects/{project_id}/market-basket/products` — product catalog and transaction count.
- `POST /api/v1/projects/{project_id}/market-basket/recommend` — request `{"cart":["BOSCH-D7050","DSC-PG9914"],"top_k":5}`.
- `POST /api/v1/projects/{project_id}/market-basket/transactions/import` — request `{"replace":false,"rows":[{"transaction_id":"ORD1","sku":"SKU1","product_name":"Example product","quantity":1}]}`.

**Important:** The demo dataset is seeded by running notebook `15`, not automatically when the API launches. Use the *same* project UUID and database location for the demo and API. The development API has **no production authentication**. Group 3 must replace authorization before deployment.

Group 1 adapter: `agent_market_basket_tool(project_id,cart,top_k=5)` (defined in `notebooks/07_fastapi_integration.ipynb`).

## Metrics

- **Support:** count of transactions containing both selected and suggested products divided by total transaction count.
- **Confidence:** count of transactions containing the cart and suggestion divided by count containing the cart.
- **Lift:** confidence divided by overall frequency of the suggested product.

`confidence` is not a model probability calibrated on new data. A proper offline evaluation against held-out real retail transactions should be performed when provided.
