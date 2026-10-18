# Group 2 — Professor Retail CSV Integration (Notebook 20)

This adds the real 1,000-transaction course CSV to Group 2 without modifying notebooks 00–19. The original synthetic market-basket demo remains intact.

1. Open the existing `group2-course-rag-engine` root in VS Code (it contains `notebooks/`, `requirements.txt`).
2. Extract `group2-real-retail-addon.zip` **into that folder**. Its `notebooks/` and `data/` folders merge with yours.
3. Open `notebooks/20_real_retail_market_basket.ipynb` and select your existing `.venv` kernel.
4. Click **Run All**. The notebook imports or safely verifies 1,000 transactions and 40 distinct products, executes Apriori, demonstrates dynamic-cart behavior, and tests the same FastAPI contracts used by Group 3.
5. If the optional widget doesn't display, `python -m pip install ipywidgets` inside your activated `.venv` then rerun the last interactive cell.

**Existing notebooks are not overwritten.** This module uses its own fixed project ID `22222222-2222-4222-8222-222222222222` to protect the synthetic tests in notebooks 15/16. Group 3 must use that project ID for the real dataset, with the same existing endpoints.

The professor CSV is `transaction_id,item_1,...,item_5`, 1,000 rows and 40 SKUs. This dataset has relatively sparse product associations. The real notebook demonstrates evidence-backed rules at 1% minimum support, 10% confidence and 1.05 lift. Some multi-product carts have **no supported association rule**: not generating a recommendation is expected and avoids fake certainty. Real customer purchase provenance isn't independently established.

Do not commit `.runtime/` or `.venv/` to Git; databases are generated on each developer's computer. The ZIP includes the original retail CSV for reproducible project setup.
