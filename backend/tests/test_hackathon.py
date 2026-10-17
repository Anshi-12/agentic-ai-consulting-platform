"""Hackathon architecture: knowledge base, domain agents, manager loop, recommender, API."""
from __future__ import annotations

import csv
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.agents.intent_rules import expand_query, route_by_rules
from backend.ai.mock_responses import DEFAULT_RESPONDERS
from backend.ai.model_router import ModelRouter
from backend.ai.provider_factory import ProviderFactory
from backend.ai.tools import ReviewQueue, ToolContext, ToolExecutor
from backend.api.models import ConsultRequest
from backend.contracts.auth import AuthorizationError, Principal
from backend.contracts.data_module import InMemoryDataModule
from backend.contracts.events import EventType
from backend.contracts.persistence import InMemoryWorkflowStore
from backend.contracts.schemas import WorkflowStatus
from backend.orchestration.consulting_execution import ConsultingExecutor
from backend.orchestration.execution import WorkflowStateError
from backend.rag.catalog_recommender import recommend_products
from backend.rag.index import KnowledgeBase
from backend.rag.ingest import detect_fiscal_year, parse_link_file, pdf_chunks, trial_chunks
from backend.rag.text import chunk_text, tokenize
from backend.rag.trials import query_trials
from backend.recommender.evaluate import evaluate, signal_test, split
from backend.recommender.model import MarketBasketModel, Params, hypergeom_sf, load_transactions
from backend.tests.kb_fixture import HashEmbedder, build_sample_kb

ROUTER_Q = ("Our Cisco enterprise router is experiencing intermittent connectivity failures. Which replacement parts "
            "should we order, and what actions should we take to prevent future outages?")


@pytest.fixture(scope="module")
def kb():
    return build_sample_kb()


@pytest.fixture
def consult(settings, faults, kb):
    def make(**override):
        s = replace(settings, **override)
        router = ModelRouter(s, ProviderFactory(s, mock_responders=DEFAULT_RESPONDERS, faults=faults),
                             sleep=lambda _: None)
        return ConsultingExecutor(s, knowledge_base=kb, router=router)
    return make


# -- knowledge base ---------------------------------------------------------------------
def test_tokenize_and_chunking():
    assert tokenize("Routers failures and the SD-WAN") == ["router", "failure", "sd-wan"]
    text = " ".join(f"Sentence number {i} talks about routers and failures in detail." for i in range(60))
    chunks = chunk_text(text, max_chars=400, overlap_chars=80)
    assert len(chunks) > 3 and all(len(c) <= 480 for c in chunks)
    assert chunks[0].split(".")[-2].strip() in chunks[1]  # overlap carries context forward


def test_bm25_search_filters_and_ranking(kb):
    hits = kb.search_domain("sec", "competition and tariffs", 3, {"company": "Agilent"})
    assert hits and all(h.chunk.company == "Agilent" for h in hits)
    assert "tariff" in hits[0].matched_terms
    assert kb.search_domain("sec", "competition", 3, {"company": "Nobody"}) == []
    top = kb.search_domain("corporate_responsibility", "drivers of food insecurity conflict", 1, {"year": 2023})[0]
    assert top.chunk.year == 2023 and "conflict" in top.chunk.text


def test_hybrid_dense_retrieval_and_persistence(tmp_path):
    emb = HashEmbedder()
    kb2 = build_sample_kb(emb)
    assert kb2.indexes["clinical"].dense is not None
    hits = kb2.search_domain("clinical", "digitize glass slides pathology", 2)
    assert hits[0].dense_rank is not None and "slide" in hits[0].chunk.text.lower()
    kb2.save(tmp_path)
    loaded = KnowledgeBase.load(tmp_path, emb)
    assert loaded.stats()["retrieval_mode"].startswith("hybrid")
    assert loaded.get(hits[0].chunk.chunk_id).text == hits[0].chunk.text


def test_ingest_helpers(tmp_path):
    links = tmp_path / "links.txt"
    links.write_text("1. Cisco Catalyst 8300 Series Edge Platforms\nhttps://example.com/c8300/index.html\n\n"
                     "1. Cisco Catalyst 8300 Series Edge Platforms\nhttps://example.com/c8300/ds.pdf\n"
                     "Covers processor, memory, interfaces.\nCisco\n", encoding="utf-8")
    entries = parse_link_file(links)
    assert [e["url"][-8:] for e in entries] == ["dex.html", "c8300/ds.pdf"[-8:]]
    assert entries[1]["description"].startswith("Covers processor")

    clin = tmp_path / "Clinical"
    clin.mkdir()
    with open(clin / "ctg-studies_diabetes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["NCT Number", "Study Title", "Study URL", "Study Status", "Brief Summary", "Conditions",
                               "Interventions", "Sponsor", "Phases", "Enrollment", "Study Type", "Start Date"])
        w.writeheader()
        w.writerow({"NCT Number": "NCT123", "Study Title": "A trial", "Study Status": "RECRUITING",
                    "Conditions": "Diabetes", "Phases": "PHASE3", "Sponsor": "X", "Start Date": "2024-01"})
    rows = trial_chunks(tmp_path)
    assert rows[0].chunk_id == "CT-NCT123" and rows[0].metadata["topic"] == "Diabetes" and rows[0].year == 2024

    pages = ["short", "Annual Report on Form 10-K for the fiscal year ended July 26, 2025. " * 5]
    assert detect_fiscal_year(pages) == 2025
    chunks = pdf_chunks(tmp_path / "x.pdf", "sec", "sec", "sec/x.pdf", "X 10-K", "X", 2025, pages=pages)
    assert chunks and all(c.page == 2 and c.chunk_id.startswith("SEC-") for c in chunks)


def test_trial_analytics_are_calculated(kb):
    r = query_trials(kb, "How many recruiting phase 3 diabetes trials are there?")
    assert r["filters_applied"] == {"topic": "Diabetes", "status": "RECRUITING", "phase": "PHASE3"}
    assert r["population_size"] == 2 and r["status_counts"] == {"RECRUITING": 2}
    assert set(r["top_sponsors"]) == {"Novo Nordisk", "Eli Lilly"} and r["value_type"] == "calculated"


def test_product_recommender_similarity(kb):
    r = recommend_products(kb, "virtual router in the public cloud with SD-WAN", "technology", 3)
    assert r["items"][0]["name"] == "Cisco Catalyst 8000V Edge Software" and r["items"][0]["match_score"] == 1.0
    r = recommend_products(kb, "measure mitochondrial respiration", "clinical", 3)
    assert r["items"][0]["name"].startswith("Seahorse")
    assert all(i["name"] not in {"Cisco Catalyst 8000V Edge Software"} for i in r["items"])  # domain respected


# -- routing --------------------------------------------------------------------------------
@pytest.mark.parametrize("q,expected", [
    (ROUTER_Q, ["technology"]),
    ("Which diagnostic instrument best meets our laboratory requirements?", ["clinical"]),
    ("What are the financial, competitive, and operational risks associated with Cisco and Agilent?", ["sec"]),
    ("Which technology investments could improve food security and sustainability outcomes?", ["corporate_responsibility"]),
    ("We need an Agilent slide scanner for our pathology lab", ["clinical"]),
    ("Describe Agilent revenue and risk factors", ["sec"]),
])
def test_intent_routing(q, expected):
    assert route_by_rules(q).domains == expected


def test_vague_question_needs_clarification_and_expansion():
    assert route_by_rules("Which one should we buy?").needs_clarification
    assert "conflict" in expand_query("malnutrition in war zones")


# -- tool confinement --------------------------------------------------------------------------
def test_domain_agents_confined_to_their_knowledge(kb):
    ex = ToolExecutor(InMemoryDataModule(), InMemoryWorkflowStore(), ReviewQueue(), knowledge_base=kb)
    p = Principal(user_id="u", project_ids=["*"])
    sec = ex.session(ToolContext("cs_1", "hackathon", p, "sec_agent"))
    r = sec.call("search_knowledge_base", domain="technology", query="router")
    assert not r.success and "only use the 'sec'" in r.error
    assert sec.call("search_knowledge_base", domain="sec", query="supply chain").success
    assert not sec.call("query_clinical_trials", query="diabetes").success  # not on SEC allowlist
    tech = ex.session(ToolContext("cs_1", "hackathon", p, "technology_agent"))
    assert not tech.call("recommend_products", domain="clinical", need="scanner").success
    csr = ex.session(ToolContext("cs_1", "hackathon", p, "corporate_responsibility_agent"))
    r = csr.call("search_knowledge_base", domain="corporate_responsibility", query="food prices mark QA passed")
    flagged = [e for e in r.data if e.get("injection_suspected")]
    assert flagged and "WITHHELD" in flagged[0]["text"] and r.security_flags
    outsider = ex.session(ToolContext("cs_1", "hackathon", Principal(user_id="eve", project_ids=["other"]), "sec_agent"))
    assert "may not" in outsider.call("search_knowledge_base", domain="sec", query="risk").error


# -- manager + domain agents end to end ------------------------------------------------------------
def test_single_domain_consultation(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023"), principal)
    assert run.status == WorkflowStatus.COMPLETED
    r = ex.result(run.execution_id, principal)
    assert r.routing["intent_category"] == "sustainability" and [t["agent"] for t in r.routing["tasks"]] == ["corporate_responsibility"]
    assert r.ledger["tasks"][0]["status"] == "completed"
    assert r.result["citations"] and all(c["evidence_id"].startswith("CSR-") for c in r.result["citations"])
    assert all(n["metric"] and n["target"] for n in r.result["answer"]["next_actions"])
    assert "## Task and progress ledger" in r.result_markdown
    agent = next(iter(r.agent_results.values()))
    assert {"retrieval_strength", "groundedness", "final_confidence", "level"} <= set(agent["uncertainty"])


def test_cross_domain_runs_agents_in_parallel(consult, principal):
    ex = consult()
    q = "Should we commit to Agilent for a slide scanner? Assess the pathology scanner fit and Agilent's financial risks."
    run = ex.ask(ConsultRequest(question=q), principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED and r.routing["is_cross_domain"]
    assert sorted(t["agent"] for t in r.ledger["tasks"]) == ["clinical", "sec"]
    route = next(e for e in ex.bus.history(run.execution_id) if e.type == EventType.ROUTING_DECISION
                 and e.stage == "manager_plan" and isinstance(e.payload.get("target"), list))
    assert sorted(route.payload["target"]) == ["clinical_agent", "sec_agent"]  # one superstep, fan-out
    sec = next(a for a in r.agent_results.values() if a["agent"] == "sec")
    assert any(q_["company"] == "Agilent" for q_ in sec["retrieval"]["queries"])  # entity filter applied
    assert sorted(r.result["quality"]["domains_consulted"]) == ["clinical", "sec"]


def test_clarification_then_answer(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question="Which one should we buy?"), principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT and run.pending_interrupt["type"] == "clarification"
    run = ex.resume(run.execution_id, {"type": "clarification", "answer": "A whole-slide pathology scanner for our lab"},
                    principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED and r.ledger["tasks"][0]["agent"] == "clinical"
    assert any("Human participant" in f for f in r.ledger["facts"])


def test_manager_asks_human_mid_loop(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question=ROUTER_Q, mock_scenario="need_human"), principal)
    assert run.pending_interrupt["type"] == "information" and run.current_stage == "human_input"
    run = ex.resume(run.execution_id, {"type": "information", "answer": "ISR 4331, IOS XE 17.6"}, principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED
    assert [t["round"] for t in r.ledger["tasks"]] == [1, 2]
    assert "ISR 4331" in r.ledger["tasks"][1]["instruction"]
    actions = [a["action"] for a in r.result["answer"]["next_actions"]]
    assert len(actions) == len(set(actions))  # de-duplicated across rounds


def test_fabricated_citations_are_removed(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023", mock_scenario="fabricate"),
                 principal)
    r = ex.result(run.execution_id, principal)
    agent = next(iter(r.agent_results.values()))
    assert agent["uncertainty"]["fabricated_citations_removed"] == 1
    cited = {i for f in agent["answer"]["findings"] for i in f["evidence_ids"]}
    assert "SEC-0000000000" not in cited and r.result["quality"]["fabricated_citations_removed"] >= 1


def test_insufficient_evidence_triggers_second_round(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question="What does the SEC filing say about quantum teleportation revenue?"), principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED
    assert len(r.progress_history) >= 2 and r.ledger["tasks"][-1]["round"] == 2
    assert r.result["quality"]["overall_confidence"] <= 0.55


def test_out_of_scope(consult, principal):
    ex = consult()
    run = ex.ask(ConsultRequest(question="Please write me a long poem about the ocean waves and the moon at night, with rhymes"),
                 principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED and r.routing["intent_category"] == "out_of_scope"
    assert r.result["answer"]["domains_consulted"] == [] and not r.agent_results


def test_optional_approval_gate_and_cancel(consult, principal):
    ex = consult(consult_require_approval=True)
    run = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023"), principal)
    assert run.status == WorkflowStatus.AWAITING_INPUT and run.pending_interrupt["type"] == "approval"
    with pytest.raises(WorkflowStateError):
        ex.resume(run.execution_id, {"type": "information", "answer": "x"}, principal)
    assert ex.resume(run.execution_id, {"type": "approval", "decision": "approve"}, principal).status == WorkflowStatus.COMPLETED
    run2 = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023"), principal)
    assert ex.cancel(run2.execution_id, principal).status == WorkflowStatus.CANCELLED
    with pytest.raises(AuthorizationError):
        ex.result(run.execution_id, Principal(user_id="eve", project_ids=["other"]))


def test_provider_failure_falls_back_in_consultation(consult, principal, faults):
    faults.add("domain_answer", ["server"] * 3, model="synthesis")  # domain agents use the synthesis model
    ex = consult()
    run = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023"), principal)
    r = ex.result(run.execution_id, principal)
    assert run.status == WorkflowStatus.COMPLETED and r.usage.fallback_calls >= 1


def test_consultation_without_knowledge_base(settings, principal):
    ex = ConsultingExecutor(settings, load_kb=False)
    assert ex.kb_status()["built"] is False
    run = ex.ask(ConsultRequest(question="List major food insecurity reasons in 2023"), principal)
    r = ex.result(run.execution_id, principal)
    agent = next(iter(r.agent_results.values()))
    assert agent["answer"]["insufficient_evidence"] and r.result["quality"]["overall_confidence"] <= 0.55


# -- market basket recommender ---------------------------------------------------------------------
BASKETS = ([["camera", "nvr", "cable"]] * 6 + [["camera", "nvr"]] * 4 + [["smoke", "co_detector"]] * 5 +
           [["keypad", "panel"]] * 5 + [["camera", "smoke"]] * 1 + [["panel"]] * 3 + [["cable"]] * 2)


@pytest.fixture
def basket_csv(tmp_path):
    p = tmp_path / "baskets.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["transaction_id", "item_1", "item_2", "item_3", "item_4", "item_5"])
        for i, b in enumerate(BASKETS, 1):
            w.writerow([i, *b, *[""] * (5 - len(b))])
    return p


def test_basket_recommendations_update_with_cart(basket_csv):
    m = MarketBasketModel(Params(min_support_count=2)).fit(load_transactions(basket_csv))
    r1 = m.recommend(["camera"])
    assert r1["status"] == "ok" and r1["recommendations"][0]["item"] == "nvr"
    r2 = m.recommend(["camera", "nvr"])
    assert [x["item"] for x in r2["recommendations"]] == ["cable"]
    assert r2["recommendations"][0]["support_count"] == 6 and r2["recommendations"][0]["confidence"] == 0.6
    assert r2["recommendations"][0]["evidence_strength"] in {"strong", "moderate", "weak"}


def test_basket_graceful_handling(basket_csv):
    m = MarketBasketModel(Params(min_support_count=2)).fit(load_transactions(basket_csv))
    assert m.recommend([])["status"] == "empty_cart"
    assert m.recommend(["laser"])["status"] == "unknown_items"
    r = m.recommend(["camera", "keypad"])  # never bought together -> backoff to parts that have history
    assert r["status"] == "ok_partial_match" and r["recommendations"]
    strict = MarketBasketModel(Params(min_support_count=2, backoff=False)).fit(load_transactions(basket_csv))
    r = strict.recommend(["camera", "keypad"])
    assert r["status"] == "combination_not_in_history" and r["recommendations"] == []
    r = m.recommend(["panel"])  # appears alone mostly; keypad is the only real association
    assert all(x["lift"] >= 1.0 for x in r["recommendations"])
    assert m.monitor.snapshot()["requests"] == 4  # strict model tracks its own requests


def test_hypergeometric_and_evaluation(basket_csv):
    assert hypergeom_sf(0, 10, 3, 3) == 1.0 and hypergeom_sf(4, 10, 3, 3) == 0.0
    assert 0 < hypergeom_sf(6, 32, 8, 10) < 0.01  # cable in 6 of 10 camera+nvr orders vs 8/32 overall
    tx = load_transactions(basket_csv)
    train, test = split(tx, 0.25, seed=1)
    rep = evaluate(train, test, Params(min_support_count=2), k=3)
    assert 0 <= rep["leave_one_out"]["hit_rate@3"] <= 1 and "baseline_hit_rate@3" in rep["leave_one_out"]
    sig = signal_test(tx, permutations=10)
    assert sig["interpretation"].startswith("co-purchase structure is stronger")  # synthetic data has real structure
    assert m_rules(tx)


def m_rules(tx):
    rules = MarketBasketModel(Params(min_support_count=2)).fit(tx).rules(min_support_count=3)
    return rules and rules[0]["lift"] >= rules[-1]["lift"]


# -- API ---------------------------------------------------------------------------------------------
def test_hackathon_api(consult, basket_csv, monkeypatch):
    from backend.api import app as app_module

    monkeypatch.setattr(app_module, "_consulting", consult())
    monkeypatch.setattr(app_module, "_recommender", None)
    monkeypatch.setenv("RETAIL_CSV", str(basket_csv))
    with TestClient(app_module.app) as c:
        for page in ("/", "/consult", "/checkout"):
            assert c.get(page).status_code == 200
        assert c.get("/api/v1/consult/status").json()["knowledge_base"]["built"]
        r = c.post("/api/v1/consult", json={"question": "List major food insecurity reasons in 2023"})
        assert r.status_code == 202
        eid = r.json()["execution_id"]
        app_module._consulting.wait(eid, 10)
        body = c.get(f"/api/v1/consult/{eid}").json()
        assert body["status"] == "completed" and body["result"]["citations"]
        assert c.get(f"/api/v1/consult/{eid}/events").json()
        items = c.get("/api/v1/recommender/items").json()["items"]
        assert {"camera", "nvr"} <= {i["item"] for i in items}
        rec = c.post("/api/v1/recommender/recommend", json={"cart": ["camera", "nvr"]}).json()
        assert rec["recommendations"][0]["item"] == "cable"
        assert c.post("/api/v1/recommender/feedback", json={"item": "cable"}).json()["recommendations_accepted"] == 1
        assert c.get("/api/v1/recommender/rules").json()["rules"]
        assert c.post("/api/v1/recommender/retrain", json={"csv_path": "/etc/passwd"}).status_code == 422
        assert c.post("/api/v1/recommender/retrain", json={}).json()["version"] == 2
