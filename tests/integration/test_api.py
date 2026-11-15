"""FastAPI backend for the React app: every endpoint, with a fake LLM, mock data and a tiny KB."""

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from src.api import main
from src.workflow.graph import build_graph
from tests.fakes import FakeLLM, make_deps


@pytest.fixture
def client(monkeypatch):
    deps = make_deps(FakeLLM())
    graph = build_graph(deps, checkpointer=InMemorySaver())
    monkeypatch.setattr(main, "get_deps", lambda: deps)
    monkeypatch.setattr(main, "get_graph", lambda: graph)
    with TestClient(main.app) as c:
        yield c


def test_health_and_status(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    s = client.get("/api/status").json()
    assert set(s) >= {"services", "risk_profiles", "categories", "retrieval_mode", "disclaimer"}
    assert "moderate" in s["risk_profiles"]


def test_chat_keeps_memory_per_thread(client):
    first = client.post("/api/chat", json={"question": "What is a bond?", "thread_id": "t1"})
    assert first.status_code == 200
    body = first.json()
    assert body["thread_id"] == "t1" and body["agents"] == ["finance_qa"] and body["disclaimer"]
    second = client.post("/api/chat", json={"question": "Any news on TSLA?", "thread_id": "t1"})
    assert second.json()["agents"] == ["news"]


def test_chat_uses_the_analyzed_portfolio(client):
    r = client.post(
        "/api/chat",
        json={
            "question": "Analyze my portfolio",
            "portfolio": {
                "holdings": [{"ticker": "VTI", "shares": 10}, {"ticker": "BND", "shares": 5}]
            },
        },
    )
    assert "portfolio" in r.json()["agents"]
    assert r.json()["data"]["portfolio_analysis"]["total_value"] > 0


def test_chat_rejects_empty_question(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_market_lookup_by_company_name(client):
    r = client.get("/api/market/lookup", params={"q": "nvidia"}).json()
    assert r["ticker"] == "NVDA" and r["name"] == "NVIDIA Corp."
    assert r["snapshot"]["quote"]["is_mock"] and len(r["history"]) > 50
    assert client.get("/api/market/lookup", params={"q": "what is money"}).status_code == 404


def test_market_indexes(client):
    rows = client.get("/api/market/indexes").json()
    assert [r["ticker"] for r in rows] == ["SPY", "QQQ", "DIA"]


def test_portfolio_samples_and_analysis(client):
    samples = client.get("/api/portfolio/samples").json()
    assert "Balanced Three Fund" in samples
    holdings = samples["Balanced Three Fund"]
    a = client.post("/api/portfolio/analyze", json={"holdings": holdings}).json()
    assert 0 <= a["diversification_score"] <= 100 and a["positions"]
    assert client.post("/api/portfolio/analyze", json={"holdings": []}).status_code == 422


def test_goal_plan(client):
    r = client.post(
        "/api/goals/plan", json={"target_amount": 40000, "years": 6, "current_savings": 5000}
    )
    plan = r.json()
    assert [s["label"] for s in plan["scenarios"]] == ["low", "expected", "high"]
    assert plan["target_in_future_dollars"] > 40000
    assert client.post("/api/goals/plan", json={"target_amount": -1, "years": 6}).status_code == 422


def test_kb_search_and_sources(client):
    s = client.get("/api/kb/search", params={"q": "Roth IRA contribution limit"}).json()
    assert s["results"] and {"citation", "url", "score", "similarity", "text"} <= set(
        s["results"][0]
    )
    kb = client.get("/api/kb/sources").json()
    assert kb["chunks"] == sum(src["chunks"] for src in kb["sources"])


def test_metrics_snapshot(client):
    m = client.get("/api/metrics").json()
    assert {"retrieval", "routing", "safety", "latency", "tests", "experiments"} <= set(m)


def test_frontend_is_served_with_client_side_routes(tmp_path, monkeypatch):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text('<div id="root"></div>', encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    from fastapi import FastAPI

    app = FastAPI()
    monkeypatch.setattr(main, "app", app)
    main._mount_frontend(tmp_path)
    c = TestClient(app)
    assert 'id="root"' in c.get("/").text
    assert 'id="root"' in c.get("/portfolio").text  # unknown paths fall back to the app
    assert c.get("/favicon.svg").text == "<svg/>"
    assert 'id="root"' in c.get("/../../etc/passwd").text  # never escapes the build folder
