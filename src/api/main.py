"""Finnie REST API — the agents and tools behind the React app, plus the built app itself.

    uvicorn src.api.main:app --reload            # API on http://localhost:8000 (docs at /docs)
    cd frontend && npm run dev                   # React dev server, proxies /api here

Every endpoint is a thin wrapper: parsing, analysis and answers come from the same code the
agents and the MCP server use. When ``frontend/dist`` exists (the Docker image builds it), the
React app is served from ``/``.
"""

from __future__ import annotations

import csv
import json
import threading
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.tools import analyze_portfolio, plan_goal
from src.core.config import PROJECT_ROOT, get_secrets, get_settings
from src.core.schemas import (
    FinalResponse,
    GoalInput,
    GoalPlan,
    Portfolio,
    PortfolioAnalysis,
    UserProfile,
)
from src.data.providers.base import ProviderError
from src.data.reference import resolve_ticker, ticker_info
from src.utils import disclaimers

SAMPLES_DIR = PROJECT_ROOT / "data" / "sample_portfolios"
METRICS_PATH = PROJECT_ROOT / "data" / "metrics" / "metrics.json"
DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
INDEXES = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "DIA": "Dow Jones"}


# ---------- services (one per process, built on first use) ----------


@lru_cache(maxsize=1)
def get_deps():
    from src.workflow.deps import Deps

    return Deps.from_settings(get_settings())


@lru_cache(maxsize=1)
def get_graph():
    from src.workflow.graph import build_graph, sqlite_checkpointer

    settings = get_settings()
    return build_graph(
        get_deps(), sqlite_checkpointer(settings.resolve(settings.memory.checkpoint_db))
    )


_chat_lock = threading.Lock()  # one SQLite checkpointer connection → one chat turn at a time


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_deps().retriever  # noqa: B018 — load the embedding model before the first request
    yield


app = FastAPI(
    title="Finnie API",
    description="Multi-agent AI finance-education assistant. Educational only — not advice.",
    version="1.0",
    lifespan=lifespan,
)


# ---------- request bodies ----------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=64)
    profile: UserProfile = Field(default_factory=UserProfile)
    portfolio: Portfolio | None = None


# ---------- endpoints ----------


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    """What the UI needs on load: which live services are configured, options, disclaimer."""
    s, settings = get_secrets(), get_settings()

    def has(v) -> bool:
        return bool(v and v.get_secret_value())

    return {
        "services": {
            "LLM (OpenAI)": has(s.openai_api_key),
            "Market data (Alpha Vantage)": has(s.alpha_vantage_api_key),
            "News (Tavily)": has(s.tavily_api_key),
            "Tracing (LangSmith)": has(s.langsmith_api_key),
        },
        "risk_profiles": list(settings.goals.risk_profiles),
        "categories": list(settings.rag.categories),
        "retrieval_mode": settings.rag.retrieval_mode,
        "disclaimer": disclaimers.GENERAL,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """One chat turn through the LangGraph workflow; memory is kept per ``thread_id``."""
    from src.workflow.graph import ask

    with _chat_lock:
        final: FinalResponse = ask(
            get_graph(),
            req.question,
            req.thread_id,
            user_profile=req.profile.model_dump(),
            portfolio=req.portfolio.model_dump() if req.portfolio else None,
        )
    return {"thread_id": req.thread_id, **final.model_dump(mode="json")}


@app.get("/api/market/indexes")
def market_indexes() -> list[dict]:
    market, rows = get_deps().market, []
    for ticker, name in INDEXES.items():
        try:
            rows.append({"name": name, **market.quote(ticker).model_dump(mode="json")})
        except ProviderError:
            rows.append({"name": name, "ticker": ticker, "price": None})
    return rows


@app.get("/api/market/lookup")
def market_lookup(q: str) -> dict:
    """Ticker or company name ("nvidia") → quote, 52-week range, returns, trend and history."""
    ticker = resolve_ticker(q)
    if ticker is None:
        raise HTTPException(404, f"Couldn't match “{q.strip()}” to a ticker — try the symbol.")
    try:
        snap, history = get_deps().market.snapshot(ticker)
    except ProviderError as exc:
        raise HTTPException(502, f"No data for {ticker}: {exc}") from exc
    name = ticker_info(ticker).name
    return {
        "ticker": ticker,
        "name": None if name == ticker else name,
        "snapshot": snap.model_dump(mode="json"),
        "history": [{"date": p.date.date().isoformat(), "close": p.close} for p in history.points],
    }


@app.get("/api/portfolio/samples")
def portfolio_samples() -> dict[str, list[dict]]:
    samples = {}
    for path in sorted(SAMPLES_DIR.glob("*.csv")):
        with path.open(encoding="utf-8") as f:
            rows = [
                {"ticker": r["ticker"], "shares": float(r["shares"])} for r in csv.DictReader(f)
            ]
        samples[path.stem.replace("_", " ").title()] = rows
    return samples


@app.post("/api/portfolio/analyze")
def portfolio_analyze(portfolio: Portfolio) -> PortfolioAnalysis:
    if not portfolio.holdings:
        raise HTTPException(422, "Add at least one holding with shares > 0.")
    try:
        return analyze_portfolio(portfolio, get_deps().market)
    except ProviderError as exc:
        raise HTTPException(502, f"Couldn't fetch prices: {exc}") from exc


@app.post("/api/goals/plan")
def goals_plan(goal: GoalInput) -> GoalPlan:
    return plan_goal(goal, get_settings().goals)


@app.get("/api/kb/search")
def kb_search(q: str, category: str | None = None, k: int = 5) -> dict:
    retriever = get_deps().retriever
    result = retriever.retrieve(q, k=min(k, 10), category=category or None)
    return {
        "confident": result.confident,
        "top_similarity": result.top_dense_score,
        "min_score": retriever.min_score,
        "results": [
            {
                "citation": ev.chunk.meta.citation,
                "url": ev.chunk.meta.url,
                "category": ev.chunk.meta.category,
                "score": ev.score,
                "similarity": ev.dense_score,
                "text": ev.chunk.text,
            }
            for ev in result.evidence
        ],
    }


@app.get("/api/kb/sources")
def kb_sources() -> dict:
    """Every document in the knowledge base, grouped by category, with chunk counts."""
    store = get_deps().retriever.store
    counts: Counter[str] = Counter()
    docs: dict[str, dict] = {}
    for chunk in store.chunks:
        doc = chunk.doc_id.split(":")[0]
        counts[doc] += 1
        m = chunk.meta
        docs.setdefault(
            doc,
            {
                "title": m.title,
                "url": m.url,
                "source": m.source,
                "category": m.category,
                "file_type": m.file_type,
                "tax_year": m.tax_year,
            },
        )
    for doc, info in docs.items():
        info["chunks"] = counts[doc]
    return {
        "chunks": len(store),
        "sources": sorted(docs.values(), key=lambda d: (d["category"], d["title"])),
    }


@app.get("/api/metrics")
def metrics() -> dict:
    """The snapshot written by ``python -m scripts.collect_metrics``."""
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No metrics snapshot — run `python -m scripts.collect_metrics`.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


# ---------- the React app ----------


def _mount_frontend(dist: Path) -> None:
    if not (dist / "index.html").exists():
        return
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (dist / path).resolve()
        if path and file.is_file() and file.is_relative_to(dist.resolve()):
            return FileResponse(file)
        return FileResponse(dist / "index.html")  # client-side routes


_mount_frontend(DIST_DIR)
