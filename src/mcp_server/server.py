"""Finnie as an MCP server — lets Claude Desktop (or any MCP client) use Finnie's tools.

Every tool is a thin wrapper over the *same* functions the agents use (market data chain,
``analyze_portfolio``, ``plan_goal``, the RAG retriever, the LangGraph app), so Claude Desktop
and the Finnie app always agree. Transport: stdio (Claude Desktop starts this process).

    python -m src.mcp_server.server        # or: python C:\\path\\to\\src\\mcp_server\\server.py

Setup for Claude Desktop: see docs/mcp.md and claude_desktop_config.example.json.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):  # launched as a script by Claude Desktop → make `src` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from src.agents.tools import analyze_portfolio, parse_holdings, plan_goal  # noqa: E402
from src.core.schemas import GoalInput, Holding, Portfolio  # noqa: E402
from src.utils import disclaimers  # noqa: E402
from src.workflow.deps import Deps  # noqa: E402

# stdout carries the MCP protocol — logs must go to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

server = MCPServer(
    name="finnie",
    title="Finnie — financial education tools",
    instructions=(
        "Tools for financial EDUCATION: market quotes and trends, news, portfolio "
        "diversification analysis, savings-goal math, and a cited knowledge base (IRS, SEC "
        "Investor.gov). Never present results as buy/sell advice. " + disclaimers.GENERAL
    ),
    version="1.0.0",
)

_deps: Deps | None = None
_app = None


def get_deps() -> Deps:
    global _deps
    if _deps is None:
        _deps = Deps.from_settings()
    return _deps


def set_deps(deps: Deps | None) -> None:
    """Inject fakes in tests (and reset the cached graph)."""
    global _deps, _app
    _deps, _app = deps, None


@server.tool(description="Latest price and daily change for a stock/ETF ticker (e.g. AAPL, VTI).")
def get_quote(ticker: str) -> dict:
    quote = get_deps().market.quote(ticker)
    return quote.model_dump(mode="json")


@server.tool(
    description="Trend snapshot for a ticker: price, 52-week high/low, 20/50-day moving averages, "
    "1- and 6-month returns, trend label, plus the last `days` daily closes."
)
def get_price_history(ticker: str, days: int = 90) -> dict:
    snap, history = get_deps().market.snapshot(ticker)
    points = history.points[-max(min(days, 365), 2) :]
    return {
        "snapshot": snap.model_dump(mode="json"),
        "closes": [{"date": p.date.date().isoformat(), "close": p.close} for p in points],
        "source": history.source,
        "is_mock": history.is_mock,
    }


@server.tool(description="Recent financial news headlines for a query or ticker, with links.")
def search_news(query: str) -> list[dict]:
    return [item.model_dump() for item in get_deps().news.search(query)]


@server.tool(
    name="analyze_portfolio",
    description="Analyze a portfolio's diversification, concentration, asset/sector mix, fund "
    "costs and volatility. `holdings` is a list of {ticker, shares}, or pass `text` like "
    "'10 AAPL, 5 MSFT, 20 VTI'. Educational observations only — no buy/sell advice.",
)
def analyze_portfolio_tool(holdings: list[dict] | None = None, text: str | None = None) -> dict:
    if holdings:
        portfolio = Portfolio(holdings=[Holding.model_validate(h) for h in holdings])
    else:
        portfolio = parse_holdings(text or "")
    if not portfolio.holdings:
        return {"error": "No holdings found. Example: [{'ticker': 'VTI', 'shares': 10}]"}
    result = analyze_portfolio(portfolio, get_deps().market).model_dump(mode="json")
    return {**result, "disclaimer": disclaimers.GENERAL}


@server.tool(
    description="Savings-goal plan: inflation-adjusted target and required monthly saving in low / "
    "expected / high return scenarios for a risk profile (conservative|moderate|aggressive)."
)
def calculate_goal_plan(
    target_amount: float,
    years: float,
    current_savings: float = 0,
    monthly_contribution: float | None = None,
    risk_profile: str = "moderate",
) -> dict:
    goal = GoalInput(
        target_amount=target_amount,
        years=years,
        current_savings=current_savings,
        monthly_contribution=monthly_contribution,
        risk_profile=risk_profile,  # type: ignore[arg-type]
    )
    plan = plan_goal(goal, get_deps().settings.goals)
    return {**plan.model_dump(mode="json"), "disclaimer": "Returns are assumptions, not promises."}


@server.tool(
    description="Search Finnie's curated knowledge base (IRS publications, SEC Investor.gov, "
    "articles). Returns cited passages and whether the match is confident. Optional category: "
    "basics, investing, diversification, retirement_tax, goals_budgeting, markets."
)
def search_knowledge_base(query: str, category: str | None = None, k: int = 5) -> dict:
    result = get_deps().retriever.retrieve(query, k=min(max(k, 1), 10), category=category)
    return {
        "confident": result.confident,
        "top_dense_score": result.top_dense_score,
        "results": [
            {
                "citation": ev.chunk.meta.citation,
                "url": ev.chunk.meta.url,
                "category": ev.chunk.meta.category,
                "text": ev.chunk.text,
            }
            for ev in result.evidence
        ],
    }


@server.tool(
    description="Ask Finnie's full multi-agent assistant (router → specialist agents → cited "
    "answer). Use thread_id to continue a conversation."
)
def ask_finnie(
    question: str, thread_id: str | None = None, knowledge_level: str = "beginner"
) -> dict:
    global _app
    from langgraph.checkpoint.memory import InMemorySaver

    from src.workflow.graph import ask, build_graph

    if _app is None:
        _app = build_graph(get_deps(), InMemorySaver())
    thread_id = thread_id or str(uuid.uuid4())
    final = ask(_app, question, thread_id, {"knowledge_level": knowledge_level})
    return {**final.model_dump(mode="json"), "thread_id": thread_id}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
