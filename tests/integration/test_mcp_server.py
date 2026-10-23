"""MCP server: tool registration and calls through the MCP API, with fake services."""

import pytest

from src.mcp_server import server
from tests.fakes import FakeLLM, make_deps

EXPECTED_TOOLS = {
    "get_quote",
    "get_price_history",
    "search_news",
    "analyze_portfolio",
    "calculate_goal_plan",
    "search_knowledge_base",
    "ask_finnie",
}


@pytest.fixture(autouse=True)
def fake_deps():
    server.set_deps(make_deps(FakeLLM()))
    yield
    server.set_deps(None)


async def test_all_tools_are_listed_with_descriptions():
    tools = await server.server.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.description for t in tools)


async def _call(name: str, args: dict):
    return await server.server.call_tool(name, args)


async def test_tools_return_data():
    for name, args in [
        ("get_quote", {"ticker": "aapl"}),
        ("get_price_history", {"ticker": "VTI", "days": 30}),
        ("search_news", {"query": "TSLA"}),
        ("analyze_portfolio", {"text": "10 AAPL, 30 BND"}),
        ("calculate_goal_plan", {"target_amount": 20000, "years": 4}),
        (
            "search_knowledge_base",
            {"query": "Roth IRA contributions", "category": "retirement_tax"},
        ),
        ("ask_finnie", {"question": "What is a bond?"}),
    ]:
        result = await _call(name, args)
        assert result is not None, name


def test_tool_functions_directly():
    assert server.get_quote("aapl")["ticker"] == "AAPL"
    history = server.get_price_history("VTI", days=30)
    assert len(history["closes"]) == 30 and history["is_mock"]
    portfolio = server.analyze_portfolio_tool(holdings=[{"ticker": "VTI", "shares": 10}])
    assert portfolio["top_holding_weight"] == 1.0 and "disclaimer" in portfolio
    assert "error" in server.analyze_portfolio_tool(text="nothing here")
    plan = server.calculate_goal_plan(20000, 4, risk_profile="conservative")
    assert [s["label"] for s in plan["scenarios"]] == ["low", "expected", "high"]
    kb = server.search_knowledge_base("Roth IRA contributions")
    assert kb["confident"] and "590-A" in kb["results"][0]["citation"]
    answer = server.ask_finnie("What is a bond?")
    assert answer["agents"] == ["finance_qa"] and answer["thread_id"]
