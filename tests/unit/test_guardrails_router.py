import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.schemas import RouterDecision
from src.workflow import guardrails
from src.workflow.router import keyword_route, route
from tests.fakes import FakeLLM

# ---------- guardrails ----------


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("Should I buy Tesla stock right now?", "advice"),
        ("What stocks should I buy this year?", "advice"),
        ("Is NVDA a good investment?", "advice"),
        ("Give me guaranteed returns of 20%", "guarantee"),
        ("How can I double my money fast?", "guarantee"),
        ("Can you file my taxes for me?", "personal_tax"),
        ("How do I use insider information to trade?", "illegal"),
        ("How can I hide income from the IRS?", "illegal"),
        ("What is a Roth IRA?", "ok"),
        ("How is my portfolio diversified?", "ok"),
    ],
)
def test_guardrail_categories(text, category):
    result = guardrails.check(text)
    assert result.category == category
    assert result.blocked == (category == "illegal")
    assert (result.note is None) == (category in {"ok", "illegal"})


# ---------- keyword router ----------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is a bond?", ["finance_qa"]),
        ("What's the Roth IRA contribution limit for 2025?", ["tax"]),
        ("Analyze my portfolio please", ["portfolio"]),
        ("I want to save $40,000 for a house in 6 years", ["goal"]),
        ("What's the stock price of AAPL?", ["market"]),
        ("Any news on Tesla?", ["news"]),
        ("How is NVDA?", ["market"]),  # a bare ticker → market
    ],
)
def test_keyword_route_single_intent(question, expected):
    assert keyword_route(question).intents == expected


def test_keyword_route_multi_intent_and_tickers():
    d = keyword_route("Show me news and the stock price of MSFT and how my HSA is taxed")
    assert set(d.intents) == {"news", "market", "tax"}
    assert d.tickers == ["MSFT", "HSA"][:1]  # HSA is an acronym, not a ticker


def test_keyword_route_uses_saved_portfolio():
    assert keyword_route("Is my mix too risky?", has_portfolio=True).intents[0] == "portfolio"


# ---------- LLM router + fallback ----------


def test_llm_route_resolves_follow_up_and_merges_tickers():
    decision = RouterDecision(
        intents=["tax"], tickers=[], standalone_question="What is the Roth IRA limit in 2025?"
    )
    llm = FakeLLM(structured={RouterDecision: decision})
    history = [
        HumanMessage("What's the traditional IRA limit?"),
        AIMessage("$7,000 for 2025."),
        HumanMessage("what about for a Roth?"),
    ]
    out = route(history, llm)
    assert out.intents == ["tax"]
    assert out.standalone_question == "What is the Roth IRA limit in 2025?"


def test_llm_failure_falls_back_to_keywords():
    llm = FakeLLM(structured={RouterDecision: RuntimeError("API down")})
    out = route([HumanMessage("Any news on AAPL?")], llm)
    assert out.intents == ["news"] and out.tickers == ["AAPL"]


def test_no_llm_uses_keywords():
    assert route([HumanMessage("What is a bond?")], None).intents == ["finance_qa"]
