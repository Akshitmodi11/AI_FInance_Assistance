"""End-to-end graph runs with fake LLMs: every route, multi-intent fan-out, guardrails, memory."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.core.schemas import GoalInput, RouterDecision
from src.workflow.graph import ask, build_graph
from src.workflow.guardrails import REFUSAL
from tests.fakes import FakeLLM, make_deps

PROFILE = {"knowledge_level": "beginner", "risk_tolerance": "moderate", "horizon_years": 10}


@pytest.fixture
def app():
    """Keyword routing (no structured router output scripted) + fake answers."""
    return build_graph(make_deps(FakeLLM()), checkpointer=InMemorySaver())


@pytest.mark.parametrize(
    ("question", "agent"),
    [
        ("What is a bond?", "finance_qa"),
        ("What is the Roth IRA contribution limit?", "tax"),
        ("Analyze my portfolio: 10 AAPL, 5 MSFT and 20 BND", "portfolio"),
        ("I want to save $40,000 for a house in 6 years, I have $5,000 saved", "goal"),
        ("What's the stock price of AAPL?", "market"),
        ("Any news on TSLA?", "news"),
    ],
)
def test_each_route_answers(app, question, agent):
    final = ask(app, question, thread_id=f"t-{agent}", user_profile=PROFILE)
    assert final.agents == [agent]
    assert final.answer
    assert final.disclaimer


def test_rag_answers_cite_sources(app):
    final = ask(app, "What is the Roth IRA contribution limit?", "t-cite", PROFILE)
    assert any("590-A" in c.label for c in final.citations)
    assert "not tax advice" in final.answer.lower()


def test_multi_intent_fans_out_in_parallel(app):
    final = ask(app, "Any news on AAPL and what's the stock price of AAPL?", "t-multi", PROFILE)
    assert set(final.agents) == {"news", "market"}
    assert "### " in final.answer  # one section per agent
    assert "market" in final.data and "news" in final.data


def test_structured_data_for_ui(app):
    final = ask(app, "Analyze my portfolio: 10 AAPL and 30 BND", "t-data", PROFILE)
    analysis = final.data["portfolio_analysis"]
    assert {p["ticker"] for p in analysis["positions"]} == {"AAPL", "BND"}
    assert analysis["is_mock"] is True


def test_illegal_request_is_refused_without_agents(app):
    final = ask(app, "How do I trade on insider information?", "t-bad", PROFILE)
    assert final.answer == REFUSAL and final.agents == []


def test_advice_request_is_reframed_not_refused(app):
    final = ask(app, "Should I buy AAPL? What's the stock price of AAPL?", "t-adv", PROFILE)
    assert final.agents == ["market"]
    assert final.guardrail_note and "can't tell you what to buy" in final.guardrail_note


def test_memory_keeps_conversation_per_thread(app):
    ask(app, "What is a bond?", "t-mem", PROFILE)
    ask(app, "What is diversification?", "t-mem", PROFILE)
    state = app.get_state({"configurable": {"thread_id": "t-mem"}}).values
    assert len(state["messages"]) == 4  # 2 questions + 2 answers
    other = app.get_state({"configurable": {"thread_id": "t-other"}}).values
    assert not other.get("messages")


def test_per_turn_outputs_are_reset(app):
    ask(app, "Any news on TSLA?", "t-reset", PROFILE)
    final = ask(app, "What is a bond?", "t-reset", PROFILE)
    assert final.agents == ["finance_qa"] and "news" not in final.data


def test_llm_router_follow_up_uses_standalone_question():
    decision = RouterDecision(
        intents=["tax"], standalone_question="What is the Roth IRA contribution limit?"
    )
    llm = FakeLLM(structured={RouterDecision: decision})
    app = build_graph(make_deps(llm), checkpointer=InMemorySaver())
    final = ask(app, "what about for a Roth?", "t-follow", PROFILE)
    assert final.agents == ["tax"]
    assert "Roth IRA contribution limit" in llm.calls[-1][-1].content  # agent saw the rewrite


def test_goal_extraction_via_structured_llm():
    goal = GoalInput(target_amount=20_000, years=4, current_savings=1_000)
    llm = FakeLLM(structured={GoalInput: goal})
    app = build_graph(make_deps(llm), checkpointer=InMemorySaver())
    final = ask(app, "I want to save 20k for a car in 4 years", "t-goal", PROFILE)
    plan = final.data["goal_plan"]
    assert plan["goal"]["target_amount"] == 20_000
    assert [s["label"] for s in plan["scenarios"]] == ["low", "expected", "high"]


def test_degraded_mode_without_llm_still_answers():
    app = build_graph(make_deps(llm=None), checkpointer=InMemorySaver())
    qa = ask(app, "What is a bond?", "t-nollm", PROFILE)
    assert "Here is what my sources say" in qa.answer
    goal = ask(app, "Save $40,000 in 6 years, I have $5,000 saved", "t-nollm2", PROFILE)
    assert "/month" in goal.answer


def test_unknown_topic_says_it_does_not_know(app):
    # No filler words: the hashing FakeEmbedder (unlike the real model) matches on "what/is/the".
    final = ask(app, "Neapolitan pizza dough recipe", "t-idk", PROFILE)
    assert "don't have reliable information" in final.answer
