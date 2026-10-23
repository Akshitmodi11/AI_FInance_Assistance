"""The LangGraph workflow.

    START → guardrail ──blocked──▶ refuse ─────────────────────────────▶ END
                 │ok
                 ▼
              router ──Send──▶ finance_qa ─┐
                     ──Send──▶ tax        ─┤
                     ──Send──▶ portfolio  ─┼──▶ synthesizer ──▶ END
                     ──Send──▶ goal       ─┤   (runs once, after all
                     ──Send──▶ market     ─┤    parallel agents finish)
                     ──Send──▶ news       ─┘

* **Fan-out with ``Send``**: the router can pick 1–3 intents; each becomes a parallel branch that
  receives only an ``AgentInput`` (question, profile, portfolio, tickers, guardrail note).
* **Shared state** (``FinnieState``) is how agents "communicate": each writes its result under
  ``agent_outputs[intent]``; the synthesizer reads them all.
* **Memory**: compiled with a checkpointer; each chat uses a ``thread_id``, so the conversation
  (``messages``) persists and the router can resolve follow-up questions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from src.agents.base import BaseAgent
from src.agents.finance_qa import FinanceQAAgent
from src.agents.goal_planning import GoalPlanningAgent
from src.agents.market import MarketAgent
from src.agents.news import NewsAgent
from src.agents.portfolio import PortfolioAgent
from src.agents.tax import TaxAgent
from src.core.schemas import FinalResponse
from src.core.state import AgentInput, FinnieState
from src.utils import disclaimers
from src.workflow import guardrails
from src.workflow.deps import Deps
from src.workflow.router import _text, route
from src.workflow.synthesizer import synthesize

AGENT_CLASSES: list[type[BaseAgent]] = [
    FinanceQAAgent,
    TaxAgent,
    PortfolioAgent,
    GoalPlanningAgent,
    MarketAgent,
    NewsAgent,
]


def build_graph(deps: Deps, checkpointer: BaseCheckpointSaver | None = None):
    agents = {cls.name: cls(deps) for cls in AGENT_CLASSES}

    def guardrail_node(state: FinnieState) -> dict:
        question = _text(state["messages"][-1]) if state.get("messages") else ""
        result = guardrails.check(question)
        # Reset per-turn fields — the checkpointer would otherwise carry them over.
        return {"guardrail": result.as_dict(), "agent_outputs": None, "final": None}

    def after_guardrail(state: FinnieState) -> str:
        return "refuse" if state["guardrail"]["blocked"] else "router"

    def refuse_node(state: FinnieState) -> dict:
        final = FinalResponse(answer=guardrails.REFUSAL, agents=[], disclaimer=disclaimers.GENERAL)
        return {"final": final.model_dump(), "messages": [AIMessage(guardrails.REFUSAL)]}

    def router_node(state: FinnieState) -> dict:
        try:
            llm = deps.llm("router")
        except Exception:  # noqa: BLE001 — no key → keyword routing
            llm = None
        decision = route(state["messages"], llm, has_portfolio=bool(state.get("portfolio")))
        return {
            "intents": list(decision.intents),
            "tickers": decision.tickers,
            "question": decision.standalone_question,
        }

    def fan_out(state: FinnieState) -> list[Send]:
        payload: AgentInput = {
            "question": state["question"],
            "user_profile": state.get("user_profile") or {},
            "portfolio": state.get("portfolio"),
            "tickers": state.get("tickers") or [],
            "guardrail_note": state["guardrail"].get("note"),
        }
        return [Send(intent, payload) for intent in state["intents"]]

    def make_agent_node(agent: BaseAgent):
        def node(inp: AgentInput) -> dict:
            return {"agent_outputs": {agent.name: agent.run(inp).model_dump(mode="json")}}

        return node

    def synthesizer_node(state: FinnieState) -> dict:
        final = synthesize(
            state.get("agent_outputs") or {}, state["intents"], state["guardrail"].get("note")
        )
        return {"final": final.model_dump(mode="json"), "messages": [AIMessage(final.answer)]}

    graph = StateGraph(FinnieState)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("router", router_node)
    graph.add_node("synthesizer", synthesizer_node)
    for name, agent in agents.items():
        graph.add_node(name, make_agent_node(agent))
        graph.add_edge(name, "synthesizer")

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", after_guardrail, ["refuse", "router"])
    graph.add_conditional_edges("router", fan_out, list(agents))
    graph.add_edge("refuse", END)
    graph.add_edge("synthesizer", END)
    return graph.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: Path):
    """Persistent conversation memory (survives app restarts)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def ask(
    app,
    question: str,
    thread_id: str,
    user_profile: dict | None = None,
    portfolio: dict | None = None,
) -> FinalResponse:
    """One chat turn. ``app`` is a compiled graph (with a checkpointer for memory)."""
    update: dict = {"messages": [HumanMessage(question)]}
    if user_profile is not None:
        update["user_profile"] = user_profile
    if portfolio is not None:
        update["portfolio"] = portfolio
    state = app.invoke(update, config={"configurable": {"thread_id": thread_id}})
    return FinalResponse.model_validate(state["final"])
