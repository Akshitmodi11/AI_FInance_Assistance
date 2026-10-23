"""LangGraph state: the shared "whiteboard" every node reads from and writes to.

Agents don't call each other. Each node returns a *partial update*; LangGraph merges it into the
state using the reducer declared on each key:

* ``messages`` uses ``add_messages`` → the conversation is appended, never overwritten. With a
  checkpointer + ``thread_id`` it persists across turns, which is what makes follow-ups work.
* ``agent_outputs`` uses :func:`merge_outputs` → parallel agents each add their own key in the
  same step without clobbering each other; ``None`` resets it at the start of every turn.
* every other key is last-write-wins.

Values are plain dicts (``model_dump()``), so the SQLite checkpointer can serialize them.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_outputs(left: dict | None, right: dict | None) -> dict:
    if right is None:  # explicit reset at the start of a turn
        return {}
    return {**(left or {}), **right}


class FinnieState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_profile: dict  # UserProfile.model_dump()
    portfolio: dict | None  # Portfolio.model_dump() from the UI, if the user entered one
    question: str  # latest question, rewritten to stand alone (router)
    intents: list[str]
    tickers: list[str]
    guardrail: dict  # GuardrailResult.as_dict()
    agent_outputs: Annotated[dict[str, dict], merge_outputs]  # intent → AgentResult dict
    final: dict | None  # FinalResponse.model_dump()


class AgentInput(TypedDict, total=False):
    """What the router sends to each agent (via ``Send``) — only what the agent needs."""

    question: str
    user_profile: dict
    portfolio: dict | None
    tickers: list[str]
    guardrail_note: str | None
