"""BaseAgent — the shared shape of all six specialist agents.

Every agent: ``run(AgentInput) -> AgentResult``. The pattern inside is always the same:
1. gather **facts** with deterministic tools (RAG, market data, finance math);
2. ask the LLM only to **explain** those facts for the user's knowledge level;
3. if the LLM is unavailable, return a plain, fact-based answer instead (degraded, not broken).
Any unexpected error becomes an ``AgentResult`` with ``error`` set — one failing agent never
takes down a multi-agent answer.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.schemas import AgentResult, Intent, UserProfile
from src.core.state import AgentInput
from src.workflow.deps import Deps

log = logging.getLogger(__name__)

BASE_RULES = """You are Finnie, a friendly financial-education assistant.
Rules:
- Educate; never tell the user to buy, sell or hold a specific investment.
- Use ONLY the facts and sources given to you. Never invent numbers, dates, limits or prices.
- If the facts don't answer the question, say so plainly.
- Be concise: short paragraphs or bullets, under ~200 words unless asked for more.
- Cover only your own topic; other specialists may answer other parts of the question in their own
  sections, so don't mention what your facts don't include about other topics.
"""

LEVELS = {
    "beginner": "The user is a beginner: use plain words, define any jargon, give a short example.",
    "intermediate": "The user knows the basics: be precise and skip elementary definitions.",
    "advanced": "The user is experienced: be compact and technical where useful.",
}


class BaseAgent(ABC):
    name: Intent

    def __init__(self, deps: Deps):
        self.deps = deps

    def run(self, inp: AgentInput) -> AgentResult:
        try:
            return self._run(inp)
        except Exception as exc:  # noqa: BLE001 — isolate failures per agent
            log.exception("%s agent failed", self.name)
            return AgentResult(
                agent=self.name,
                answer=f"Sorry — the {self.name.replace('_', ' ')} part of this answer failed "
                f"({type(exc).__name__}). Please try again.",
                confident=False,
                error=str(exc),
            )

    @abstractmethod
    def _run(self, inp: AgentInput) -> AgentResult: ...

    # ----- helpers -----

    @staticmethod
    def profile(inp: AgentInput) -> UserProfile:
        return UserProfile.model_validate(inp.get("user_profile") or {})

    def explain(self, inp: AgentInput, task: str, facts: dict | str, fallback: str) -> str:
        """Ask the answer LLM to explain ``facts``; return ``fallback`` if no LLM is available."""
        level = LEVELS[self.profile(inp).knowledge_level]
        facts_text = facts if isinstance(facts, str) else json.dumps(facts, indent=1, default=str)
        system = f"{BASE_RULES}\n{level}\n\n{task}"
        user = f"Question: {inp.get('question', '')}\n\nFacts:\n{facts_text}"
        if note := inp.get("guardrail_note"):
            user += f"\n\nImportant: start by saying, in your own words: {note}"
        try:
            reply = self.deps.llm("answer").invoke([SystemMessage(system), HumanMessage(user)])
        except Exception as exc:  # noqa: BLE001 — no key, timeout, provider outage
            log.warning("%s: LLM unavailable, returning facts only: %s", self.name, exc)
            return fallback
        content = reply.content
        if not isinstance(content, str):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content.strip() or fallback
