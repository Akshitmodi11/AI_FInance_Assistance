"""Router: which agent(s) should answer, plus the tickers and a standalone version of the question.

Primary path — an LLM with **structured output** (``RouterDecision``), given recent conversation
turns so it can resolve follow-ups ("what about for a Roth?" → "What is the contribution limit for
a Roth IRA?"). It may pick up to 3 intents; the graph runs them in parallel.

Fallback — keyword rules, used when the LLM is unavailable, errors or returns something invalid.
Finnie never fails to answer just because routing had a hiccup.
"""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from src.agents.tools import extract_tickers
from src.core.schemas import INTENTS, RouterDecision

log = logging.getLogger(__name__)

# Checked in this order; first match wins priority in the list of intents.
KEYWORDS: dict[str, re.Pattern] = {
    "portfolio": re.compile(
        r"\bmy (portfolio|holdings|investments|allocation)\b|\b(analy[sz]e|review|check)\b.*\b"
        r"(portfolio|holdings)\b|\bi (own|have|hold) \d+|\bdiversif\w* enough\b",
        re.I,
    ),
    "goal": re.compile(
        r"\b(save|saving) (for|up)\b|\bgoal\b|\bretire (at|by|in)\b|\bhow much (do|should|must|"
        r"would) i (need to )?(save|invest)\b|\bin \d+ years\b|\b(down ?payment|emergency fund|"
        r"college fund)\b|\bper month to reach\b",
        re.I,
    ),
    "news": re.compile(r"\bnews\b|\bheadlines?\b|\bwhat'?s happening\b|\blatest on\b", re.I),
    "market": re.compile(
        r"\b(stock )?price of\b|\bquote\b|\btrend(ing)?\b|\bstock price\b|"
        r"\b52[- ]week\b|\bmarkets? (today|doing|performance)\b|\bperform(ing|ance)\b|"
        r"\bmoving average\b|\bS&P 500 (today|doing)\b",
        re.I,
    ),
    "tax": re.compile(
        r"\b(roth|ira|iras|401\(?k\)?|403\(?b\)?|hsa|fsa|tax(es|able)?|deduct\w*|irs|rmd|rollover|"
        r"capital gains?|1099|w-?2|contribution limits?|form \d{3,5})\b",
        re.I,
    ),
}

SYSTEM = """You route questions for Finnie, a financial-education assistant. Pick 1-3 intents:
- finance_qa: general financial concepts and definitions (stocks, bonds, funds, diversification…)
- tax: retirement accounts and tax rules (IRA, Roth, 401(k), HSA, deductions, contribution limits)
- portfolio: analyzing the user's own holdings / allocation / diversification
- goal: saving for a target amount by a date, retirement or savings plans, "how much should I save"
- market: current prices, quotes, trends or performance of specific tickers or the market
- news: recent news or headlines about companies or markets
Also return any stock/ETF tickers mentioned (uppercase), and rewrite the latest user message as a
standalone question, using earlier turns to resolve references like "it", "that" or "what about…".
"""


def keyword_route(question: str, has_portfolio: bool = False) -> RouterDecision:
    intents = [intent for intent, pattern in KEYWORDS.items() if pattern.search(question)]
    tickers = extract_tickers(question)
    if has_portfolio and "portfolio" not in intents and re.search(r"\bmy\b", question, re.I):
        intents.insert(0, "portfolio")
    if not intents:
        intents = ["market"] if tickers else ["finance_qa"]
    return RouterDecision(intents=intents[:3], tickers=tickers, standalone_question=question)


def route(
    messages: list[AnyMessage], llm: BaseChatModel | None, has_portfolio: bool = False
) -> RouterDecision:
    question = _text(messages[-1]) if messages else ""
    if llm is not None:
        try:
            history = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Finnie'}: {_text(m)[:400]}"
                for m in messages[-7:-1]
            )
            prompt = (
                f"Conversation so far:\n{history or '(none)'}\n\nLatest user message: {question}"
            )
            if has_portfolio:
                prompt += "\n(The user has entered a portfolio in the app.)"
            decision = llm.with_structured_output(RouterDecision).invoke(
                [SystemMessage(SYSTEM), HumanMessage(prompt)]
            )
            if isinstance(decision, RouterDecision) and all(i in INTENTS for i in decision.intents):
                found = extract_tickers(decision.standalone_question) + extract_tickers(question)
                tickers = list(dict.fromkeys([t.upper() for t in decision.tickers] + found))
                return decision.model_copy(update={"tickers": tickers})
        except Exception as exc:  # noqa: BLE001 — any LLM/parsing failure → keyword fallback
            log.warning("LLM router failed, using keyword fallback: %s", exc)
    return keyword_route(question, has_portfolio)


def _text(message: AnyMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
