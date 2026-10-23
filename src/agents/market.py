"""Market agent — quotes and simple, explainable trend facts for tickers (or the broad market).

Facts are computed in Python (``MarketData.snapshot``): price, day change, 52-week range,
20/50-day moving averages, 1- and 6-month returns. The LLM explains them in plain language and
always states the data source and time, including when it's cached or sample data.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.tools import extract_tickers
from src.core.schemas import AgentResult
from src.core.state import AgentInput
from src.data.providers.base import ProviderError
from src.utils import disclaimers

MARKET_PROXIES = ["SPY", "QQQ", "DIA"]  # S&P 500, Nasdaq-100, Dow — for "how's the market?"
MAX_TICKERS = 3

TASK = """Explain these market facts for each ticker: price and today's move, where it sits in its
52-week range, and the trend (price vs its 20- and 50-day moving averages, 1- and 6-month returns).
Explain what a moving average is if the user is a beginner. State the data source and time
("as of …"), and say clearly if data is cached or sample data. No predictions, no buy/sell calls."""


class MarketAgent(BaseAgent):
    name = "market"

    def _run(self, inp: AgentInput) -> AgentResult:
        tickers = list(inp.get("tickers") or extract_tickers(inp.get("question", "")))
        tickers = tickers[:MAX_TICKERS] or MARKET_PROXIES
        facts, data, failed, any_mock = [], [], [], False
        for ticker in tickers:
            try:
                snap, history = self.deps.market.snapshot(ticker)
            except ProviderError:
                failed.append(ticker)
                continue
            any_mock = any_mock or snap.quote.is_mock
            facts.append(snap.model_dump(mode="json"))
            data.append(
                {
                    "snapshot": snap.model_dump(mode="json"),
                    "history": [
                        {"date": p.date.date().isoformat(), "close": p.close}
                        for p in history.points
                    ],
                }
            )
        if not facts:
            return AgentResult(
                agent=self.name,
                answer=f"I couldn't get market data for {', '.join(failed)} right now.",
                confident=False,
            )

        fallback = "\n".join(
            f"- **{f['quote']['ticker']}** ${f['quote']['price']:,.2f} "
            f"({f['quote']['change_pct']:+.2f}% today), trend: {f['trend']}, "
            f"source: {f['quote']['source']}"
            for f in facts
        )
        answer = self.explain(inp, TASK, {"tickers": facts, "unavailable": failed}, fallback)
        if any_mock:
            answer += f"\n\n_{disclaimers.MOCK_DATA}_"
        return AgentResult(agent=self.name, answer=answer, data={"market": data})
