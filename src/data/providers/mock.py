"""Deterministic sample data — the last fallback, always clearly marked ``is_mock=True``.

Prices are a seeded random walk around the reference base price, so the same ticker gives the
same series on every run (stable tests, stable demo) without pretending to be live.
"""

from __future__ import annotations

import hashlib
import random
from datetime import UTC, datetime, timedelta

from src.core.schemas import NewsItem, PriceHistory, PricePoint, Quote
from src.data.reference import ticker_info


def _seed(ticker: str) -> int:
    return int(hashlib.md5(ticker.upper().encode()).hexdigest()[:8], 16)


def _walk(ticker: str, days: int) -> list[float]:
    rng = random.Random(_seed(ticker))
    base = ticker_info(ticker).base_price or 50 + _seed(ticker) % 200
    drift = rng.uniform(-0.0003, 0.0008)
    price, out = base * rng.uniform(0.85, 1.0), []
    for _ in range(days):
        price *= 1 + drift + rng.gauss(0, 0.012)
        out.append(round(price, 2))
    return out


class MockMarketProvider:
    name = "mock"

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        trading_days = max(int(days * 252 / 365), 2)
        closes = _walk(ticker, trading_days)
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        points = [
            PricePoint(date=today - timedelta(days=(trading_days - 1 - i) * 365 / 252), close=c)
            for i, c in enumerate(closes)
        ]
        return PriceHistory(ticker=ticker.upper(), points=points, source=self.name, is_mock=True)

    def get_quote(self, ticker: str) -> Quote:
        closes = self.get_history(ticker, 10).closes
        price, prev = closes[-1], closes[-2]
        return Quote(
            ticker=ticker.upper(),
            price=price,
            change=round(price - prev, 2),
            change_pct=(price / prev - 1) * 100,
            as_of=datetime.now(UTC),
            source=self.name,
            is_mock=True,
        )


class MockNewsProvider:
    name = "mock"

    def search(self, query: str, max_results: int) -> list[NewsItem]:
        topic = query.strip() or "markets"
        items = [
            NewsItem(
                title=f"[Sample] What investors are watching: {topic}",
                url="https://www.investor.gov/introduction-investing",
                source="Finnie sample news",
                summary="Sample headline shown because live news is unavailable. "
                "Check a real news source for current events.",
                is_mock=True,
            ),
            NewsItem(
                title="[Sample] Why diversification matters when headlines are loud",
                url="https://www.investor.gov/introduction-investing/investing-basics/save-and-invest/diversify-your-investments",
                source="Finnie sample news",
                summary="Sample item: short-term news moves prices; long-term plans rely on "
                "diversification and costs.",
                is_mock=True,
            ),
        ]
        return items[:max_results]
