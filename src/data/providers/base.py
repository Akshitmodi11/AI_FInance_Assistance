"""Provider interfaces. Each live source (Alpha Vantage, yfinance, Tavily) and the mock implement
the same small protocol, so :class:`~src.data.market.MarketData` can try them in order."""

from __future__ import annotations

from typing import Protocol

from src.core.schemas import NewsItem, PriceHistory, Quote


class ProviderError(RuntimeError):
    """A provider couldn't answer (no key, rate limit, network, bad symbol, malformed data)."""


class MarketDataProvider(Protocol):
    name: str

    def get_quote(self, ticker: str) -> Quote: ...

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        """Daily closes for roughly the last ``days`` calendar days, oldest first."""
        ...


class NewsProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[NewsItem]: ...
