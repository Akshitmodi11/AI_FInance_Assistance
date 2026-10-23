"""Market data service: provider fallback chain + 30-minute disk cache + computed snapshot.

    request ─▶ cache hit? ──yes──▶ return (is_cached=True)
                 │no
                 ▼
     Alpha Vantage ─fail─▶ yfinance ─fail─▶ mock (is_mock=True)
                 │ok
                 ▼
            cache (live results only — mock is never cached, so live data is retried)

The cache lives on disk (diskcache), so restarts don't burn Alpha Vantage's 25 requests/day.
Every result says where it came from (``source``, ``as_of``, ``is_cached``, ``is_mock``) and the
UI shows that label — users always know how fresh the numbers are.
"""

from __future__ import annotations

import logging
from collections import Counter

import diskcache

from src.core.config import Secrets, Settings
from src.core.schemas import MarketSnapshot, PriceHistory, Quote
from src.data.providers.base import MarketDataProvider, ProviderError
from src.data.providers.mock import MockMarketProvider
from src.utils import finance_math as fm

log = logging.getLogger(__name__)


class MarketData:
    def __init__(
        self,
        providers: list[MarketDataProvider],
        cache: diskcache.Cache | None = None,
        ttl_s: int = 1800,
    ):
        if not providers:
            providers = [MockMarketProvider()]
        self.providers, self.cache, self.ttl_s = providers, cache, ttl_s
        self.stats: Counter[str] = Counter()  # cache_hits, fallbacks, provider_errors, ...

    @classmethod
    def from_settings(cls, settings: Settings, secrets: Secrets | None = None) -> MarketData:
        secrets = secrets or Secrets()
        cfg = settings.market_data
        cache = diskcache.Cache(str(settings.resolve(cfg.cache_dir)))
        providers: list[MarketDataProvider] = []
        for name in cfg.providers:
            if name == "alpha_vantage":
                key = secrets.alpha_vantage_api_key
                if key is not None and key.get_secret_value():
                    from src.data.providers.alpha_vantage import AlphaVantageProvider
                    from src.data.rate_limit import RateLimiter

                    av = cfg.alpha_vantage
                    limiter = RateLimiter(
                        cache, "alpha_vantage", av.requests_per_minute, av.requests_per_day
                    )
                    providers.append(
                        AlphaVantageProvider(
                            key.get_secret_value(), limiter, cfg.request_timeout_s, cfg.max_retries
                        )
                    )
            elif name == "yfinance":
                from src.data.providers.yfinance_provider import YFinanceProvider

                providers.append(YFinanceProvider())
            elif name == "mock":
                providers.append(MockMarketProvider())
        return cls(providers, cache, cfg.cache_ttl_s)

    # ----- core -----

    def _fetch(self, kind: str, ticker: str, days: int = 0):
        ticker = ticker.strip().upper()
        key = f"{kind}:{ticker}:{days}"
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            self.stats["cache_hits"] += 1
            model = Quote if kind == "quote" else PriceHistory
            return model.model_validate({**hit, "is_cached": True})

        errors = []
        for i, provider in enumerate(self.providers):
            try:
                result = (
                    provider.get_quote(ticker)
                    if kind == "quote"
                    else provider.get_history(ticker, days)
                )
            except ProviderError as exc:
                self.stats["provider_errors"] += 1
                errors.append(f"{provider.name}: {exc}")
                log.info("market %s %s via %s failed: %s", kind, ticker, provider.name, exc)
                continue
            if i > 0:
                self.stats["fallbacks"] += 1
            if self.cache is not None and not result.is_mock:
                self.cache.set(key, result.model_dump(mode="json"), expire=self.ttl_s)
            self.stats[f"source_{provider.name}"] += 1
            return result
        raise ProviderError(f"No provider could return {kind} for {ticker}: {'; '.join(errors)}")

    def quote(self, ticker: str) -> Quote:
        return self._fetch("quote", ticker)

    def history(self, ticker: str, days: int = 365) -> PriceHistory:
        return self._fetch("history", ticker, days)

    def snapshot(self, ticker: str) -> tuple[MarketSnapshot, PriceHistory]:
        """Quote + 1-year history → trend facts computed in Python for the Market agent."""
        quote, history = self.quote(ticker), self.history(ticker, 365)
        closes = history.closes
        last_year = closes[-252:]
        sma_20, sma_50 = fm.sma(closes, 20), fm.sma(closes, 50)
        snap = MarketSnapshot(
            quote=quote,
            high_52w=max(last_year) if last_year else None,
            low_52w=min(last_year) if last_year else None,
            sma_20=sma_20,
            sma_50=sma_50,
            trend=fm.trend_label(quote.price, sma_20, sma_50),
            return_1m=fm.pct_change(closes, 21),
            return_6m=fm.pct_change(closes, 126),
        )
        return snap, history
