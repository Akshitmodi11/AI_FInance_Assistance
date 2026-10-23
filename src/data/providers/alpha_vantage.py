"""Alpha Vantage REST API (free key: 25 requests/day, 5/minute) — first choice when a key is set."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.schemas import PriceHistory, PricePoint, Quote
from src.data.providers.base import ProviderError
from src.data.rate_limit import RateLimiter

BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider:
    name = "alpha_vantage"

    def __init__(
        self,
        api_key: str,
        limiter: RateLimiter,
        timeout_s: float = 10,
        retries: int = 2,
        client: httpx.Client | None = None,
    ):
        if not api_key:
            raise ProviderError("ALPHA_VANTAGE_API_KEY is not set")
        self.api_key, self.limiter = api_key, limiter
        self.client = client or httpx.Client(timeout=timeout_s)
        self._get = retry(
            retry=retry_if_exception_type(httpx.TransportError),  # network blips, not bad data
            stop=stop_after_attempt(retries + 1),
            wait=wait_exponential(multiplier=0.5, max=4),
            reraise=True,
        )(self._get_once)

    def _get_once(self, params: dict) -> dict:
        response = self.client.get(BASE_URL, params={**params, "apikey": self.api_key})
        response.raise_for_status()
        return response.json()

    def _call(self, params: dict) -> dict:
        if not self.limiter.try_acquire():
            raise ProviderError("Alpha Vantage request budget used up (5/min, 25/day)")
        try:
            data = self._get(params)
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Alpha Vantage request failed: {exc}") from exc
        for key in ("Note", "Information", "Error Message"):  # rate limit / bad symbol / key
            if key in data:
                raise ProviderError(f"Alpha Vantage: {data[key]}")
        return data

    def get_quote(self, ticker: str) -> Quote:
        data = self._call({"function": "GLOBAL_QUOTE", "symbol": ticker}).get("Global Quote") or {}
        try:
            price = float(data["05. price"])
            change = float(data["09. change"])
            change_pct = float(str(data["10. change percent"]).rstrip("%"))
            day = datetime.fromisoformat(data["07. latest trading day"]).replace(tzinfo=UTC)
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"Alpha Vantage: malformed quote for {ticker}") from exc
        return Quote(
            ticker=ticker.upper(),
            price=price,
            change=change,
            change_pct=change_pct,
            as_of=day,
            source=self.name,
        )

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        # "compact" = last 100 trading days (~140 calendar days); "full" = 20+ years.
        size = "compact" if days <= 140 else "full"
        data = self._call(
            {"function": "TIME_SERIES_DAILY", "symbol": ticker, "outputsize": size}
        ).get("Time Series (Daily)")
        if not data:
            raise ProviderError(f"Alpha Vantage: no daily series for {ticker}")
        try:
            rows = sorted(
                (datetime.fromisoformat(d).replace(tzinfo=UTC), float(v["4. close"]))
                for d, v in data.items()
            )
        except (KeyError, ValueError) as exc:
            raise ProviderError(f"Alpha Vantage: malformed series for {ticker}") from exc
        cutoff = datetime.now(UTC).timestamp() - days * 86_400
        points = [PricePoint(date=d, close=c) for d, c in rows if d.timestamp() >= cutoff]
        if len(points) < 2:
            points = [PricePoint(date=d, close=c) for d, c in rows[-max(days, 2) :]]
        return PriceHistory(ticker=ticker.upper(), points=points, source=self.name)
