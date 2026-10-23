"""Yahoo Finance via ``yfinance`` — no API key; the second choice after Alpha Vantage."""

from __future__ import annotations

from datetime import UTC

from src.core.schemas import PriceHistory, PricePoint, Quote
from src.data.providers.base import ProviderError


class YFinanceProvider:
    name = "yfinance"

    def get_history(self, ticker: str, days: int) -> PriceHistory:
        import yfinance as yf

        try:
            frame = yf.Ticker(ticker).history(period=f"{max(days, 5)}d", auto_adjust=True)
        except Exception as exc:  # noqa: BLE001 — network / parsing errors from yfinance
            raise ProviderError(f"yfinance failed for {ticker}: {exc}") from exc
        if frame is None or frame.empty or "Close" not in frame:
            raise ProviderError(f"yfinance returned no data for {ticker}")
        points = [
            PricePoint(date=idx.to_pydatetime().astimezone(UTC), close=float(close))
            for idx, close in frame["Close"].dropna().items()
        ]
        if len(points) < 2:
            raise ProviderError(f"yfinance returned too little data for {ticker}")
        return PriceHistory(ticker=ticker.upper(), points=points, source=self.name)

    def get_quote(self, ticker: str) -> Quote:
        closes = self.get_history(ticker, 7).points
        last, prev = closes[-1], closes[-2]
        return Quote(
            ticker=ticker.upper(),
            price=round(last.close, 2),
            change=round(last.close - prev.close, 2),
            change_pct=(last.close / prev.close - 1) * 100,
            as_of=last.date if last.date.tzinfo else last.date.replace(tzinfo=UTC),
            source=self.name,
        )
