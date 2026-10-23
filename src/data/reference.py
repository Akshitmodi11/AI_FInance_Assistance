"""Static ticker reference data (data/reference/tickers.yaml): asset class, sector, fund costs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.core.config import PROJECT_ROOT

REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "tickers.yaml"


class TickerInfo(BaseModel):
    ticker: str
    name: str
    asset_class: str = "stock"
    sector: str = "Unknown"
    expense_ratio: float | None = None
    base_price: float | None = None


@lru_cache(maxsize=1)
def _load(path: Path = REFERENCE_PATH) -> dict[str, TickerInfo]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["tickers"]
    return {t: TickerInfo(ticker=t, **info) for t, info in raw.items()}


def known_tickers() -> set[str]:
    return set(_load())


def ticker_info(ticker: str) -> TickerInfo:
    """Reference info, or a neutral default for tickers we don't list (treated as a stock)."""
    ticker = ticker.upper()
    return _load().get(ticker) or TickerInfo(ticker=ticker, name=ticker)
