"""Static ticker reference data (data/reference/tickers.yaml): asset class, sector, fund costs, and
company-name aliases ("nvidia" → NVDA) for lookups."""

from __future__ import annotations

import re
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


# Known symbols that are also everyday words; matched only when typed in capitals ("SPY").
_LOWERCASE_WORDS = {"SPY"}
_SYMBOL = re.compile(r"[A-Za-z]{1,5}(?:\.[A-Za-z])?")


@lru_cache(maxsize=1)
def _raw(path: Path = REFERENCE_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load() -> dict[str, TickerInfo]:
    return {t: TickerInfo(ticker=t, **info) for t, info in _raw()["tickers"].items()}


@lru_cache(maxsize=1)
def company_aliases() -> dict[str, str]:
    """Lowercase company name → ticker ("nvidia" → "NVDA")."""
    return {name.lower(): t.upper() for name, t in (_raw().get("aliases") or {}).items()}


def known_tickers() -> set[str]:
    return set(_load())


def ticker_info(ticker: str) -> TickerInfo:
    """Reference info, or a neutral default for tickers we don't list (treated as a stock)."""
    ticker = ticker.upper()
    return _load().get(ticker) or TickerInfo(ticker=ticker, name=ticker)


def resolve_ticker(query: str) -> str | None:
    """A ticker from what someone typed: "nvidia", "NVIDIA Corp", "nvda", "$NVDA" → "NVDA".

    Known names and symbols first; anything else shaped like a symbol is passed through uppercased
    (so unlisted tickers still work); otherwise the first company named in the text ("nvidia stock
    price"). ``None`` when nothing matches.
    """
    q = query.strip().lstrip("$").strip()
    if not q:
        return None
    key = re.sub(r"[\s,.]+(inc|corp|corporation|co|company|ltd|plc)\.?$", "", q.lower()).strip()
    if key in company_aliases():
        return company_aliases()[key]
    for t, info in _load().items():  # full reference names, e.g. "nvidia corp."
        if key == info.name.lower().rstrip("."):
            return t
    if _SYMBOL.fullmatch(q):
        return q.upper()
    in_text = find_company_tickers(q)  # "nvidia stock price", "price of apple"
    return min(in_text)[1] if in_text else None


def find_company_tickers(text: str) -> list[tuple[int, str]]:
    """(position, ticker) for every company name or lowercase known symbol in free text."""
    lower = text.lower()
    found: list[tuple[int, str]] = []
    names = sorted(company_aliases(), key=len, reverse=True)  # "exxon mobil" before "exxon"
    taken: list[range] = []
    for name in names:
        for m in re.finditer(rf"(?<![a-z]){re.escape(name)}(?:'s)?(?![a-z])", lower):
            if not any(m.start() in r for r in taken):
                taken.append(range(m.start(), m.end()))
                found.append((m.start(), company_aliases()[name]))
    for m in re.finditer(r"(?<![a-z$])([a-z]{3,5})(?![a-z])", lower):  # "nvda", "voo"
        symbol = m.group(1).upper()
        overlaps = any(m.start() in r for r in taken)
        if symbol in _load() and symbol not in _LOWERCASE_WORDS and not overlaps:
            found.append((m.start(), symbol))
    return found
