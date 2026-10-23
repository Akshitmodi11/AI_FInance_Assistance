"""Deterministic finance calculations. Pure functions, no I/O, exhaustively unit-tested.

Rule of the project: **the LLM never does arithmetic.** Agents compute every number here and
the LLM only explains numbers it is given — so answers can't contain made-up figures.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

TRADING_DAYS = 252


# ---------- Portfolio ----------


def weights(values: Mapping[str, float]) -> dict[str, float]:
    """Position value → share of the total. Empty or all-zero portfolio → {}."""
    total = sum(v for v in values.values() if v > 0)
    if total <= 0:
        return {}
    return {k: v / total for k, v in values.items() if v > 0}


def hhi(ws: Iterable[float]) -> float:
    """Herfindahl–Hirschman index Σw². 1/n for n equal holdings, 1.0 for a single holding."""
    return sum(w * w for w in ws)


def effective_n(ws: Iterable[float]) -> float:
    """1 / HHI — "this portfolio behaves like N equally weighted holdings"."""
    h = hhi(ws)
    return 1 / h if h > 0 else 0.0


def top_weight(ws: Mapping[str, float]) -> float:
    return max(ws.values(), default=0.0)


def group_weights(ws: Mapping[str, float], group_of: Mapping[str, str]) -> dict[str, float]:
    """Sum weights by a label (asset class, sector). Unknown labels → "Unknown"."""
    out: dict[str, float] = defaultdict(float)
    for key, w in ws.items():
        out[group_of.get(key, "Unknown")] += w
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def weighted_expense_ratio(
    ws: Mapping[str, float], expense_ratios: Mapping[str, float | None]
) -> float | None:
    """Σ wᵢ·ERᵢ over holdings with a known expense ratio (individual stocks count as 0).
    None if no holding is a fund."""
    known = {k: er for k, er in expense_ratios.items() if er is not None}
    if not known:
        return None
    return sum(w * known.get(k, 0.0) for k, w in ws.items())


def daily_returns(prices: Sequence[float]) -> list[float]:
    return [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices)) if prices[i - 1] > 0]


def annualized_volatility(prices: Sequence[float]) -> float | None:
    """Sample std-dev of daily returns × √252. None with fewer than 3 prices."""
    rets = daily_returns(prices)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(TRADING_DAYS)


def portfolio_price_series(
    histories: Mapping[str, Sequence[float]], shares: Mapping[str, float]
) -> list[float]:
    """Daily portfolio value from aligned close series (truncated to the shortest)."""
    series = [histories[t] for t in shares if histories.get(t)]
    if not series:
        return []
    n = min(len(s) for s in series)
    tickers = [t for t in shares if histories.get(t)]
    return [
        sum(shares[t] * histories[t][len(histories[t]) - n + i] for t in tickers) for i in range(n)
    ]


def diversification_score(eff_n: float, top: float, n_asset_classes: int) -> int:
    """0–100 heuristic for the UI gauge (educational, not a rating):
    breadth (effective N, saturating at 20) 50 pts · concentration (top weight) 30 pts ·
    asset-class variety (up to 3) 20 pts."""
    breadth = min(eff_n / 20, 1.0) * 50
    concentration = max(0.0, 1 - max(top - 0.05, 0) / 0.95) * 30
    variety = min(n_asset_classes / 3, 1.0) * 20
    return round(breadth + concentration + variety)


# ---------- Market trend ----------


def sma(prices: Sequence[float], n: int) -> float | None:
    if n <= 0 or len(prices) < n:
        return None
    return sum(prices[-n:]) / n


def pct_change(prices: Sequence[float], lookback: int) -> float | None:
    if len(prices) <= lookback or prices[-1 - lookback] <= 0:
        return None
    return prices[-1] / prices[-1 - lookback] - 1


def trend_label(price: float, sma_20: float | None, sma_50: float | None) -> str:
    """Simple, explainable rule: price above both averages and SMA20 > SMA50 → uptrend."""
    if sma_20 is None or sma_50 is None:
        return "unknown"
    if price > sma_20 > sma_50:
        return "uptrend"
    if price < sma_20 < sma_50:
        return "downtrend"
    return "sideways"


# ---------- Goals ----------


def future_value(present: float, monthly: float, annual_rate: float, years: float) -> float:
    """Value after ``years`` with monthly compounding and end-of-month contributions."""
    months = round(years * 12)
    r = annual_rate / 12
    if r == 0:
        return present + monthly * months
    growth = (1 + r) ** months
    return present * growth + monthly * (growth - 1) / r


def required_monthly(target: float, present: float, annual_rate: float, years: float) -> float:
    """Monthly saving needed to reach ``target`` (PMT). 0 if savings already get there."""
    months = round(years * 12)
    if months <= 0:
        return max(target - present, 0.0)
    r = annual_rate / 12
    if r == 0:
        return max((target - present) / months, 0.0)
    growth = (1 + r) ** months
    needed = (target - present * growth) * r / (growth - 1)
    return max(needed, 0.0)


def inflate(amount: float, inflation: float, years: float) -> float:
    """Today's dollars → future dollars (what the goal will actually cost)."""
    return amount * (1 + inflation) ** years
