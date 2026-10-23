"""Deterministic tools shared by the agents, the Streamlit UI and the MCP server.

Agents call these to get *facts* (numbers computed in Python), then ask the LLM only to explain
them. The MCP server exposes the very same functions, so Claude Desktop and Finnie can't disagree.
"""

from __future__ import annotations

import contextlib
import re

from src.core.config import GoalsConfig
from src.core.schemas import (
    GoalInput,
    GoalPlan,
    GoalScenario,
    Holding,
    Portfolio,
    PortfolioAnalysis,
    PositionMetrics,
)
from src.data.market import MarketData
from src.data.providers.base import ProviderError
from src.data.reference import ticker_info
from src.utils import finance_math as fm

# ---------- parsing ----------

_NOT_TICKERS = {
    "I", "A", "AN", "AND", "OR", "THE", "IRA", "IRAS", "HSA", "FSA", "HRA", "MSA", "ETF", "ETFS",
    "USD", "US", "USA", "IRS", "SEC", "FDIC", "AGI", "MAGI", "RMD", "CEO", "IPO", "EPS", "PE", "AI",
    "IT", "OK", "SO", "MY", "ME", "IS", "AM", "TO", "IN", "ON", "AT", "OF", "DO", "BE", "NO", "IF",
    "VS", "FAQ", "W", "K", "S", "P", "Q", "Y", "TAX", "ROTH", "HDHP", "SEP", "SIMPLE", "GDP", "CPI",
}  # fmt: skip
_TICKER = r"\$?([A-Z]{1,5}(?:\.[A-Z])?)"
_NUM = r"(\d+(?:\.\d+)?)"


def extract_tickers(text: str) -> list[str]:
    """Uppercase symbols like ``AAPL`` or ``$TSLA``; ignores finance acronyms (IRA, ETF, …)."""
    found: list[str] = []
    for match in re.finditer(rf"(?<![A-Za-z]){_TICKER}(?![A-Za-z])", text):
        symbol = match.group(1)
        explicit = text[match.start()] == "$"
        allowed = symbol not in _NOT_TICKERS or explicit
        if allowed and symbol not in found and (explicit or len(symbol) >= 2):
            found.append(symbol)
    return found


def parse_holdings(text: str) -> Portfolio:
    """Holdings from free text: "10 AAPL", "AAPL 10", "10 shares of MSFT", "VTI: 25"."""
    patterns = [
        rf"{_NUM}\s*(?:shares?\s+(?:of\s+)?)?{_TICKER}\b",  # 10 AAPL / 10 shares of AAPL
        rf"{_TICKER}\s*[:=x-]?\s*{_NUM}\s*(?:shares?)?",  # AAPL 10 / AAPL: 10
    ]
    shares: dict[str, float] = {}
    for i, pattern in enumerate(patterns):
        for match in re.finditer(pattern, text):
            num, sym = (match.group(1), match.group(2)) if i == 0 else match.group(2, 1)
            if sym in _NOT_TICKERS or sym in shares:
                continue
            if float(num) > 0:
                shares[sym] = float(num)
    return Portfolio(holdings=[Holding(ticker=t, shares=s) for t, s in shares.items()])


def parse_goal(text: str, risk_profile: str = "moderate") -> GoalInput | None:
    """Regex fallback for "save $50,000 in 5 years, I have $5k, can add $500 a month"."""

    def money(s: str) -> float:
        s = s.replace(",", "").lower().strip()
        mult = 1_000_000 if s.endswith(("m", "million")) else 1_000 if s.endswith("k") else 1
        return float(re.match(r"[\d.]+", s).group()) * mult

    amount = r"\$?\s?(\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million)\b)?)"
    amounts = [money(m) for m in re.findall(amount, text, re.I) if re.search(r"\d", m)]
    years = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?(?:years?|yrs?)", text, re.I)
    if not amounts or not years:
        return None
    monthly = re.search(amount + r"\s*(?:a|per|/|each)\s*month", text, re.I)
    have = re.search(r"(?:have|saved|savings of|already)\D{0,15}" + amount, text, re.I)
    target = max(amounts)
    return GoalInput(
        target_amount=target,
        years=float(years.group(1)),
        current_savings=money(have.group(1)) if have and money(have.group(1)) < target else 0,
        monthly_contribution=money(monthly.group(1)) if monthly else None,
        risk_profile=risk_profile,  # type: ignore[arg-type]
    )


# ---------- portfolio ----------


def analyze_portfolio(portfolio: Portfolio, market: MarketData) -> PortfolioAnalysis:
    """Prices → values → weights, concentration, mix, costs, volatility, educational notes."""
    if not portfolio.holdings:
        raise ValueError("The portfolio has no holdings")
    prices, sources, is_mock, histories = {}, set(), False, {}
    for h in portfolio.holdings:
        quote = market.quote(h.ticker)
        prices[h.ticker], is_mock = quote.price, is_mock or quote.is_mock
        sources.add(quote.source)
        with contextlib.suppress(ProviderError):  # no history → volatility just omitted
            histories[h.ticker] = market.history(h.ticker, 365).closes

    shares = {h.ticker: h.shares for h in portfolio.holdings}
    values = {t: shares[t] * prices[t] for t in shares}
    ws = fm.weights(values)
    infos = {t: ticker_info(t) for t in shares}
    asset_mix = fm.group_weights(ws, {t: i.asset_class for t, i in infos.items()})
    sector_mix = fm.group_weights(ws, {t: i.sector for t, i in infos.items()})
    expense = fm.weighted_expense_ratio(ws, {t: i.expense_ratio for t, i in infos.items()})
    vol = fm.annualized_volatility(fm.portfolio_price_series(histories, shares))
    eff_n, top = fm.effective_n(ws.values()), fm.top_weight(ws)

    positions = [
        PositionMetrics(
            ticker=t,
            shares=shares[t],
            price=prices[t],
            value=values[t],
            weight=ws.get(t, 0.0),
            asset_class=infos[t].asset_class,  # type: ignore[arg-type]
            sector=infos[t].sector,
            expense_ratio=infos[t].expense_ratio,
        )
        for t in sorted(shares, key=lambda t: -values[t])
    ]
    return PortfolioAnalysis(
        total_value=sum(values.values()),
        positions=positions,
        hhi=fm.hhi(ws.values()),
        effective_n=eff_n,
        top_holding_weight=top,
        asset_class_mix=asset_mix,
        sector_mix=sector_mix,
        weighted_expense_ratio=expense,
        annualized_volatility=vol,
        diversification_score=fm.diversification_score(eff_n, top, len(asset_mix)),
        observations=_observations(positions, asset_mix, sector_mix, expense, vol, eff_n),
        data_source=", ".join(sorted(sources)),
        is_mock=is_mock,
    )


def _observations(positions, asset_mix, sector_mix, expense, vol, eff_n) -> list[str]:
    """Educational observations — describe, never instruct (no buy/sell)."""
    notes = []
    top = positions[0]
    if top.weight > 0.25:
        notes.append(
            f"{top.ticker} is {top.weight:.0%} of the portfolio — a single holding above ~20–25% "
            "means its ups and downs drive most of the result."
        )
    big_sector, big_sector_w = next(iter(sector_mix.items()))
    if big_sector_w > 0.4 and big_sector not in {"Broad Market", "Bonds"}:
        notes.append(f"{big_sector_w:.0%} sits in one sector ({big_sector}).")
    if not {"bond", "cash"} & set(asset_mix):
        notes.append(
            "There are no bond or cash holdings; bonds often move differently from stocks."
        )
    if eff_n < 5:
        notes.append(f"It behaves like about {eff_n:.1f} equally weighted holdings.")
    if expense is not None and expense > 0.005:
        notes.append(f"The weighted fund cost is {expense:.2%} a year, which compounds over time.")
    if vol is not None and vol > 0.25:
        notes.append(f"Annualized volatility is about {vol:.0%}, higher than a broad index.")
    return notes or ["Holdings are spread across several positions and asset classes."]


# ---------- goals ----------


def plan_goal(goal: GoalInput, cfg: GoalsConfig) -> GoalPlan:
    """Required monthly saving and projections for the low / expected / high return bands."""
    band = cfg.risk_profiles[goal.risk_profile]
    target = fm.inflate(goal.target_amount, cfg.inflation, goal.years)
    scenarios = []
    for label in ("low", "expected", "high"):
        rate = getattr(band, label)
        projected = (
            fm.future_value(goal.current_savings, goal.monthly_contribution, rate, goal.years)
            if goal.monthly_contribution is not None
            else None
        )
        scenarios.append(
            GoalScenario(
                label=label,  # type: ignore[arg-type]
                annual_return=rate,
                required_monthly=fm.required_monthly(
                    target, goal.current_savings, rate, goal.years
                ),
                projected_value=projected,
                on_track=None if projected is None else projected >= target,
            )
        )
    monthly = goal.monthly_contribution
    if monthly is None:  # chart the path at the "expected" required saving
        monthly = scenarios[1].required_monthly
    yearly = [
        {
            "year": float(y),
            **{
                s.label: fm.future_value(goal.current_savings, monthly, s.annual_return, y)
                for s in scenarios
            },
        }
        for y in range(0, int(goal.years) + 1)
    ]
    return GoalPlan(
        goal=goal,
        inflation=cfg.inflation,
        target_in_future_dollars=target,
        scenarios=scenarios,
        yearly_projection=yearly,
    )
