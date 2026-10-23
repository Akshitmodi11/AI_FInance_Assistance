import pytest

from src.agents.tools import (
    analyze_portfolio,
    extract_tickers,
    parse_goal,
    parse_holdings,
    plan_goal,
)
from src.core.config import GoalsConfig, ReturnBand
from src.core.schemas import GoalInput, Holding, Portfolio
from src.data.market import MarketData
from src.data.providers.mock import MockMarketProvider

GOALS = GoalsConfig(
    risk_profiles={
        "conservative": ReturnBand(low=0.02, expected=0.04, high=0.06),
        "moderate": ReturnBand(low=0.03, expected=0.06, high=0.09),
        "aggressive": ReturnBand(low=0.03, expected=0.08, high=0.12),
    },
    inflation=0.025,
)


@pytest.fixture
def market() -> MarketData:
    return MarketData([MockMarketProvider()])


def test_parse_holdings_formats():
    p = parse_holdings("I have 10 AAPL, MSFT 5 and 20 shares of VTI plus BND: 30")
    assert {h.ticker: h.shares for h in p.holdings} == {
        "AAPL": 10,
        "MSFT": 5,
        "VTI": 20,
        "BND": 30,
    }
    assert parse_holdings("My IRA has 5000 dollars").holdings == []


def test_extract_tickers_ignores_acronyms():
    assert extract_tickers("How is NVDA doing vs $TSLA? Is my IRA in an ETF ok?") == [
        "NVDA",
        "TSLA",
    ]
    assert extract_tickers("what is a roth ira") == []


def test_parse_goal():
    goal = parse_goal(
        "I want to save $50,000 for a house in 5 years. I have $5k saved and can add $500 a month"
    )
    assert goal.target_amount == 50_000
    assert goal.years == 5
    assert goal.current_savings == 5_000
    assert goal.monthly_contribution == 500
    assert parse_goal("$1.2 million in 30 years").target_amount == 1_200_000
    assert parse_goal("$900 monthly for 3 years").target_amount == 900  # "m" isn't "million"
    assert parse_goal("save money someday") is None


def test_analyze_portfolio(market):
    p = Portfolio(holdings=[Holding(ticker="aapl", shares=10), Holding(ticker="BND", shares=50)])
    a = analyze_portfolio(p, market)
    assert a.total_value == pytest.approx(sum(pos.value for pos in a.positions))
    assert sum(pos.weight for pos in a.positions) == pytest.approx(1)
    assert set(a.asset_class_mix) == {"stock", "bond"}
    assert a.is_mock and a.data_source == "mock"
    assert 0 <= a.diversification_score <= 100
    assert a.annualized_volatility is not None
    assert not any("buy" in o.lower() or "sell" in o.lower() for o in a.observations)


def test_single_stock_is_flagged_concentrated(market):
    a = analyze_portfolio(Portfolio(holdings=[Holding(ticker="TSLA", shares=3)]), market)
    assert a.top_holding_weight == 1.0
    assert any("TSLA is 100%" in o for o in a.observations)
    assert any("no bond" in o for o in a.observations)


def test_empty_portfolio_rejected(market):
    with pytest.raises(ValueError):
        analyze_portfolio(Portfolio(), market)


def test_plan_goal_scenarios_order_and_projection():
    plan = plan_goal(
        GoalInput(target_amount=50_000, years=5, current_savings=5_000, monthly_contribution=500),
        GOALS,
    )
    low, expected, high = plan.scenarios
    assert low.required_monthly > expected.required_monthly > high.required_monthly
    assert plan.target_in_future_dollars == pytest.approx(50_000 * 1.025**5)
    assert all(s.projected_value is not None for s in plan.scenarios)
    assert len(plan.yearly_projection) == 6 and plan.yearly_projection[0]["expected"] == 5_000


def test_plan_goal_without_monthly_charts_required_path():
    plan = plan_goal(GoalInput(target_amount=10_000, years=2), GOALS)
    assert all(s.projected_value is None and s.on_track is None for s in plan.scenarios)
    assert plan.yearly_projection[-1]["expected"] == pytest.approx(
        plan.target_in_future_dollars, rel=1e-6
    )
