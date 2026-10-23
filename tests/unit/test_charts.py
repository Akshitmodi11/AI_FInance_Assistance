"""Chart builders are pure functions — test them without running Streamlit."""

import plotly.graph_objects as go

from src.agents.tools import analyze_portfolio, plan_goal
from src.core.config import get_settings
from src.core.schemas import GoalInput, Holding, Portfolio
from src.data.market import MarketData
from src.data.providers.mock import MockMarketProvider
from src.web_app.components import charts


def _analysis() -> dict:
    portfolio = Portfolio(
        holdings=[Holding(ticker="VTI", shares=10), Holding(ticker="BND", shares=20)]
    )
    return analyze_portfolio(portfolio, MarketData([MockMarketProvider()])).model_dump(mode="json")


def test_allocation_donut_variants():
    a = _analysis()
    for by, labels in [("holding", {"VTI", "BND"}), ("asset_class", {"etf", "bond"})]:
        fig = charts.allocation_donut(a, by)
        assert isinstance(fig.data[0], go.Pie) and set(fig.data[0].labels) == labels
    assert sum(charts.allocation_donut(a, "sector").data[0].values) == 1


def test_sector_bar_and_gauge():
    a = _analysis()
    assert len(charts.sector_bar(a).data[0].y) == len(a["sector_mix"])
    assert (
        charts.diversification_gauge(a["diversification_score"]).data[0].value
        == (a["diversification_score"])
    )


def test_price_chart_smas_use_full_history():
    history = [{"date": f"d{i}", "close": float(i)} for i in range(1, 101)]
    fig = charts.price_chart(history, "TEST", show_last=21)
    assert [t.name for t in fig.data] == ["TEST close", "SMA 20", "SMA 50"]
    assert len(fig.data[0].x) == 21
    assert all(v is not None for v in fig.data[2].y)  # 50-day SMA available in a 1-month view
    assert fig.data[2].y[-1] == sum(range(51, 101)) / 50


def test_price_chart_short_history_skips_long_sma():
    fig = charts.price_chart([{"date": "a", "close": 1.0}, {"date": "b", "close": 2.0}], "X")
    assert len(fig.data) == 1


def test_goal_projection_has_three_scenarios_and_target():
    plan = plan_goal(
        GoalInput(target_amount=30_000, years=5, current_savings=2_000, monthly_contribution=300),
        get_settings().goals,
    ).model_dump(mode="json")
    fig = charts.goal_projection(plan)
    assert [t.name for t in fig.data] == ["High return", "Low return", "Expected"]
    assert fig.layout.shapes[0].y0 == plan["target_in_future_dollars"]
