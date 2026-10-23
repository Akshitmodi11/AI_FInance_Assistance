import math

import pytest

from src.utils import finance_math as fm

# ---------- portfolio ----------


def test_weights_normal_and_edge_cases():
    assert fm.weights({"A": 300, "B": 100}) == {"A": 0.75, "B": 0.25}
    assert fm.weights({}) == {}
    assert fm.weights({"A": 0, "B": 0}) == {}
    assert fm.weights({"A": 100, "B": 0}) == {"A": 1.0}  # zero-value positions dropped


def test_hhi_and_effective_n():
    assert fm.hhi([1.0]) == 1.0
    assert fm.hhi([0.25] * 4) == pytest.approx(0.25)
    assert fm.effective_n([0.25] * 4) == pytest.approx(4)
    assert fm.effective_n([0.5, 0.3, 0.2]) == pytest.approx(1 / 0.38)
    assert fm.effective_n([]) == 0.0


def test_top_weight_and_groups():
    ws = {"AAPL": 0.5, "MSFT": 0.3, "BND": 0.2}
    assert fm.top_weight(ws) == 0.5
    assert fm.top_weight({}) == 0.0
    groups = fm.group_weights(ws, {"AAPL": "stock", "MSFT": "stock", "BND": "bond"})
    assert groups == {"stock": pytest.approx(0.8), "bond": pytest.approx(0.2)}
    assert list(groups) == ["stock", "bond"]  # sorted by weight
    assert fm.group_weights({"X": 1.0}, {}) == {"Unknown": 1.0}


def test_weighted_expense_ratio():
    ws = {"VTI": 0.5, "AAPL": 0.5}
    assert fm.weighted_expense_ratio(ws, {"VTI": 0.0003, "AAPL": None}) == pytest.approx(0.00015)
    assert fm.weighted_expense_ratio(ws, {"VTI": None, "AAPL": None}) is None


def test_volatility():
    assert fm.annualized_volatility([100, 100, 100, 100]) == 0.0
    assert fm.annualized_volatility([100, 101]) is None  # too short
    prices = [100, 101, 99, 102, 100]
    rets = fm.daily_returns(prices)
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
    assert fm.annualized_volatility(prices) == pytest.approx(sd * math.sqrt(252))


def test_portfolio_price_series_aligns_to_shortest():
    series = fm.portfolio_price_series({"A": [1, 2, 3], "B": [10, 20]}, {"A": 2, "B": 1})
    assert series == [2 * 2 + 10, 2 * 3 + 20]
    assert fm.portfolio_price_series({}, {"A": 1}) == []


def test_diversification_score_bounds():
    assert fm.diversification_score(1, 1.0, 1) < 25  # single stock
    assert fm.diversification_score(25, 0.05, 3) == 100
    assert 0 <= fm.diversification_score(5, 0.4, 2) <= 100


# ---------- market ----------


def test_sma_pct_change_trend():
    prices = list(range(1, 61))  # steadily rising
    assert fm.sma(prices, 20) == pytest.approx(sum(range(41, 61)) / 20)
    assert fm.sma(prices, 100) is None
    assert fm.pct_change(prices, 10) == pytest.approx(60 / 50 - 1)
    assert fm.pct_change(prices, 100) is None
    assert fm.trend_label(60, fm.sma(prices, 20), fm.sma(prices, 50)) == "uptrend"
    falling = prices[::-1]
    assert fm.trend_label(1, fm.sma(falling, 20), fm.sma(falling, 50)) == "downtrend"
    assert fm.trend_label(10, None, 5) == "unknown"
    assert fm.trend_label(10, 12, 9) == "sideways"


# ---------- goals ----------


def test_future_value_zero_and_positive_rate():
    assert fm.future_value(1000, 100, 0.0, 1) == 1000 + 1200
    # 10,000 at 6% compounded monthly for 10 years
    assert fm.future_value(10_000, 0, 0.06, 10) == pytest.approx(10_000 * (1.005**120))


def test_required_monthly_roundtrip():
    pmt = fm.required_monthly(100_000, 5_000, 0.06, 10)
    assert fm.future_value(5_000, pmt, 0.06, 10) == pytest.approx(100_000)


def test_required_monthly_edge_cases():
    assert fm.required_monthly(1_000, 5_000, 0.05, 5) == 0.0  # already there
    assert fm.required_monthly(1_200, 0, 0.0, 1) == pytest.approx(100)
    assert fm.required_monthly(1_000, 200, 0.05, 0) == 800  # no time left


def test_inflate():
    assert fm.inflate(100, 0.0, 10) == 100
    assert fm.inflate(100, 0.025, 2) == pytest.approx(105.0625)
