"""Portfolio tab — enter or upload holdings, see metrics and charts, ask Finnie to explain."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.tools import analyze_portfolio
from src.core.config import PROJECT_ROOT
from src.core.schemas import Holding, Portfolio
from src.web_app.components import charts
from src.web_app.components.ui import get_deps, source_badge

SAMPLES_DIR = PROJECT_ROOT / "data" / "sample_portfolios"


def _samples() -> dict[str, Path]:
    return {p.stem.replace("_", " ").title(): p for p in sorted(SAMPLES_DIR.glob("*.csv"))}


def _to_portfolio(frame: pd.DataFrame) -> Portfolio:
    holdings = []
    for row in frame.dropna(subset=["ticker", "shares"]).itertuples():
        if str(row.ticker).strip() and float(row.shares) > 0:
            holdings.append(Holding(ticker=str(row.ticker), shares=float(row.shares)))
    return Portfolio(holdings=holdings)


def render() -> None:
    st.subheader("📊 Portfolio analysis")
    st.caption(
        "Educational analysis of diversification, concentration, costs and risk — not advice."
    )

    samples = _samples()
    left, right = st.columns([2, 1])
    with right:
        choice = st.selectbox("Start from a sample", ["—", *samples])
        upload = st.file_uploader("…or upload a CSV (columns: ticker, shares)", type="csv")
    if upload is not None:
        frame = pd.read_csv(upload)
    elif choice != "—":
        frame = pd.read_csv(samples[choice])
    else:
        frame = st.session_state.get("portfolio_frame") or pd.DataFrame(
            {"ticker": ["VTI", "VXUS", "BND", "AAPL"], "shares": [20.0, 30.0, 25.0, 5.0]}
        )
    frame.columns = [c.strip().lower() for c in frame.columns]
    if not {"ticker", "shares"} <= set(frame.columns):
        st.error("The CSV needs `ticker` and `shares` columns.")
        return

    with left:
        edited = st.data_editor(
            frame[["ticker", "shares"]],
            num_rows="dynamic",
            key=f"editor-{choice}-{upload.name if upload else ''}",
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "shares": st.column_config.NumberColumn("Shares", min_value=0.0, step=1.0),
            },
        )
    if not st.button("Analyze portfolio", type="primary"):
        if "portfolio_analysis" in st.session_state:
            _show(st.session_state.portfolio_analysis)
        return

    portfolio = _to_portfolio(edited)
    if not portfolio.holdings:
        st.warning("Add at least one holding with shares > 0.")
        return
    with st.spinner("Fetching prices and computing metrics…"):
        try:
            analysis = analyze_portfolio(portfolio, get_deps().market).model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't analyze the portfolio: {exc}")
            return
    st.session_state.portfolio = portfolio.model_dump()  # the Chat tab can now say "my portfolio"
    st.session_state.portfolio_analysis = analysis
    _show(analysis)


def _show(a: dict) -> None:
    er, vol = a["weighted_expense_ratio"], a["annualized_volatility"]
    cols = st.columns(5)
    cols[0].metric("Total value", f"${a['total_value']:,.0f}")
    cols[1].metric("Holdings behave like", f"{a['effective_n']:.1f}", help="1 / HHI of weights")
    cols[2].metric("Largest holding", f"{a['top_holding_weight']:.0%}")
    cols[3].metric("Fund costs / yr", "—" if er is None else f"{er:.2%}")
    cols[4].metric("Volatility (1y)", "—" if vol is None else f"{vol:.0%}")
    source_badge(a["data_source"], mock=a["is_mock"])

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(charts.allocation_donut(a, "holding"), key="pf-holding")
    c2.plotly_chart(charts.allocation_donut(a, "asset_class"), key="pf-asset")
    c3.plotly_chart(charts.diversification_gauge(a["diversification_score"]), key="pf-gauge")
    st.plotly_chart(charts.sector_bar(a), key="pf-sector")

    st.markdown("**Observations** (educational)")
    for note in a["observations"]:
        st.markdown(f"- {note}")
    with st.expander("Positions"):
        st.dataframe(
            pd.DataFrame(a["positions"]).assign(weight=lambda d: (d.weight * 100).round(1)),
            hide_index=True,
        )
    st.caption(
        "Tip: ask about it in the **Chat** tab — e.g. *“Is my portfolio diversified enough?”*"
    )
