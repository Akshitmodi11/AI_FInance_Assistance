"""Markets tab — index overview, ticker lookup, price chart with SMAs, freshness labels."""

from __future__ import annotations

import streamlit as st

from src.data.providers.base import ProviderError
from src.web_app.components import charts
from src.web_app.components.ui import get_deps, source_badge

INDEXES = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "DIA": "Dow Jones"}
TREND_TEXT = {
    "uptrend": "📈 Uptrend — price above its 20-day average, which is above the 50-day.",
    "downtrend": "📉 Downtrend — price below its 20-day average, which is below the 50-day.",
    "sideways": "↔️ Sideways — the moving averages don't line up in one direction.",
    "unknown": "Not enough history to judge the trend.",
}


def render() -> None:
    st.subheader("📈 Markets")
    market = get_deps().market

    cols = st.columns(len(INDEXES))
    for col, (ticker, name) in zip(cols, INDEXES.items(), strict=True):
        try:
            q = market.quote(ticker)
            col.metric(
                f"{name} ({ticker})", f"${q.price:,.2f}", f"{q.change_pct:+.2f}%", border=True
            )
        except ProviderError:
            col.metric(f"{name} ({ticker})", "n/a", border=True)

    ticker = st.text_input("Look up a ticker", value="VTI", max_chars=10).strip().upper()
    if not ticker:
        return
    with st.spinner(f"Loading {ticker}…"):
        try:
            snap, history = market.snapshot(ticker)
        except ProviderError as exc:
            st.error(f"No data for {ticker}: {exc}")
            return

    q = snap.quote
    c = st.columns(4)
    c[0].metric("Price", f"${q.price:,.2f}", f"{q.change:+.2f} ({q.change_pct:+.2f}%)")
    c[1].metric("52-week range", f"${snap.low_52w:,.0f} – ${snap.high_52w:,.0f}")
    c[2].metric("1-month return", "—" if snap.return_1m is None else f"{snap.return_1m:+.1%}")
    c[3].metric("6-month return", "—" if snap.return_6m is None else f"{snap.return_6m:+.1%}")
    source_badge(q.source, q.as_of.isoformat(), q.is_cached, q.is_mock)

    days = st.segmented_control("Range", ["1M", "3M", "6M", "1Y"], default="6M") or "6M"
    n = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}[days]
    points = [{"date": p.date.date().isoformat(), "close": p.close} for p in history.points]
    st.plotly_chart(charts.price_chart(points, ticker, show_last=n))
    st.info(TREND_TEXT[snap.trend])
    st.caption(
        "Moving averages smooth daily noise: the 20-day shows the recent direction, the 50-day the "
        "medium-term one. These describe the past — they don't predict prices."
    )
