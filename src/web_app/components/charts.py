"""Plotly figures — pure functions (data in, figure out) so they're testable without Streamlit."""

from __future__ import annotations

import plotly.graph_objects as go

PALETTE = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
BAND = {"low": "#f59e0b", "expected": "#2563eb", "high": "#16a34a"}


def _layout(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        height=height,
        margin={"l": 10, "r": 10, "t": 45, "b": 10},
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def allocation_donut(analysis: dict, by: str = "holding") -> go.Figure:
    """Donut of portfolio weights by holding, asset class or sector."""
    if by == "holding":
        labels = [p["ticker"] for p in analysis["positions"]]
        values = [p["weight"] for p in analysis["positions"]]
        title = "Allocation by holding"
    else:
        mix = analysis["asset_class_mix" if by == "asset_class" else "sector_mix"]
        labels, values = list(mix), list(mix.values())
        title = "Allocation by asset class" if by == "asset_class" else "Allocation by sector"
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker={"colors": PALETTE},
            textinfo="label+percent",
            sort=False,
        )
    )
    return _layout(fig, title)


def sector_bar(analysis: dict) -> go.Figure:
    mix = analysis["sector_mix"]
    fig = go.Figure(
        go.Bar(
            x=[v * 100 for v in mix.values()],
            y=list(mix),
            orientation="h",
            marker_color=PALETTE[0],
            text=[f"{v:.0%}" for v in mix.values()],
            textposition="auto",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="% of portfolio", range=[0, 100])
    return _layout(fig, "Sector exposure")


def diversification_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1e293b"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#bbf7d0"},
                ],
            },
        )
    )
    return _layout(fig, "Diversification score (educational)", height=260)


def price_chart(
    history: list[dict],
    ticker: str,
    sma_windows: tuple[int, ...] = (20, 50),
    show_last: int | None = None,
) -> go.Figure:
    """Close prices with simple moving averages. ``history`` = [{"date", "close"}, …].

    Averages are computed on the full history, then only the last ``show_last`` days are drawn —
    so a 1-month view still shows a real 50-day average.
    """
    dates = [h["date"] for h in history]
    closes = [h["close"] for h in history]
    start = max(len(closes) - show_last, 0) if show_last else 0
    fig = go.Figure(
        go.Scatter(x=dates[start:], y=closes[start:], name=f"{ticker} close", line={"width": 2})
    )
    for i, n in enumerate(sma_windows):
        if len(closes) >= n:
            sma = [None] * (n - 1) + [
                sum(closes[j - n + 1 : j + 1]) / n for j in range(n - 1, len(closes))
            ]
            fig.add_trace(
                go.Scatter(
                    x=dates[start:], y=sma[start:], name=f"SMA {n}",
                    line={"width": 1.5, "dash": "dot"}, marker_color=PALETTE[i + 1],
                )
            )  # fmt: skip
    fig.update_yaxes(title="Price (USD)")
    return _layout(fig, f"{ticker} — price and moving averages", height=380)


def goal_projection(plan: dict) -> go.Figure:
    """Savings path in low / expected / high scenarios vs the inflation-adjusted target."""
    rows = plan["yearly_projection"]
    years = [r["year"] for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=years, y=[r["high"] for r in rows], name="High return",
                   line={"color": BAND["high"], "width": 1}),
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=years, y=[r["low"] for r in rows], name="Low return", fill="tonexty",
                   fillcolor="rgba(37,99,235,0.10)", line={"color": BAND["low"], "width": 1}),
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=years, y=[r["expected"] for r in rows], name="Expected",
                   line={"color": BAND["expected"], "width": 3}),
    )  # fmt: skip
    target = plan["target_in_future_dollars"]
    fig.add_hline(
        y=target, line_dash="dash", line_color="#dc2626",
        annotation_text=f"Target ${target:,.0f} (inflation-adjusted)",
    )  # fmt: skip
    fig.update_xaxes(title="Years from now")
    fig.update_yaxes(title="Projected savings (USD)", tickprefix="$")
    return _layout(fig, "Savings projection by return scenario", height=380)
