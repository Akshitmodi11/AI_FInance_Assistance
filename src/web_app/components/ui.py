"""Small shared Streamlit helpers: cached services, data-source badges, rendering an answer."""

from __future__ import annotations

import streamlit as st

from src.core.config import get_settings
from src.web_app.components import charts


@st.cache_resource(show_spinner="Starting Finnie…")
def get_deps():
    from src.workflow.deps import Deps

    return Deps.from_settings(get_settings())


@st.cache_resource(show_spinner=False)
def get_app():
    from src.workflow.graph import build_graph, sqlite_checkpointer

    settings = get_settings()
    return build_graph(
        get_deps(), sqlite_checkpointer(settings.resolve(settings.memory.checkpoint_db))
    )


def source_badge(source: str, as_of: str | None = None, cached: bool = False, mock: bool = False):
    """One-line freshness label shown next to every number from a data provider."""
    parts = [f"Source: **{source}**"]
    if as_of:
        parts.append(f"as of {as_of[:16].replace('T', ' ')} UTC")
    if cached:
        parts.append("♻️ cached (≤30 min)")
    if mock:
        parts.append("⚠️ **sample data** — live providers unavailable")
    st.caption(" · ".join(parts))


AGENT_BADGES = {
    "finance_qa": "📚 Finance Q&A",
    "tax": "🧾 Tax",
    "portfolio": "📊 Portfolio",
    "goal": "🎯 Goals",
    "market": "📈 Market",
    "news": "📰 News",
}


def render_final(final: dict, key: str) -> None:
    """Show one assistant turn: agents used, guardrail note, answer, charts, sources."""
    agents = final.get("agents") or []
    st.caption(
        "Answered by: " + (" · ".join(AGENT_BADGES.get(a, a) for a in agents) or "🛡️ guardrail")
    )
    if note := final.get("guardrail_note"):
        st.info(note, icon="🛡️")
    st.markdown(final["answer"])

    data = final.get("data") or {}
    if analysis := data.get("portfolio_analysis"):
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.allocation_donut(analysis), key=f"{key}-donut")
        c2.plotly_chart(
            charts.diversification_gauge(analysis["diversification_score"]), key=f"{key}-gauge"
        )
    if plan := data.get("goal_plan"):
        st.plotly_chart(charts.goal_projection(plan), key=f"{key}-goal")
    for i, item in enumerate(data.get("market") or []):
        quote = item["snapshot"]["quote"]
        st.plotly_chart(
            charts.price_chart(item["history"], quote["ticker"], show_last=126),
            key=f"{key}-mkt{i}",
        )
        source_badge(quote["source"], quote["as_of"], quote["is_cached"], quote["is_mock"])

    if final.get("citations"):
        with st.expander(f"Sources ({len(final['citations'])})"):
            for i, c in enumerate(final["citations"], start=1):
                st.markdown(f"{i}. [{c['label']}]({c['url']}) — {c['source']}")
    st.caption(f"_{final['disclaimer']}_")
