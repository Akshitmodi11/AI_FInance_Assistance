"""Finnie — Streamlit app.  Run:  streamlit run src/web_app/app.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `src` importable

import streamlit as st  # noqa: E402

from src.core.config import get_secrets  # noqa: E402
from src.core.schemas import UserProfile  # noqa: E402
from src.utils import disclaimers  # noqa: E402
from src.web_app.tabs import chat, goals, knowledge, markets, portfolio  # noqa: E402

st.set_page_config(page_title="Finnie — AI Finance Assistant", page_icon="💸", layout="wide")


def sidebar() -> dict:
    with st.sidebar:
        st.title("💸 Finnie")
        st.caption("Your AI guide to investing, taxes and financial planning — for learning.")
        st.markdown("#### Your profile")
        level = st.radio(
            "Knowledge level", ["beginner", "intermediate", "advanced"], horizontal=True
        )
        risk = st.select_slider(
            "Risk appetite", ["conservative", "moderate", "aggressive"], value="moderate"
        )
        horizon = st.slider("Investment horizon (years)", 1, 40, 10)
        if st.button("🗑️ New chat", width="stretch"):
            chat.new_chat()
            st.rerun()

        st.markdown("#### Live data")
        s = get_secrets()

        def has(v) -> bool:
            return bool(v and v.get_secret_value())

        status = {
            "LLM (OpenAI)": has(s.openai_api_key),
            "Market data (Alpha Vantage)": has(s.alpha_vantage_api_key),
            "News (Tavily)": has(s.tavily_api_key),
            "Tracing (LangSmith)": has(s.langsmith_api_key),
        }
        for name, ok in status.items():
            st.caption(f"{'🟢' if ok else '⚪'} {name}" + ("" if ok else " — fallback in use"))
        st.divider()
        st.caption(f"⚠️ {disclaimers.GENERAL}")
    return UserProfile(
        knowledge_level=level, risk_tolerance=risk, horizon_years=horizon
    ).model_dump()


def main() -> None:
    profile = sidebar()
    tabs = st.tabs(["💬 Chat", "📊 Portfolio", "📈 Markets", "🎯 Goals", "📚 Knowledge"])
    with tabs[0]:
        chat.render(profile)
    with tabs[1]:
        portfolio.render()
    with tabs[2]:
        markets.render()
    with tabs[3]:
        goals.render(profile)
    with tabs[4]:
        knowledge.render()


main()
