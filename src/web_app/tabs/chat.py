"""Chat tab — the multi-agent assistant with memory, agent badges, charts and citations."""

from __future__ import annotations

import uuid

import streamlit as st

from src.web_app.components.ui import get_app, render_final

EXAMPLES = [
    "What's the difference between a traditional and a Roth IRA?",
    "Analyze my portfolio",
    "I want $40,000 for a house in 6 years. I have $5,000 and can add $400 a month.",
    "How is VTI doing, and any news on the stock market?",
    "Should I buy Tesla?",
]


def new_chat() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []


def render(profile: dict) -> None:
    if "thread_id" not in st.session_state:
        new_chat()

    if not st.session_state.chat:
        st.markdown("#### 👋 Hi, I'm Finnie — ask me about investing, taxes, goals or markets.")
        st.caption("Try one of these:")
        cols = st.columns(len(EXAMPLES))
        for col, example in zip(cols, EXAMPLES, strict=True):
            if col.button(example, key=f"ex-{example[:20]}", width="stretch"):
                st.session_state.pending = example

    for i, turn in enumerate(st.session_state.chat):
        with st.chat_message(turn["role"], avatar="🧑" if turn["role"] == "user" else "💸"):
            if turn["role"] == "user":
                st.markdown(turn["content"])
            else:
                render_final(turn["final"], key=f"turn{i}")

    question = st.chat_input("Ask Finnie…") or st.session_state.pop("pending", None)
    if question:
        _answer(question, profile)


def _answer(question: str, profile: dict) -> None:
    from src.workflow.graph import ask

    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="💸"), st.spinner("Finnie is thinking…"):
        try:
            final = ask(
                get_app(),
                question,
                st.session_state.thread_id,
                user_profile=profile,
                portfolio=st.session_state.get("portfolio"),
            ).model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001 — show a friendly error instead of a traceback
            st.error(f"Something went wrong: {type(exc).__name__}. Please try again.")
            st.session_state.chat.pop()
            return
        render_final(final, key=f"turn{len(st.session_state.chat)}")
    st.session_state.chat.append({"role": "assistant", "final": final})
