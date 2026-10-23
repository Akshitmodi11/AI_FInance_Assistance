"""Knowledge tab — search and browse the cited knowledge base behind Finnie's answers."""

from __future__ import annotations

from collections import Counter, defaultdict

import streamlit as st

from src.core.config import get_settings
from src.web_app.components.ui import get_deps


def render() -> None:
    st.subheader("📚 Knowledge base")
    deps = get_deps()
    store = deps.retriever.store
    categories = get_settings().rag.categories

    by_doc: dict[str, dict] = {}
    chunk_counts: Counter[str] = Counter()
    for chunk in store.chunks:
        doc = chunk.doc_id.split(":")[0]
        chunk_counts[doc] += 1
        by_doc.setdefault(doc, {"meta": chunk.meta})

    c = st.columns(3)
    c[0].metric("Sources", len(by_doc))
    c[1].metric("Indexed chunks", len(store))
    c[2].metric("Retrieval", "Hybrid (FAISS + BM25, RRF)")

    query = st.text_input("Search the knowledge base", placeholder="e.g. HSA contribution limit")
    category = st.selectbox("Category", ["All", *categories])
    if query:
        result = deps.retriever.retrieve(
            query, k=5, category=None if category == "All" else category
        )
        if not result.confident:
            st.warning(
                f"No confident match (best similarity {result.top_dense_score or 0:.2f} < "
                f"{deps.retriever.min_score}). Finnie would say it doesn't know."
            )
        for ev in result.evidence:
            meta = ev.chunk.meta
            with st.container(border=True):
                st.markdown(f"**[{meta.citation}]({meta.url})** · {meta.category}")
                st.caption(f"relevance {ev.score:.3f} · similarity {ev.dense_score:.2f}")
                st.write(ev.chunk.text[:700] + ("…" if len(ev.chunk.text) > 700 else ""))

    st.markdown("#### Browse sources")
    grouped: dict[str, list] = defaultdict(list)
    for doc, info in by_doc.items():
        grouped[info["meta"].category].append((doc, info["meta"]))
    for cat in sorted(grouped):
        with st.expander(f"{cat} ({len(grouped[cat])})"):
            for doc, meta in sorted(grouped[cat], key=lambda d: d[1].title):
                st.markdown(
                    f"- [{meta.title}]({meta.url}) — {meta.source}, {meta.file_type.upper()}, "
                    f"{chunk_counts[doc]} chunks"
                    + (f", tax year {meta.tax_year}" if meta.tax_year else "")
                )
