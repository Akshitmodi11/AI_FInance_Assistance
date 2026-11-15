"""Insights tab — every measured number in one place: retrieval quality, routing, safety, latency,
cost, tests, knowledge-base make-up, experiment history and live system status.

Reads the snapshot written by ``python -m scripts.collect_metrics`` (``data/metrics/metrics.json``),
so the tab opens instantly and works in Docker; live parts (services, cache counters) come from the
running app.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.core.config import get_secrets, get_settings
from src.web_app.components import charts
from src.web_app.components.ui import get_deps

METRICS_PATH = Path("data/metrics/metrics.json")
MODE_LABELS = {
    "dense": "dense",
    "sparse": "sparse",
    "hybrid": "hybrid (RRF)",
    "rerank": "hybrid → rerank",
}


def load_metrics(path: Path | None = None) -> dict | None:
    path = path or get_settings().resolve(METRICS_PATH)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def render() -> None:
    st.subheader("🔬 Insights")
    m = load_metrics()
    if m is None:
        st.info("No metrics snapshot yet. Run `python -m scripts.collect_metrics`, then reload.")
        return
    st.caption(
        f"Snapshot {m['generated']} · retrieval and knowledge base measured then; routing, safety, "
        "latency and tests copied from their last recorded run (dates below). "
        "Refresh: `python -m scripts.collect_metrics`."
    )

    rows = {r["mode"]: r for r in m["retrieval"]["rows"]}
    active = rows.get(m["config"]["retrieval_mode"]) or next(iter(rows.values()))
    llm_router = m["routing"]["rows"][0]
    guard = m["safety"]["rows"][0]  # the 18-prompt set; the other rows are subsets of it
    lat = m["latency"]["rows"]
    tests = m["tests"]

    k = st.columns(6)
    k[0].metric(
        f"Recall@{m['retrieval']['k']}",
        f"{active['recall']:.2f}",
        help=f"{MODE_LABELS.get(active['mode'], active['mode'])}, "
        f"{m['retrieval']['questions']} labelled questions",
    )
    k[1].metric("MRR", f"{active['mrr']:.2f}", help="How high the first correct source ranks")
    k[2].metric("Routing accuracy", _pct(llm_router["exact"]), help=llm_router["router"])
    k[3].metric("Safety prompts", f"{guard['passed']}/{guard['total']}", help=guard["check"])
    k[4].metric("Tests · coverage", f"{tests['passed']} · {_pct(tests['coverage'])}")
    k[5].metric(
        "Cost / question",
        f"${min(r['cost_usd'] for r in lat):.4f}–{max(r['cost_usd'] for r in lat):.4f}",
    )

    # ----- Retrieval -----
    st.markdown("#### Retrieval — which search mode finds the right source")
    left, right = st.columns([3, 2])
    ordered = [dict(r, mode=MODE_LABELS.get(r["mode"], r["mode"])) for r in rows.values()]
    left.plotly_chart(charts.retrieval_modes_bar(ordered), width="stretch")
    types = sorted({t for r in rows.values() for t in r["by_type"]})
    right.dataframe(
        [
            {
                "mode": MODE_LABELS.get(r["mode"], r["mode"]),
                **{f"R@5 {t}": round(r["by_type"].get(t, 0), 2) for t in types},
                "off-topic refused": _pct(r["no_answer"]),
                "p50 ms": round(r["p50_ms"]),
            }
            for r in rows.values()
        ],
        hide_index=True,
        width="stretch",
    )
    right.caption(f"Active mode: **{MODE_LABELS.get(active['mode'], active['mode'])}**")
    with st.expander("Missed questions per mode"):
        for r in rows.values():
            st.markdown(f"**{MODE_LABELS.get(r['mode'], r['mode'])}** — {len(r['misses'])} misses")
            st.code("\n".join(r["misses"]) or "none", language=None)

    # ----- Answer pipeline -----
    st.markdown("#### Answers — routing, safety, speed and cost")
    a, b, c = st.columns([2, 2, 3])
    a.caption(f"Routing · {m['routing']['questions']} questions · {m['routing']['date']}")
    a.dataframe(
        [
            {"router": r["router"], "exact": _pct(r["exact"]), "primary": _pct(r["primary"])}
            for r in m["routing"]["rows"]
        ],
        hide_index=True,
        width="stretch",
    )
    b.caption(f"Safety · {m['safety']['date']}")
    b.dataframe(
        [
            {"check": r["check"], "passed": f"{r['passed']}/{r['total']}"}
            for r in m["safety"]["rows"]
        ],
        hide_index=True,
        width="stretch",
    )
    c.caption(f"End-to-end latency and cost per question · {m['latency']['date']}")
    c.dataframe(
        [
            {
                "agent": r["intent"],
                "p50 s": r["p50_s"],
                "p95 s": r["p95_s"],
                "tokens": r["tokens"],
                "cost $": r["cost_usd"],
            }
            for r in lat
        ],
        hide_index=True,
        width="stretch",
    )

    # ----- Knowledge base -----
    kb = m["kb"]
    st.markdown(f"#### Knowledge base — {kb['chunks']:,} chunks from {kb['sources']} sources")
    left, right = st.columns(2)
    left.plotly_chart(charts.count_bar(kb["by_category"], "Chunks by category"), width="stretch")
    right.plotly_chart(charts.count_bar(kb["by_file_type"], "Chunks by file type"), width="stretch")

    # ----- Experiments -----
    st.markdown("#### Experiment log")
    st.dataframe(m["experiments"], hide_index=True, width="stretch")
    st.caption("Full write-ups: `docs/benchmarks.md`.")

    # ----- Live system -----
    st.markdown("#### System — live")
    cfg = m["config"]
    s = get_secrets()
    left, right = st.columns(2)
    left.markdown(
        f"- Retrieval: **{MODE_LABELS.get(cfg['retrieval_mode'], cfg['retrieval_mode'])}** "
        f"on **{cfg['vector_store']}**, gate ≥ {cfg['min_score']}\n"
        f"- Embeddings: `{cfg['embedding_model']}`\n"
        f"- Reranker: `{cfg['rerank_model']}`\n"
        f"- LLM provider: **{cfg['llm_provider']}**"
    )
    stats = dict(get_deps().market.stats)
    tracing = bool(s.langsmith_api_key and s.langsmith_api_key.get_secret_value())
    right.markdown(
        "- Market data this session: "
        + (", ".join(f"{k.replace('_', ' ')} {v}" for k, v in stats.items()) or "no requests yet")
        + "\n- Traces: "
        + (
            "[LangSmith → project `finnie`](https://smith.langchain.com)"
            if tracing
            else "off (no key)"
        )
    )
