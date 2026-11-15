"""Snapshot every measured number into one JSON file for the Insights tab.

    python -m scripts.collect_metrics                  # all retrieval modes + KB stats
    python -m scripts.collect_metrics --modes hybrid rerank

Retrieval numbers and knowledge-base stats are **measured now**. Routing, safety, latency, cost and
test results come from runs that call the LLM or take minutes, so they are copied from the last
recorded run in ``docs/benchmarks.md`` (``RECORDED`` below, each with its date). Update them after
re-running ``tests.eval.run_agent_eval``, ``scripts.benchmarks`` or ``pytest --cov``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from src.core.config import Settings, get_settings
from src.rag.retriever import Retriever
from src.rag.store import ChunkStore
from tests.eval.run_retrieval_eval import MODES, evaluate, load_dataset

OUT = Path("data/metrics/metrics.json")

# (intent, p50 s, p95 s, tokens, cost $) — docs/benchmarks.md "Performance", before the reranker
LATENCY = [
    ("finance_qa", 3.5, 3.7, 1322, 0.0039),
    ("tax", 3.1, 3.8, 1896, 0.0053),
    ("portfolio", 5.4, 7.0, 1639, 0.0063),
    ("goal", 5.2, 5.2, 1470, 0.0043),
    ("market", 3.8, 4.9, 1097, 0.0039),
    ("news", 9.0, 9.4, 1548, 0.0044),
]

# (date, experiment, result, decision) — docs/benchmarks.md
EXPERIMENTS = [
    ("2026-10-22", "Baseline: dense vs sparse vs hybrid (43 Q)",
     "hybrid R@5 0.92, MRR 0.73 (dense 0.86, sparse 0.78)", "hybrid default"),
    ("2026-10-22", "Exp 1: PDF section rule",
     "hybrid R@5 0.92 in all 3 variants", "kept current rule"),
    ("2026-10-22", "Exp 2: FAISS vs Qdrant Cloud",
     "identical quality; 4 ms vs ~110 ms; N+1 fix 488 → 100 ms", "FAISS default"),
    ("2026-11-05", "Exp 3: +14 plain-language articles (59 Q)",
     "new topics R@5 0.00 → 1.00; all 59: 0.63 → 0.92", "kept; q001 regressed"),
    ("2026-11-14", "Exp 4: cross-encoder reranker",
     "R@5 0.92 → 0.96, MRR 0.79 → 0.90, +~350 ms", "rerank default"),
]  # fmt: skip

RECORDED = {
    "routing": {
        "date": "2026-10-22",
        "questions": 60,
        "rows": [
            {"router": "LLM router (gpt-4o-mini)", "exact": 0.95, "primary": 0.98},
            {"router": "Keyword fallback", "exact": 0.88, "primary": 0.92},
        ],
    },
    "safety": {
        "date": "2026-10-22",
        "rows": [
            {"check": "Guardrail category (18 prompts)", "passed": 18, "total": 18},
            {"check": "Advice reframed (live answers)", "passed": 9, "total": 9},
            {"check": "Illegal requests refused", "passed": 4, "total": 4},
        ],
    },
    "latency": {
        "date": "2026-10-22",
        "rows": [
            dict(zip(("intent", "p50_s", "p95_s", "tokens", "cost_usd"), row, strict=True))
            for row in LATENCY
        ],
    },
    "tests": {"date": "2026-11-14", "passed": 188, "coverage": 0.91, "gate": 0.80},
    "experiments": [
        dict(zip(("date", "experiment", "result", "decision"), row, strict=True))
        for row in EXPERIMENTS
    ],
}

RETRIEVAL_KEYS = ("mode", "recall", "mrr", "no_answer", "by_type", "p50_ms")


def kb_stats(settings: Settings) -> dict:
    """Chunk counts by category, file type and source — read from the built index (fast)."""
    store = ChunkStore.load(settings.resolve(settings.rag.index_dir))
    chunks = [store[i] for i in range(len(store))]
    return {
        "chunks": len(chunks),
        "sources": len({c.meta.title for c in chunks}),
        "by_category": dict(Counter(c.meta.category for c in chunks).most_common()),
        "by_file_type": dict(Counter(c.meta.file_type for c in chunks).most_common()),
    }


def retrieval_metrics(retriever: Retriever, modes: list[str], k: int = 5) -> dict:
    dataset = load_dataset()
    rows = []
    for mode in modes:
        r = evaluate(retriever, dataset, mode, k)
        row = {key: r[key] for key in RETRIEVAL_KEYS}
        row["misses"] = [m.strip() for m in r["misses"]]
        rows.append(row)
    return {"questions": len(dataset), "k": k, "rows": rows}


def collect(settings: Settings, modes: list[str]) -> dict:
    return {
        "generated": date.today().isoformat(),
        "config": {
            "retrieval_mode": settings.rag.retrieval_mode,
            "vector_store": settings.rag.vector_store,
            "embedding_model": settings.rag.embedding_model,
            "rerank_model": settings.rag.rerank_model,
            "min_score": settings.rag.min_score,
            "llm_provider": settings.llm.provider,
        },
        "kb": kb_stats(settings),
        "retrieval": retrieval_metrics(Retriever.from_settings(settings), modes),
        **RECORDED,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    settings = get_settings()
    metrics = collect(settings, args.modes)
    out = settings.resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    n_q, n_c = metrics["retrieval"]["questions"], metrics["kb"]["chunks"]
    print(f"wrote {out} — {n_q} questions, {n_c} chunks")


if __name__ == "__main__":
    main()
