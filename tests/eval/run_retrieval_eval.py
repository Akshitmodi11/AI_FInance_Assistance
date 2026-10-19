"""Compare dense vs sparse vs hybrid retrieval on the labeled question set.

Usage (after ``python -m src.rag.ingest``):
    python -m tests.eval.run_retrieval_eval            # all modes, k=5
    python -m tests.eval.run_retrieval_eval --k 3 --show-misses

Dataset: tests/eval/retrieval_set.jsonl, one JSON object per line:
    {"id", "question", "expected_sources": [file stems], "type"}
``expected_sources`` are file stems (``"diversification-basics"``, ``"p590a"``) — a hit is any
chunk from that file. Use ``[]`` for out-of-scope questions; those are scored on whether the
retriever reports ``confident=False``.
``type`` groups results so you can show *where* each mode wins: "exact_term" (form numbers,
acronyms), "paraphrase" (no shared keywords), "concept", "out_of_scope".
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.core.config import get_settings
from src.rag.retriever import Retriever
from tests.eval.metrics import first_hit_rank, mean_reciprocal_rank, recall_at_k

DATASET = Path(__file__).with_name("retrieval_set.jsonl")
MODES = ("dense", "sparse", "hybrid")


def load_dataset(path: Path = DATASET) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def source_of(chunk_id: str) -> str:
    """``"p590a:p12:0"`` → ``"p590a"``."""
    return chunk_id.split(":", 1)[0]


def evaluate(retriever: Retriever, dataset: list[dict], mode: str, k: int) -> dict:
    ranks: list[int | None] = []
    ranks_by_type: dict[str, list[int | None]] = defaultdict(list)
    no_answer_correct = no_answer_total = 0
    misses: list[str] = []
    latencies: list[float] = []

    for item in dataset:
        start = time.perf_counter()
        result = retriever.retrieve(item["question"], k=k, mode=mode)
        latencies.append((time.perf_counter() - start) * 1000)

        expected = set(item["expected_sources"])
        if not expected:
            no_answer_total += 1
            no_answer_correct += not result.confident
            if result.confident:
                misses.append(f"  [{item['id']}] should be out-of-scope: {item['question']}")
            continue

        retrieved = [source_of(e.chunk.chunk_id) for e in result.evidence]
        rank = first_hit_rank(retrieved, expected)
        ranks.append(rank)
        ranks_by_type[item.get("type", "other")].append(rank)
        if rank is None:
            misses.append(f"  [{item['id']}] {item['question']}  → got {retrieved[:3]}")

    latencies.sort()
    return {
        "mode": mode,
        "recall": recall_at_k(ranks, k),
        "mrr": mean_reciprocal_rank(ranks),
        "no_answer": no_answer_correct / no_answer_total if no_answer_total else None,
        "by_type": {t: recall_at_k(r, k) for t, r in sorted(ranks_by_type.items())},
        "p50_ms": latencies[len(latencies) // 2] if latencies else 0.0,
        "misses": misses,
    }


def print_table(rows: list[dict], k: int) -> None:
    types = sorted({t for r in rows for t in r["by_type"]})
    header = f"| Mode | Recall@{k} | MRR | No-answer acc. | " + " | ".join(
        f"R@{k} {t}" for t in types
    ) + " | p50 ms |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for r in rows:
        na = "n/a" if r["no_answer"] is None else f"{r['no_answer']:.2f}"
        by_type = " | ".join(f"{r['by_type'].get(t, 0):.2f}" for t in types)
        print(
            f"| {r['mode']} | {r['recall']:.2f} | {r['mrr']:.2f} | {na} | {by_type} | "
            f"{r['p50_ms']:.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    retriever = Retriever.from_settings(get_settings())
    dataset = load_dataset()
    retriever.retrieve("warm up the embedding model")  # keep model load out of latency

    rows = [evaluate(retriever, dataset, mode, args.k) for mode in args.modes]
    print(f"\n{len(dataset)} questions\n")
    print_table(rows, args.k)  # paste this Markdown table into README / docs/benchmarks.md

    if args.show_misses:
        for r in rows:
            if r["misses"]:
                print(f"\nMisses ({r['mode']}):")
                print("\n".join(r["misses"]))


if __name__ == "__main__":
    main()
