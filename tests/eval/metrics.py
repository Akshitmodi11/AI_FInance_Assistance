"""Retrieval metrics.

* **Recall@k** — fraction of questions where at least one expected source appears in the
  top k. "Did the right article make it into the context the LLM sees?"
* **MRR (Mean Reciprocal Rank)** — average of 1/rank of the first correct hit (0 if none).
  Rewards putting the right source *first*, not just somewhere in the top k.
* **No-answer accuracy** — for out-of-scope questions (expected = []), the fraction where
  the retriever correctly reported ``confident=False``.
"""

from __future__ import annotations


def first_hit_rank(retrieved: list[str], expected: set[str]) -> int | None:
    """1-based rank of the first retrieved source that is in ``expected``, else None."""
    for rank, source in enumerate(retrieved, start=1):
        if source in expected:
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)
