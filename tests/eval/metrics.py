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


def matches_label(chunk_id: str, label: str) -> bool:
    """Does a retrieved chunk satisfy an expected-source label?

    Labels are ``doc_id`` prefixes at any granularity:
    ``"p590a"`` (any chunk of the file), ``"p590a:p9"`` (page 9 only),
    ``"investor-gov-stocks:s2"`` (that section only).
    Matching is on whole ``:``-separated parts, so ``"p590a:p1"`` does not match page 10.
    """
    doc_id = chunk_id.rsplit(":", 1)[0]  # drop the chunk number: "p590a:p9:2" → "p590a:p9"
    return doc_id == label or doc_id.startswith(label + ":")


def first_match_rank(chunk_ids: list[str], labels: list[str]) -> int | None:
    """1-based rank of the first chunk matching any of ``labels`` (alternatives), else None."""
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        if any(matches_label(chunk_id, label) for label in labels):
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
