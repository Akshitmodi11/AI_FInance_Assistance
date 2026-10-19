"""Reciprocal Rank Fusion.

Why ranks, not scores: cosine lives in [-1, 1], BM25 in [0, ~30]. Adding them lets BM25
drown out the dense retriever. RRF ignores the raw scores and uses each result's *position*:

    rrf(d) = Σ over rankers r of   1 / (k + rank_r(d))        rank starts at 1

A document missing from a ranker's list simply gets nothing from that ranker. ``k = 60``
comes from the original paper (Cormack et al., 2009); it damps the gap between rank 1 and
rank 2 so a single ranker can't dominate.
"""

from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse several ranked lists of row ids into one.

    ``rankings``: e.g. ``[[4, 1, 9], [1, 7]]`` — each inner list is best-first.
    Returns ``[(row, rrf_score), ...]`` sorted by score descending; break ties by row id
    ascending so the output is deterministic (tests depend on it).
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, row in enumerate(ranking, start=1):
            if row in seen:
                continue  # ignore duplicates in a single ranking
            seen.add(row)
            scores[row] = scores.get(row, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
