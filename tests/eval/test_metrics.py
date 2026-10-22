import pytest

from tests.eval.metrics import (
    first_hit_rank,
    first_match_rank,
    matches_label,
    mean_reciprocal_rank,
    recall_at_k,
)


def test_matches_label_granularity():
    assert matches_label("p590a:p9:2", "p590a")  # file level
    assert matches_label("p590a:p9:2", "p590a:p9")  # page level
    assert not matches_label("p590a:p9:2", "p590a:p10")
    assert not matches_label("p590a:p10:0", "p590a:p1")  # whole parts, not string prefix
    assert matches_label("investor-gov-stocks:s2:0", "investor-gov-stocks:s2")
    assert not matches_label("p590b:p9:0", "p590")  # file stems must match exactly


def test_first_match_rank_with_alternatives():
    retrieved = ["p969:p3:0", "p590a:p12:1", "p590a:p9:0"]
    assert first_match_rank(retrieved, ["p590a:p9", "p590a:p12"]) == 2
    assert first_match_rank(retrieved, ["p590a:p40"]) is None


def test_first_hit_rank():
    assert first_hit_rank(["a", "b", "c"], {"b"}) == 2
    assert first_hit_rank(["a", "b"], {"z"}) is None
    assert first_hit_rank(["a", "a", "b"], {"a", "b"}) == 1


def test_recall_and_mrr():
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 5) == 0.5
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)


def test_empty():
    assert recall_at_k([], 5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0
