import pytest

from tests.eval.metrics import first_hit_rank, mean_reciprocal_rank, recall_at_k


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
