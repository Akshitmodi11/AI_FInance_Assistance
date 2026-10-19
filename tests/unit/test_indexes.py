"""Dense index, sparse index (incl. tokenizer) and RRF fusion."""

from pathlib import Path

import numpy as np
import pytest

from src.rag.dense_index import DenseIndex
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.sparse_index import SparseIndex, tokenize

# ---------- Dense ----------


def _unit(rows: list[list[float]]) -> np.ndarray:
    a = np.array(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_dense_search_orders_by_cosine():
    index = DenseIndex.build(_unit([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]))
    results = index.search(_unit([[1, 0, 0]])[0], k=2)
    assert [row for row, _ in results] == [0, 2]
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(index) == 3


def test_dense_search_clamps_k():
    index = DenseIndex.build(_unit([[1, 0], [0, 1]]))
    assert len(index.search(np.array([1, 0], dtype=np.float32), k=10)) == 2


def test_dense_save_load_roundtrip(tmp_path: Path):
    vectors = _unit([[1, 0], [0, 1]])
    DenseIndex.build(vectors).save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.search(vectors[1], k=1)[0][0] == 1


# ---------- Tokenizer ----------


@pytest.mark.parametrize("text", ["401(k)", "401k", "401 (k)", "401(K)"])
def test_tokenize_401k_variants(text):
    assert tokenize(text) == ["401k"]


def test_tokenize_403b():
    assert tokenize("a 403(b) plan") == ["403b", "plan"]


def test_tokenize_keeps_finance_symbols():
    assert tokenize("S&P 500") == ["s&p", "500"]
    assert tokenize("The P/E ratio") == ["p/e", "ratio"]


def test_tokenize_lowercases_and_drops_stopwords():
    tokens = tokenize("What is a Roth IRA?")
    assert tokens == ["roth", "ira"]


# ---------- Sparse ----------

CORPUS = [
    "Roth IRA contributions are made with after-tax money.",
    "A 401(k) is an employer-sponsored retirement plan.",
    "Diversification spreads risk across investments.",
    "Form 8606 reports nondeductible IRA contributions.",
]


def test_sparse_exact_term_match():
    index = SparseIndex.build(CORPUS)
    assert index.search("form 8606", k=3)[0][0] == 3
    assert index.search("401k match", k=3)[0][0] == 1


def test_sparse_returns_only_positive_scores():
    index = SparseIndex.build(CORPUS)
    assert index.search("cryptocurrency", k=5) == []


def test_sparse_save_load_roundtrip(tmp_path: Path):
    SparseIndex.build(CORPUS).save(tmp_path)
    loaded = SparseIndex.load(tmp_path)
    assert len(loaded) == 4
    assert loaded.search("diversification", k=1)[0][0] == 2


# ---------- RRF ----------


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]], k=60)
    # 2 is rank 2 in both lists: 2/(62). 1 and 3: 1/61 + 1/63. Those are higher.
    assert [row for row, _ in fused][:2] == [1, 3]  # tie broken by row id
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_handles_items_in_one_list_only():
    fused = dict(reciprocal_rank_fusion([[7], [8, 7]], k=60))
    assert fused[7] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[8] == pytest.approx(1 / 61)


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []
