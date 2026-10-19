import pytest

from src.rag.dense_index import DenseIndex
from src.rag.retriever import Retriever
from src.rag.sparse_index import SparseIndex
from src.rag.store import ChunkStore
from tests.conftest import make_chunk

TAX, DIV = "retirement_tax", "diversification"
CHUNKS = [
    make_chunk("Roth IRA contributions use after-tax money", "roth:s0:0", category=TAX),
    make_chunk("A 401(k) plan is offered by an employer", "k401:s0:0", category=TAX),
    make_chunk("Diversification spreads risk across many stocks", "div:s0:0", category=DIV),
    make_chunk("Compound interest grows savings over time", "ci:s0:0", category="basics"),
    make_chunk("Stocks give partial ownership of a company", "stk:s0:0", category="investing"),
]


@pytest.fixture
def retriever(fake_embedder) -> Retriever:
    texts = [c.embedding_text for c in CHUNKS]
    return Retriever(
        store=ChunkStore(CHUNKS),
        dense=DenseIndex.build(fake_embedder.embed(texts)),
        sparse=SparseIndex.build(texts),
        embedder=fake_embedder,
        top_k=3,
        candidate_k=5,
        min_score=0.3,
    )


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_each_mode_finds_the_obvious_match(retriever, mode):
    result = retriever.retrieve("roth ira after-tax contributions", mode=mode)
    assert result.mode == mode
    assert result.evidence[0].chunk.chunk_id == "roth:s0:0"
    assert result.confident


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_dense_score_always_filled(retriever, mode):
    result = retriever.retrieve("diversification risk stocks", mode=mode)
    assert all(e.dense_score is not None for e in result.evidence)
    assert result.top_dense_score == max(e.dense_score for e in result.evidence)


def test_k_limits_results(retriever):
    assert len(retriever.retrieve("stocks risk money", k=2, mode="hybrid").evidence) <= 2


def test_category_filter(retriever):
    result = retriever.retrieve("money plan stocks", category="retirement_tax", mode="hybrid")
    assert result.evidence
    assert {e.chunk.meta.category for e in result.evidence} == {"retirement_tax"}


def test_off_topic_query_is_not_confident(retriever):
    result = retriever.retrieve("bitcoin mining hardware", mode="hybrid")
    assert not result.confident


def test_empty_query(retriever):
    result = retriever.retrieve("   ")
    assert result.evidence == []
    assert not result.confident


def test_hybrid_scores_are_rrf(retriever):
    result = retriever.retrieve("roth ira", mode="hybrid")
    assert result.evidence[0].score <= 2 / 61 + 1e-9  # max possible with two rankers, k=60
