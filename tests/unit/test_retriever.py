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


class FakeReranker:
    """Scores by how many query words appear in the text; records what it was asked."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(texts)
        words = set(query.lower().split())
        return [float(len(words & set(t.lower().split()))) for t in texts]


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
        reranker=FakeReranker(),
    )


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "rerank"])
def test_each_mode_finds_the_obvious_match(retriever, mode):
    result = retriever.retrieve("roth ira after-tax contributions", mode=mode)
    assert result.mode == mode
    assert result.evidence[0].chunk.chunk_id == "roth:s0:0"
    assert result.confident


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "rerank"])
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


def test_rerank_orders_by_reranker_score(retriever):
    result = retriever.retrieve("compound interest grows savings", mode="rerank")
    assert result.mode == "rerank"
    assert result.evidence[0].chunk.chunk_id == "ci:s0:0"
    scores = [e.score for e in result.evidence]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 4.0  # FakeReranker: 4 shared words — score is the reranker's, not RRF


def test_rerank_sees_only_filtered_candidates_with_headers(retriever):
    retriever.retrieve("money plan stocks", category="retirement_tax", mode="rerank")
    texts = retriever.reranker.calls[-1]
    assert texts and all(("Roth" in t) or ("401(k)" in t) for t in texts)
    assert all("\n" in t for t in texts)  # embedding_text = contextual header + chunk


def test_rerank_keeps_dense_confidence_gate(retriever):
    result = retriever.retrieve("bitcoin mining hardware", mode="rerank")
    assert not result.confident
    assert all(e.dense_score is not None for e in result.evidence)


def test_rerank_without_reranker_fails_loudly(retriever):
    retriever.reranker = None
    with pytest.raises(ValueError, match="reranker"):
        retriever.retrieve("roth ira", mode="rerank")
