"""QdrantIndex tests — run against Qdrant's in-memory mode, so no network or account needed."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.rag.dense_index import DenseIndex
from src.rag.qdrant_index import QdrantIndex
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
]


@pytest.fixture
def vectors(fake_embedder) -> np.ndarray:
    return fake_embedder.embed([c.embedding_text for c in CHUNKS])


@pytest.fixture
def qdrant(vectors) -> QdrantIndex:
    return QdrantIndex.build(vectors, QdrantClient(":memory:"), "test_chunks", CHUNKS)


def test_search_matches_faiss(qdrant, vectors, fake_embedder):
    faiss_index = DenseIndex.build(vectors)
    query = fake_embedder.embed(["roth ira after-tax money"])[0]
    q_hits, f_hits = qdrant.search(query, 4), faiss_index.search(query, 4)
    assert q_hits[0][0] == f_hits[0][0] == 0
    # Same cosine for every row. (Rows that tie at 0.0 may come back in a different order.)
    assert dict(q_hits) == pytest.approx(dict(f_hits), abs=1e-5)


def test_vector_roundtrip_and_len(qdrant, vectors):
    assert len(qdrant) == len(CHUNKS)
    assert np.allclose(qdrant.vector(2), vectors[2], atol=1e-5)  # inputs were already normalized


def test_payload_is_browsable(qdrant):
    record = qdrant.client.retrieve("test_chunks", ids=[2], with_payload=True)[0]
    assert record.payload["chunk_id"] == "div:s0:0"
    assert record.payload["category"] == "diversification"


def test_k_and_dimension_guards(qdrant, vectors):
    assert qdrant.search(vectors[0], 0) == []
    assert len(qdrant.search(vectors[0], 99)) == len(CHUNKS)
    with pytest.raises(ValueError):
        qdrant.search(np.ones(3, dtype=np.float32), 2)


def test_rebuild_is_idempotent(vectors):
    client = QdrantClient(":memory:")
    QdrantIndex.build(vectors, client, "c", CHUNKS)
    assert len(QdrantIndex.build(vectors[:2], client, "c", CHUNKS[:2])) == 2


def test_connect_missing_collection_is_a_clear_error():
    with pytest.raises(ValueError, match="qdrant_index"):
        QdrantIndex.connect(QdrantClient(":memory:"), "nope")


def test_rejects_misaligned_chunks(vectors):
    with pytest.raises(ValueError, match="row-aligned"):
        QdrantIndex.build(vectors, QdrantClient(":memory:"), "c", CHUNKS[:2])


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_retriever_gives_same_results_on_both_backends(qdrant, vectors, fake_embedder, mode):
    def make(dense):
        texts = [c.embedding_text for c in CHUNKS]
        return Retriever(ChunkStore(CHUNKS), dense, SparseIndex.build(texts), fake_embedder,
                         top_k=3, candidate_k=4, min_score=0.3)  # fmt: skip

    on_faiss = make(DenseIndex.build(vectors)).retrieve("diversification risk", mode=mode)
    on_qdrant = make(qdrant).retrieve("diversification risk", mode=mode)
    assert [e.chunk.chunk_id for e in on_qdrant.evidence] == [
        e.chunk.chunk_id for e in on_faiss.evidence
    ]
    assert on_qdrant.top_dense_score == pytest.approx(on_faiss.top_dense_score, abs=1e-5)
