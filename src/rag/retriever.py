"""Retriever: the one entry point agents call — TODO(you).

    query ─┬─▶ dense  top candidate_k ─┐
           └─▶ sparse top candidate_k ─┴─▶ RRF ─▶ category filter ─▶ top k ─▶ RetrievalResult

The confidence gate: RRF scores only encode *position*, so even nonsense queries get a
"top" result with a normal-looking RRF score. Confidence therefore comes from the best
**dense cosine** score among the returned evidence — if it is below ``min_score``, the
agent must say it doesn't have reliable information.
"""

from __future__ import annotations

from typing import Literal

from src.core.config import Settings
from src.rag.dense_index import DenseIndex
from src.rag.embeddings import Embedder, SentenceTransformerEmbedder
from src.rag.models import RetrievalResult
from src.rag.sparse_index import SparseIndex
from src.rag.store import ChunkStore

Mode = Literal["dense", "sparse", "hybrid"]


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        dense: DenseIndex,
        sparse: SparseIndex,
        embedder: Embedder,
        *,
        mode: Mode = "hybrid",
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
        min_score: float = 0.35,
    ):
        self.store = store
        self.dense = dense
        self.sparse = sparse
        self.embedder = embedder
        self.mode = mode
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.min_score = min_score

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder | None = None) -> Retriever:
        """Load the indexes built by ``python -m src.rag.ingest``."""
        rag = settings.rag
        index_dir = settings.resolve(rag.index_dir)
        return cls(
            store=ChunkStore.load(index_dir),
            dense=DenseIndex.load(index_dir),
            sparse=SparseIndex.load(index_dir),
            embedder=embedder or SentenceTransformerEmbedder(rag.embedding_model),
            mode=rag.retrieval_mode,
            top_k=rag.top_k,
            candidate_k=rag.candidate_k,
            rrf_k=rag.rrf_k,
            min_score=rag.min_score,
        )

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        category: str | None = None,
        mode: Mode | None = None,
    ) -> RetrievalResult:
        """Return the top ``k`` Evidence for ``query``.

        Requirements (see tests/unit/test_retriever.py):
        * ``mode``/``k`` default to the instance settings.
        * dense:  order by cosine; ``Evidence.score = dense_score = cosine``.
        * sparse: order by BM25;   ``Evidence.score = bm25``; still fill ``dense_score``
          for the returned chunks so the confidence gate works in every mode.
          (Hint: the query vector · ``embedder.embed([chunk.embedding_text])`` — or cache
          the chunk vectors; your call. Note the trade-off in a comment.)
        * hybrid: fuse the two candidate lists with RRF; ``score = rrf``,
          ``dense_score = cosine`` (compute it if the chunk only came from BM25).
        * ``category``: drop evidence from other categories. Filter *before* cutting to k,
          and fetch ``candidate_k`` candidates so filtering doesn't leave you with nothing.
        * ``confident = top_dense_score >= min_score`` where ``top_dense_score`` is the max
          ``dense_score`` among returned evidence; no evidence → ``confident=False``.
        * Empty / whitespace query → empty, non-confident result (don't call the indexes).
        """
        raise NotImplementedError("TODO(you): implement Retriever.retrieve")
