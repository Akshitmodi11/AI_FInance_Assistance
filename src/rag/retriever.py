"""Retriever: the one entry point agents call.

    query ─┬─▶ dense  top candidate_k ─┐
           └─▶ sparse top candidate_k ─┴─▶ RRF ─▶ category filter ─▶ top k ─▶ RetrievalResult

The confidence gate: RRF scores only encode *position*, so even nonsense queries get a
"top" result with a normal-looking RRF score. Confidence therefore comes from the best
**dense cosine** score among the returned evidence — if it is below ``min_score``, the
agent must say it doesn't have reliable information.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from src.core.config import Settings
from src.rag.dense_index import DenseIndex
from src.rag.embeddings import Embedder, SentenceTransformerEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.models import Evidence, RetrievalResult
from src.rag.sparse_index import SparseIndex
from src.rag.store import ChunkStore

Mode = Literal["dense", "sparse", "hybrid"]


class VectorIndex(Protocol):
    """What the retriever needs from a dense backend — FAISS ``DenseIndex`` or ``QdrantIndex``."""

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]: ...
    def vectors(self, rows: list[int]) -> np.ndarray: ...
    def __len__(self) -> int: ...


def load_vector_index(settings: Settings) -> VectorIndex:
    """FAISS files from ``index_dir``, or the Qdrant collection, per ``rag.vector_store``."""
    rag = settings.rag
    if rag.vector_store == "qdrant":
        from src.rag.qdrant_index import QdrantIndex, make_client  # optional dependency

        return QdrantIndex.connect(make_client(settings), rag.qdrant_collection)
    return DenseIndex.load(settings.resolve(rag.index_dir))


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        dense: VectorIndex,
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
        """Load the indexes built by ``python -m src.rag.ingest`` (dense backend per config)."""
        rag = settings.rag
        index_dir = settings.resolve(rag.index_dir)
        store = ChunkStore.load(index_dir)
        dense = load_vector_index(settings)
        if len(dense) != len(store):  # row numbers must line up across all three
            raise ValueError(
                f"{rag.vector_store} has {len(dense)} vectors but the chunk store has {len(store)}"
                " — re-run ingest (and `python -m src.rag.qdrant_index` for Qdrant)"
            )
        return cls(
            store=store,
            dense=dense,
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
        * hybrid: fuse the two candidate lists with RRF; ``score = rrf``,
          ``dense_score = cosine`` (compute it if the chunk only came from BM25).
        * ``category``: drop evidence from other categories. Filter *before* cutting to k,
          and fetch ``candidate_k`` candidates so filtering doesn't leave you with nothing.
        * ``confident = top_dense_score >= min_score`` where ``top_dense_score`` is the max
          ``dense_score`` among returned evidence; no evidence → ``confident=False``.
        * Empty / whitespace query → empty, non-confident result (don't call the indexes).
        """
        mode = mode or self.mode
        k = k or self.top_k
        if not query.strip():
            return RetrievalResult(mode=mode, confident=False)

        query_vec = self.embedder.embed([query])[0]

        # 1. Candidates from each ranker as {row: score}. Dicts keep insertion order → best first.
        dense_hits = (
            dict(self.dense.search(query_vec, self.candidate_k)) if mode != "sparse" else {}
        )
        sparse_hits = dict(self.sparse.search(query, self.candidate_k)) if mode != "dense" else {}

        # 2. Order the candidates.
        if mode == "dense":
            ranked = list(dense_hits.items())
        elif mode == "sparse":
            ranked = list(sparse_hits.items())
        else:
            ranked = reciprocal_rank_fusion([list(dense_hits), list(sparse_hits)], k=self.rrf_k)

        # 3. Category filter BEFORE cutting to k.
        if category:
            ranked = [(row, s) for row, s in ranked if self.store[row].meta.category == category]
        ranked = ranked[:k]

        # 4. Evidence — dense_score must be filled even for BM25-only hits (gate needs it).
        missing = [row for row, _ in ranked if row not in dense_hits]
        cosines = {**dense_hits, **self._cosines(query_vec, missing)}  # one batched lookup
        evidence = [
            Evidence(chunk=self.store[row], score=score, dense_score=cosines[row])
            for row, score in ranked
        ]

        # 5. Confidence gate on the best dense cosine (RRF scores are not calibrated).
        top = max((e.dense_score for e in evidence), default=None)
        return RetrievalResult(
            evidence=evidence,
            mode=mode,
            confident=top is not None and top >= self.min_score,
            top_dense_score=top,
        )

    def _cosines(self, query_vec: np.ndarray, rows: list[int]) -> dict[int, float]:
        """Cosine between the query and indexed chunks, without re-embedding the chunks.

        Trade-off: both backends keep the (normalized) vectors, so this is a lookup — an array
        read for FAISS, **one batched** round trip for Qdrant (not one per row: that N+1 pattern
        made sparse mode 5x slower on Qdrant Cloud). Re-embedding ``chunk.embedding_text``
        instead would cost a model call per BM25-only hit (~10 ms each on CPU).
        """
        if not rows:
            return {}
        query = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if not norm:
            return dict.fromkeys(rows, 0.0)
        scores = self.dense.vectors(rows) @ query / norm
        return {row: float(s) for row, s in zip(rows, scores, strict=True)}
