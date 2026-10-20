"""Retriever: the one entry point agents call.

    query ─┬─▶ dense  top candidate_k ─┐
           └─▶ sparse top candidate_k ─┴─▶ RRF ─▶ category filter ─▶ top k ─▶ RetrievalResult

The confidence gate: RRF scores only encode *position*, so even nonsense queries get a
"top" result with a normal-looking RRF score. Confidence therefore comes from the best
**dense cosine** score among the returned evidence — if it is below ``min_score``, the
agent must say it doesn't have reliable information.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.core.config import Settings
from src.rag.dense_index import DenseIndex
from src.rag.embeddings import Embedder, SentenceTransformerEmbedder
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.models import Evidence, RetrievalResult
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
        evidence = []
        for row, score in ranked:
            dense_score = dense_hits.get(row)
            if dense_score is None:
                dense_score = self._cosine(query_vec, row)
            evidence.append(Evidence(chunk=self.store[row], score=score, dense_score=dense_score))

        # 5. Confidence gate on the best dense cosine (RRF scores are not calibrated).
        top = max((e.dense_score for e in evidence), default=None)
        return RetrievalResult(
            evidence=evidence,
            mode=mode,
            confident=top is not None and top >= self.min_score,
            top_dense_score=top,
        )

    def _cosine(self, query_vec: np.ndarray, row: int) -> float:
        """Cosine between the query and an indexed chunk, without re-embedding the chunk.

        Trade-off: ``IndexFlatIP`` keeps the (normalized) vectors, so ``reconstruct`` is a cheap
        array lookup. Re-embedding ``chunk.embedding_text`` instead would cost one model call
        per BM25-only hit (~10 ms each on CPU). An approximate index (IVF/PQ) would need
        ``make_direct_map()`` or a separate vector cache for this to work.
        """
        chunk_vec = self.dense.index.reconstruct(int(row))
        query = np.asarray(query_vec, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        return float(chunk_vec @ query / norm) if norm else 0.0
