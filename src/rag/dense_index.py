"""Dense (semantic) index over chunk embeddings.

Concepts this teaches:
* FAISS stores vectors and finds nearest neighbours. ``IndexFlatIP`` = exact, brute-force
  inner-product search. With < 100k vectors it's fast (milliseconds) and has no accuracy
  loss; approximate indexes (IVF, HNSW) only pay off at millions of vectors. Write that
  trade-off in the design doc.
* Vectors are normalized, so inner product = cosine similarity in [-1, 1].
* FAISS only knows row numbers 0..n-1. Row i ↔ chunks[i] is kept by the ChunkStore.

API: ``faiss.IndexFlatIP(dim)``, ``index.add(vectors)``, ``scores, ids = index.search(q, k)``
(``q`` must be 2-D: shape (1, dim)), ``faiss.write_index`` / ``faiss.read_index``.
FAISS returns id ``-1`` when k > number of vectors — filter those out.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

INDEX_FILE = "dense.faiss"


class DenseIndex:
    def __init__(self, index=None):
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> DenseIndex:
        """Create an ``IndexFlatIP`` sized to ``vectors.shape[1]`` and add all vectors.

        Vectors are L2-normalized on a copy, so inner product = cosine even if the
        embedder didn't normalize; the caller's array is left untouched.
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (number_of_vectors, dimension)")
        if vectors.shape[1] == 0:
            raise ValueError("vectors must have a non-zero dimension")

        vectors = np.ascontiguousarray(vectors.copy())
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        if len(vectors) > 0:
            index.add(vectors)

        return cls(index)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, cosine_score)`` pairs, best first.

        Accepts a 1-D ``(dim,)`` query and reshapes it. ``k`` is clamped to the index size.
        Raises ``ValueError`` if the query dimension doesn't match the index.
        """
        if self.index is None or k <= 0 or self.index.ntotal == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError("query dimension must match the indexed vector dimension")

        query = np.ascontiguousarray(query.copy())
        faiss.normalize_L2(query)

        search_k = min(k, self.index.ntotal)
        scores, ids = self.index.search(query, search_k)

        return [
            (int(row_id), float(score))
            for row_id, score in zip(ids[0], scores[0], strict=True)
            if row_id >= 0
        ]

    def save(self, directory: Path) -> None:
        if self.index is None:
            raise ValueError("cannot save an uninitialized index")

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))

    @classmethod
    def load(cls, directory: Path) -> DenseIndex:
        index = faiss.read_index(str(directory / INDEX_FILE))
        return cls(index)

    def __len__(self) -> int:
        return 0 if self.index is None else self.index.ntotal
