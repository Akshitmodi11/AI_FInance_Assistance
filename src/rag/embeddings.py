"""Text → vector.

An embedding model maps text to a point in a 384-dimensional space where texts with similar
*meaning* land close together. We L2-normalize every vector (length 1), which makes
dot product == cosine similarity — that's why the dense index can use FAISS ``IndexFlatIP``
(inner product).

``Embedder`` is a Protocol so tests can inject a tiny fake instead of downloading the model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a float32 array of shape ``(len(texts), dim)`` with unit-length rows."""
        ...


class SentenceTransformerEmbedder:
    """Local, free embeddings via sentence-transformers. Model loads lazily on first use."""

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import (torch)

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)
