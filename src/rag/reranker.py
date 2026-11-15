"""Second-stage reranking with a cross-encoder.

The embedder is a **bi-encoder**: query and chunk are embedded separately, and similarity is a
dot product. That is fast enough to search every chunk, but the model never sees the query and
the chunk together. A **cross-encoder** reads ``(query, chunk)`` as one input and outputs a
relevance score, so it can weigh which words in the chunk actually answer the question. It is too
slow to run on the whole corpus, so it only reorders the ~20 hybrid candidates (retrieve →
rerank).

Cross-encoder scores are unbounded logits, not cosines, so the confidence gate still uses the
dense cosine (see ``retriever.py``).

``Reranker`` is a Protocol so tests can inject a fake instead of downloading the model.
"""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each text to ``query`` — higher is more relevant."""
        ...


class CrossEncoderReranker:
    """Local cross-encoder via sentence-transformers. Model loads lazily on first use."""

    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # heavy import (torch)

            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
