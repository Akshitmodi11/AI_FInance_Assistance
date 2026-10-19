"""Build all RAG indexes from data/kb — TODO(you).

Usage:  python -m src.rag.ingest

Idempotent: always rebuilds from scratch into ``rag.index_dir``, so re-running after adding
articles is safe. Writes three files that must stay row-aligned:
``chunks.json`` (ChunkStore), ``dense.faiss`` (DenseIndex), ``sparse_tokens.json`` (SparseIndex).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.core.config import Settings, get_settings
from src.rag.embeddings import Embedder, SentenceTransformerEmbedder

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    documents: int
    chunks: int
    by_file_type: dict[str, int]  # chunks per file type, e.g. {"md": 120, "pdf": 340}
    by_category: dict[str, int]  # chunks per category — watch for corpus imbalance!
    seconds: float


def build_index(settings: Settings, embedder: Embedder) -> IngestStats:
    """Load → chunk → embed → build dense + sparse → save. Return stats.

    Steps:
    1. ``load_corpus(settings)``                                   (src.rag.loaders)
    2. ``chunk_documents(docs, rag.chunk_size, rag.chunk_overlap)`` (src.rag.chunking)
    3. ``embedder.embed([c.embedding_text for c in chunks])``      — use embedding_text, not text
    4. ``DenseIndex.build(vectors)`` and ``SparseIndex.build([c.embedding_text ...])``
    5. Save store + both indexes into ``settings.resolve(rag.index_dir)``
    6. Return IngestStats (time it with ``time.perf_counter()``) — these numbers go straight
       into docs/benchmarks.md.
    Raise ``ValueError`` if the corpus produces zero chunks.
    """
    raise NotImplementedError("TODO(you): implement build_index")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    start = time.perf_counter()
    stats = build_index(settings, SentenceTransformerEmbedder(settings.rag.embedding_model))
    log.info(
        "Indexed %d documents → %d chunks in %.1fs (wall %.1fs)",
        stats.documents,
        stats.chunks,
        stats.seconds,
        time.perf_counter() - start,
    )
    log.info("Chunks by file type: %s", stats.by_file_type)
    log.info("Chunks by category:  %s", stats.by_category)


if __name__ == "__main__":
    main()
