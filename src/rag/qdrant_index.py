"""Dense index backed by Qdrant — a drop-in alternative to the FAISS :class:`DenseIndex`.

Same interface the retriever relies on: ``search(query_vector, k) -> [(row, cosine)]``,
``vector(row)`` and ``len()``. Point ids are the ChunkStore row numbers, so BM25, the chunk
store and Qdrant stay row-aligned exactly like the FAISS files do.

Where it runs (see :func:`make_client`):
* ``QDRANT_URL`` + ``QDRANT_API_KEY`` set in ``.env`` → Qdrant Cloud.
* otherwise → embedded local mode in ``rag.qdrant_local_path`` (no server, no account).

Vectors are uploaded from the FAISS index rather than re-embedded, so both stores hold
*identical* vectors and the FAISS-vs-Qdrant benchmark compares only the stores.
Searches use ``exact=True`` (brute force) by default to match ``IndexFlatIP``; HNSW
(approximate) only pays off far beyond our ~600 points.

Upload / refresh the collection after ``python -m src.rag.ingest``:
    python -m src.rag.qdrant_index
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from src.core.config import Secrets, Settings, get_settings
from src.rag.models import Chunk

log = logging.getLogger(__name__)

UPLOAD_BATCH = 128


def make_client(settings: Settings, secrets: Secrets | None = None) -> QdrantClient:
    """Qdrant Cloud if ``QDRANT_URL`` is set, else embedded local mode on disk."""
    secrets = secrets or Secrets()
    if secrets.qdrant_url:
        key = secrets.qdrant_api_key.get_secret_value() if secrets.qdrant_api_key else None
        return QdrantClient(url=secrets.qdrant_url, api_key=key or None, timeout=30)
    return QdrantClient(path=str(settings.resolve(settings.rag.qdrant_local_path)))


def _payload(chunk: Chunk) -> dict:
    """Stored next to each vector — makes the collection browsable in the Qdrant dashboard
    and enables server-side filters (e.g. ``category``) later."""
    meta = chunk.meta
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": meta.title,
        "section": meta.section,
        "page": meta.page,
        "category": meta.category,
        "file_type": meta.file_type,
        "source": meta.source,
        "url": meta.url,
        "text": chunk.text,
    }


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str, *, exact: bool = True):
        self.client = client
        self.collection = collection
        self.exact = exact
        # Local/in-memory mode is always brute force and warns if given search params.
        is_local = (
            client.init_options.get("url") is None and client.init_options.get("host") is None
        )
        self._search_params = models.SearchParams(exact=True) if exact and not is_local else None
        info = client.get_collection(collection)
        self.dim: int = info.config.params.vectors.size
        self._size: int = client.count(collection, exact=True).count  # cached: no call per query

    @classmethod
    def connect(cls, client: QdrantClient, collection: str, *, exact: bool = True) -> QdrantIndex:
        """Open an existing collection, with a clear message if it was never uploaded."""
        if not client.collection_exists(collection):
            raise ValueError(
                f"Qdrant collection {collection!r} not found — run `python -m src.rag.qdrant_index`"
            )
        return cls(client, collection, exact=exact)

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        client: QdrantClient,
        collection: str,
        chunks: Sequence[Chunk] | None = None,
        *,
        exact: bool = True,
    ) -> QdrantIndex:
        """(Re)create ``collection`` and upload ``vectors`` with point id = row number."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("vectors must have shape (number_of_vectors, dimension)")
        if chunks is not None and len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors — not row-aligned")

        if client.collection_exists(collection):
            client.delete_collection(collection)  # idempotent rebuild, like ingest
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        for start in range(0, len(vectors), UPLOAD_BATCH):
            rows = range(start, min(start + UPLOAD_BATCH, len(vectors)))
            client.upsert(
                collection,
                points=[
                    models.PointStruct(
                        id=row,
                        vector=vectors[row].tolist(),
                        payload=_payload(chunks[row]) if chunks is not None else {},
                    )
                    for row in rows
                ],
                wait=True,
            )
        return cls(client, collection, exact=exact)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, cosine_score)`` pairs, best first."""
        if k <= 0 or self._size == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError("query dimension must match the indexed vector dimension")
        response = self.client.query_points(
            self.collection,
            query=query.tolist(),
            limit=k,
            search_params=self._search_params,
        )
        return [(int(point.id), float(point.score)) for point in response.points]

    def vector(self, row: int) -> np.ndarray:
        """The stored vector for ``row`` (Qdrant normalizes vectors in COSINE collections)."""
        return self.vectors([row])[0]

    def vectors(self, rows: list[int]) -> np.ndarray:
        """Stored vectors for several rows in **one** round trip, in the given order.

        Fetching them one by one was an N+1-query problem: sparse-mode eval latency on Qdrant
        Cloud went from 488 ms to ~2 round trips once batched (see docs/benchmarks.md).
        """
        if not rows:
            return np.empty((0, self.dim), dtype=np.float32)
        ids = [int(row) for row in rows]
        records = self.client.retrieve(self.collection, ids=ids, with_vectors=True)
        by_id = {int(r.id): r.vector for r in records}  # retrieve doesn't promise order
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise KeyError(missing)
        return np.asarray([by_id[i] for i in ids], dtype=np.float32)

    def __len__(self) -> int:
        return self._size


def sync_from_faiss(settings: Settings, client: QdrantClient | None = None) -> QdrantIndex:
    """Upload the vectors + chunks built by ``python -m src.rag.ingest`` into Qdrant."""
    from src.rag.dense_index import DenseIndex
    from src.rag.store import ChunkStore

    index_dir = settings.resolve(settings.rag.index_dir)
    store = ChunkStore.load(index_dir)
    dense = DenseIndex.load(index_dir)
    vectors = dense.index.reconstruct_n(0, len(dense))
    client = client or make_client(settings)
    return QdrantIndex.build(vectors, client, settings.rag.qdrant_collection, store.chunks)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    secrets = Secrets()
    where = f"Qdrant Cloud ({secrets.qdrant_url})" if secrets.qdrant_url else "Qdrant local mode"

    start = time.perf_counter()
    index = sync_from_faiss(settings, make_client(settings, secrets))
    seconds = time.perf_counter() - start

    # Sanity check: a stored vector must find itself first.
    top_row, top_score = index.search(index.vector(0), 1)[0]
    log.info("Uploaded %d points to %s, collection %r in %.1fs", len(index), where,
             index.collection, seconds)  # fmt: skip
    log.info("Self-search check: row 0 → row %d (cosine %.3f) %s", top_row, top_score,
             "✓" if top_row == 0 else "✗")  # fmt: skip


if __name__ == "__main__":
    main()
