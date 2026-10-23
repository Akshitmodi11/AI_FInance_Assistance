# Vector store: FAISS and Qdrant

Finnie's dense retrieval runs on either backend, chosen in `config.yaml`:

```yaml
rag:
  vector_store: faiss            # faiss | qdrant
  qdrant_collection: finnie_chunks
  qdrant_local_path: data/qdrant # only used when QDRANT_URL is empty
```

Both implement the same small interface the retriever depends on (`VectorIndex` in
`src/rag/retriever.py`): `search(query_vector, k) -> [(row, cosine)]`, `vectors(rows)` and `len()`.
Row numbers are shared with the chunk store and the BM25 index, so sparse search, RRF fusion, the
category filter and the confidence gate are identical on both. The retriever refuses to start if the
vector count and the chunk count differ.

| | FAISS (`src/rag/dense_index.py`) | Qdrant (`src/rag/qdrant_index.py`) |
|---|---|---|
| Runs | in-process, 3 files in `data/index/` | Qdrant Cloud, or embedded local mode (`data/qdrant/`) |
| Search | `IndexFlatIP`, exact | `exact=True` (brute force) — matches FAISS; HNSW not needed at ~600 points |
| Setup | none | `.env`: `QDRANT_URL`, `QDRANT_API_KEY` (or nothing for local mode) |
| Payload | none (metadata in `chunks.json`) | title, section, page, category, URL, text on every point — browsable in the dashboard |
| Updates | full rebuild (~10 s) | full rebuild today; per-point upserts possible |
| Best for | this project's size, offline dev, CI | shared/deployed index, live updates, server-side filters, scale |

## Setup

**Local mode (no account):** leave `QDRANT_URL` empty and set `vector_store: qdrant`.

**Qdrant Cloud:**
1. cloud.qdrant.io → create a Free cluster → copy the cluster URL and create an API key.
2. Put them in `.env` (never committed):
   ```
   QDRANT_URL=https://<cluster-id>.<region>.cloud.qdrant.io
   QDRANT_API_KEY=<key>
   ```
3. Check the connection (never prints the key): `python -m scripts.check_qdrant`

## Build / refresh the collection

```
python -m src.rag.ingest          # load → chunk → embed → FAISS + BM25 + chunks.json
python -m src.rag.qdrant_index    # upload those exact vectors + payloads to Qdrant
```

Vectors are copied from the FAISS index instead of re-embedded, so both stores hold identical vectors
and any benchmark difference comes from the store alone. The upload recreates the collection
(idempotent) and ends with a self-search check (a stored vector must find itself with cosine 1.0).
Current run: 590 points uploaded to Qdrant Cloud (AWS us-east-1) in 4.8 s.

## Evaluate either store

```
python -m tests.eval.run_retrieval_eval --store faiss
python -m tests.eval.run_retrieval_eval --store qdrant
```

## Results (43 questions, k = 5) — full table in `docs/benchmarks.md`

| Store | hybrid R@5 | hybrid MRR | dense p50 | sparse p50 | hybrid p50 |
|---|---|---|---|---|---|
| FAISS (local) | 0.92 | 0.73 | 4 ms | 4 ms | 5 ms |
| Qdrant Cloud (us-east-1, from India) | 0.92 | 0.73 | 108 ms | 100 ms | 111 ms |

- **Quality is identical** in every mode and question type — expected, since the vectors are the same
  and both searches are exact. This confirms the backend swap is correct.
- **Latency is network round trips.** Qdrant's own search is fast; ~100 ms per call is the distance from
  India to us-east-1. A region closer to the users (or local mode) would cut most of it.
- **N+1 fix found by this benchmark:** the confidence gate needs a dense cosine for BM25-only hits.
  Fetching those vectors one call per row made sparse mode 488 ms on Qdrant Cloud; fetching them in one
  batched `retrieve` brought it to 100 ms (4.9x) with no change on FAISS.

## Why FAISS stays the default

At ~600 chunks exact FAISS search takes milliseconds in-process with zero infrastructure, works offline
and in CI, and the index rebuilds in seconds. Qdrant becomes the better choice when the index must be
shared by several app instances, updated per document while the app runs, filtered server-side on
metadata, or grown past what fits comfortably in memory (≈1M+ vectors).

## Tests

`tests/unit/test_qdrant_index.py` runs against Qdrant's in-memory mode (no network): same scores as
FAISS, vector round-trip, payload contents, k/dimension guards, idempotent rebuild, clear error for a
missing collection, misaligned chunks rejected, and identical retriever results on both backends in all
three modes.
