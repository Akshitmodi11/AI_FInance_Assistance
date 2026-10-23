# Benchmarks

## Retrieval — dense vs sparse vs hybrid (baseline, 2026-10-22)

**Setup:** 590 chunks (IRS Pub 590-A + 969, 3 Investor.gov pages, 1 Markdown article),
`all-MiniLM-L6-v2` embeddings, chunk 800 / overlap 120, `k = 5`, `candidate_k = 20`, RRF `k = 60`,
confidence threshold `min_score = 0.35`. 43 hand-checked questions in `tests/eval/retrieval_set.jsonl`
(10 exact_term, 18 paraphrase, 8 concept, 7 out_of_scope). Labels are **page/section-level**
(e.g. `p590a:p40`), so a large PDF doesn't score a hit by accident.

Reproduce: `python -m src.rag.ingest` then `python -m tests.eval.run_retrieval_eval --show-misses`.

| Mode | Recall@5 | MRR | No-answer acc. | R@5 concept | R@5 exact_term | R@5 paraphrase | p50 ms |
|---|---|---|---|---|---|---|---|
| dense | 0.86 | 0.71 | 1.00 | 0.88 | 1.00 | 0.78 | 4 |
| sparse | 0.78 | 0.67 | 1.00 | 0.88 | 1.00 | 0.61 | 5 |
| **hybrid (RRF)** | **0.92** | **0.73** | 1.00 | **1.00** | 1.00 | **0.83** | 4 |

**Findings**
- Hybrid is best overall: +6 points recall over dense, +14 over sparse.
- Sparse (BM25) is weakest on **paraphrases** (0.61): "My wife doesn't have a job" shares no keywords with
  "Spousal IRA". This is the gap dense embeddings close.
- Exact-term questions (form numbers, named limits) score 1.00 in every mode — the contextual chunk header
  (title + section) gives the dense model the form names too.
- All 7 out-of-scope questions are refused (dense-cosine confidence gate).
- Remaining hybrid misses: p02 (spousal IRA, phrased with no shared terms), p04 (60-day rollover — retrieves
  the inherited-IRA page instead), p09 (age-55 HSA catch-up — retrieves the base-limit page).

## Experiment 1 — PDF section rule (2026-10-22)

Each PDF page gets a `section` from the PDF bookmarks, which goes into the contextual header. The current rule
labels a page with the **last** heading that starts on or before it, so a page whose final heading starts near
the bottom is labelled by that heading. Two alternatives, each with a full rebuild + eval:

| Rule | dense R@5 | sparse R@5 | hybrid R@5 | hybrid MRR | hybrid misses |
|---|---|---|---|---|---|
| last heading on/before page (current) | 0.86 | 0.78 | **0.92** | 0.73 | p02, p04, p09 |
| heading in effect at top of page | 0.89 | 0.75 | **0.92** | 0.74 | p02, p09, p12 |
| all headings touching the page | 0.89 | 0.81 | **0.92** | 0.72 | p07, p09, p12 |

**Decision: keep the current rule.** Hybrid recall is identical; the variants only change *which* 3 questions
miss. With 36 answerable questions, one question ≈ 0.03 — these differences are within noise.

## Experiment 2 — FAISS vs Qdrant Cloud (2026-10-22)

Same 590 vectors (copied from FAISS, not re-embedded), same 43 questions, exact search on both.
Qdrant Cloud cluster in AWS us-east-1, queried from India. Setup and commands: `docs/vector_store.md`.

| Store | Mode | Recall@5 | MRR | No-answer acc. | p50 ms |
|---|---|---|---|---|---|
| FAISS | dense | 0.86 | 0.71 | 1.00 | 4 |
| FAISS | sparse | 0.78 | 0.67 | 1.00 | 4 |
| FAISS | hybrid | 0.92 | 0.73 | 1.00 | 5 |
| Qdrant Cloud | dense | 0.86 | 0.71 | 1.00 | 108 |
| Qdrant Cloud | sparse | 0.78 | 0.67 | 1.00 | 100 |
| Qdrant Cloud | hybrid | 0.92 | 0.73 | 1.00 | 111 |

- **Quality identical** in every mode and question type → the backend swap is correct.
- **Latency = network round trips** (~100 ms India → us-east-1).
- **N+1 bug found and fixed:** sparse mode first measured **488 ms** on Qdrant, because the confidence
  gate fetched each BM25-only hit's vector in a separate call. Batching into one `retrieve` → **100 ms**
  (4.9x faster); FAISS unchanged.
- Decision: FAISS stays the default for this corpus size; Qdrant is a config switch away.

## Next experiments
- Corpus balance: add plain-language Markdown articles, re-run (before/after).
- Chunk size sweep: 400 / 800 / 1200.
- Fixed-size chunking baseline and "no contextual header" ablation (see `docs/chunking_strategy.md`).
- Stretch: Qdrant native hybrid (server-side dense + sparse fusion); cross-encoder reranker as a 4th row.
