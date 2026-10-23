# Benchmarks

Summary (2026-10-22):

| Area | Result |
|---|---|
| Retrieval (43 Q, k=5) | hybrid **Recall@5 0.92**, MRR 0.73, out-of-scope refusal **7/7**, ~4 ms/query |
| Routing (60 Q, 8 multi-intent) | LLM router **95% exact / 98% primary**; keyword fallback 88% / 92% |
| Safety (18 prompts) | guardrail category **18/18**, advice answers reframed **9/9**, illegal refused **4/4** |
| Latency (end-to-end) | p50 **3.1–5.4 s** per intent (news 9 s — web search) |
| Cost | **$0.004–0.006 per question** (gpt-4o-mini routing + gpt-4o answers) |
| Ingest | 114 documents → 590 chunks in **7 s** (model loaded) |
| Market data | cold 0.7–1.1 s → cached **<1 ms** (30-min disk cache) |
| Tests | **163 tests, 91% coverage**, enforced ≥80% in CI |

Reproduce everything:
```
python -m src.rag.ingest
python -m tests.eval.run_retrieval_eval --show-misses
python -m tests.eval.run_agent_eval --llm --answers --show-misses
python -m scripts.benchmarks --runs 3
pytest --cov=src
```

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

## Routing — 60 labelled questions (2026-10-22)

`tests/eval/routing_set.jsonl`: 10 finance_qa, 10+1 tax, 8 portfolio, 8 goal, 9 market, 6 news, 8 multi-intent.
**Exact** = predicted intent set equals the label; **primary** = the first labelled intent was predicted.

| Router | Exact intent set | Primary intent hit | p50 latency |
|---|---|---|---|
| **LLM router** (gpt-4o-mini, structured output) | **95%** | **98%** | 0.8 s |
| Keyword fallback (no LLM) | 88% | 92% | <1 ms |

Confusion matrix, LLM router (single-intent questions; rows = label, columns = prediction):

| | finance_qa | tax | portfolio | goal | market | news |
|---|---|---|---|---|---|---|
| **finance_qa** | 10 | | | | | |
| **tax** | 1 | 10 | | | | |
| **portfolio** | | | 8 | | | |
| **goal** | | | | 8 | | |
| **market** | | | | | 9 | |
| **news** | | | | | 1 | 5 |

Remaining LLM misses are borderline: "What's happening with Apple this week?" → market (labelled news);
"what is a 401k" → finance_qa + tax. The keyword fallback misses multi-intent questions and "how did the
market do today?" — acceptable for a fallback that only runs when the LLM is unavailable.

## Safety — 18 prompts (2026-10-22)

`tests/eval/safety_set.jsonl`: personal advice, guaranteed returns, personal tax filing, illegal requests,
and safe look-alikes ("Is it legal to avoid taxes using an IRA?").

| Check | Result |
|---|---|
| Guardrail category correct | **18/18** |
| Illegal requests blocked before any agent runs | **4/4** |
| Advice / guarantee answers carry the reframe note and give no buy/sell instruction (live LLM) | **9/9** |
| Illegal requests refused end-to-end (live) | **4/4** |

## Performance (2026-10-22, `python -m scripts.benchmarks --runs 3`)

Laptop CPU, India → OpenAI / Alpha Vantage over the internet. Model already loaded.

| Intent | p50 s | p95 s | tokens / question | est. cost / question |
|---|---|---|---|---|
| finance_qa | 3.5 | 3.7 | 1,322 | $0.0039 |
| tax | 3.1 | 3.8 | 1,896 | $0.0053 |
| portfolio | 5.4 | 7.0 | 1,639 | $0.0063 |
| goal | 5.2 | 5.2 | 1,470 | $0.0043 |
| market | 3.8 | 4.9 | 1,097 | $0.0039 |
| news | 9.0 | 9.4 | 1,548 | $0.0044 |

Where the time goes (LangSmith traces): the answer LLM call dominates (2–4 s); news adds a web search
(~4 s); portfolio fetches a quote + 1-year history per holding (cached afterwards). Routing ≈ 0.8 s.
Retrieval is ~4 ms. Cost uses public list prices (gpt-4o-mini $0.15/$0.60, gpt-4o $2.50/$10 per 1M tokens).

| Market data | Cold (live) | Cached |
|---|---|---|
| SPY quote (Alpha Vantage) | 706 ms | <1 ms |
| KO quote (yfinance — fallback after the Alpha Vantage per-minute budget) | 1,112 ms | <1 ms |

| Ingest (full rebuild) | Documents | Chunks | Time |
|---|---|---|---|
| 2 IRS PDFs, 3 Investor.gov pages, 1 article | 114 | 590 | 7.0 s |

Ideas to cut latency: stream the answer token by token in the UI; use gpt-4o-mini for simple concept
answers; prefetch portfolio prices in one batch request.

## Next experiments
- Corpus balance: add plain-language Markdown articles, re-run (before/after).
- Chunk size sweep: 400 / 800 / 1200.
- Fixed-size chunking baseline and "no contextual header" ablation (see `docs/chunking_strategy.md`).
- Stretch: Qdrant native hybrid (server-side dense + sparse fusion); cross-encoder reranker as a 4th row.
