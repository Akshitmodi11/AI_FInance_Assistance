# Benchmarks

Summary (updated 2026-11-14):

| Area | Result |
|---|---|
| Retrieval (59 Q, k=5) | **rerank Recall@5 0.96, MRR 0.90**, out-of-scope refusal **7/7**, ~360 ms/query · hybrid 0.92 / 0.79, ~4 ms |
| Reranker (Exp. 4) | cross-encoder over 20 hybrid candidates: **+4 pts recall, +11 pts MRR**, no extra false refusals |
| Corpus balance (Exp. 3) | 14 new articles: new-topic questions **0.00 → 1.00** recall; original 43: 0.92 → 0.89 |
| Routing (60 Q, 8 multi-intent) | LLM router **95% exact / 98% primary**; keyword fallback 88% / 92% |
| Safety (18 prompts) | guardrail category **18/18**, advice answers reframed **9/9**, illegal refused **4/4** |
| Latency (end-to-end) | p50 **3.1–5.4 s** per intent (news 9 s — web search) |
| Cost | **$0.004–0.006 per question** (gpt-4o-mini routing + gpt-4o answers) |
| Ingest | 192 documents → 668 chunks in **15 s** (incl. model load) |
| Market data | cold 0.7–1.1 s → cached **<1 ms** (30-min disk cache) |
| Tests | **210 tests, 91% coverage**, enforced ≥80% in CI |

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

## Experiment 3 — Corpus balance: 14 plain-language articles (2026-11-05)

**Why:** the knowledge base covered only 3 of the 6 categories in `config.yaml` (investing, retirement_tax,
diversification), and 544 of 590 chunks were IRS PDF pages. Questions about budgeting, emergency funds,
compounding or market basics had nothing to retrieve.

**Change:** 14 Markdown articles written in plain language, each citing a public reference page
(SEC Investor.gov, CFPB, FINRA): 4 `basics`, 4 `goals_budgeting`, 4 `markets`, 1 `investing`,
1 `diversification`. Index: 590 → **668 chunks** (PDF share 92% → 81%). 16 new labelled questions
(ids `k01`–`k16`) added to `retrieval_set.jsonl`, written **before** re-indexing and measured on both
indexes. Thresholds and retrieval settings unchanged.

| Question set | Mode | Before R@5 | After R@5 | Before MRR | After MRR |
|---|---|---|---|---|---|
| New topics (16) | dense | 0.00 | **1.00** | 0.00 | 0.90 |
| New topics (16) | sparse | 0.00 | **0.94** | 0.00 | 0.85 |
| New topics (16) | hybrid | 0.00 | **1.00** | 0.00 | **0.93** |
| Original (43) | dense | 0.86 | 0.86 | 0.71 | 0.71 |
| Original (43) | sparse | 0.78 | 0.78 | 0.67 | 0.64 |
| Original (43) | hybrid | 0.92 | 0.89 | 0.73 | 0.72 |
| All (59) | hybrid | 0.63 | **0.92** | 0.51 | **0.79** |

Out-of-scope refusal (dense-cosine gate, `min_score = 0.35`): 7/7 before → **6/7** after.

**Findings**
- **The old corpus answered confidently from the wrong documents.** Before the change, 11 of the 16
  new-topic questions passed the "I don't know" gate on unrelated chunks, e.g. *"What is the 50/30/20
  rule?"* → IRS Pub 590-A. Missing content is a correctness problem, not just a coverage gap.
- **One original question regressed:** q001 *"Why shouldn't I put all my money in one stock?"* now
  retrieves the new *How Stock Markets Work* article and Investor.gov *Stocks* page instead of the
  diversification article — the word "stock" pulls toward stock-market content. Labels were not
  widened after the fact.
- **One off-topic question now passes the gate:** q901 *"best options strategy for earnings week"*
  scores 0.403 against the new market articles. The lowest *answerable* question scores 0.412 (p12),
  so a threshold separating them would be fitted to a single point; kept 0.35. The answer prompt's
  "if the facts don't answer the question, say so" rule is the second line of defence.
- **Dense gate vs exact terms:** k06 *"50/30/20 rule"* is retrieved correctly (hybrid rank 1) but its
  dense cosine is 0.343, just under the gate — embeddings handle number patterns poorly. A candidate
  fix is letting a strong BM25 hit also count as confident.

## Experiment 4 — Cross-encoder reranker (2026-11-14)

**Why:** hybrid retrieval ranks with two *bi-encoder*-style signals (embedding cosine, BM25) that score query
and chunk separately. A **cross-encoder** reads the (query, chunk) pair together, so it can tell whether the
chunk actually answers the question — too slow for the whole corpus, cheap on 20 candidates.

**Change:** new `rerank` mode = hybrid candidates (`candidate_k = 20`, category filter first) →
`cross-encoder/ms-marco-MiniLM-L-6-v2` reorders them → top 5. Chunks are scored with their contextual header
(title — section). Confidence gate unchanged (dense cosine ≥ 0.35; cross-encoder logits are not calibrated).
`src/rag/reranker.py`, `config.yaml: rag.retrieval_mode / rerank_model`. Same 59 questions, same index.

| Mode | Recall@5 | MRR | No-answer acc. | R@5 concept | R@5 exact_term | R@5 paraphrase | answerable refused | p50 ms |
|---|---|---|---|---|---|---|---|---|
| dense | 0.90 | 0.77 | 0.86 | 0.92 | 1.00 | 0.85 | 1/52 | 4 |
| sparse | 0.83 | 0.70 | 1.00 | 0.92 | 1.00 | 0.69 | 2/52 | 5 |
| hybrid (RRF) | 0.92 | 0.79 | 0.86 | 1.00 | 1.00 | 0.85 | 1/52 | 4 |
| **rerank** | **0.96** | **0.90** | **1.00** | 1.00 | 1.00 | **0.92** | 1/52 | 358 (p95 587) |

**Findings**
- **MRR +11 points** (0.79 → 0.90): the right chunk moves to the top, which is what the answer LLM reads first.
  Recall +4 points; the gain is all on **paraphrases** (0.85 → 0.92), where keyword and embedding overlap is weak.
- Fixed three hybrid misses — p02 (spousal IRA), p09 (HSA catch-up), q001 (the Experiment 3 regression) — and
  introduced one, p03 (excess contributions → retrieves the filing-deadline page). p04 (60-day rollover) misses in every mode.
- **Off-topic refusal 6/7 → 7/7, as a side effect:** the gate scores the evidence that is *returned*; for q901
  the reranked top 5 have lower cosines than the RRF top 5. No answerable question lost confidence (still only k06).
- **Cost:** ~350 ms on CPU per query (20 pairs, MiniLM-L6), vs a 3–5 s end-to-end answer — under 10%.
  No API cost; the model (~90 MB) is downloaded once and baked into the Docker image.
- **Decision: `rerank` is the default.** `retrieval_mode: hybrid` in `config.yaml` switches it off.

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
- Confidence gate: also accept a strong BM25 hit (fixes k06-style exact-term refusals); re-check q901.
- Chunk size sweep: 400 / 800 / 1200.
- Fixed-size chunking baseline and "no contextual header" ablation (see `docs/chunking_strategy.md`).
- Stretch: Qdrant native hybrid (server-side dense + sparse fusion).
- Reranker: larger cross-encoder (e.g. `BAAI/bge-reranker-base`) vs latency; rerank 10 vs 20 vs 40 candidates.
