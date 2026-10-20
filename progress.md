# Finnie — Progress Log

_Last updated: 2026-10-19_

## Where we are
**Week 1 — RAG layer.** Guided build: Claude scaffolds + reviews, I write the core logic.
Goal: finish earlier than the 4-week plan, in small daily steps.
Tests: **62 / 62 passing** ✅ (`pytest -q`). **RAG layer complete (2026-10-19).**

### My 9 RAG files
| # | File | Functions to write | Status | Tests |
|---|---|---|---|---|
| 1 | `src/rag/fusion.py` | `reciprocal_rank_fusion` | ✅ done | 3/3 |
| 2 | `src/rag/sparse_index.py` | `tokenize`, `SparseIndex` | ✅ done | 10/10 |
| 3 | `src/rag/dense_index.py` | `DenseIndex` | ✅ done | 3/3 |
| 4 | `src/rag/chunking.py` | `chunk_document` (+ 3 helpers) | ✅ done | 6/6 |
| 5 | `src/rag/loaders/pdf.py` | `load_pdf`, `strip_repeated_lines`, `clean_page_text`, `page_sections` | ✅ done | 8/8 |
| 6 | `src/rag/loaders/html.py` | `load_html` | ✅ done | 3/3 |
| 7 | `src/rag/loaders/docx.py` | `load_docx` | ✅ done | 2/2 |
| 8 | `src/rag/retriever.py` | `Retriever.retrieve` | ✅ done | 11/11 |
| 9 | `src/rag/ingest.py` | `build_index` | ✅ done | 1/1 |

Already passing (Claude's code): config 5, KB articles 2, Markdown loader + registry 5, eval metrics 3.

## Git
- `main` — only README. Merge into it only when CI is green.
- `dev-branch` — **current working branch**, pushed at `b7f1b7c`. CI fixed (`activate-environment: true`);
  lint ✅, tests ❌ (all 62 tests should pass once pushed).
- `corpus-profiling` — **merged into `dev-branch` (fast-forward) and deleted** on 2026-10-19. Its commits:
  - `6d065c0` corpus (5 docs + manifest), profiler, chunking strategy doc, loader specs + 5 tests
  - `3ff8a2a` chunking.py · `b913de5` pdf.py · `b7f1b7c` html.py
- Workflow from now on: work on `dev-branch` (or a short feature branch → merge back) → `main` only when CI is green.
- Cleanup still open: delete the local typo branch → `git branch -d dev-branc`

## Corpus (data/kb)
| File | Type | Size | Category |
|---|---|---|---|
| `p590a.pdf` | IRS Pub 590-A (IRAs), 2025 | 62 pages | retirement_tax |
| `p969.pdf` | IRS Pub 969 (HSAs), 2025 | 22 pages | retirement_tax |
| `investor-gov-stocks.html` | SEC Investor.gov | 6 sections | investing |
| `investor-gov-bonds.html` | SEC Investor.gov | 6 sections | investing |
| `investor-gov-mutual-funds-etfs.html` | SEC Investor.gov | 13 sections | investing |
| `articles/diversification-basics.md` | my article | 5 sections | diversification |
| ~~`p550.pdf`~~ | parked — 117 pages = 62% of corpus | | |

Folder layout (renamed 2026-10-19: `data/kb/documents/` → `data/kb/sources/`, config key `rag.sources_dir`):
- `data/kb/articles/` — my Markdown articles (metadata in front-matter) · `data/kb/sources/` — downloaded PDF/HTML/DOCX
  → both are **read by Finnie** (indexed + cited).
- `docs/` — project documentation for humans (design, chunking strategy, benchmarks) → **never indexed**.

`manifest.yaml` (top-level key `sources:`) = metadata (title, url, source, category, tax_year) for every file in `sources/`.
Ingestion fails on any file without an entry → every answer is citable.

## Corpus profiling (2026-10-18) — `python notebooks/profile_corpus.py .`
| Source | Units | Median chars | % > 800 | Paragraph breaks |
|---|---|---|---|---|
| Markdown | sections | 214 | 0% | yes |
| IRS PDFs | pages | ~5,200 | 99–100% | **no** |
| Investor.gov HTML | sections | 612–4,309 | 33–50% | yes |

Findings → what we did:
1. PDFs have no blank-line paragraphs → sentence splitting is the main path for PDFs.
2. PDF lines hard-wrapped + hyphenated (`retire-\nment`) → `clean_page_text` unwraps + de-hyphenates.
3. PDF header changes per page (`Page 10 of 62 Fileid: …`) → regex noise patterns in `strip_repeated_lines`.
4. Mutual Funds HTML uses `<h4>` → HTML splits on h2/h3/h4 (2 sections → 13).
5. Markdown sections already small → chunking leaves them alone.
6. Corpus imbalance (PDFs 99.8%) → parked p550; still need many more Markdown articles.
7. IRS PDFs have **bookmarks** → `page_sections` gives every PDF chunk a section header.

## Chunking strategy — DECIDED 2026-10-19 → `docs/chunking_strategy.md`
**Structure-aware recursive chunking with overlap and contextual headers.**
```
STRUCTURE (loaders)   → section / page; chunks never cross these
SIZE      (chunking)  → ≤ 800 chars: paragraphs → sentences → hard cut, greedy packing
OVERLAP   (chunking)  → 120 chars, snapped to a word boundary
CONTEXT   (models.py) → embedding_text = "Title — Section\n" + text
```
Why not the others: fixed-size = baseline only; token-based = still a blind cut; semantic / parent-child =
Week-4 stretch; LLM chunking = too costly.
**Verified with MY pdf.py + chunking.py on the real IRS PDFs:** p590a → 62 pages / 390 chunks / 30 sections,
p969 → 22 / 154 / 16; 0 junk chunks; max chunk 920 (= 800 + 120 overlap).
Benchmark later: fixed-size baseline vs ours vs no-headers vs 400 vs 1200 (recall@5, MRR) → README.

## First live retrieval (2026-10-19, real index, `k=3`)
| Query | dense | sparse | hybrid | #1 citation (hybrid) |
|---|---|---|---|---|
| How much can I contribute to an IRA in 2025? | 0.73 ✅ | 0.72 ✅ | 0.72 ✅ | IRS Pub 590-A, p. 5 |
| What is diversification? | 0.80 ✅ | 0.80 ✅ | 0.80 ✅ | What Is Diversification? (my article) |
| Form 8606 | 0.52 ✅ | 0.52 ✅ | 0.52 ✅ | IRS Pub 590-A, p. 16 |
| best pizza recipe in Naples | 0.17 ❌ | 0.00 ❌ | 0.17 ❌ | → "I don't know" gate works |
(numbers = top dense cosine; ✅ = confident ≥ 0.35)

Rebuild the index any time: `python -m src.rag.ingest` (set `HF_HUB_OFFLINE=1` after the first run to skip HF checks).

## Next session — start here
1. Commit + push everything on `dev-branch`; check CI is fully green (62/62) → then merge `dev-branch` into `main`.
2. **Corpus balance** — chunks are 92% PDF / 7% HTML / 1% Markdown. Write ~10 Markdown articles (2 per category:
   basics, investing, diversification, retirement_tax, goals_budgeting, markets), re-run ingest.
3. **Eval set** — label ~40 questions in `tests/eval/retrieval_set.jsonl` (exact_term / paraphrase / concept /
   out_of_scope), then `python -m tests.eval.run_retrieval_eval --show-misses` → dense vs sparse vs hybrid table → README.
4. Then Week 2: LangGraph + the six agents (Claude scaffolds `core/schemas.py` + `core/llm.py` first).

## Done log
- 2026-10-18: fusion, sparse_index, dense_index; CI fix; corpus downloaded + profiled.
- 2026-10-19: chunking strategy decided (`docs/chunking_strategy.md`); p550 parked; loader specs + 5 tests;
  **chunking.py done** (my 3 helpers: `_split_into_pieces`, `_pack`, `_add_overlap` — all blanks right first try).
  **pdf.py done** — noise regex, de-hyphenation, bookmark sections; 0 junk chunks on real IRS PDFs.
  (Editor issue: my typed answers didn't reach the saved file 3×; Claude filled the blanks from the skeleton.)
  **html.py done** — h2/h3/h4 sections, noise + cookie/banner removal; real pages → 6 / 6 / 13 sections,
  12 / 11 / 18 chunks, no nav/footer leaks. (Blanks filled by Claude again — same editor issue.)
  Renamed `data/kb/documents/` → `data/kb/sources/`. **docx.py done** — all 4 loaders finished.
  **retriever.py done** — dense / sparse / hybrid, category filter before top-k, dense cosine for BM25-only
  hits via `index.reconstruct` (no re-embedding), confidence gate on the best dense cosine.
  **ingest.py done → real index built:** 114 docs → 590 chunks (pdf 544 · html 41 · md 5) in 68 s
  (first run incl. model download; embedding itself ~5 s). **62 / 62 tests — RAG layer complete.**

## My TODOs outside code
- Sign up: Alpha Vantage, Tavily → put keys in `.env` (never commit it).
- Write ~10 Markdown articles (2 per category) → grow to ~50 (also fixes the corpus imbalance).
- Label ~40 eval questions in `tests/eval/retrieval_set.jsonl` (exact_term / paraphrase / concept / out_of_scope).

## Interview talking points so far
- RRF instead of adding scores: cosine and BM25 live on different scales.
- BM25 alongside embeddings: exact terms ("Form 8606", "401k") vs meaning.
- IndexFlatIP: exact search, milliseconds at this size; HNSW/IVF only pay off at ~1M vectors.
- Profiled the corpus before chunking; every finding became a loader rule or a test.
- Manifest = data-provenance registry; unregistered files can't be ingested.
- Retriever: confidence gate uses dense cosine, not RRF (RRF = position only; nonsense still gets a "#1");
  category filter BEFORE top-k (fetch candidate_k=20); BM25-only hits get cosine from `index.reconstruct`.
- Chunks live in `data/index/` (chunks.json, dense.faiss, sparse_tokens.json) — derived, gitignored, rebuildable.
- Chunker details: `+ 2` in packing = the `"\n\n"` separator; overlap snaps to a word start (half-words are
  noise for BM25 and embeddings); max chunk = `chunk_size + overlap` (800 + 120 = 920) because overlap is
  prepended after packing.
- PDF loader: `.search` not `.match` (IRS page 1 has `AH XSL/XML Fileid:` mid-line — found on real data);
  repeated-line rule only with ≥ 3 pages (with 2 pages everything "repeats"); `retire-\nment` → join, but
  `1040-\nSR` → keep the hyphen; section = **last** bookmark starting on/before the page (most specific heading);
  blank lines kept through stripping because they are paragraph breaks for the chunker.

## Habit reminders from mistakes so far
- Paste code *into* the file — twice now the whole file got replaced (docstrings / imports / functions lost).
- Save before asking for a review (turn on File → Auto Save).
- If edits don't reach the file: close the tab (Don't Save), reopen from Explorer; check for read-only,
  preview mode (italic tab title), or the file being open in a second VS Code window.
- Missing imports (`faiss`, `BM25Okapi`, `re`) — run `ruff check` before saying "done"; it flags them as F821.

## Habits
- Replace only the `raise NotImplementedError(...)` line — keep docstrings.
- Before saying "done": `pytest` + `ruff check --fix` + `ruff format`.
- Work on a branch, never directly on `main`. `git status` before and after `git add`.
