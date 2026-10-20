# Finnie — Progress Log

_Last updated: 2026-10-18_

## Where we are
**Week 1 — RAG layer.** Guided build: Claude scaffolds + reviews, I write the core logic.
Tests: **31 / 57 passing** (`pytest -q`). The 26 failures are the stubs I haven't written yet.

### My 9 RAG files
| # | File | Status | Tests |
|---|---|---|---|
| 1 | `src/rag/fusion.py` (RRF) | ✅ done | 3/3 |
| 2 | `src/rag/sparse_index.py` (tokenizer + BM25) | ✅ done | 10/10 |
| 3 | `src/rag/dense_index.py` (FAISS IndexFlatIP) | ✅ done | 3/3 |
| 4 | `src/rag/chunking.py` | ⬜ **next** | 0/6 |
| 5 | `src/rag/loaders/pdf.py` | ⬜ | 0/4 |
| 6 | `src/rag/loaders/docx.py` | ⬜ | 0/2 |
| 7 | `src/rag/loaders/html.py` | ⬜ | 0/2 |
| 8 | `src/rag/retriever.py` | ⬜ | 0/11 |
| 9 | `src/rag/ingest.py` | ⬜ | 0/1 |

## Git
- `main` — only README (merge into it only when CI is green).
- `dev-branch` — all work so far, pushed. CI fixed (`activate-environment: true`); lint ✅, tests ❌ (expected until RAG is done).
- `corpus-profiling` — **current branch, nothing committed yet**:
  - 6 downloaded docs in `data/kb/documents/` (p590a, p550, p969 PDFs; 3 Investor.gov HTML pages)
  - `data/kb/manifest.yaml` entries for all 6
  - `notebooks/profile_corpus.py` (corpus profiler)
  - this `progress.md`
- Cleanup: delete the typo branch → `git branch -d dev-branc`

## Corpus profiling results (run: `python notebooks/profile_corpus.py .`)
| Source | Units | Median chars | % > 800 | Paragraphs/unit |
|---|---|---|---|---|
| Markdown (1 article) | 5 sections | 214 | 0% | 1.0 |
| IRS PDFs (3) | 201 pages | ~5,200 | 99–100% | ~1.1 |
| Investor.gov HTML (3) | 14 sections | 612–4,309 | 33–50% | – |

Findings:
1. **PDFs have no blank-line paragraphs** → sentence splitting is the main chunking path for PDFs.
2. **PDF lines hard-wrapped at ~47 chars with hyphenation** (`retire-\nment`) → loader must unwrap lines + de-hyphenate.
3. **PDF page header varies per page** (`Page 10 of 62 Fileid: …`) → exact repeated-line stripping misses it; add a regex.
   The line `The type and rule above prints on all proofs…` repeats on every page (exact match catches it).
4. **Mutual Funds HTML uses `<h4>` for questions** → split HTML on h2/h3/h4, not just h2.
5. Markdown sections are already small → chunking won't touch them (good).
6. **Corpus imbalance: PDFs = 99.8% of text** (p550 alone = 62%) → beginner questions get drowned by IRS chunks.

## Decisions (2026-10-19)
- [x] `p550.pdf` dropped (parked in a manifest comment) — corpus balance.
- [x] Loader specs updated for the real data:
  - `pdf.py`: noise patterns (`Page N of M`, `Userid:`, `…Fileid:`), new `clean_page_text` (unwrap + de-hyphenate),
    new `page_sections` (section from PDF bookmarks).
  - `html.py`: split on h2/h3/h4, skip empty headings, join blocks with blank lines.
  - `chunking.py`: better sentence-split regex.
  - 5 new tests (62 total). A reference solution passes all 62 and yields 0 junk chunks on the real PDFs.

## Chunking strategy — DECIDED 2026-10-19 → see `docs/chunking_strategy.md`
Structure-aware recursive chunking: **structure** (loaders) → **size** 800 (paragraphs → sentences → hard cut)
→ **overlap** 120 → **context header** (title — section). New: PDF section comes from the PDF's **bookmarks**.
Benchmark later vs fixed-size baseline, no-headers, 400, 1200.

## Next session — start here
1. `git checkout corpus-profiling` (check VS Code bottom-left).
2. Decide the 2 open questions above.
3. Write `src/rag/chunking.py` — 3 helpers: `_split_into_pieces`, `_pack`, `_add_overlap` (skeleton is in the chat).
4. `pytest tests/unit/test_chunking.py -v` → `ruff check --fix` → `ruff format` → ask Claude to review.
5. Commit on `corpus-profiling`, then merge into `dev-branch`.

## My TODOs outside code
- Sign up: Alpha Vantage, Tavily → put keys in `.env` (never commit it).
- Write ~10 Markdown articles (2 per category) → grow to ~50.
- Label ~40 eval questions in `tests/eval/retrieval_set.jsonl` (exact_term / paraphrase / concept / out_of_scope).

## Habits
- Replace only the `raise NotImplementedError(...)` line — keep docstrings.
- Before saying "done": `pytest` + `ruff check --fix` + `ruff format`.
- Work on a branch, never directly on `main`.
