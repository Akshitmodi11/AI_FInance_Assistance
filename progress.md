# Finnie — Progress Log

_Last updated: 2026-10-19_

## Where we are
**Week 1 — RAG layer.** Guided build: Claude scaffolds + reviews, I write the core logic.
Goal: finish earlier than the 4-week plan, in small daily steps.
Tests: **37 / 62 passing** (`pytest -q`). All 25 failures are in the stubs I haven't written yet.

### My 9 RAG files
| # | File | Functions to write | Status | Tests |
|---|---|---|---|---|
| 1 | `src/rag/fusion.py` | `reciprocal_rank_fusion` | ✅ done | 3/3 |
| 2 | `src/rag/sparse_index.py` | `tokenize`, `SparseIndex` | ✅ done | 10/10 |
| 3 | `src/rag/dense_index.py` | `DenseIndex` | ✅ done | 3/3 |
| 4 | `src/rag/chunking.py` | `chunk_document` (+ 3 helpers) | ✅ done | 6/6 |
| 5 | `src/rag/loaders/pdf.py` | `load_pdf`, `strip_repeated_lines`, `clean_page_text`, `page_sections` | ⬜ **next** | 0/8 |
| 6 | `src/rag/loaders/docx.py` | `load_docx` | ⬜ | 0/2 |
| 7 | `src/rag/loaders/html.py` | `load_html` | ⬜ | 0/3 |
| 8 | `src/rag/retriever.py` | `Retriever.retrieve` | ⬜ | 0/11 |
| 9 | `src/rag/ingest.py` | `build_index` | ⬜ | 0/1 |

Already passing (Claude's code): config 5, KB articles 2, Markdown loader + registry 5, eval metrics 3.

## Git
- `main` — only README. Merge into it only when CI is green.
- `dev-branch` — fusion / sparse / dense done, pushed. CI fixed (`activate-environment: true`);
  lint ✅, tests ❌ (expected until the RAG layer is done).
- `corpus-profiling` — **current branch**. Commit `6d065c0` pushed (2026-10-19).
  **Not yet committed:** `src/rag/chunking.py` (done), `progress.md` updates. Commit 6d065c0 contains:
  - 5 documents in `data/kb/documents/` (p590a, p969 PDFs; Investor.gov stocks / bonds / mutual-funds HTML)
  - `data/kb/manifest.yaml` entries for all 5 (+ parked p550 note)
  - `notebooks/profile_corpus.py`, `docs/chunking_strategy.md`, this `progress.md`
  - updated specs: `loaders/pdf.py`, `loaders/html.py`, `chunking.py`; 5 new tests; bookmark support in `tests/conftest.py`
- Commit the chunker:
  ```
  git status
  git add src/rag/chunking.py progress.md
  git commit -m "Implement recursive chunker with overlap"
  git push
  ```
- Cleanup: delete the typo branch → `git branch -d dev-branc`

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

`manifest.yaml` = metadata (title, url, source, category, tax_year) for every non-Markdown file.
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
Week-4 stretch; LLM chunking = too costly. Verified with a reference implementation on the real corpus:
p590a → 388 chunks / 30 sections, p969 → 154 / 16, 0 junk chunks, 0 broken words.
Benchmark later: fixed-size baseline vs ours vs no-headers vs 400 vs 1200 (recall@5, MRR) → README.

## Next session — start here
1. Commit + push the chunker (commands above) if not done.
2. Write `src/rag/loaders/pdf.py`, easiest first — ask Claude for each function's skeleton:
   | Order | Function | Tests | Core idea |
   |---|---|---|---|
   | 1 | `clean_page_text` | 2 | 2 `re.sub` for hyphens → split paragraphs → collapse whitespace |
   | 2 | `strip_repeated_lines` | 3 | 1 compiled noise regex + `Counter` of lines per page |
   | 3 | `page_sections` | 1 | flatten `reader.outline` recursively → last bookmark on/before each page |
   | 4 | `load_pdf` | 2 | glue: extract → strip → clean → sections → `Document`s |
   Run one at a time: `pytest tests/unit/test_loaders.py -k clean_page -v`
3. Then `html.py` (3 tests) → `docx.py` (2).
4. Then `retriever.py` → `ingest.py` → run the retrieval eval → merge into `dev-branch`.

## Done log
- 2026-10-18: fusion, sparse_index, dense_index; CI fix; corpus downloaded + profiled.
- 2026-10-19: chunking strategy decided (`docs/chunking_strategy.md`); p550 parked; loader specs + 5 tests;
  **chunking.py done** (my 3 helpers: `_split_into_pieces`, `_pack`, `_add_overlap` — all blanks right first try).

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
- Chunker details: `+ 2` in packing = the `"\n\n"` separator; overlap snaps to a word start (half-words are
  noise for BM25 and embeddings); max chunk = `chunk_size + overlap` (800 + 120 = 920) because overlap is
  prepended after packing.

## Habit reminders from mistakes so far
- Paste code *into* the file — twice now the whole file got replaced (docstrings / imports / functions lost).
- Save before asking for a review (turn on File → Auto Save).
- Missing imports (`faiss`, `BM25Okapi`, `re`) — run `ruff check` before saying "done"; it flags them as F821.

## Habits
- Replace only the `raise NotImplementedError(...)` line — keep docstrings.
- Before saying "done": `pytest` + `ruff check --fix` + `ruff format`.
- Work on a branch, never directly on `main`. `git status` before and after `git add`.
