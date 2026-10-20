# Chunking Strategy — Decision Record

_Status: **decided** 2026-10-19 · Evidence: `notebooks/profile_corpus.py` · Validated later by the retrieval eval_

## Context
The embedding model (`all-MiniLM-L6-v2`) reads ~256 tokens (~1,000 chars) and silently truncates the rest,
and one vector per chunk blurs multiple ideas together. Chunks must be **small enough to be precise,
large enough to stand alone, and always citable** (page or section).

Corpus profile (6 documents + 1 article):

| Source | Units | Median chars | % > 800 | Paragraph breaks |
|---|---|---|---|---|
| Markdown | sections | 214 | 0% | yes |
| Investor.gov HTML | sections | 612 – 4,309 | 33–50% | yes (block tags) |
| IRS PDFs | pages | ~5,200 | 99–100% | **no** (~1 per page) |

PDF quirks: lines hard-wrapped at ~47 chars with hyphenation (`retire-\nment`), a per-page header that varies
(`Page 10 of 62 Fileid: …`), a fixed proof-printer line on every page, and **bookmarks (outline) that map
every page to a section** (590-A: 75 bookmarks, 969: 53).

## Decision
**Structure-aware recursive chunking with overlap and contextual headers.**

```
Layer 1  STRUCTURE  (loaders)     one Document per MD/DOCX/HTML section or PDF page — chunks never cross these
Layer 2  SIZE       (chunking.py) ≤ 800 chars: paragraphs → sentences → hard cut, packed greedily
Layer 3  OVERLAP    (chunking.py) 120 chars of the previous chunk, snapped to a word boundary
Layer 4  CONTEXT    (models.py)   embedding_text = "Title — Section\n" + chunk text  (stored text stays clean)
```

### Parameters (in `config.yaml`, not code)
| Setting | Value | Why |
|---|---|---|
| `chunk_size` | 800 chars | ≈ 200 tokens; stays under the 256-token window even with the header |
| `chunk_overlap` | 120 chars (15%) | carries one boundary sentence; more only duplicates |
| sentence boundary | `(?<=[.!?])\s+(?=[A-Z0-9"($])` | splits "…limit. The…" but not "$7,000.50" or "e.g. a"; known miss: "U.S. Treasury" |

### Per-format preparation (loader responsibilities)
| Format | Split on | Cleaning before chunking | Section for the header |
|---|---|---|---|
| Markdown | `##` | strip H1 | the `##` heading |
| DOCX | `Heading*` styles | drop empty paragraphs | the heading text |
| HTML | **`h2`, `h3`, `h4`** | drop script/style/nav/header/footer/cookie banners; join block tags with blank lines | the heading text |
| PDF | page | 1. regex-drop `Page N of M` / `Fileid:` / `Userid:` lines · 2. drop lines repeated on ≥ 50% of pages · 3. **de-hyphenate** `word-\nword` → `wordword` · 4. unwrap single newlines → spaces | **deepest bookmark covering that page** (fallback: none) |

Citations stay `"Title, p. N"` for PDFs; the bookmark section improves *retrieval* (via the header), not the citation.

## Alternatives considered
| Strategy | Verdict | Reason |
|---|---|---|
| Fixed-size (every N chars) | **baseline only** | cuts mid-word/sentence; kept as the benchmark to beat |
| Token-based | rejected | still a blind cut; 800 chars is already safely under the token limit |
| Pure sentence-based | absorbed | it *is* our layer-2 fallback, the main path for PDFs |
| Semantic (embedding-shift splits) | stretch | our text already has headings; ~20k extra embeddings at ingest; hard to unit-test |
| Parent-child / small-to-big | stretch | retrieve small chunks, send the parent section to the LLM |
| LLM / proposition chunking | rejected | API cost per page, non-deterministic, overkill for ~1 MB |
| One chunk per section, no cap | rejected | 5,000-char PDF pages would be truncated by the model |
| Merge tiny trailing chunks | deferred | revisit only if the eval shows short chunks ranking badly |

## Consequences
- Same algorithm for all formats: Markdown/HTML use the paragraph path; PDFs use the sentence path.
- Measured on the real corpus with a reference implementation (p550 parked, see `manifest.yaml`):

  | File | Docs | Chunks | Median chars | Sections found |
  |---|---|---|---|---|
  | p590a.pdf | 62 pages | 388 | 816 | 30 (bookmarks) |
  | p969.pdf | 22 pages | 154 | 814 | 16 (bookmarks) |
  | investor-gov-stocks.html | 6 | 12 | 677 | 6 |
  | investor-gov-bonds.html | 6 | 11 | 624 | 6 |
  | investor-gov-mutual-funds-etfs.html | 13 (was 2 with h2-only) | 18 | 536 | 13 |

  0 header-junk chunks, 0 leftover line-break hyphens. PDFs are still ~93% of chunks — chunking can't
  fix that; more Markdown articles + category filters do.
- Cross-page sentences get split at the page boundary — accepted in exchange for page-accurate citations.
- Chunk size can exceed 800 by up to the overlap (≤ 920); the tests enforce `≤ chunk_size + overlap`.

## Validation plan (results go in the README)
Same eval questions, same embedder, `retrieval_mode=hybrid`:

| Variant | chunks | recall@5 | MRR |
|---|---|---|---|
| Fixed-size 800, no overlap (baseline) | | | |
| **Recursive 800 + overlap 120 + headers (chosen)** | | | |
| Recursive 800 + overlap 120, **no headers** | | | |
| Recursive **400** + overlap 60 + headers | | | |
| Recursive **1200** + overlap 180 + headers | | | |

If a variant beats the chosen one by > 0.05 recall@5, switch and update this record.
