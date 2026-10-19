"""PDF loader — TODO(you).

Concepts this teaches:
* PDFs have no "paragraphs", only positioned text. Extraction gives you a string per page.
* Pages are the natural structural unit, and page numbers make citations verifiable:
  "IRS Publication 590-A, p. 12" lets a reader check the claim.
* Every page repeats a running header/footer ("Publication 590-A (2025)", "Page 12").
  Left in, those lines end up in every chunk, pollute embeddings and match every BM25 query
  for "590-A". Strip them.

Library: ``from pypdf import PdfReader`` → ``PdfReader(path).pages[i].extract_text()``.
"""

from __future__ import annotations

from pathlib import Path

from src.rag.models import Document


def load_pdf(path: Path, meta: dict | None = None) -> list[Document]:
    """Load a PDF into one Document per non-empty page.

    Requirements (see tests/unit/test_loaders.py):
    * ``meta`` (from manifest.yaml) supplies title/url/source/category[/tax_year/level].
      Raise ``ValueError`` if it is None — a PDF without a manifest entry can't be cited.
    * ``file_type="pdf"``, ``page`` = 1-based page number.
    * ``doc_id = f"{path.stem}:p{page}"``.
    * Run the page texts through :func:`strip_repeated_lines` before building Documents.
    * Skip pages that are empty after stripping (cover pages, blank pages).
    """
    raise NotImplementedError("TODO(you): implement load_pdf")


def strip_repeated_lines(pages: list[str], min_fraction: float = 0.5) -> list[str]:
    """Remove running headers/footers from a list of page texts.

    Remove a line from every page if either:
    * it is a bare page number such as ``"12"``, ``"Page 12"`` or ``"- 12 -"``; or
    * the same line (after ``.strip()``) appears on at least ``min_fraction`` of the pages.
      Only apply this rule when there are 3+ pages — with 1–2 pages everything "repeats".

    Hint: ``collections.Counter`` over ``set(page.splitlines())`` per page counts how many
    pages each line appears on.
    """
    raise NotImplementedError("TODO(you): implement strip_repeated_lines")
