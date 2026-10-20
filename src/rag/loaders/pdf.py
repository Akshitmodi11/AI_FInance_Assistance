"""PDF loader.

Concepts this teaches:
* PDFs have no "paragraphs", only positioned text. Extraction gives you a string per page,
  hard-wrapped at the printed line width (~47 chars in IRS publications) with words
  hyphenated across lines: ``"retire-\\nment"``. Left alone, BM25 sees "retire" and "ment"
  and never matches "retirement".
* Pages are the natural structural unit, and page numbers make citations verifiable:
  "IRS Publication 590-A, p. 12" lets a reader check the claim.
* Every page repeats running headers/footers. Some are identical on every page (exact-match
  them); some change per page (``"Page 10 of 62  Fileid: …"``) and need a pattern.
  Left in, they end up in every chunk and pollute embeddings and BM25.
* Many PDFs carry **bookmarks** (the outline in a PDF viewer's sidebar) that map pages to
  section titles. Using them as ``section`` gives PDF chunks a contextual header
  ("IRS Publication 969 — Limit on Contributions") instead of just the document title.

Decided in docs/chunking_strategy.md. Profiled with notebooks/profile_corpus.py.

Library: ``from pypdf import PdfReader`` → ``reader.pages[i].extract_text()``,
``reader.outline`` (nested lists of bookmarks), ``reader.get_destination_page_number(item)``
(0-based).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from src.rag.models import Document, SourceMeta

_NOISE_LINE = re.compile(
    r"^[-\s]*(page\s*)?\d+[-\s]*$"  # "12", "Page 12", "- 12 -"
    r"|^page\s+\d+\s+of\s+\d+"  # "Page 10 of 62  Fileid: …"
    r"|^userid:"  # "Userid: CPM Schema: …"
    r"|fileid:",  # "AH XSL/XML Fileid: …" (anywhere in the line)
    re.IGNORECASE,
)


def load_pdf(path: Path, meta: dict | None = None) -> list[Document]:
    """Load a PDF into one Document per non-empty page.

    Requirements (see tests/unit/test_loaders.py):
    * ``meta`` (from manifest.yaml) supplies title/url/source/category[/tax_year/level].
      Raise ``ValueError`` if it is None — a PDF without a manifest entry can't be cited.
    * Pipeline per page: extract text → :func:`strip_repeated_lines` (over all pages at
      once) → :func:`clean_page_text`.
    * Skip pages that are empty after cleaning (cover pages, blank pages).
    * ``file_type="pdf"``, ``page`` = 1-based page number,
      ``section`` = :func:`page_sections` for that page (``None`` if the PDF has no bookmarks).
    * ``doc_id = f"{path.stem}:p{page}"``.
    """
    if meta is None:
        raise ValueError(f"{path.name}: a PDF needs a manifest.yaml entry to be citable")

    reader = PdfReader(path)
    raw_pages = [page.extract_text() or "" for page in reader.pages]
    pages = [clean_page_text(p) for p in strip_repeated_lines(raw_pages)]
    sections = page_sections(reader)

    docs = []
    for number, text in enumerate(pages, start=1):
        if not text:
            continue  # cover / blank page
        docs.append(
            Document(
                doc_id=f"{path.stem}:p{number}",
                text=text,
                meta=SourceMeta(**meta, file_type="pdf", page=number, section=sections.get(number)),
            )
        )
    return docs


def strip_repeated_lines(pages: list[str], min_fraction: float = 0.5) -> list[str]:
    """Remove running headers/footers from a list of page texts.

    Remove a line (compare after ``.strip()``) from every page if either:
    * it matches a **noise pattern** — checked on any number of pages:
      - a bare page number: ``"12"``, ``"Page 12"``, ``"- 12 -"``;
      - a line starting with ``"Page <n> of <m>"`` (IRS: ``"Page 10 of 62  Fileid: …"``);
      - a line starting with ``"Userid:"`` or **containing** ``"Fileid:"`` (IRS print-proof
        metadata; page 1 has ``"AH XSL/XML Fileid: …"``, so "starts with" isn't enough); or
    * the same line appears on at least ``min_fraction`` of the pages.
      Only apply this rule when there are 3+ pages — with 1–2 pages everything "repeats".

    Blank lines are kept: they are paragraph breaks for :func:`clean_page_text`.
    """
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({line.strip() for line in page.splitlines() if line.strip()})

    repeated: set[str] = set()
    if len(pages) >= 3:
        repeated = {line for line, n in counts.items() if n / len(pages) >= min_fraction}

    cleaned = []
    for page in pages:
        keep = [
            line
            for line in page.splitlines()
            if not line.strip()  # keep blank lines: they're paragraph breaks
            or (line.strip() not in repeated and not _NOISE_LINE.search(line.strip()))
        ]
        cleaned.append("\n".join(keep))
    return cleaned


def clean_page_text(text: str) -> str:
    """Undo PDF line wrapping so the chunker sees normal prose.

    * Join words hyphenated across a line break, in this order:
      1. letter + ``-`` + newline + lowercase letter → drop both: ``"retire-\\nment"`` →
         ``"retirement"``. Known trade-off: a real compound split there (``"after-\\ntax"``)
         becomes ``"aftertax"`` — accepted, it's rare.
      2. any other ``-`` + newline → keep the hyphen, drop the newline:
         ``"1040-\\nSR"`` → ``"1040-SR"``.
    * Keep blank lines (``"\\n\\n"``) as paragraph breaks; turn every other newline into a
      space.
    * Collapse runs of spaces/tabs into one space and strip each paragraph.
    """
    text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)  # retire-\nment → retirement
    text = text.replace("-\n", "-")  # 1040-\nSR → 1040-SR
    paragraphs = re.split(r"\n\s*\n", text)  # keep blank-line paragraph breaks
    cleaned = [" ".join(p.split()) for p in paragraphs]  # all whitespace runs → one space
    return "\n\n".join(p for p in cleaned if p)


def page_sections(reader) -> dict[int, str]:
    """Map each 1-based page number to the section it belongs to, using the PDF's bookmarks.

    * ``reader.outline`` is a list whose items are bookmarks or *nested lists* of child
      bookmarks — flatten it recursively, keeping document order.
    * For each bookmark: ``page = reader.get_destination_page_number(item) + 1``,
      title = ``item.title`` (strip it).
    * A page's section is the title of the **last bookmark (in document order) that starts on
      or before that page** — i.e. the most specific heading in effect.
    * Pages before the first bookmark get no entry. A PDF without bookmarks → ``{}``.
    """
    bookmarks: list[tuple[int, str]] = []

    def walk(items) -> None:
        for item in items:
            if isinstance(item, list):  # a nested list = the children of the previous bookmark
                walk(item)
            else:
                page = reader.get_destination_page_number(item) + 1  # pypdf is 0-based
                bookmarks.append((page, item.title.strip()))

    walk(reader.outline)

    sections: dict[int, str] = {}
    for page in range(1, len(reader.pages) + 1):
        current = None
        for start, title in bookmarks:
            if start <= page:
                current = title  # later bookmarks overwrite → the most specific heading wins
        if current:
            sections[page] = current
    return sections
