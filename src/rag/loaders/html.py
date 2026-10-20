"""HTML loader — TODO(you).

Concepts this teaches:
* A saved web page is mostly *boilerplate*: navigation menus, cookie banners, footers,
  scripts. On a typical government page the article is under a third of the text.
  Boilerplate in the index means every query about "investing" matches the site menu.
* Main-content extraction: delete known-noise tags, then prefer the semantic container
  (``<main>`` or ``<article>``) if the page has one. Investor.gov has neither, so the
  ``<body>`` fallback matters.
* Heading levels aren't consistent across pages: Investor.gov "Stocks" uses ``<h2>`` for its
  questions, "Mutual Funds" uses ``<h4>`` (and has an empty ``<h2>``). Splitting on ``<h2>``
  only would turn that page into one 8,000-char section.

Decided in docs/chunking_strategy.md. Profiled with notebooks/profile_corpus.py.

Library: ``from bs4 import BeautifulSoup`` → ``BeautifulSoup(html, "html.parser")``.
"""

from __future__ import annotations

from pathlib import Path

from src.rag.models import Document

NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe"]
HEADING_TAGS = ["h2", "h3", "h4"]
BLOCK_TAGS = ["p", "li"]


def load_html(path: Path, meta: dict | None = None) -> list[Document]:
    """Load a saved HTML page into one Document per heading section of its main content.

    Requirements (see tests/unit/test_loaders.py):
    * ``meta`` from manifest.yaml is required (``ValueError`` if None).
    * Remove every tag in ``NOISE_TAGS`` (``tag.decompose()``), plus elements whose
      ``id``/``class`` contains "cookie" or "banner".
    * Use ``<main>`` if present, else ``<article>``, else ``<body>``.
    * Walk ``HEADING_TAGS`` + ``BLOCK_TAGS`` in document order
      (``root.find_all(HEADING_TAGS + BLOCK_TAGS)``):
      - a heading **with text** starts a new section (heading text → ``section``);
        headings with no text are ignored;
      - a block adds its text to the current section; skip an ``<li>`` that contains a
        ``<p>`` (the ``<p>`` is collected on its own — otherwise the text appears twice).
      - content before the first heading is a section named after the title.
    * Collapse whitespace inside each block (``" ".join(text.split())``) and join a
      section's blocks with ``"\\n\\n"`` so the chunker can pack paragraphs.
    * Drop sections with no text. ``file_type="html"``, ``doc_id = f"{path.stem}:s{i}"``
      numbered after dropping.
    """
    raise NotImplementedError("TODO(you): implement load_html")
