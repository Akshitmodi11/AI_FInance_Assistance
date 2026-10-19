"""HTML loader — TODO(you).

Concepts this teaches:
* A saved web page is mostly *boilerplate*: navigation menus, cookie banners, footers,
  scripts. On a typical government page the article is under a third of the text.
  Boilerplate in the index means every query about "investing" matches the site menu.
* Main-content extraction: delete known-noise tags, then prefer the semantic container
  (``<main>`` or ``<article>``) if the page has one.

Library: ``from bs4 import BeautifulSoup`` → ``BeautifulSoup(html, "html.parser")``.
"""

from __future__ import annotations

from pathlib import Path

from src.rag.models import Document

NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe"]


def load_html(path: Path, meta: dict | None = None) -> list[Document]:
    """Load a saved HTML page into one Document per ``<h2>`` section of its main content.

    Requirements (see tests/unit/test_loaders.py):
    * ``meta`` from manifest.yaml is required (``ValueError`` if None).
    * Remove every tag in ``NOISE_TAGS`` (``tag.decompose()``), plus elements whose
      ``id``/``class`` contains "cookie" or "banner".
    * Use ``<main>`` if present, else ``<article>``, else ``<body>``.
    * Split at ``<h2>`` headings (heading text → ``section``); content before the first
      ``<h2>`` is a section named after the title.
    * Collapse whitespace (``" ".join(text.split())`` per block) — HTML text is full of
      indentation and newlines.
    * ``file_type="html"``, ``doc_id = f"{path.stem}:s{i}"``.
    """
    raise NotImplementedError("TODO(you): implement load_html")
