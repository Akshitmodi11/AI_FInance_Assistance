"""Markdown loader — the worked example the other loaders follow.

Structure-aware loading: one Document per ``##`` section, so a chunk never mixes two topics.
Metadata comes from the YAML front-matter at the top of the file.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

from src.rag.models import Document, SourceMeta

_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def load_markdown(path: Path, meta: dict | None = None) -> list[Document]:
    """Load a Markdown article into one Document per ``##`` section.

    Text before the first ``##`` (the intro) becomes its own section named after the title.
    ``meta`` is ignored for Markdown: the front-matter is the source of truth.
    """
    post = frontmatter.load(path)
    fm = post.metadata
    base = SourceMeta(
        title=fm["title"],
        url=fm["url"],
        source=fm["source"],
        category=fm["category"],
        file_type="md",
        tax_year=fm.get("tax_year"),
        level=fm.get("level"),
    )

    body = _strip_h1(post.content)
    sections = _split_sections(body, intro_name=base.title)

    return [
        Document(
            doc_id=f"{path.stem}:s{i}",
            text=text,
            meta=base.model_copy(update={"section": name}),
        )
        for i, (name, text) in enumerate(sections)
    ]


def _strip_h1(text: str) -> str:
    """Drop the leading ``# Title`` line — the title is already in the metadata."""
    return re.sub(r"\A\s*#\s+.*\n", "", text, count=1)


def _split_sections(text: str, intro_name: str) -> list[tuple[str, str]]:
    """Split on ``##`` headings → ``[(section_name, section_text), ...]``, skipping empties."""
    sections: list[tuple[str, str]] = []
    matches = list(_H2.finditer(text))

    intro = text[: matches[0].start()] if matches else text
    if intro.strip():
        sections.append((intro_name, intro.strip()))

    for m, nxt in zip(matches, [*matches[1:], None], strict=True):
        end = nxt.start() if nxt else len(text)
        content = text[m.end() : end].strip()
        if content:
            sections.append((m.group(1), content))
    return sections
