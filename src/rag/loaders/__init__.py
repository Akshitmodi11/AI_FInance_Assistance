"""Loader registry: picks a loader by file extension and gathers the whole corpus.

Adding a format = write ``load_xxx(path, meta) -> list[Document]`` and register it below.
Nothing downstream (chunking, indexing, retrieval) changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from src.core.config import Settings
from src.rag.loaders.docx import load_docx
from src.rag.loaders.html import load_html
from src.rag.loaders.markdown import load_markdown
from src.rag.loaders.pdf import load_pdf
from src.rag.models import Document

log = logging.getLogger(__name__)

Loader = Callable[[Path, dict | None], list[Document]]

LOADERS: dict[str, Loader] = {
    ".md": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
}


def load_manifest(path: Path) -> dict[str, dict]:
    """Read manifest.yaml → ``{filename: metadata}``. Missing file → empty manifest."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("documents", {})


def load_file(path: Path, meta: dict | None = None) -> list[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader for {path.suffix!r} ({path.name})")
    return loader(path, meta)


def load_corpus(settings: Settings) -> list[Document]:
    """Load every supported file from the articles and documents folders."""
    articles_dir = settings.resolve(settings.rag.articles_dir)
    documents_dir = settings.resolve(settings.rag.documents_dir)
    manifest = load_manifest(settings.resolve(settings.rag.manifest_path))

    docs: list[Document] = []
    for path in sorted(articles_dir.glob("*.md")):
        docs.extend(load_file(path))

    if documents_dir.exists():
        for path in sorted(p for p in documents_dir.iterdir() if p.is_file()):
            if path.suffix.lower() not in LOADERS:
                log.warning("Skipping unsupported file %s", path.name)
                continue
            if path.name not in manifest:
                raise ValueError(f"{path.name} has no entry in manifest.yaml")
            docs.extend(load_file(path, manifest[path.name]))

    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate doc_ids (two files share a name?): {dupes}")
    return docs
