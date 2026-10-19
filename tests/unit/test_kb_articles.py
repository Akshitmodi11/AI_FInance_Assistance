"""Validates every knowledge-base article has the metadata RAG citations depend on."""

from datetime import date
from pathlib import Path

import frontmatter
import pytest

from src.core.config import get_settings

settings = get_settings()
ARTICLES = sorted(settings.resolve(settings.rag.articles_dir).glob("*.md"))
REQUIRED = {"title", "url", "source", "category", "reviewed_date"}


def test_kb_has_articles():
    assert ARTICLES, "add articles to data/kb/articles/"


@pytest.mark.parametrize("path", ARTICLES, ids=lambda p: p.name)
def test_article_metadata(path: Path):
    post = frontmatter.load(path)
    missing = REQUIRED - post.metadata.keys()
    assert not missing, f"missing front-matter: {missing}"
    assert post["category"] in settings.rag.categories
    assert str(post["url"]).startswith("http")
    assert isinstance(post["reviewed_date"], date)
    assert len(post.content.split()) >= 150, "article body is too short to be useful"
