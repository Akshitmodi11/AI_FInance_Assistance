"""End-to-end: files on disk → build_index → Retriever.from_settings → answer."""

from pathlib import Path

import pytest
import yaml

from src.core.config import Settings
from src.rag.ingest import build_index
from src.rag.retriever import Retriever
from tests.conftest import make_pdf
from tests.unit.test_loaders import MD

pytestmark = pytest.mark.integration


def test_ingest_then_retrieve(tmp_path: Path, fake_embedder, manifest_meta):
    (tmp_path / "articles").mkdir()
    (tmp_path / "articles" / "compound-interest.md").write_text(MD, encoding="utf-8")
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "p590a.pdf").write_bytes(
        make_pdf([["Roth IRA contributions are not deductible."], ["Excess contributions."]])
    )
    (tmp_path / "manifest.yaml").write_text(
        yaml.safe_dump({"documents": {"p590a.pdf": manifest_meta}}), encoding="utf-8"
    )
    settings = Settings(
        rag={
            "articles_dir": tmp_path / "articles",
            "documents_dir": tmp_path / "documents",
            "manifest_path": tmp_path / "manifest.yaml",
            "index_dir": tmp_path / "index",
            "min_score": 0.2,
        }
    )

    stats = build_index(settings, fake_embedder)
    assert stats.documents == 5  # 3 markdown sections + 2 pdf pages
    assert stats.chunks >= 5
    assert stats.by_file_type.keys() == {"md", "pdf"}

    retriever = Retriever.from_settings(settings, embedder=fake_embedder)
    result = retriever.retrieve("roth ira deductible", mode="hybrid")
    assert result.evidence[0].chunk.meta.citation == "IRS Publication 590-A, p. 1"

    # Idempotent: rebuilding gives the same number of chunks.
    assert build_index(settings, fake_embedder).chunks == stats.chunks
