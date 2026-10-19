"""Data contracts passed between RAG stages.

    loader ──▶ Document ──▶ chunker ──▶ Chunk ──▶ indexes ──▶ retriever ──▶ Evidence

Every stage carries a :class:`SourceMeta` along so that, at the end, an agent can cite exactly
where a sentence came from ("IRS Publication 590-A, p. 12").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FileType = Literal["md", "pdf", "docx", "html"]


class SourceMeta(BaseModel):
    """Where a piece of text came from. Shared by Document, Chunk and Evidence."""

    title: str
    url: str
    source: str
    category: str
    file_type: FileType
    page: int | None = None  # PDFs: 1-based page number
    section: str | None = None  # heading the text sits under, if known
    tax_year: int | None = None  # tax figures change yearly — say which year
    level: str | None = None  # beginner / intermediate

    @property
    def citation(self) -> str:
        """Human-readable label, e.g. ``"IRS Publication 590-A, p. 12"``."""
        label = self.title
        if self.page is not None:
            label += f", p. {self.page}"
        elif self.section:
            label += f" — {self.section}"
        return label


class Document(BaseModel):
    """A structural unit produced by a loader: a Markdown/Word section, a PDF page, etc.

    ``doc_id`` must be unique across the whole corpus, e.g. ``"p590a:p12"`` or
    ``"diversification-basics:s2"``.
    """

    doc_id: str
    text: str
    meta: SourceMeta


class Chunk(BaseModel):
    """A size-bounded piece of a Document; the unit that gets embedded and indexed."""

    chunk_id: str  # f"{doc_id}:{n}"
    doc_id: str
    text: str
    meta: SourceMeta

    @property
    def embedding_text(self) -> str:
        """Text sent to the embedder and BM25.

        Prepending the title and section ("contextual chunk headers") lets a chunk that
        never repeats its topic still match it — e.g. a paragraph about "contribution
        limits" inside the Roth IRA article also matches the query "Roth IRA limits".
        """
        header = self.meta.title
        if self.meta.section and self.meta.section != self.meta.title:
            header += f" — {self.meta.section}"
        return f"{header}\n{self.text}"


class Evidence(BaseModel):
    """A retrieved chunk plus how it scored."""

    chunk: Chunk
    score: float  # the score results are ordered by (cosine, BM25 or RRF depending on mode)
    dense_score: float | None = None  # cosine similarity, used for the confidence gate


class RetrievalResult(BaseModel):
    evidence: list[Evidence] = Field(default_factory=list)
    mode: Literal["dense", "sparse", "hybrid"]
    confident: bool  # False -> the agent should say "I don't have reliable info on that"
    top_dense_score: float | None = None
