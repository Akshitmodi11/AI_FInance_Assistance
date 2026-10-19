"""Chunking — TODO(you).

Concepts this teaches:
* Loaders already split on *structure* (sections, pages). Chunking enforces a *size* limit,
  because embedding models only "see" a limited window (all-MiniLM-L6-v2 truncates at
  256 tokens ≈ 1,000 characters) and a smaller chunk gives a more precise match.
* Too small → a chunk loses context ("it" refers to what?). Too big → one vector averages
  several ideas and matches none of them well. You'll tune ``chunk_size`` with the eval.
* Overlap: repeat the tail of one chunk at the start of the next so a sentence that spans
  the boundary is fully present in at least one chunk.
"""

from __future__ import annotations

from src.rag.models import Chunk, Document


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split one Document into Chunks of at most ``chunk_size`` characters.

    Algorithm (see tests/unit/test_chunking.py):
    1. If ``len(doc.text) <= chunk_size`` → a single chunk with the whole text.
    2. Otherwise split into paragraphs on blank lines (``"\\n\\n"``) and pack them greedily:
       keep adding paragraphs to the current chunk while it fits in ``chunk_size``.
    3. A single paragraph longer than ``chunk_size`` is split further — on sentence ends
       (". ") if possible, otherwise hard-cut at ``chunk_size``.
    4. Each new chunk starts with the last ``overlap`` characters of the previous chunk.
       (Bonus: snap the overlap start forward to a word boundary so it doesn't begin
       mid-word.) The total may then slightly exceed ``chunk_size`` — that's fine, but it
       must stay ≤ ``chunk_size + overlap``.
    5. ``chunk_id = f"{doc.doc_id}:{n}"`` with n = 0, 1, 2, …; copy ``doc.meta``.
    6. Never emit empty / whitespace-only chunks.
    """
    raise NotImplementedError("TODO(you): implement chunk_document")


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
