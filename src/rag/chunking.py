"""Chunking. Strategy and alternatives: docs/chunking_strategy.md.

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

import re

from src.rag.models import Chunk, Document

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"($])")


def _split_into_pieces(text: str, chunk_size: int) -> list[str]:
    """Split text into paragraph/sentence pieces, each at most ``chunk_size`` chars."""
    pieces: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            pieces.append(para)
            continue

        for sentence in _SENTENCE_END.split(para):
            while len(sentence) > chunk_size:  # no usable sentence break → hard cut
                pieces.append(sentence[:chunk_size])
                sentence = sentence[chunk_size:]
            if sentence:
                pieces.append(sentence)

    return pieces


def _pack(pieces: list[str], chunk_size: int) -> list[str]:
    """Greedily join pieces with blank lines while the result still fits in ``chunk_size``."""
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= chunk_size:  # + 2 for the "\n\n" separator
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            current = piece

    if current:
        chunks.append(current)

    return chunks


def _add_overlap(texts: list[str], overlap: int) -> list[str]:
    """Prefix each chunk with the last ``overlap`` chars of the previous one, from a word start."""
    if overlap <= 0:
        return texts

    result = texts[:1]
    for prev, text in zip(texts, texts[1:], strict=False):
        tail = prev[-overlap:]
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1 :]
        result.append(f"{tail} {text}")

    return result


def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split one Document into Chunks of at most ``chunk_size`` characters.

    Algorithm (see tests/unit/test_chunking.py):
    1. If ``len(doc.text) <= chunk_size`` → a single chunk with the whole text.
    2. Otherwise split into paragraphs on blank lines (``"\\n\\n"``) and pack them greedily:
       keep adding paragraphs to the current chunk while it fits in ``chunk_size``.
    3. A single paragraph longer than ``chunk_size`` is split further — on sentence ends if
       possible, otherwise hard-cut at ``chunk_size``. Sentence end = ``.``/``!``/``?`` +
       whitespace + a capital, digit, quote, ``(`` or ``$`` — so "$7,000.50" and "e.g. a"
       don't split. This is the *main* path for PDF pages, which have no blank lines.
    4. Each new chunk starts with the last ``overlap`` characters of the previous chunk,
       snapped forward to a word boundary. The total may then exceed ``chunk_size`` but
       stays ≤ ``chunk_size + overlap``.
    5. ``chunk_id = f"{doc.doc_id}:{n}"`` with n = 0, 1, 2, …; copy ``doc.meta``.
    6. Never emit empty / whitespace-only chunks.
    """
    text = doc.text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        texts = [text]
    else:
        texts = _add_overlap(_pack(_split_into_pieces(text, chunk_size), chunk_size), overlap)
    return [
        Chunk(chunk_id=f"{doc.doc_id}:{i}", doc_id=doc.doc_id, text=t, meta=doc.meta)
        for i, t in enumerate(texts)
    ]


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, chunk_size, overlap)]
