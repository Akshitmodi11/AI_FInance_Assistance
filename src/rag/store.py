"""Chunk store: row number ↔ Chunk. Both indexes return row numbers; this turns them back
into text + citation metadata. Saved as JSON so it's human-inspectable."""

from __future__ import annotations

import json
from pathlib import Path

from src.rag.models import Chunk

STORE_FILE = "chunks.json"


class ChunkStore:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def __getitem__(self, row: int) -> Chunk:
        return self.chunks[row]

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = [c.model_dump() for c in self.chunks]
        (directory / STORE_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> ChunkStore:
        payload = json.loads((directory / STORE_FILE).read_text(encoding="utf-8"))
        return cls([Chunk.model_validate(c) for c in payload])
