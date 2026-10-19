"""Sparse (keyword) index with BM25.

Concepts this teaches:
* "Sparse" = each text is a vector over the whole vocabulary that is almost all zeros;
  only the words it contains are non-zero.
* BM25 scores a chunk higher when it contains the query's words, especially *rare* words
  (inverse document frequency: "8606" is rare → strong signal; "the" is everywhere → none),
  with diminishing returns for repeating a word, normalized for chunk length.
* Tokenization decides what "the same word" means. Finance has awkward tokens:
  "401(k)", "S&P 500", "P/E", "Roth". A naive split turns "401(k)" into "401" and "k" and
  the query "401k" then never matches. That's why ``tokenize`` is worth testing.

API: ``from rank_bm25 import BM25Okapi`` → ``BM25Okapi(tokenized_corpus)``,
``bm25.get_scores(query_tokens)`` → numpy array with one score per document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

INDEX_FILE = "sparse_tokens.json"

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
        "has", "have", "how", "i", "in", "is", "it", "its", "my", "of", "on", "or", "should",
        "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
        "to", "was", "what", "when", "where", "which", "who", "why", "will", "with", "you",
        "your",
    }
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercase, normalize finance terms, split into word tokens, drop stopwords.

    * ``"401(k)"``, ``"401k"`` and ``"401 (k)"`` all → ``"401k"``; ``403(b)`` → ``"403b"``.
    * ``"S&P 500"`` → ``["s&p", "500"]`` (keep the ampersand inside the token).
    * ``"P/E ratio"`` → ``["p/e", "ratio"]``.
    * Punctuation elsewhere is a separator:
      ``"Roth IRA's limits!"`` → ``["roth", "ira", "s", "limits"]``.
    """
    text = text.lower()
    text = re.sub(r"\b(\d+)\s*\(([a-z])\)", r"\1\2", text)
    tokens = re.findall(r"[a-z0-9]+(?:[&/][a-z0-9]+)*", text)
    return [token for token in tokens if token not in STOPWORDS]


class SparseIndex:
    def __init__(self, tokenized_corpus: list[list[str]]):
        self.tokenized_corpus = tokenized_corpus
        # BM25 can't be built if there are no documents or no tokens.
        self.bm25 = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None

    @classmethod
    def build(cls, texts: list[str]) -> SparseIndex:
        """Tokenize every text and build the index."""
        return cls([tokenize(text) for text in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, bm25_score)`` pairs, best first.

        Only rows with score > 0 are returned — a chunk sharing no query word is not a match.
        Ties keep corpus order (stable sort) so results are deterministic.
        """
        if self.bm25 is None or k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        best_rows = np.argsort(-scores, kind="stable")[:k]
        return [(int(row), float(scores[row])) for row in best_rows if scores[row] > 0]

    def save(self, directory: Path) -> None:
        """Persist the tokenized corpus as JSON (BM25 itself is rebuilt on load — it's fast,
        and JSON avoids pickle's security and version-compatibility problems)."""
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / INDEX_FILE).open("w", encoding="utf-8") as file:
            json.dump(self.tokenized_corpus, file)

    @classmethod
    def load(cls, directory: Path) -> SparseIndex:
        with (directory / INDEX_FILE).open(encoding="utf-8") as file:
            return cls(json.load(file))

    def __len__(self) -> int:
        return len(self.tokenized_corpus)
