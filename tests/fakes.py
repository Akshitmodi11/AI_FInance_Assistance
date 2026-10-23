"""Test doubles for the agent layer: a scripted chat model and a ready-made Deps."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import get_settings
from src.core.llm import LLMUnavailableError
from src.data.market import MarketData
from src.data.news import News
from src.data.providers.mock import MockMarketProvider, MockNewsProvider
from src.rag.dense_index import DenseIndex
from src.rag.retriever import Retriever
from src.rag.sparse_index import SparseIndex
from src.rag.store import ChunkStore
from src.workflow.deps import Deps
from tests.conftest import FakeEmbedder, make_chunk


class _Structured:
    def __init__(self, value):
        self.value = value

    def invoke(self, _messages):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeLLM:
    """Echoes a deterministic answer; returns scripted objects for ``with_structured_output``."""

    def __init__(self, structured: dict | None = None):
        self.structured = structured or {}  # schema class → instance (or Exception to raise)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        return AIMessage(f"[fake answer] {human.content[:60]}")

    def with_structured_output(self, schema):
        if schema not in self.structured:
            raise NotImplementedError(f"no scripted output for {schema.__name__}")
        return _Structured(self.structured[schema])


KB = [
    make_chunk("Roth IRA contributions use after-tax money; the 2025 limit is $7,000.",
               "p590a:p40:0", category="retirement_tax", title="IRS Publication 590-A",
               page=40, tax_year=2025),
    make_chunk("A 401(k) plan is offered by an employer and contributions are pre-tax.",
               "k401:s0:0", category="retirement_tax"),
    make_chunk("Diversification spreads risk across many different investments.",
               "diversification-basics:s0:0", category="diversification"),
    make_chunk("A bond is a debt security, like an IOU, that pays interest.",
               "investor-gov-bonds:s0:0", category="investing"),
    make_chunk("Concentration risk: one holding driving most of a portfolio's result.",
               "diversification-basics:s4:0", category="diversification"),
]  # fmt: skip


def make_retriever() -> Retriever:
    embedder = FakeEmbedder()
    texts = [c.embedding_text for c in KB]
    return Retriever(
        ChunkStore(KB),
        DenseIndex.build(embedder.embed(texts)),
        SparseIndex.build(texts),
        embedder,
        top_k=3,
        candidate_k=5,
        min_score=0.3,
    )


def make_deps(llm: FakeLLM | None = None) -> Deps:
    """``llm=None`` simulates a missing API key (degraded mode)."""

    def factory(_role: str):
        if llm is None:
            raise LLMUnavailableError("no key in tests")
        return llm

    return Deps(
        settings=get_settings(),
        llm=factory,
        market=MarketData([MockMarketProvider()]),
        news=News([MockNewsProvider()]),
        retriever_factory=make_retriever,
    )
