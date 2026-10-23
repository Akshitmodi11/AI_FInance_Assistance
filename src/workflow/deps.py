"""Dependencies the agents need, bundled so the graph can be built with real or fake services.

Production: ``Deps.from_settings()`` → real LLMs, RAG retriever, market data, news.
Tests: pass fakes (a scripted LLM, an in-memory retriever, the mock market provider).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property

from langchain_core.language_models import BaseChatModel

from src.core.config import Settings, get_settings
from src.data.market import MarketData
from src.data.news import News
from src.rag.retriever import Retriever

LLMFactory = Callable[[str], BaseChatModel]  # role ("router" | "answer") → model; may raise


@dataclass
class Deps:
    settings: Settings
    llm: LLMFactory
    market: MarketData
    news: News
    retriever_factory: Callable[[], Retriever] = field(repr=False)

    @cached_property
    def retriever(self) -> Retriever:
        """Loaded on first use — the embedding model takes a few seconds to load."""
        return self.retriever_factory()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Deps:
        from dotenv import load_dotenv

        from src.core.config import PROJECT_ROOT
        from src.core.llm import get_llm

        # LangChain/LangSmith read LANGSMITH_* from the process environment, not from Settings.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        settings = settings or get_settings()
        return cls(
            settings=settings,
            llm=get_llm,
            market=MarketData.from_settings(settings),
            news=News.from_settings(settings),
            retriever_factory=lambda: Retriever.from_settings(settings),
        )
