"""Financial news: Tavily web search (key in .env) → clearly-labelled sample news fallback."""

from __future__ import annotations

import logging

from src.core.config import Secrets, Settings
from src.core.schemas import NewsItem
from src.data.providers.base import NewsProvider, ProviderError
from src.data.providers.mock import MockNewsProvider

log = logging.getLogger(__name__)


class TavilyNewsProvider:
    name = "tavily"

    def __init__(self, api_key: str):
        from tavily import TavilyClient

        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int) -> list[NewsItem]:
        try:
            data = self.client.search(query=query, topic="news", max_results=max_results, days=7)
        except Exception as exc:  # noqa: BLE001 — SDK raises various HTTP/key errors
            raise ProviderError(f"Tavily search failed: {exc}") from exc
        items = [
            NewsItem(
                title=r.get("title") or "(untitled)",
                url=r.get("url", ""),
                source=_domain(r.get("url", "")),
                published=r.get("published_date"),
                summary=(r.get("content") or "")[:500],
            )
            for r in data.get("results", [])
            if r.get("url")
        ]
        if not items:
            raise ProviderError("Tavily returned no results")
        return items


def _domain(url: str) -> str:
    host = url.split("//", 1)[-1].split("/", 1)[0]
    return host.removeprefix("www.") or "web"


class News:
    def __init__(self, providers: list[NewsProvider], max_results: int = 5):
        self.providers = providers or [MockNewsProvider()]
        self.max_results = max_results

    @classmethod
    def from_settings(cls, settings: Settings, secrets: Secrets | None = None) -> News:
        secrets = secrets or Secrets()
        providers: list[NewsProvider] = []
        for name in settings.news.providers:
            key = secrets.tavily_api_key
            if name == "tavily" and key is not None and key.get_secret_value():
                providers.append(TavilyNewsProvider(key.get_secret_value()))
            elif name == "mock":
                providers.append(MockNewsProvider())
        return cls(providers, settings.news.max_results)

    def search(self, query: str) -> list[NewsItem]:
        for provider in self.providers:
            try:
                return provider.search(query, self.max_results)
            except ProviderError as exc:
                log.info("news via %s failed: %s", provider.name, exc)
        return MockNewsProvider().search(query, self.max_results)
