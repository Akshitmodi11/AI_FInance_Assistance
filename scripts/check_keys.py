"""Verify every configured API key with one cheap live call each. Never prints key values.

Usage:  python -m scripts.check_keys
Alpha Vantage uses 1 of its 25 daily requests (the result is then cached for 30 minutes).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from src.core.config import PROJECT_ROOT, get_secrets


def _ok(name: str, detail: str) -> None:
    print(f"✓ {name:14} {detail}")


def _fail(name: str, detail: str) -> None:
    print(f"✗ {name:14} {detail}")


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")  # LangSmith reads its settings from the environment
    s = get_secrets()
    failures = 0

    def secret(value) -> str:
        return value.get_secret_value() if value is not None else ""

    # OpenAI
    try:
        from langchain_openai import ChatOpenAI

        reply = ChatOpenAI(model="gpt-4o-mini", api_key=secret(s.openai_api_key), max_tokens=5)
        _ok("OpenAI", f"gpt-4o-mini replied: {reply.invoke('Say OK').content!r}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        _fail("OpenAI", f"{type(exc).__name__}: {str(exc)[:120]}")

    # Tavily
    if secret(s.tavily_api_key):
        try:
            from src.data.news import TavilyNewsProvider

            items = TavilyNewsProvider(secret(s.tavily_api_key)).search("stock market", 2)
            _ok("Tavily", f"{len(items)} live headlines, e.g. {items[0].title[:60]!r}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _fail("Tavily", str(exc)[:140])
    else:
        _fail("Tavily", "TAVILY_API_KEY empty → mock news")

    # Alpha Vantage (through the real chain → rate limiter + cache)
    if secret(s.alpha_vantage_api_key):
        try:
            from src.core.config import get_settings
            from src.data.market import MarketData

            market = MarketData.from_settings(get_settings())
            quote = market.quote("IBM")
            source = "cache" if quote.is_cached else quote.source
            _ok("Alpha Vantage", f"IBM ${quote.price:,.2f} via {source}")
            if quote.source != "alpha_vantage":
                print("    (answered by a fallback — Alpha Vantage may be rate-limited today)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _fail("Alpha Vantage", str(exc)[:140])
    else:
        _fail("Alpha Vantage", "ALPHA_VANTAGE_API_KEY empty → yfinance")

    # LangSmith
    if secret(s.langsmith_api_key):
        try:
            from langsmith import Client

            client = Client(api_key=secret(s.langsmith_api_key))
            projects = [p.name for p in client.list_projects(limit=5)]
            tracing = os.getenv("LANGSMITH_TRACING", "false")
            project = os.getenv("LANGSMITH_PROJECT")
            _ok("LangSmith", f"key valid, tracing={tracing}, project={project} "
                f"(existing projects: {len(projects)})")  # fmt: skip
        except Exception as exc:  # noqa: BLE001
            failures += 1
            _fail("LangSmith", f"{type(exc).__name__}: {str(exc)[:120]}")
    else:
        _fail("LangSmith", "LANGSMITH_API_KEY empty → no tracing")

    print("\nGoogle (Gemini) key: " + ("set" if secret(s.google_api_key) else "not set (optional)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
