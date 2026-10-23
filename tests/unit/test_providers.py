"""Market/news providers, rate limiter, fallback chain and cache — no real network calls.

Alpha Vantage is exercised through ``httpx.MockTransport``: good data, rate-limit "Note",
malformed JSON, HTTP 500 and timeouts. yfinance and Tavily are monkeypatched.
"""

from datetime import UTC, datetime

import diskcache
import httpx
import pandas as pd
import pytest

from src.core.config import Secrets, Settings
from src.core.schemas import Quote
from src.data.market import MarketData
from src.data.news import News, TavilyNewsProvider
from src.data.providers.alpha_vantage import AlphaVantageProvider
from src.data.providers.base import ProviderError
from src.data.providers.mock import MockMarketProvider, MockNewsProvider
from src.data.rate_limit import RateLimiter

GOOD_QUOTE = {
    "Global Quote": {
        "05. price": "190.50",
        "09. change": "-1.25",
        "10. change percent": "-0.6518%",
        "07. latest trading day": "2026-10-21",
    }
}


def _series(n: int = 60) -> dict:
    days = pd.bdate_range(end=datetime.now(UTC).date(), periods=n)
    return {
        "Time Series (Daily)": {
            d.date().isoformat(): {"4. close": str(100 + i)} for i, d in enumerate(days)
        }
    }


@pytest.fixture
def cache(tmp_path):
    with diskcache.Cache(str(tmp_path / "cache")) as c:
        yield c


def _av(cache, handler, per_minute=5, per_day=25, retries=0) -> AlphaVantageProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    limiter = RateLimiter(cache, "av-test", per_minute, per_day)
    return AlphaVantageProvider("demo-key", limiter, retries=retries, client=client)


# ---------- Alpha Vantage ----------


def test_alpha_vantage_quote_and_history(cache):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["apikey"] == "demo-key"
        fn = request.url.params["function"]
        return httpx.Response(200, json=GOOD_QUOTE if fn == "GLOBAL_QUOTE" else _series())

    av = _av(cache, handler)
    quote = av.get_quote("AAPL")
    assert (quote.price, quote.change, quote.source) == (190.5, -1.25, "alpha_vantage")
    assert quote.change_pct == pytest.approx(-0.6518)
    history = av.get_history("AAPL", 60)
    assert history.closes == sorted(history.closes) and len(history.points) >= 2


@pytest.mark.parametrize(
    "payload",
    [
        {"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25/day."},
        {"Information": "Invalid API key"},
        {"Error Message": "Invalid API call"},
        {"Global Quote": {"05. price": "not-a-number"}},  # malformed
        {"Global Quote": {}},
    ],
)
def test_alpha_vantage_errors_become_provider_errors(cache, payload):
    av = _av(cache, lambda r: httpx.Response(200, json=payload))
    with pytest.raises(ProviderError):
        av.get_quote("AAPL")


def test_alpha_vantage_http_error_and_bad_json(cache):
    with pytest.raises(ProviderError):
        _av(cache, lambda r: httpx.Response(500)).get_quote("AAPL")
    with pytest.raises(ProviderError):
        _av(cache, lambda r: httpx.Response(200, content=b"<html>oops")).get_quote("AAPL")


def test_alpha_vantage_retries_network_errors_then_succeeds(cache):
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("timed out", request=request)
        return httpx.Response(200, json=GOOD_QUOTE)

    assert _av(cache, flaky, retries=2).get_quote("AAPL").price == 190.5
    assert calls["n"] == 2


def test_alpha_vantage_stops_before_exceeding_budget(cache):
    av = _av(cache, lambda r: httpx.Response(200, json=GOOD_QUOTE), per_minute=2)
    av.get_quote("A")
    av.get_quote("B")
    with pytest.raises(ProviderError, match="budget"):
        av.get_quote("C")


def test_alpha_vantage_requires_key(cache):
    with pytest.raises(ProviderError):
        AlphaVantageProvider("", RateLimiter(cache, "x", 5, 25))


# ---------- rate limiter ----------


def test_rate_limiter_minute_and_day_budgets(cache):
    limiter = RateLimiter(cache, "rl", per_minute=10, per_day=3)
    assert [limiter.try_acquire() for _ in range(4)] == [True, True, True, False]
    assert limiter.remaining_today() == 0
    persisted = RateLimiter(cache, "rl", per_minute=10, per_day=3)  # e.g. after a restart
    assert persisted.try_acquire() is False


# ---------- yfinance ----------


def test_yfinance_provider(monkeypatch):
    from src.data.providers import yfinance_provider

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, period, auto_adjust):
            if self.symbol == "BAD":
                return pd.DataFrame()
            idx = pd.date_range(end="2026-10-21", periods=5, tz="America/New_York")
            return pd.DataFrame({"Close": [10.0, 11.0, 12.0, 11.5, 12.5]}, index=idx)

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    provider = yfinance_provider.YFinanceProvider()
    quote = provider.get_quote("msft")
    assert (quote.ticker, quote.price, quote.change) == ("MSFT", 12.5, 1.0)
    assert provider.get_history("MSFT", 30).closes[-1] == 12.5
    with pytest.raises(ProviderError):
        provider.get_quote("BAD")


# ---------- fallback chain + cache ----------


class Failing:
    name = "failing"

    def get_quote(self, ticker):
        raise ProviderError("down")

    def get_history(self, ticker, days):
        raise ProviderError("down")


class Live:
    name = "live"

    def __init__(self):
        self.calls = 0

    def get_quote(self, ticker):
        self.calls += 1
        return Quote(ticker=ticker, price=10, change=1, change_pct=10,
                     as_of=datetime.now(UTC), source="live")  # fmt: skip

    def get_history(self, ticker, days):
        return (
            MockMarketProvider()
            .get_history(ticker, days)
            .model_copy(update={"source": "live", "is_mock": False})
        )


def test_fallback_chain_and_cache(cache):
    live = Live()
    market = MarketData([Failing(), live, MockMarketProvider()], cache=cache)
    first = market.quote("aapl")
    assert first.source == "live" and not first.is_cached
    second = market.quote("AAPL")
    assert second.is_cached and live.calls == 1  # served from cache
    assert market.stats["fallbacks"] == 1 and market.stats["cache_hits"] == 1


def test_mock_results_are_never_cached(cache):
    market = MarketData([Failing(), MockMarketProvider()], cache=cache)
    assert market.quote("AAPL").is_mock
    assert not market.quote("AAPL").is_cached


def test_all_providers_failing_raises(cache):
    with pytest.raises(ProviderError, match="No provider"):
        MarketData([Failing()], cache=cache).quote("AAPL")


def test_snapshot_computes_trend_facts():
    snap, history = MarketData([MockMarketProvider()]).snapshot("VTI")
    assert snap.low_52w <= snap.quote.price <= snap.high_52w or snap.quote.is_mock
    assert snap.sma_20 and snap.sma_50 and snap.trend in {"uptrend", "downtrend", "sideways"}
    assert len(history.points) > 200


def test_from_settings_respects_keys(tmp_path):
    settings = Settings(market_data={"cache_dir": tmp_path / "c"})
    no_keys = MarketData.from_settings(settings, Secrets(_env_file=None))
    assert [p.name for p in no_keys.providers] == ["yfinance", "mock"]
    with_key = MarketData.from_settings(
        settings, Secrets(_env_file=None, alpha_vantage_api_key="k")
    )
    assert [p.name for p in with_key.providers] == ["alpha_vantage", "yfinance", "mock"]


# ---------- news ----------


def test_tavily_provider_and_news_fallback(monkeypatch):
    import tavily

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, **kwargs):
            if kwargs["query"] == "boom":
                raise RuntimeError("401 unauthorized")
            item = {
                "title": "Stocks rally",
                "url": "https://www.example.com/a",
                "content": "Markets rose.",
                "published_date": "2026-10-21",
            }
            return {"results": [item]}

    monkeypatch.setattr(tavily, "TavilyClient", FakeTavily)
    provider = TavilyNewsProvider("key")
    items = provider.search("markets", 3)
    assert items[0].source == "example.com" and not items[0].is_mock
    news = News([provider, MockNewsProvider()])
    assert news.search("boom")[0].is_mock  # Tavily failed → labelled sample news
    assert News.from_settings(Settings(), Secrets(_env_file=None)).providers[0].name == "mock"
