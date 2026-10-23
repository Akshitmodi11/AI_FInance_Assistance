"""Performance benchmarks → Markdown tables for docs/benchmarks.md.

Usage:  python -m scripts.benchmarks [--runs 3]

* End-to-end latency per intent (p50 / p95 over ``--runs``), LLM tokens and estimated cost.
* Market data: cold (live provider) vs cached quote latency.
* Ingest: full rebuild time into a temporary directory (load → chunk → embed → index).
Uses real OpenAI / market APIs — costs a few cents and a few Alpha Vantage requests.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.memory import InMemorySaver

from src.core.config import PROJECT_ROOT, Settings, get_settings

QUESTIONS = {
    "finance_qa": "What is an index fund?",
    "tax": "What is the 2025 Roth IRA contribution limit?",
    "portfolio": "Analyze my portfolio: 10 VTI, 5 BND, 3 AAPL",
    "goal": "I want $30,000 in 5 years, I have $2,000 and can add $300 a month",
    "market": "How is VOO doing?",
    "news": "Any news on the Federal Reserve?",
}
# USD per 1M tokens (input, output) — public list prices, for an order-of-magnitude estimate.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


def cost(usage: dict) -> float:
    total = 0.0
    for model, u in usage.items():
        name = "gpt-4o-mini" if "mini" in model else "gpt-4o"
        price_in, price_out = PRICES[name]
        total += u.get("input_tokens", 0) / 1e6 * price_in
        total += u.get("output_tokens", 0) / 1e6 * price_out
    return total


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))]


def bench_intents(runs: int) -> None:
    from src.workflow.deps import Deps
    from src.workflow.graph import ask, build_graph

    app = build_graph(Deps.from_settings(), InMemorySaver())
    ask(app, "What is a bond?", "warmup")  # load the embedding model outside the timings
    print("\n## End-to-end latency per intent\n")
    print("| Intent | Agents | p50 s | p95 s | tokens/question | est. cost/question |")
    print("|---|---|---|---|---|---|")
    for intent, question in QUESTIONS.items():
        times, tokens, costs, agents = [], [], [], []
        for _ in range(runs):
            with get_usage_metadata_callback() as cb:
                start = time.perf_counter()
                final = ask(app, question, str(uuid.uuid4()))
                times.append(time.perf_counter() - start)
            tokens.append(sum(u.get("total_tokens", 0) for u in cb.usage_metadata.values()))
            costs.append(cost(cb.usage_metadata))
            agents = final.agents
        print(
            f"| {intent} | {', '.join(agents)} | {statistics.median(times):.1f} | "
            f"{pct(times, 0.95):.1f} | {statistics.mean(tokens):,.0f} | "
            f"${statistics.mean(costs):.4f} |"
        )


def bench_market() -> None:
    from src.data.market import MarketData

    print("\n## Market data: cold vs cached\n")
    print("| Ticker | Cold (live) ms | Source | Cached ms |")
    print("|---|---|---|---|")
    with tempfile.TemporaryDirectory() as tmp:
        base = get_settings()
        settings = base.model_copy(
            update={"market_data": base.market_data.model_copy(update={"cache_dir": Path(tmp)})}
        )
        market = MarketData.from_settings(settings)
        for ticker in ("SPY", "KO"):
            start = time.perf_counter()
            quote = market.quote(ticker)
            cold = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            market.quote(ticker)
            warm = (time.perf_counter() - start) * 1000
            print(f"| {ticker} | {cold:,.0f} | {quote.source} | {warm:.1f} |")
        market.cache.close()


def bench_ingest() -> None:
    from src.rag.embeddings import SentenceTransformerEmbedder
    from src.rag.ingest import build_index

    print("\n## Ingest (full rebuild)\n")
    base = get_settings()
    embedder = SentenceTransformerEmbedder(base.rag.embedding_model)
    embedder.embed(["warm up"])  # model load is a one-time cost, measured separately below
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(rag={**base.rag.model_dump(), "index_dir": Path(tmp)})
        stats = build_index(settings, embedder)
    print("| Documents | Chunks | Seconds | By file type |")
    print("|---|---|---|---|")
    print(f"| {stats.documents} | {stats.chunks} | {stats.seconds:.1f} | {stats.by_file_type} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    bench_ingest()
    bench_market()
    bench_intents(args.runs)


if __name__ == "__main__":
    main()
