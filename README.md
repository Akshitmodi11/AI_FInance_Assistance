# Finnie — AI Finance Assistant

A multi-agent conversational tutor that helps beginners learn about investing. A LangGraph
router sends each question to one of six specialist agents (Finance Q&A, Portfolio, Market,
Goal Planning, News, Tax Education). Answers are grounded in a cited knowledge base (FAISS RAG)
and live market data. **Educational only — not financial advice.**

> Work in progress. Full architecture, API docs, usage and troubleshooting sections land in Week 4.

## Setup
Requires **Python 3.12** (3.13+ breaks some dependencies).
```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
copy .env.example .env   # then fill in your keys
```

## Test
```bash
pytest --cov=src
ruff check .
```

## Project structure
```
src/core/        config, LLM factory, schemas, graph state
src/agents/      the six specialist agents
src/workflow/    LangGraph graph, router, guardrails
src/rag/         ingestion, chunking, FAISS store, retriever
src/data/        market/news providers, cache, rate limiting
src/utils/       deterministic finance math, formatting
src/mcp_server/  MCP server for Claude Desktop
src/web_app/     Streamlit UI
data/kb/         knowledge-base articles (markdown + front-matter)
tests/           unit, integration, eval
config.yaml      non-secret settings (secrets go in .env)
```
