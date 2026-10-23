# Finnie MCP server (Claude Desktop)

Finnie exposes its tools over the **Model Context Protocol**, so Claude Desktop (or any MCP client) can
call them directly. Every tool is a thin wrapper over the same code the Finnie agents use — market data
chain, portfolio analysis, goal math, the RAG retriever and the LangGraph app — so answers can't diverge.

Implementation: `src/mcp_server/server.py` (MCP Python SDK v2, `MCPServer`, stdio transport).

## Tools

| Tool | What it returns |
|---|---|
| `get_quote(ticker)` | price, daily change, source, `as_of`, `is_cached`, `is_mock` |
| `get_price_history(ticker, days=90)` | trend snapshot (52-week range, SMA 20/50, 1m/6m returns, trend label) + daily closes |
| `search_news(query)` | recent headlines with links (Tavily → labelled sample fallback) |
| `analyze_portfolio(holdings \| text)` | value, weights, effective N, asset/sector mix, fund costs, volatility, diversification score, educational observations |
| `calculate_goal_plan(target_amount, years, current_savings, monthly_contribution, risk_profile)` | inflation-adjusted target, required monthly saving in low/expected/high scenarios, yearly projection |
| `search_knowledge_base(query, category, k)` | cited passages from IRS / SEC Investor.gov / articles + a confidence flag |
| `ask_finnie(question, thread_id, knowledge_level)` | the full multi-agent answer (router → agents → synthesizer) with citations; reuse `thread_id` for follow-ups |

## Setup (Windows)

1. Make sure the project works: `.venv` exists, `python -m src.rag.ingest` has built `data/index/`, keys are in `.env`.
2. Open Claude Desktop → **Settings → Developer → Edit Config**. This opens
   `%APPDATA%\Claude\claude_desktop_config.json`.
3. Add the `finnie` entry from `src/mcp_server/claude_desktop_config.example.json` (adjust the paths if
   the project isn't in `C:\dev\capstone-project`). Use the **absolute path to the venv's `python.exe`** and
   to `server.py` — Claude Desktop doesn't start in the project folder.
4. Fully quit Claude Desktop (tray icon → Quit) and reopen it. The tools icon in the chat box should list
   the 7 Finnie tools.
5. Try: *"Use Finnie to analyze a portfolio of 10 VTI, 5 BND and 3 AAPL"* or *"Ask Finnie what the 2025
   Roth IRA limit is, with sources."*

API keys are read from the project's `.env` (the server finds it by its own location), so they never go
into the Claude Desktop config.

## Test without Claude Desktop

```
python -m scripts.mcp_smoke_test          # starts the server over stdio, lists tools, calls 3 of them
pytest tests/integration/test_mcp_server.py
```

MCP Inspector (needs Node.js): `npx @modelcontextprotocol/inspector .venv\Scripts\python.exe src\mcp_server\server.py`

## Troubleshooting

| Symptom | Fix |
|---|---|
| Finnie tools don't appear | Check the JSON is valid and paths use `\\`; fully quit and restart Claude Desktop; see `%APPDATA%\Claude\logs\mcp-server-finnie.log` |
| First knowledge-base call is slow | The embedding model loads on first use (~5–10 s); later calls are fast |
| Market data says `is_mock: true` | Live providers failed (no key / Alpha Vantage's 25/day used up / network); the chain fell back to labelled sample data |
| Server crashes on start | Run `python src\mcp_server\server.py` in a terminal to see the error (stdout is the protocol channel; logs go to stderr) |

## Why MCP and not A2A?

MCP standardizes how an assistant calls **tools**; it gives Finnie an immediate, demonstrable integration
with Claude Desktop at low cost. A2A (agent-to-agent) is for independent agents negotiating with each
other across services — useful if Finnie's agents were deployed as separate services, which they aren't.
A2A stays a possible extension (expose the Market agent as an A2A service).
