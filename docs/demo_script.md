# Demo script (target 8–9 minutes)

**Before recording:** `python -m scripts.check_keys` (all ✓) · `streamlit run src/web_app/app.py` open in
the browser · LangSmith project `finnie` open in another tab · Claude Desktop with the Finnie MCP server
connected · sidebar: knowledge level **beginner**, risk **moderate**. Close other notifications.

| Time | Show | Say (key points) | Rubric |
|---|---|---|---|
| 0:00–0:40 | README top + architecture diagram | "Finnie: 6 specialist agents coordinated by LangGraph, grounded in IRS/SEC sources and live market data. The LLM explains; Python computes every number." | architecture |
| 0:40–2:00 | **Chat**: *What's the traditional IRA contribution limit for 2025?* then *what about for a Roth?* | Point at "Answered by: 🧾 Tax", open **Sources** (IRS Pub 590-A, page numbers). Follow-up resolved from memory — router rewrote it into a standalone question. | RAG, memory, education |
| 2:00–2:45 | Sidebar → **advanced**, ask *What is an index fund?* | Answers adapt to knowledge level. | conversational |
| 2:45–3:30 | Chat: *How is NVDA doing, and any news on it?* | Two agents in parallel (Market + News), live price chart with 20/50-day averages, "Source · as of" label, real headlines with links. | multi-agent, real-time |
| 3:30–4:00 | Chat: *Should I buy Tesla?* then *How do I trade on insider information?* | Guardrail: educational reframe, no recommendation; illegal request refused before any agent runs. | safety |
| 4:00–5:15 | **Portfolio** tab: sample **Single Stock** → Analyze, then **Balanced Three Fund** → Analyze | Score/gauge, concentration warning, donuts, sector bar, costs, volatility. Then Chat: *Is my portfolio diversified enough?* — uses the portfolio from the tab. | portfolio, visualization |
| 5:15–6:00 | **Goals** tab: $40,000 house in 6 years, $5,000 saved, $400/month; switch risk to aggressive | Inflation-adjusted target, 3 scenarios, on-track flags, projection band. Risk appetite changes the assumptions. | goal planning, viz |
| 6:00–6:30 | **Markets** tab: index cards, look up VTI, switch 1M/1Y | Freshness labels; mention fallback Alpha Vantage → yfinance → mock and the 30-min cache. | market intelligence |
| 6:30–7:00 | **Knowledge** tab: search *HSA contribution limit*, then *best pizza in Naples* | Hybrid retrieval with scores; the second shows the "I don't know" gate. | RAG |
| 7:00–7:45 | **Claude Desktop**: *Use Finnie to analyze 10 VTI, 5 BND, 3 AAPL* | Same tools via MCP — 7 tools, shared code with the app. | MCP bonus |
| 7:45–8:15 | **LangSmith** trace of the NVDA question | guardrail → router → market ∥ news → synthesizer, tokens and latency per step. | observability |
| 8:15–8:50 | `docs/benchmarks.md` summary table + GitHub Actions green run | Recall@5 0.92 hybrid vs 0.86 dense; routing 95%; safety 18/18; 163 tests, 91% coverage; ~$0.005/question. | testing, docs |

**Backup if something is slow/fails live:** the fallbacks are a feature — say "Alpha Vantage's free tier is
25 calls a day, so this switched to Yahoo Finance; the label shows it." Keep a second browser tab with a
finished answer ready.
