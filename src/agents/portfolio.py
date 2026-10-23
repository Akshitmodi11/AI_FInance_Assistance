"""Portfolio agent — diversification, concentration, costs and risk of the user's holdings.

Holdings come from the app (sidebar / CSV upload) or are parsed from the message ("I own 10 AAPL
and 20 VTI"). Every metric is computed by ``analyze_portfolio``; the LLM explains them and adds
educational context from the knowledge base. Observations describe — they never say buy or sell.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.agents.tools import analyze_portfolio, parse_holdings
from src.core.schemas import AgentResult, Citation, Portfolio
from src.core.state import AgentInput
from src.utils import disclaimers

ASK_FOR_HOLDINGS = (
    "I can analyze a portfolio once I know what's in it. Enter your holdings in the Portfolio tab, "
    "or write them like: **10 AAPL, 5 MSFT, 20 VTI, 30 BND** (shares and ticker)."
)

TASK = """Explain this portfolio analysis to the user: total value, how concentrated it is
(top holding, effective number of holdings), the asset-class and sector mix, costs and volatility.
Then go through the observations and explain *why* each matters, using the education sources if
given. Describe trade-offs; never say what to buy or sell. Mention if the data is sample data."""


class PortfolioAgent(BaseAgent):
    name = "portfolio"

    def _run(self, inp: AgentInput) -> AgentResult:
        portfolio = parse_holdings(inp.get("question", ""))
        if not portfolio.holdings and inp.get("portfolio"):
            portfolio = Portfolio.model_validate(inp["portfolio"])
        if not portfolio.holdings:
            return AgentResult(agent=self.name, answer=ASK_FOR_HOLDINGS, confident=False)

        analysis = analyze_portfolio(portfolio, self.deps.market)
        citations, education = [], []
        try:  # a little "why diversification matters" context from the knowledge base
            result = self.deps.retriever.retrieve(
                "why diversification and concentration risk matter", k=2
            )
            for ev in result.evidence if result.confident else []:
                education.append(ev.chunk.text[:600])
                meta = ev.chunk.meta
                citations.append(Citation(label=meta.citation, url=meta.url, source=meta.source))
        except Exception:  # noqa: BLE001 — education context is optional
            pass

        facts = {
            "analysis": analysis.model_dump(exclude={"positions"}),
            "positions": [
                {
                    "ticker": p.ticker,
                    "weight": round(p.weight, 4),
                    "value": round(p.value, 2),
                    "asset_class": p.asset_class,
                    "sector": p.sector,
                }
                for p in analysis.positions
            ],
            "education_sources": education,
        }
        fallback = (
            f"Total value ${analysis.total_value:,.0f} across {len(analysis.positions)} holdings "
            f"(diversification score {analysis.diversification_score}/100).\n\n"
            + "\n".join(f"- {o}" for o in analysis.observations)
        )
        answer = self.explain(inp, TASK, facts, fallback)
        if analysis.is_mock:
            answer += f"\n\n_{disclaimers.MOCK_DATA}_"
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=citations,
            data={"portfolio_analysis": analysis.model_dump(mode="json")},
        )
