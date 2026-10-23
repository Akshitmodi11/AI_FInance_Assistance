"""News agent — recent headlines (Tavily web search) summarized with links and dates."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.core.schemas import AgentResult, Citation
from src.core.state import AgentInput
from src.utils import disclaimers

TASK = """Summarize these news items in 3-5 bullets. Each bullet: what happened, why it might
matter to an investor in general terms, and its source number like [1]. Don't predict prices or
recommend trades. If items are marked as sample data, say so."""


class NewsAgent(BaseAgent):
    name = "news"

    def _run(self, inp: AgentInput) -> AgentResult:
        tickers = inp.get("tickers") or []
        query = (
            f"{' '.join(tickers)} stock news" if tickers else inp.get("question", "markets news")
        )
        items = self.deps.news.search(query)
        is_mock = any(i.is_mock for i in items)
        numbered = [
            f"[{n}] {i.title} — {i.source}{f' ({i.published})' if i.published else ''}\n{i.summary}"
            for n, i in enumerate(items, start=1)
        ]
        fallback = "\n".join(f"- [{i.title}]({i.url}) — {i.source}" for i in items)
        answer = self.explain(inp, TASK, "\n\n".join(numbered), fallback)
        if is_mock:
            answer += f"\n\n_{disclaimers.MOCK_DATA}_"
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=[Citation(label=i.title, url=i.url, source=i.source) for i in items],
            data={"news": [i.model_dump() for i in items]},
        )
