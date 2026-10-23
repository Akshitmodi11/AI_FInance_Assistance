"""Finance Q&A agent — answers concept questions from the knowledge base, with citations.

Uses the hybrid retriever; if the best dense match is below the confidence threshold it says it
doesn't know instead of guessing (the "I don't know" gate measured in docs/benchmarks.md).
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.core.schemas import AgentResult, Citation
from src.core.state import AgentInput
from src.rag.models import RetrievalResult

NO_ANSWER = (
    "I don't have reliable information on that in my knowledge base, so I'd rather not guess. "
    "Try rephrasing, or ask about investing basics, funds, diversification or retirement accounts."
)

TASK = """Answer the question using the numbered sources. Cite each fact with its number, like [1].
If sources disagree or a figure depends on the tax year, say which year the source covers."""


class FinanceQAAgent(BaseAgent):
    name = "finance_qa"
    category: str | None = None  # restrict retrieval to one KB category (Tax agent uses this)
    extra_task: str = ""

    def retrieve(self, question: str) -> RetrievalResult:
        result = self.deps.retriever.retrieve(question, category=self.category)
        if self.category and not result.confident:  # widen the search before giving up
            result = self.deps.retriever.retrieve(question)
        return result

    def _run(self, inp: AgentInput) -> AgentResult:
        question = inp.get("question", "")
        result = self.retrieve(question)
        if not result.confident:
            return AgentResult(agent=self.name, answer=NO_ANSWER, confident=False)

        citations: list[Citation] = []
        sources = []
        for i, ev in enumerate(result.evidence, start=1):
            meta = ev.chunk.meta
            year = f" (tax year {meta.tax_year})" if meta.tax_year else ""
            sources.append(f"[{i}] {meta.citation}{year}\n{ev.chunk.text}")
            citations.append(Citation(label=meta.citation, url=meta.url, source=meta.source))

        fallback = "Here is what my sources say:\n\n" + "\n\n".join(
            f"[{i}] {ev.chunk.text[:400]}…" for i, ev in enumerate(result.evidence[:3], start=1)
        )
        answer = self.explain(inp, TASK + self.extra_task, "\n\n".join(sources), fallback)
        return AgentResult(
            agent=self.name,
            answer=answer,
            citations=citations,
            data={"top_dense_score": result.top_dense_score, "retrieval_mode": result.mode},
        )
