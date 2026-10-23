"""Synthesizer — merges the parallel agents' results into one response.

Deterministic on purpose: one agent → its answer as-is; several → one section per agent, in the
order the router chose. A second LLM call to "blend" them would add latency, cost and a new place
to hallucinate, for little benefit. Citations are de-duplicated; the right disclaimer is attached.
"""

from __future__ import annotations

from src.core.schemas import AgentResult, Citation, FinalResponse
from src.utils import disclaimers

TITLES = {
    "finance_qa": "📚 Explained",
    "tax": "🧾 Tax & retirement accounts",
    "portfolio": "📊 Your portfolio",
    "goal": "🎯 Goal plan",
    "market": "📈 Market",
    "news": "📰 News",
}


def synthesize(
    outputs: dict[str, dict], intents: list[str], guardrail_note: str | None = None
) -> FinalResponse:
    results = [AgentResult.model_validate(outputs[i]) for i in intents if i in outputs]
    if len(results) == 1:
        body = results[0].answer
    else:
        body = "\n\n".join(f"### {TITLES[r.agent]}\n{r.answer}" for r in results)

    citations: list[Citation] = []
    seen = set()
    for c in (c for r in results for c in r.citations):
        if (c.label, c.url) not in seen:
            seen.add((c.label, c.url))
            citations.append(c)

    data = {}
    for r in results:
        data.update(r.data)
    disclaimer = disclaimers.GENERAL
    if any(r.agent == "tax" for r in results):
        disclaimer = f"{disclaimers.TAX} {disclaimers.GENERAL}"
    return FinalResponse(
        answer=body or "Sorry, I couldn't produce an answer.",
        agents=[r.agent for r in results],
        citations=citations,
        data=data,
        guardrail_note=guardrail_note,
        disclaimer=disclaimer,
    )
