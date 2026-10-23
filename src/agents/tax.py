"""Tax Education agent — retirement accounts and tax rules from IRS publications.

Same retrieval-and-cite pattern as Finance Q&A, but searches the ``retirement_tax`` category first
and always ends with the fixed "not tax advice" boundary.
"""

from __future__ import annotations

from src.agents.finance_qa import FinanceQAAgent
from src.core.schemas import AgentResult
from src.core.state import AgentInput
from src.utils import disclaimers


class TaxAgent(FinanceQAAgent):
    name = "tax"
    category = "retirement_tax"
    extra_task = (
        "\nWhen comparing accounts (traditional vs Roth IRA, 401(k) vs IRA, HSA), use a short "
        "side-by-side list. Always state the tax year that limits and thresholds apply to."
    )

    def _run(self, inp: AgentInput) -> AgentResult:
        result = super()._run(inp)
        if result.confident:
            result.answer = f"{result.answer}\n\n_{disclaimers.TAX}_"
        return result
