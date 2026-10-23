"""Goal Planning agent — "how much should I save each month to reach $X in N years?"

The goal is extracted from the message (LLM structured output, regex fallback), the user's risk
appetite picks the return band (conservative / moderate / aggressive from config.yaml), and
``plan_goal`` computes the inflation-adjusted target and the low / expected / high scenarios.
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent
from src.agents.tools import parse_goal, plan_goal
from src.core.schemas import AgentResult, GoalInput
from src.core.state import AgentInput

log = logging.getLogger(__name__)

ASK_FOR_GOAL = (
    "To plan a goal I need a **target amount** and a **time frame** — for example: "
    '*"I want $40,000 for a house down payment in 6 years; I have $5,000 saved."* '
    "You can also use the Goals tab."
)

EXTRACT = """Extract the savings goal from the user's message. target_amount and years are
required; current_savings defaults to 0; monthly_contribution only if the user states one.
goal_name is a 1-3 word label (e.g. "house down payment"). Use risk_profile={risk}."""

TASK = """Explain this savings plan: the target in today's vs future dollars (inflation), the
monthly saving needed in the low / expected / high return scenarios, and — if the user gave a
monthly amount — whether they are on track in each. Explain why the scenarios differ (return
assumptions for their risk profile) and that returns are not guaranteed."""


class GoalPlanningAgent(BaseAgent):
    name = "goal"

    def extract_goal(self, inp: AgentInput) -> GoalInput | None:
        risk = self.profile(inp).risk_tolerance
        question = inp.get("question", "")
        if not re.search(r"\d", question):  # no numbers → nothing to plan; don't let an LLM invent
            return None
        try:
            llm = self.deps.llm("router").with_structured_output(GoalInput)
            goal = llm.invoke([SystemMessage(EXTRACT.format(risk=risk)), HumanMessage(question)])
            if isinstance(goal, GoalInput):
                return goal.model_copy(update={"risk_profile": risk})
        except Exception as exc:  # noqa: BLE001 — no LLM or nothing extractable → regex
            log.info("goal extraction via LLM failed, using regex: %s", exc)
        return parse_goal(question, risk)

    def _run(self, inp: AgentInput) -> AgentResult:
        goal = self.extract_goal(inp)
        if goal is None:
            return AgentResult(agent=self.name, answer=ASK_FOR_GOAL, confident=False)
        plan = plan_goal(goal, self.deps.settings.goals)
        lines = [
            f"- {s.label} ({s.annual_return:.0%}/yr): save **${s.required_monthly:,.0f}/month**"
            + ("" if s.on_track is None else f" — on track: {'yes' if s.on_track else 'no'}")
            for s in plan.scenarios
        ]
        fallback = (
            f"Goal: ${goal.target_amount:,.0f} in {goal.years:g} years "
            f"(≈ ${plan.target_in_future_dollars:,.0f} in future dollars at "
            f"{plan.inflation:.1%} inflation), starting from ${goal.current_savings:,.0f}.\n"
            + "\n".join(lines)
        )
        answer = self.explain(inp, TASK, plan.model_dump(exclude={"yearly_projection"}), fallback)
        return AgentResult(
            agent=self.name, answer=answer, data={"goal_plan": plan.model_dump(mode="json")}
        )
