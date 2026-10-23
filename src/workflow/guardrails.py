"""Guardrails — run before routing, deterministic (regex), so they can't be talked out of a rule.

* ``illegal``       insider trading, tax evasion, laundering, market manipulation → refuse.
* ``advice``        "should I buy X?", "what stock should I pick?" → answer *educationally*:
                    the agents still run, with a note that Finnie won't pick investments.
* ``guarantee``     "guaranteed returns", "double my money fast" → education + risk note.
* ``personal_tax``  "file my taxes", "how much tax do I owe" → general rules + see a professional.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from src.utils import disclaimers

Category = Literal["ok", "advice", "guarantee", "personal_tax", "illegal"]

_RULES: list[tuple[Category, re.Pattern]] = [
    (
        "illegal",
        re.compile(
            r"insider (trading|info)|non-?public info|evade (my |paying )?tax|tax evasion|"
            r"hide (income|money) from (the )?irs|launder|pump(ing)?[- ]and[- ]dump|"
            r"front[- ]?run|manipulate (the )?(stock|price|market)",
            re.I,
        ),
    ),
    (
        "advice",
        re.compile(
            r"should i (buy|sell|short|invest in|put (my|all|\$?\d)|hold)|"
            r"(which|what) (stocks?|etfs?|funds?|crypto) should i|"
            r"is [\w$.]+ a (good|bad|smart) (buy|investment|stock)|tell me what to (buy|sell)|"
            r"(best|top) (stocks?|etfs?|funds?) to buy|pick (a |some )?stocks? for me",
            re.I,
        ),
    ),
    (
        "guarantee",
        re.compile(
            r"guarantee[ds]? (return|profit|income)|risk[- ]?free (return|profit)|can'?t lose|"
            r"double my money|get rich quick",
            re.I,
        ),
    ),
    (
        "personal_tax",
        re.compile(
            r"(file|prepare|do) my (tax|taxes|return)|how much (tax|taxes) (do|will) i owe|"
            r"fill (out|in) my (1040|w-?4|tax form)",
            re.I,
        ),
    ),
]

NOTES: dict[Category, str | None] = {
    "ok": None,
    "advice": disclaimers.ADVICE_REFRAME,
    "guarantee": "No investment can guarantee returns — higher potential return always comes "
    "with higher risk. Here's how to think about it:",
    "personal_tax": "I can't prepare or file your return, but here are the general rules:",
    "illegal": None,
}

REFUSAL = (
    "I can't help with that — it describes an illegal activity (such as insider trading, tax "
    "evasion or market manipulation). I'm happy to explain how the rules work or how to invest "
    "and pay taxes legally."
)


@dataclass(frozen=True)
class GuardrailResult:
    category: Category
    blocked: bool
    note: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def check(text: str) -> GuardrailResult:
    for category, pattern in _RULES:
        if pattern.search(text):
            return GuardrailResult(category, category == "illegal", NOTES[category])
    return GuardrailResult("ok", False, None)
