"""Chat with Finnie in the terminal.

python -m src.workflow.cli                    # interactive, remembers the conversation
python -m src.workflow.cli "What is a Roth IRA?"   # one question and exit
/new  starts a new conversation · /level beginner|intermediate|advanced · /quit
"""

from __future__ import annotations

import logging
import sys
import uuid

from src.core.config import get_settings
from src.core.schemas import UserProfile
from src.workflow.deps import Deps
from src.workflow.graph import ask, build_graph, sqlite_checkpointer


def _print(final) -> None:
    print(f"\n[{', '.join(final.agents) or 'guardrail'}]\n")
    print(final.answer)
    if final.citations:
        print("\nSources:")
        for i, c in enumerate(final.citations, start=1):
            print(f"  [{i}] {c.label} — {c.url}")
    print(f"\n({final.disclaimer})\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    app = build_graph(
        Deps.from_settings(settings),
        sqlite_checkpointer(settings.resolve(settings.memory.checkpoint_db)),
    )
    profile = UserProfile()
    thread = str(uuid.uuid4())

    if len(sys.argv) > 1:
        _print(ask(app, " ".join(sys.argv[1:]), thread, profile.model_dump()))
        return

    print("Finnie — financial education. /new · /level beginner|intermediate|advanced · /quit")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/new":
            thread = str(uuid.uuid4())
            print("(new conversation)")
            continue
        if question.startswith("/level"):
            level = question.split(maxsplit=1)[-1]
            profile = profile.model_copy(update={"knowledge_level": level})
            print(f"(knowledge level: {level})")
            continue
        _print(ask(app, question, thread, profile.model_dump()))


if __name__ == "__main__":
    main()
