"""Fill in API keys in .env from the terminal — input is hidden, values are never printed.

Usage (in your own terminal, from the project root):
    python -m scripts.set_keys            # asks only for keys that are still empty
    python -m scripts.set_keys --all      # asks for every key (Enter keeps the current value)

Creates .env from .env.example if it doesn't exist. .env is gitignored.
"""

from __future__ import annotations

import getpass
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV, EXAMPLE = ROOT / ".env", ROOT / ".env.example"
PLAIN = {"LANGSMITH_TRACING", "LANGSMITH_PROJECT", "QDRANT_URL"}  # not secret → visible input


def main() -> int:
    if not ENV.exists():
        shutil.copy(EXAMPLE, ENV)
    lines = ENV.read_text(encoding="utf-8").splitlines()
    ask_all = "--all" in sys.argv
    changed = []

    for i, line in enumerate(lines):
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if value and not ask_all:
            continue
        hint = "set — Enter keeps it" if value else "empty — Enter skips"
        prompt = f"{name} ({hint}): "
        new = input(prompt) if name in PLAIN else getpass.getpass(prompt)
        new = new.strip().strip('"').strip("'")
        if new:
            lines[i] = f"{name}={new}"
            changed.append(name)

    if any(n == "LANGSMITH_API_KEY" for n in changed):
        lines = ["LANGSMITH_TRACING=true" if ln.startswith("LANGSMITH_TRACING=") else ln
                 for ln in lines]  # fmt: skip
        changed.append("LANGSMITH_TRACING (→ true)")

    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved .env — updated: {', '.join(changed) or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
