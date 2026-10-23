"""Check that the Qdrant credentials in .env work — without ever printing the key.

Usage:  python -m scripts.check_qdrant
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient

from src.core.config import Secrets


def main() -> int:
    secrets = Secrets()
    if not secrets.qdrant_url:
        print(
            "QDRANT_URL is empty in .env → Finnie will use Qdrant local mode (no account needed)."
        )
        return 0
    if secrets.qdrant_api_key is None or not secrets.qdrant_api_key.get_secret_value():
        print("QDRANT_URL is set but QDRANT_API_KEY is empty — add the key to .env.")
        return 1

    client = QdrantClient(
        url=secrets.qdrant_url,
        api_key=secrets.qdrant_api_key.get_secret_value(),
        timeout=10,
    )
    try:
        collections = client.get_collections().collections
    except Exception as exc:  # noqa: BLE001 — report any connection/auth problem plainly
        print(f"✗ Could not reach Qdrant at {secrets.qdrant_url}: {type(exc).__name__}: {exc}")
        return 1
    names = [c.name for c in collections] or ["(none yet)"]
    print(f"✓ Connected to Qdrant Cloud at {secrets.qdrant_url}")
    print(f"  Collections: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
