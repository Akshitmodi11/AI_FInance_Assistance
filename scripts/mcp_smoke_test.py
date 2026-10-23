"""Launch the Finnie MCP server over stdio (exactly like Claude Desktop does) and call it.

Usage:  python -m scripts.mcp_smoke_test
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "mcp_server" / "server.py"


def _payload(result) -> str:
    if getattr(result, "structured_content", None):
        return json.dumps(result.structured_content)[:300]
    return " ".join(getattr(c, "text", "") for c in result.content)[:300]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        print(f"✓ connected — {len(tools)} tools: {', '.join(t.name for t in tools)}")
        for name, args in [
            ("get_quote", {"ticker": "VOO"}),
            ("calculate_goal_plan", {"target_amount": 30000, "years": 5, "current_savings": 2000}),
            ("analyze_portfolio", {"text": "10 VTI, 5 BND"}),
        ]:
            result = await session.call_tool(name, args)
            status = "✗" if result.is_error else "✓"
            print(f"{status} {name}: {_payload(result)}…")


if __name__ == "__main__":
    asyncio.run(main())
