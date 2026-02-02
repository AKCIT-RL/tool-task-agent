"""Quick test: connect to MCP and call get_task_instruction (or another tool)."""
import asyncio
import json
import logging
import os

from dotenv import load_dotenv, find_dotenv

# Allow running from project root: python scripts/test_mcp.py
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcp_benchmark import MCPClient, build_tool_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def main() -> None:
    load_dotenv(find_dotenv())
    url = os.getenv("MCP_URL", "http://localhost:8000/mcp")
    mcp = MCPClient(url)
    try:
        await mcp.connect()
        tools = await mcp.load_tools()
        registry = build_tool_registry(tools)
        if "get_task_instruction" not in registry:
            print("ERROR: get_task_instruction not available")
            return
        result = await registry["get_task_instruction"]()
        print("Result type:", type(result))
        print("Result:", result)
        if isinstance(result, list) and result and isinstance(result[0], str):
            try:
                parsed = json.loads(result[0])
                print("Parsed JSON:", json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("Not valid JSON")
    finally:
        await mcp.close()
        print("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
