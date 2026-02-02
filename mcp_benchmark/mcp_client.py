"""MCP client: connect to server and load tools as LangChain StructuredTools."""
import logging
from typing import Any, Callable

from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

logger = logging.getLogger(__name__)


class MCPClient:
    """Async MCP client: connect, load tools, close."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._client: Client | None = None
        self._tools: list[StructuredTool] = []

    async def connect(self) -> None:
        """Open MCP connection and keep it alive."""
        self._client = Client(self.url)
        await self._client.__aenter__()
        logger.info("Connected to MCP: %s", self.url)

    async def close(self) -> None:
        """Close the MCP connection."""
        if self._client:
            await self._client.__aexit__(None, None, None)
            self._client = None
            logger.info("MCP connection closed")

    async def load_tools(self) -> list[StructuredTool]:
        """List MCP tools and convert them to LangChain StructuredTools."""
        if not self._client:
            raise RuntimeError("MCPClient not connected")
        mcp_tools = await self._client.list_tools()
        self._tools = self._convert_tools(mcp_tools)
        logger.info("MCP tools loaded: %s", [t.name for t in self._tools])
        return self._tools

    def _convert_tools(self, mcp_tools: list) -> list[StructuredTool]:
        result = []
        for tool in mcp_tools:
            schema = self._schema_to_pydantic(f"{tool.name}Schema", tool.inputSchema or {})
            coro = self._make_tool_coro(tool.name)
            result.append(
                StructuredTool.from_function(
                    func=None,
                    coroutine=coro,
                    name=tool.name,
                    description=tool.description or "",
                    args_schema=schema,
                )
            )
        return result

    def _make_tool_coro(self, tool_name: str) -> Callable[..., Any]:
        async def call(**kwargs: Any) -> list[str]:
            result = await self._client.call_tool(tool_name, arguments=kwargs)
            return [
                c.text
                for c in result.content
                if hasattr(c, "text") and c.text is not None
            ]
        return call

    def _schema_to_pydantic(self, name: str, schema: dict) -> type:
        fields = {}
        for fname, finfo in schema.get("properties", {}).items():
            t = finfo.get("type", "string")
            py_type = (
                int if t == "integer"
                else float if t == "number"
                else bool if t == "boolean"
                else str
            )
            fields[fname] = (py_type, Field(description=finfo.get("description", "")))
        return create_model(name, **fields)


def build_tool_registry(tools: list[StructuredTool]) -> dict[str, Callable[..., Any]]:
    """Build a name -> coroutine mapping for tool execution."""
    return {t.name: t.coroutine for t in tools}
