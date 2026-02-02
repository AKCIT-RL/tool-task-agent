"""Entry point: load config, connect MCP, run one benchmark episode, save history."""
import asyncio
import json
import logging
import os

from dotenv import load_dotenv, find_dotenv

from .config import BenchmarkConfig, REQUIRED_TOOLS
from .history import save_llm_history
from .agent_loop import agent_loop
from .llm import PromptLLM
from .mcp_client import MCPClient, build_tool_registry

logger = logging.getLogger(__name__)


def _configure_logging(cfg: BenchmarkConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if cfg.log_level == "DEBUG":
        for name in ("httpcore", "httpx", "openai", "urllib3", "langchain", "langchain_core", "langchain_openai", "mcp", "mcp.client", "mcp.client.streamable_http"):
            logging.getLogger(name).setLevel(logging.WARNING)
        for name in ("__main__", "mcp_benchmark", "agent_loop", "perception", "executor", "decision", "world_state", "mcp_client"):
            logging.getLogger(name).setLevel(logging.DEBUG)


async def run_single_episode(cfg: BenchmarkConfig | None = None) -> dict:
    """Run one benchmark episode: connect MCP, get task, run agent_loop, save history. Returns result dict."""
    load_dotenv(find_dotenv())
    cfg = cfg or BenchmarkConfig.from_env()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Add to .env or export.")

    _configure_logging(cfg)

    required = REQUIRED_TOOLS - {"reset_environment"} | ({"get_image"} if cfg.use_image else set())

    llm = PromptLLM(model=cfg.model)
    mcp = MCPClient(cfg.mcp_url)
    logger.info("Connecting to MCP at %s", cfg.mcp_url)
    await mcp.connect()

    try:
        tools = await mcp.load_tools()
        registry = build_tool_registry(tools)
        missing = required - set(registry)
        if missing:
            raise SystemExit(f"Missing required tools: {', '.join(sorted(missing))}")

        raw = await registry["get_task_instruction"]()
        data = json.loads(raw[0]) if isinstance(raw[0], str) else raw[0]
        goal = (data.get("task_instruction") or data.get("task", "")) + (f" {cfg.goal_suffix}" if cfg.goal_suffix else "").strip()
        logger.info("Starting agent_loop with goal: %s", goal)

        result = await agent_loop(llm, registry, tools, goal, use_image=cfg.use_image)
        llm_history = result.get("llm_history") or []
        logger.info("History received: %s interactions", len(llm_history))
        save_llm_history(llm_history, goal)
        return result
    finally:
        logger.info("Closing MCP connection")
        await mcp.close()


def main() -> None:
    asyncio.run(run_single_episode())


if __name__ == "__main__":
    main()
