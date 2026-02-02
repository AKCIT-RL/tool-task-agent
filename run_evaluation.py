"""
Run multiple benchmark episodes: get task from MCP get_task_instruction,
run agent_loop per episode, save per-episode JSON, reset between episodes.
"""
import argparse
import asyncio
import json
import logging
import os

from dotenv import load_dotenv, find_dotenv

from mcp_benchmark import (
    BenchmarkConfig,
    MCPClient,
    PromptLLM,
    agent_loop,
    build_tool_registry,
    REQUIRED_TOOLS,
    save_llm_history,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP benchmark over multiple episodes.")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes (default: 1)")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory for JSON (default: eval_results)")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode (default: 100)")
    parser.add_argument("--no-image", action="store_true", help="Disable get_image in perception")
    return parser.parse_args()


async def run_evaluation() -> None:
    args = parse_args()
    load_dotenv(find_dotenv())

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Add to .env or export.")

    cfg = BenchmarkConfig.from_env()
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    use_image = not args.no_image
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    llm = PromptLLM(model=cfg.model)
    mcp = MCPClient(cfg.mcp_url)
    logger.info("Connecting to MCP at %s", cfg.mcp_url)
    await mcp.connect()

    try:
        tools = await mcp.load_tools()
        registry = build_tool_registry(tools)
        missing = REQUIRED_TOOLS - set(registry)
        if missing:
            raise SystemExit(f"Missing required tools: {', '.join(sorted(missing))}")

        for episode_id in range(args.num_episodes):
            logger.info("=" * 60)
            logger.info("Episode %s / %s", episode_id + 1, args.num_episodes)
            logger.info("=" * 60)

            raw = await registry["get_task_instruction"]()
            data = json.loads(raw[0]) if isinstance(raw[0], str) else raw[0]
            goal = (data.get("task_instruction") or data.get("task", "")) + (f" {cfg.goal_suffix}" if cfg.goal_suffix else "").strip()
            goal = goal

            result = await agent_loop(
                llm,
                registry,
                tools,
                goal,
                max_steps=args.max_steps,
                use_image=use_image,
            )

            step_infos = result["step_infos"]
            llm_history = result.get("llm_history") or []
            payload = {
                "episode": episode_id,
                "goal": goal,
                "task_success": result["task_success"],
                "num_steps": len(step_infos),
                "steps": step_infos,
                "llm_history": llm_history,
            }
            path = os.path.join(args.output_dir, f"episode_{episode_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info("Saved %s (task_success=%s, num_steps=%s)", path, result["task_success"], len(step_infos))

            save_llm_history(llm_history, goal, episode_id=episode_id)

            if episode_id < args.num_episodes - 1:
                await registry["reset_environment"]()
                logger.info("Reset environment for next episode")

        logger.info("Evaluation finished. Results in %s", args.output_dir)
    finally:
        await mcp.close()
        logger.info("MCP connection closed")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
