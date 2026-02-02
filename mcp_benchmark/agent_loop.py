"""Main agent loop: perceive, build state, decide, execute until task_success or max_steps."""
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from .config import EXCLUDED_ACTION_TOOLS
from .decision import select_action
from .executor import execute_tool
from .perception import interpret_scene
from .prompts import TASK_REASONING_PROMPT
from .world_state import build_world_state
from .utils.message_utils import limit_message_history

logger = logging.getLogger(__name__)

GREEN = "\033[32m"
RESET = "\033[0m"


async def _check_task_success(tool_registry: dict[str, Any]) -> bool:
    """Return True if get_environment_state reports task_success."""
    get_env = tool_registry.get("get_environment_state")
    if not get_env:
        return False
    try:
        raw = await get_env()
        data = raw[0]
        if isinstance(data, str):
            data = json.loads(data)
        return bool(data.get("task_success", False)) if isinstance(data, dict) else False
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def _get_tool_call_ids(ai_message: AIMessage) -> list[str]:
    """Return all tool_call_ids from AIMessage. OpenAI requires a ToolMessage per id."""
    if not getattr(ai_message, "tool_calls", None):
        return []
    ids = []
    for tc in ai_message.tool_calls:
        tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        if tid:
            ids.append(tid)
        else:
            ids.append(f"call_{int(time.time() * 1000)}_{len(ids)}")
    return ids


async def agent_loop(
    llm: Any,
    tool_registry: dict[str, Any],
    structured_tools: list,
    goal: str,
    max_steps: int = 100,
    use_image: bool = True,
) -> dict[str, Any]:
    """
    Run the agent loop until task_success or max_steps.
    Returns dict with llm_history, step_infos, goal, task_success.
    """
    last_action: str | None = None
    last_result: Any = None
    message_history: list = []
    llm_interaction_history: list[dict] = []
    episode_step_infos: list[dict] = []

    logger.info("=" * 80)
    logger.info("Starting agent_loop")
    logger.info("=" * 80)

    reasoning_prompt = TASK_REASONING_PROMPT.format(goal=goal)
    logger.info("Reasoning about the task before starting...")
    initial_reasoning = await llm.ainvoke(reasoning_prompt)
    initial_reasoning = initial_reasoning if isinstance(initial_reasoning, str) else (str(initial_reasoning) if initial_reasoning else "")
    llm_interaction_history.append({
        "type": "reasoning",
        "step": None,
        "prompt": reasoning_prompt,
        "reasoning": None,
        "response": initial_reasoning,
        "tool_calls": None,
    })
    logger.info("Initial reasoning: %s", (initial_reasoning[:200] + "...") if len(initial_reasoning) > 200 else initial_reasoning)

    get_visible = tool_registry["get_visible_objects"]
    get_image = tool_registry.get("get_image") or get_visible
    get_agent_status = tool_registry["get_agent_status"]
    action_tools = [t for t in structured_tools if t.name not in EXCLUDED_ACTION_TOOLS]

    for step in range(max_steps):
        logger.info("\n" + "-" * 80)
        logger.info("STEP %s", step)
        logger.info("-" * 80)

        scene, inferred_objects, perception_interaction = await interpret_scene(
            llm, get_visible, get_image, use_image=use_image
        )
        if perception_interaction:
            perception_interaction["step"] = step
            llm_interaction_history.append(perception_interaction)

        world_state = await build_world_state(
            inferred_objects, scene, get_agent_status, last_action, last_result
        )
        logger.debug(
            "Inferred world: objs=%s, front_blocked=%s, agent_status=%s, last_action=%s, last_result=%s",
            world_state.objects,
            world_state.front_blocked,
            world_state.agent_status,
            world_state.last_action,
            world_state.last_result,
        )

        tool_calls_list, ai_message, decision_prompt = await select_action(
            llm, goal, world_state, action_tools, message_history, initial_reasoning=initial_reasoning
        )

        reasoning = getattr(ai_message, "content", None) or None
        llm_interaction_history.append({
            "type": "decision",
            "step": step,
            "prompt": decision_prompt,
            "reasoning": reasoning,
            "response": getattr(ai_message, "content", str(ai_message)),
            "tool_calls": getattr(ai_message, "tool_calls", None),
        })

        logger.info("\n📋 DECISION:")
        if reasoning:
            logger.info("  Reasoning: %s%s%s", GREEN, reasoning, RESET)
        else:
            logger.info("  Response: %s%s%s", GREEN, getattr(ai_message, "content", "No response") or "No response", RESET)

        if tool_calls_list is None:
            logger.info("LLM did not call a tool; checking task_success and continuing.")
            if await _check_task_success(tool_registry):
                logger.info("✓ GOAL ACHIEVED (task_success from environment)")
                break
            continue

        message_history.append(ai_message)
        all_tool_call_ids = _get_tool_call_ids(ai_message)
        last_result = None
        last_action = None

        for idx, (tool_name, args) in enumerate(tool_calls_list):
            tool_call_id = all_tool_call_ids[idx] if idx < len(all_tool_call_ids) else f"call_{int(time.time() * 1000)}_{idx}"
            logger.info("🔧 Executing tool %s/%s: %s", idx + 1, len(tool_calls_list), tool_name)
            if args:
                logger.info("  Arguments: %s", args)
            logger.info("⚙️  Executing %s...", tool_name)
            result = await execute_tool(tool_registry, tool_name, args)

            tool_result = result.raw_response if result.raw_response else f"Success: {result.success}" + (f". Error: {result.error}" if result.error else "")
            message_history.append(ToolMessage(content=tool_result, tool_call_id=tool_call_id))

            last_action = f"{tool_name}({args})"
            last_result = result
            if result.step_info is not None:
                episode_step_infos.append(result.step_info)

            if not result.success:
                logger.warning("❌ FAILURE: %s", tool_name)
                if result.error:
                    logger.warning("  Error: %s", result.error)
            else:
                logger.info("✓ SUCCESS: %s executed", tool_name)

        if await _check_task_success(tool_registry):
            logger.info("✓ GOAL ACHIEVED (task_success from environment)")
            break

        message_history = limit_message_history(message_history, max_size=20)

    if episode_step_infos:
        final_task_success = bool(episode_step_infos[-1].get("task_success", False))
    else:
        final_task_success = await _check_task_success(tool_registry)

    logger.info("=" * 80)
    logger.info("Agent loop finished. Total interactions: %s", len(llm_interaction_history))
    logger.info("=" * 80)

    return {
        "llm_history": llm_interaction_history,
        "step_infos": episode_step_infos,
        "goal": goal,
        "task_success": final_task_success,
    }
