"""Action selection: LLM with tools, optional object_id validation and correction."""
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage

from .agent_types import InferredWorldState
from .prompts import ACTION_SELECTOR_PROMPT

logger = logging.getLogger(__name__)


def _format_initial_reasoning_section(initial_reasoning: str | None) -> str:
    if not initial_reasoning or not initial_reasoning.strip():
        return ""
    return "\nYour initial plan (from reasoning before starting):\n" + initial_reasoning.strip() + "\n"


def validate_and_correct_object_id(
    args: dict,
    world_state: InferredWorldState,
    prompt: str,
) -> tuple[str, dict, AIMessage] | None:
    """
    If args contain object_id, validate against world_state; correct or suggest navigation.
    Returns (nav_action, args, fake_ai_message) when navigation is needed, else None.
    """
    if "object_id" not in args or args["object_id"] is None:
        return None
    valid_ids = {obj.object_id for obj in world_state.objects if obj.object_id is not None}
    requested_id = args["object_id"]
    if requested_id in valid_ids:
        return None
    base_name = requested_id.split("|")[0] if "|" in requested_id else requested_id
    matching = [
        obj for obj in world_state.objects
        if obj.object_id and obj.object_id.split("|")[0].lower() == base_name.lower()
    ]
    if matching:
        correct_id = matching[0].object_id
        logger.warning(
            "LLM used invalid object_id '%s', correcting to '%s'",
            requested_id,
            correct_id,
        )
        args["object_id"] = correct_id
        return None
    without_id = [
        obj for obj in world_state.objects
        if obj.object_id is None and obj.label.lower() == base_name.lower()
    ]
    if without_id:
        obj = without_id[0]
        logger.warning(
            "LLM tried object_id '%s' for '%s' which has no ID yet (at %s, %s). Need to approach.",
            requested_id,
            obj.label,
            obj.relative_position,
            obj.distance,
        )
        nav = "rotate_left" if "left" in (obj.relative_position or "") else "rotate_right" if "right" in (obj.relative_position or "") else "move_ahead"
        fake = AIMessage(
            content=f"Need to approach object {obj.label} at {obj.relative_position}",
            tool_calls=[{"name": nav, "args": {}, "id": f"nav_{int(time.time() * 1000)}"}],
        )
        return nav, {}, fake
    logger.warning(
        "LLM used invalid object_id '%s'. Valid: %s. Trying move_ahead to explore.",
        requested_id,
        sorted(valid_ids),
    )
    fake = AIMessage(
        content=f"Object '{requested_id}' not found. Exploring.",
        tool_calls=[{"name": "move_ahead", "args": {}, "id": f"nav_{int(time.time() * 1000)}"}],
    )
    return "move_ahead", {}, fake


def _parse_tool_call(tc: dict | Any) -> tuple[str, dict]:
    """Extract (name, args) from a single tool call."""
    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
    return name, args


async def select_action(
    llm: Any,
    goal: str,
    world_state: InferredWorldState,
    tools: list,
    message_history: list | None = None,
    initial_reasoning: str | None = None,
) -> tuple[list[tuple[str, dict]] | None, AIMessage, str]:
    """Select actions via LLM with bound tools. Returns (list of (tool_name, args), ai_message, prompt) or (None, ai_message, prompt)."""
    initial_section = _format_initial_reasoning_section(initial_reasoning)
    prompt = ACTION_SELECTOR_PROMPT.format(
        goal=goal,
        initial_reasoning_section=initial_section,
        world_state=world_state,
    )
    logger.debug("Sending action prompt: %s", prompt)
    response = await llm.ainvoke_with_tools(prompt, tools, message_history)

    if not isinstance(response, AIMessage):
        ai_message = AIMessage(
            content=response.content if hasattr(response, "content") else str(response),
        )
        if hasattr(response, "tool_calls"):
            ai_message.tool_calls = response.tool_calls
    else:
        ai_message = response

    if not getattr(ai_message, "tool_calls", None):
        return None, ai_message, prompt

    available = {t.name for t in tools}
    tool_calls_list: list[tuple[str, dict]] = []

    for i, tc in enumerate(ai_message.tool_calls):
        tool_name, args = _parse_tool_call(tc)
        if tool_name not in available:
            raise ValueError(
                f"LLM chose invalid tool: '{tool_name}'. Available: {sorted(available)}"
            )
        # Only validate/correct object_id for the first tool; if it needs nav, return single nav action
        if i == 0:
            nav_result = validate_and_correct_object_id(args, world_state, prompt)
            if nav_result is not None:
                return [(nav_result[0], nav_result[1])], nav_result[2], prompt
        tool_calls_list.append((tool_name, args))

    if len(tool_calls_list) > 1:
        logger.info("LLM returned %s tool calls, executing all", len(tool_calls_list))
    return tool_calls_list, ai_message, prompt
