"""Build world state from inferred objects, scene, and agent status."""
import logging
from typing import Any

from .agent_types import InferredWorldState
from .utils import normalize_payload

logger = logging.getLogger(__name__)


async def build_world_state(
    inferred_objects: list,
    scene: Any,
    get_agent_status_tool: Any,
    last_action: str | None,
    last_result: Any,
) -> InferredWorldState:
    """Fetch agent status via MCP and build InferredWorldState."""
    raw = await get_agent_status_tool()
    agent_status = normalize_payload(raw, "get_agent_status")
    if not isinstance(agent_status, dict):
        agent_status = {}
    logger.debug("Agent status: %s", agent_status)
    return InferredWorldState(
        objects=inferred_objects,
        front_blocked=scene.front_blocked,
        agent_status=agent_status,
        last_action=last_action,
        last_result=last_result,
        visual_notes=scene.notes,
    )
