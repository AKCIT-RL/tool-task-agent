"""Execute a single MCP tool and normalize result to ActionResult."""
import json
import logging
from typing import Any

from .agent_types import ActionResult
from .utils import normalize_payload

logger = logging.getLogger(__name__)


async def execute_tool(
    tool_registry: dict[str, Any],
    tool_name: str,
    args: dict[str, Any],
) -> ActionResult:
    """Call tool from registry with args; return ActionResult with success/error/raw_response/step_info."""
    tool = tool_registry.get(tool_name)
    if tool is None:
        logger.error("Tool %s not found", tool_name)
        return ActionResult(success=False, error="Tool not found")

    try:
        logger.debug("Executing tool %s with args=%s", tool_name, args)
        result = await tool(**args)
        try:
            normalized = normalize_payload(result, f"tool_{tool_name}", allow_list=True)
            if isinstance(normalized, list):
                if not normalized:
                    logger.warning("Tool %s returned empty list", tool_name)
                    return ActionResult(success=False, error="Empty result from tool", raw_response="[]")
                normalized = normalized[0]
            if not isinstance(normalized, dict):
                raw = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result
                logger.debug("Tool %s returned unexpected type: %s", tool_name, type(normalized))
                return ActionResult(success=True, raw_response=raw)
            success = normalized.get("success", normalized.get("lastActionSuccess", True))
            error = normalized.get("error") or normalized.get("errorMessage")
            step_info = normalized.get("info")
            raw_response = json.dumps(normalized, ensure_ascii=False)
            if success:
                logger.info("Tool %s executed successfully", tool_name)
            else:
                logger.warning("Tool %s failed: %s", tool_name, error)
            return ActionResult(success=success, error=error, raw_response=raw_response, step_info=step_info)
        except ValueError:
            raw = json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else str(result)
            logger.debug("Tool %s returned non-normalizable format, assuming success", tool_name)
            return ActionResult(success=True, raw_response=raw)
    except Exception as e:
        logger.exception("Error executing tool %s", tool_name)
        err_msg = str(e)
        return ActionResult(
            success=False,
            error=err_msg,
            raw_response=json.dumps({"success": False, "error": err_msg}, ensure_ascii=False),
        )
