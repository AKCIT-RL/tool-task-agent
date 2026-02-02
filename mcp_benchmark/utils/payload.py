"""Parse and normalize JSON payloads from MCP tools."""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_json(payload: Any, name: str) -> dict[str, Any]:
    """Parse payload (str/dict/list or message with .content) to a dict. Raises ValueError on failure."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"{name} returned empty list")
        first = payload[0]
        if isinstance(first, dict):
            return first
        if isinstance(first, str):
            try:
                return json.loads(first)
            except json.JSONDecodeError as e:
                raise ValueError(f"{name} returned unparseable string: {first}") from e
        if hasattr(first, "content"):
            return parse_json(first.content, name)
        raise ValueError(f"{name} returned unexpected list item: {type(first)}")
    if isinstance(payload, str):
        cleaned = payload.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} returned unparseable string: {payload}") from e
    if hasattr(payload, "content"):
        return parse_json(payload.content, name)
    raise ValueError(f"{name} returned unexpected type: {type(payload)}")


def normalize_payload(payload: Any, name: str, allow_list: bool = False) -> dict | list:
    """Normalize tool return (list/str/dict) to dict or list. Raises ValueError on failure."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if not payload:
            logger.debug("%s returned empty list (may be valid)", name)
            return []
        first = payload[0]
        if isinstance(first, dict):
            if allow_list and all(isinstance(x, dict) for x in payload):
                return payload
            return first
        if isinstance(first, str):
            try:
                parsed = json.loads(first)
            except json.JSONDecodeError as e:
                raise ValueError(f"{name} returned unparseable string: {first}") from e
            if allow_list and isinstance(parsed, list):
                return parsed
            return parsed
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"{name} returned unparseable string: {payload}") from e
        if allow_list and isinstance(parsed, list):
            return parsed
        return parsed
    raise ValueError(f"{name} returned unexpected type: {type(payload)}")
