"""Scene interpretation: MCP visible objects + optional image-based detection."""
import logging
from typing import Any

from .agent_types import InferredObject, SceneInterpretation, VisualObject
from .prompts import VISUAL_INTERPRETER_PROMPT
from .utils import parse_json, normalize_payload

logger = logging.getLogger(__name__)


async def interpret_scene(
    llm: Any,
    get_visible_objects_tool: Any,
    get_image_tool: Any,
    use_image: bool = True,
) -> tuple[SceneInterpretation, list[InferredObject], dict | None]:
    """
    Build scene and inferred objects from MCP get_visible_objects; optionally
    use get_image + LLM to add objects not yet in MCP (e.g. distant).
    Returns (SceneInterpretation, inferred_objects, perception_interaction or None).
    """
    raw_visible = await get_visible_objects_tool()
    visible_objects = normalize_payload(raw_visible, "get_visible_objects", allow_list=True)
    if isinstance(visible_objects, dict) and "objects" in visible_objects:
        visible_objects = visible_objects["objects"]
    if not isinstance(visible_objects, list):
        logger.warning("get_visible_objects returned unexpected type, converting to list")
        visible_objects = [visible_objects] if visible_objects else []

    inferred_objects: list[InferredObject] = []
    visible_object_ids: set[str] = set()
    for obj in visible_objects:
        if isinstance(obj, dict):
            oid = obj.get("objectId")
            if oid:
                visible_object_ids.add(oid)
            inferred_objects.append(
                InferredObject(label=obj.get("objectType", "Unknown"), raw_data=obj)
            )

    visual_objects: list[VisualObject] = []
    notes: str | None = None
    visual_objects_raw: list[dict] = []
    perception_interaction: dict | None = None

    if use_image:
        logger.debug("Using get_image for visual interpretation")
        raw_image = await get_image_tool()
        image = normalize_payload(raw_image, "get_image")
        image_b64 = image["image_base64"]
        visual_response = await llm.ainvoke(VISUAL_INTERPRETER_PROMPT, images=[image_b64])
        visual_data = parse_json(visual_response, "VISUAL_INTERPRETER_PROMPT")
        visual_objects_raw = visual_data.get("visual_objects", [])
        notes = visual_data.get("notes")
        logger.debug("Visual interpretation: %s objects detected", len(visual_objects_raw))
        for obj in visual_objects_raw:
            vo = VisualObject(**obj)
            visual_objects.append(vo)
            label_lower = vo.label.lower()
            if not any(inf.label.lower() == label_lower for inf in inferred_objects):
                inferred_objects.append(
                    InferredObject(
                        label=vo.label,
                        raw_data=None,
                        relative_position=vo.relative_position,
                        distance=vo.distance,
                    )
                )
        perception_interaction = {
            "type": "perception",
            "prompt": VISUAL_INTERPRETER_PROMPT,
            "response": visual_response,
        }
    else:
        logger.debug("get_image disabled, using only MCP objects")

    logger.debug(
        "Inferred objects: %s (MCP: %s, image: %s)",
        len(inferred_objects),
        len(visible_object_ids),
        len(visual_objects_raw),
    )

    scene = SceneInterpretation(
        visual_objects=visual_objects,
        front_blocked=False,
        camera_orientation="level",
        notes=notes,
    )
    return scene, inferred_objects, perception_interaction
