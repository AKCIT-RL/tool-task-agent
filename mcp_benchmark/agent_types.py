"""Data types for the benchmark agent."""
from dataclasses import dataclass
from typing import Any


@dataclass
class ActionResult:
    success: bool
    error: str | None = None
    raw_response: str | None = None
    step_info: dict[str, Any] | None = None


@dataclass
class VisualObject:
    label: str
    relative_position: str
    distance: str
    salient: bool


@dataclass
class SceneInterpretation:
    visual_objects: list[VisualObject]
    front_blocked: bool
    camera_orientation: str
    notes: str | None = None


@dataclass
class InferredObject:
    label: str
    raw_data: dict[str, Any] | None = None
    relative_position: str | None = None
    distance: str | None = None

    @property
    def object_id(self) -> str | None:
        if self.raw_data:
            return self.raw_data.get("objectId")
        return None

    @property
    def pickupable(self) -> bool:
        if self.raw_data:
            return self.raw_data.get("properties", {}).get("pickupable", False)
        return False

    def _format_properties(self, properties: dict[str, Any]) -> list[str]:
        keys = ["pickupable", "openable", "receptacle", "sliceable", "toggleable"]
        return [k for k in keys if properties.get(k)]

    def _format_state(self, state: dict[str, Any]) -> list[str]:
        parts = []
        if "isOpen" in state:
            parts.append(f"open={state['isOpen']}")
        if "isToggled" in state:
            parts.append(f"on={state['isToggled']}")
        if "isDirty" in state:
            parts.append(f"dirty={state['isDirty']}")
        return parts

    def __str__(self) -> str:
        parts = [f"Object: {self.label}"]
        if self.object_id:
            parts.append(f"ID: {self.object_id}")
        if self.raw_data:
            dist = self.raw_data.get("distance")
            if dist is not None:
                parts.append(f"Distance: {dist:.2f}" if isinstance(dist, (int, float)) else f"Distance: {dist}")
            props = self.raw_data.get("properties")
            if not isinstance(props, dict):
                props = {k: self.raw_data.get(k) for k in ["pickupable", "openable", "receptacle", "sliceable", "toggleable"] if self.raw_data.get(k)} or {}
            prop_str = ", ".join(self._format_properties(props or {})) or "-"
            parts.append(f"Properties: {prop_str}")
            state = self.raw_data.get("state", {})
            states = self._format_state(state)
            if states:
                parts.append(f"State: {', '.join(states)}")
            sr = self.raw_data.get("screen_region")
            if sr is not None:
                if isinstance(sr, dict):
                    w, h = sr.get("width") or sr.get("w"), sr.get("height") or sr.get("h")
                    parts.append(f"Screen region: x={sr.get('x')}, y={sr.get('y')}, w={w}, h={h}")
                else:
                    parts.append(f"Screen region: {sr}")
        else:
            if self.relative_position:
                parts.append(f"Position: {self.relative_position}")
            if self.distance:
                parts.append(f"Distance: {self.distance}")
        return " | ".join(parts)


@dataclass
class InferredWorldState:
    objects: list[InferredObject]
    front_blocked: bool
    agent_status: dict[str, Any]
    last_action: str | None
    last_result: ActionResult | None
    visual_notes: str | None = None

    def _format_agent_status(self) -> list[str]:
        if not self.agent_status or not isinstance(self.agent_status, dict):
            return []
        lines = []
        pos = self.agent_status.get("position")
        if pos:
            lines.append(f"  Position: {pos}" if not isinstance(pos, dict) else f"  Position: x={pos.get('x', '?')}, y={pos.get('y', '?')}, z={pos.get('z', '?')}")
        rot = self.agent_status.get("rotation")
        if rot:
            lines.append(f"  Rotation: {rot}" if not isinstance(rot, dict) else f"  Rotation: x={rot.get('x', '?')}, y={rot.get('y', '?')}, z={rot.get('z', '?')}")
        inv = self.agent_status.get("inventory")
        if inv is not None:
            lines.append(f"  Inventory: {', '.join(str(i) for i in inv)}" if isinstance(inv, list) and inv else "  Inventory: empty" if isinstance(inv, list) else f"  Inventory: {inv}")
        if self.agent_status.get("cameraHorizon") is not None:
            lines.append(f"  Camera (horizon): {self.agent_status['cameraHorizon']}")
        for key, value in self.agent_status.items():
            if key not in ("position", "rotation", "inventory", "cameraHorizon", "image_base64"):
                lines.append(f"  {key}: {value}")
        return lines

    def _format_objects(self) -> list[str]:
        return [f"  - {obj}" for obj in self.objects] if self.objects else ["No objects visible at the moment."]

    def _format_last_action(self) -> list[str]:
        if not self.last_action:
            return []
        lines = [f"Last action: {self.last_action}"]
        if self.last_result is not None and isinstance(self.last_result, ActionResult):
            lines.append(f"Result: {'success' if self.last_result.success else 'failed'}")
            if self.last_result.error:
                lines.append(f"Error: {self.last_result.error}")
        return lines

    def __str__(self) -> str:
        lines = []
        if self.visual_notes:
            lines.extend([f"Visual observations: {self.visual_notes}", ""])
        lines.append("Visible objects:")
        lines.extend(self._format_objects())
        if self.agent_status:
            lines.extend(["\nAgent status:", *self._format_agent_status()])
        last = self._format_last_action()
        if last:
            lines.extend(["", *last])
        return "\n".join(lines)
