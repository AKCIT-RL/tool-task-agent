"""LLM interaction history logging."""
import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

LOG_DIR = "logs"


def _write_header(f: Any, goal: str, count: int) -> None:
    f.write("=" * 80 + "\n")
    f.write("LLM EXECUTION HISTORY\n")
    f.write(f"Goal: {goal}\n")
    f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total interactions: {count}\n")
    f.write("=" * 80 + "\n\n")


def _format_value(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def save_llm_history(llm_history: list[dict] | None, goal: str, episode_id: int | None = None) -> None:
    """Save LLM interaction history to a file under logs/ (timestamped; optional episode_id in filename)."""
    history = llm_history or []
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"llm_history_ep{episode_id}_{timestamp}" if episode_id is not None else f"llm_history_{timestamp}"
    path = os.path.join(LOG_DIR, f"{name}.txt")

    try:
        with open(path, "w", encoding="utf-8") as f:
            _write_header(f, goal, len(history))
            if not history:
                f.write("No LLM interactions were recorded during this execution.\n")
                logger.info("Empty history file created at: %s", path)
                return

            for i, interaction in enumerate(history, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"INTERACTION {i} - Step {interaction.get('step', 'N/A')}\n")
                f.write(f"Type: {interaction.get('type', 'unknown').upper()}\n")
                f.write(f"{'=' * 80}\n\n")

                f.write("PROMPT:\n")
                f.write("-" * 80 + "\n")
                f.write(_format_value(interaction.get("prompt", "")) + "\n\n")

                if interaction.get("reasoning"):
                    f.write("REASONING:\n")
                    f.write("-" * 80 + "\n")
                    f.write(_format_value(interaction["reasoning"]) + "\n\n")

                f.write("RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(_format_value(interaction.get("response", "")) + "\n\n")

                tool_calls = interaction.get("tool_calls")
                if tool_calls:
                    f.write("TOOL CALLS:\n")
                    f.write("-" * 80 + "\n")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                f.write(f"  Tool: {tc.get('name', 'unknown')}\n")
                                f.write(f"  Args: {json.dumps(tc.get('args', {}), indent=2, ensure_ascii=False)}\n")
                            else:
                                f.write(f"  {tc}\n")
                    else:
                        f.write(str(tool_calls))
                    f.write("\n")
                f.write("\n")

        logger.info("LLM history saved at: %s (%s interactions)", path, len(history))
    except OSError as e:
        logger.error("Error saving LLM history: %s", e, exc_info=True)
