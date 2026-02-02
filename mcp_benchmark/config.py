"""Central configuration from environment and constants."""
import os
from dataclasses import dataclass


# MCP and LLM
DEFAULT_MCP_URL = "http://localhost:8000/mcp"
DEFAULT_MODEL = "gpt-4o-mini"

# Tools required to run the benchmark (must be provided by MCP server)
REQUIRED_TOOLS = frozenset({
    "get_visible_objects",
    "get_agent_status",
    "get_task_instruction",
    "reset_environment",
})

# Tools used only for perception/control; excluded from action selection
EXCLUDED_ACTION_TOOLS = frozenset({
    "get_visible_objects",
    "get_observation",
    "get_agent_status",
    "get_image",
    "reset_scene",
    "get_environment_state",
    "get_available_objects",
    "get_current_observation",
    "reset_environment",
    "drop_object",
})


@dataclass(frozen=True)
class BenchmarkConfig:
    """Runtime configuration for the MCP benchmark."""
    mcp_url: str
    model: str
    use_image: bool
    log_level: str
    goal_suffix: str

    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        use_image_raw = os.getenv("USE_IMAGE", "true").lower()
        use_image = use_image_raw in ("true", "1", "yes", "on")
        return cls(
            mcp_url=os.getenv("MCP_URL", DEFAULT_MCP_URL),
            model=os.getenv("BENCHMARK_MODEL", DEFAULT_MODEL),
            use_image=use_image,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            goal_suffix=os.getenv("GOAL_SUFFIX", ""),
        )
