"""LLM wrapper for the benchmark agent."""
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from .utils.message_utils import validate_message_history

logger = logging.getLogger(__name__)


class PromptLLM:
    """Wrapper that exposes ainvoke(prompt, images) and ainvoke_with_tools for the agent."""

    def __init__(self, model: str):
        self._client = ChatOpenAI(model=model, temperature=0)

    async def ainvoke(self, prompt: str, images: list[str] | None = None) -> str:
        if images:
            content = [{"type": "text", "text": prompt}]
            for img_b64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            message = [HumanMessage(content=content)]
        else:
            message = prompt

        response = await self._client.ainvoke(message)
        return response.content if hasattr(response, "content") else str(response)

    async def ainvoke_with_tools(
        self,
        prompt: str,
        tools: list,
        message_history: list | None = None,
    ) -> Any:
        """Invoke LLM with bound tools and optional message history."""
        llm_with_tools = self._client.bind_tools(tools)
        messages = validate_message_history(message_history) if message_history else []
        current_human = HumanMessage(content=prompt)
        messages.append(current_human)
        validated = validate_message_history(messages)

        response = await llm_with_tools.ainvoke(validated)
        if message_history is not None:
            message_history.append(current_human)
        return response
