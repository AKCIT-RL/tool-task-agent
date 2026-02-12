"""Message history validation and trimming."""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


def validate_message_history(messages: list) -> list:
    """Remove orphaned ToolMessages and ensure history starts with SystemMessage or HumanMessage."""
    if not messages:
        return []
    cleaned = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            if i > 0 and isinstance(messages[i - 1], AIMessage):
                cleaned.append(msg)
            else:
                logger.warning("Removing orphaned ToolMessage at position %s", i)
        else:
            cleaned.append(msg)
    # Preserve SystemMessage at the beginning, then ensure HumanMessage follows
    system_messages = []
    while cleaned and isinstance(cleaned[0], SystemMessage):
        system_messages.append(cleaned.pop(0))
    while cleaned and not isinstance(cleaned[0], HumanMessage):
        logger.warning("History did not start with SystemMessage/HumanMessage, removing leading messages")
        cleaned.pop(0)
    validated = system_messages + cleaned
    # Re-validate ToolMessages
    final_validated = []
    for i, msg in enumerate(validated):
        if isinstance(msg, ToolMessage):
            if i > 0 and isinstance(validated[i - 1], AIMessage):
                final_validated.append(msg)
            else:
                logger.warning("Removing invalid ToolMessage at position %s (no AIMessage before)", i)
        else:
            final_validated.append(msg)
    return final_validated


def limit_message_history(messages: list, max_size: int = 20) -> list:
    """Limit history to max_size while keeping SystemMessage at start and complete HumanMessage/AIMessage/ToolMessage trios."""
    if len(messages) <= max_size:
        return messages
    # Preserve SystemMessage at the beginning
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    cleaned = [m for m in messages if isinstance(m, (HumanMessage, AIMessage, ToolMessage))]
    while cleaned and not isinstance(cleaned[0], HumanMessage):
        cleaned.pop(0)
    if len(cleaned) <= max_size:
        return system_messages + cleaned
    new_history = system_messages + [cleaned[0]]
    remaining = cleaned[1:]
    if len(remaining) > max_size - 1:
        trios = []
        i = len(remaining) - 1
        while i >= 2:
            if (
                isinstance(remaining[i], ToolMessage)
                and isinstance(remaining[i - 1], AIMessage)
                and isinstance(remaining[i - 2], HumanMessage)
            ):
                trios.insert(0, remaining[i - 2 : i + 1])
                i -= 3
            else:
                i -= 1
        if len(trios) > 6:
            trios = trios[-6:]
        for trio in trios:
            new_history.extend(trio)
    else:
        new_history.extend(remaining)
    logger.debug("History limited to %s messages", len(new_history))
    return new_history
