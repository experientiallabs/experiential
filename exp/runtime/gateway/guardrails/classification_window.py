"""Bounded, role-labelled request window that a classification check sends to its model."""

from __future__ import annotations

from typing import Final

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest

MAX_WINDOW_CHARACTERS: Final = 30_000
"""Total kept characters; at six JSON bytes per character this leaves room for questions."""
_ELISION: Final = "\n[...]\n"
_INSTRUCTION_ROLES: Final = frozenset({"system", "developer"})


class ClassificationWindow(ContractModel):
    """Which request text a classification check classifies, and how much of it.

    The window ends at the latest user message that carries text. Messages after
    it, such as assistant tool calls and tool results in an agent loop, are left
    out, so one user turn keeps the same window, and therefore the same verdict,
    across its whole tool loop. User text that arrives right after tool results,
    such as a client-injected reminder, is a user message like any other.

    Attributes:
        system_chars: Leading characters of system and developer text, default 1,200.
        latest_user_chars: Head and tail characters of the latest user message, default 6,000.
        prior_messages: Text-bearing messages kept just before it, default 2.
        prior_chars: Head and tail characters of each prior message, default 1,500.
    """

    system_chars: int = Field(default=1_200, ge=0, le=MAX_WINDOW_CHARACTERS)
    latest_user_chars: int = Field(default=6_000, ge=1, le=MAX_WINDOW_CHARACTERS)
    prior_messages: int = Field(default=2, ge=0, le=8)
    prior_chars: int = Field(default=1_500, ge=1, le=MAX_WINDOW_CHARACTERS)

    @model_validator(mode="after")
    def _bounded_total(self) -> ClassificationWindow:
        """Keep the whole window inside the decision request input limit.

        Returns:
            The validated window.

        Raises:
            ValueError: The sections together keep more than 30,000 characters.
        """
        if self.kept_characters > MAX_WINDOW_CHARACTERS:
            raise ValueError("a classification window keeps at most 30000 characters in total")
        return self

    @property
    def kept_characters(self) -> int:
        """Return the most message characters this window can keep, before labels."""
        return self.system_chars + self.latest_user_chars + self.prior_messages * self.prior_chars


def classification_window(request: GatewayRequest, window: ClassificationWindow) -> str:
    """Return the labelled window text, or an empty string when the request has no text.

    Sections appear in conversation order: the head of the system and developer
    text, the prior messages, then the latest user message. Each starts with its
    role in brackets. Attachments, reasoning, and opaque provider items carry no
    classifiable text and are skipped, so no message shape is ever refused here.

    Args:
        request: Canonical request after continuation expansion.
        window: Section bounds.

    Returns:
        The state text for the decision model.
    """
    messages = request.messages
    latest = _latest_user_index(messages)
    end = len(messages) if latest is None else latest
    sections: list[str] = []
    instructions = "\n\n".join(
        text
        for message in messages[:end]
        if message.role in _INSTRUCTION_ROLES and (text := message_text(message))
    )
    if instructions and window.system_chars:
        sections.append(f"[system]\n{_head(instructions, window.system_chars)}")
    prior: list[str] = []
    for message in reversed(messages[:end]):
        if len(prior) == window.prior_messages:
            break
        if message.role in _INSTRUCTION_ROLES:
            continue
        text = message_text(message)
        if text:
            prior.append(f"[{message.role}]\n{_head_and_tail(text, window.prior_chars)}")
    sections.extend(reversed(prior))
    if latest is not None:
        text = message_text(messages[latest])
        sections.append(f"[latest user]\n{_head_and_tail(text, window.latest_user_chars)}")
    return "\n\n".join(sections)


def message_text(message: GatewayMessage) -> str:
    """Return the classifiable text of one canonical message.

    ``content`` already holds the flattened text of content parts and Anthropic
    text blocks, with a failed tool result's error prefix folded in. Only two
    assistant shapes keep text elsewhere: one carried Anthropic block, such as
    cited text, and an ordered Anthropic block list with no flattened content.

    Args:
        message: One canonical gateway message.

    Returns:
        The message text, or an empty string when it carries none.
    """
    if message.provider_native_item is not None:
        return ""
    if message.content is not None:
        return message.folded_tool_error_content()
    if message.provider_anthropic_block is not None:
        return _block_text(message.provider_anthropic_block)
    if message.provider_anthropic_blocks is not None:
        return "".join(_block_text(block) for block in message.provider_anthropic_blocks)
    return ""


def _latest_user_index(messages: tuple[GatewayMessage, ...]) -> int | None:
    """Return the index of the last user message with non-blank text, if any."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "user" and message_text(message).strip():
            return index
    return None


def _block_text(block: JsonObject) -> str:
    """Return the text of one Anthropic text block, or nothing for other block types."""
    text = block.get("text")
    return text if block.get("type") == "text" and isinstance(text, str) else ""


def _head(text: str, limit: int) -> str:
    """Keep the first ``limit`` characters and mark any cut."""
    return text if len(text) <= limit else text[:limit] + _ELISION.rstrip("\n")


def _head_and_tail(text: str, limit: int) -> str:
    """Keep ``limit`` characters split between the start and the end of ``text``."""
    if len(text) <= limit:
        return text
    tail = limit // 2
    return text[: limit - tail] + _ELISION + text[len(text) - tail :]
