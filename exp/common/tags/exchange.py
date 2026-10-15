"""One captured exchange in the minimal chat shape that tagging reads.

``TagExchange`` keeps only what a tag question can see: request messages and response turns in
OpenAI chat form (role, content, tool calls, tool call id), the tool list, and the two exchange
facts that messages alone cannot show. Reasoning is never carried, and image bytes are never
copied: an image part keeps at most its media type and a remote URL.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.tags.records import TagFacts

MessageRole = Literal["system", "developer", "user", "assistant", "tool"]


class TagTextPart(ContractModel):
    """Text inside a multipart message.

    Attributes:
        type: Part discriminator, always ``text``.
        text: The text.
    """

    type: Literal["text"] = "text"
    text: str


class TagImagePart(ContractModel):
    """An image inside a multipart message, recorded without its bytes.

    Attributes:
        type: Part discriminator, always ``image``.
        media_type: Declared media type, when known.
        url: Remote URL of the image, or None for inline data.
    """

    type: Literal["image"] = "image"
    media_type: str | None = None
    url: str | None = None


TagContentPart = Annotated[TagTextPart | TagImagePart, Field(discriminator="type")]


class TagFunction(ContractModel):
    """The function a tool call invokes.

    Attributes:
        name: Tool name.
        arguments: Arguments as JSON text, exactly as the model emitted them when known.
    """

    name: str = Field(min_length=1)
    arguments: str = "{}"


class TagToolCall(ContractModel):
    """One tool call made by the assistant.

    Attributes:
        id: Call identifier that the matching tool result repeats.
        type: Call kind, always ``function``.
        function: Invoked tool and its arguments.
    """

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: TagFunction


class TagMessage(ContractModel):
    """One chat message.

    Attributes:
        role: Speaker.
        content: Text, ordered parts for a message with images, or None.
        tool_calls: Assistant tool calls, in order.
        tool_call_id: For a tool result, the call it answers.
    """

    role: MessageRole
    content: str | tuple[TagContentPart, ...] | None = None
    tool_calls: tuple[TagToolCall, ...] = ()
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _role_fields(self) -> TagMessage:
        """Keep tool calls on assistant messages and call ids on tool results.

        Returns:
            The validated message.

        Raises:
            ValueError: A field does not belong to the message role.
        """
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages carry tool_calls")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("only tool messages carry tool_call_id")
        return self

    def text(self) -> str:
        """Return the message text with each image shown as a space-separated ``[image]``.

        Returns:
            The text, or an empty string when the message has none.
        """
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        text = ""
        for part in self.content:
            piece = part.text if isinstance(part, TagTextPart) else "[image]"
            marker = isinstance(part, TagImagePart) or text.endswith("[image]")
            if marker and text and piece and not (text[-1].isspace() or piece[0].isspace()):
                text += " "
            text += piece
        return text

    def has_image(self) -> bool:
        """Report whether any content part is an image."""
        return not isinstance(self.content, str | None) and any(
            isinstance(part, TagImagePart) for part in self.content
        )


class TagTool(ContractModel):
    """One tool offered to the model.

    Attributes:
        name: Tool name.
        description: Caller-supplied description, when any.
        parameters: JSON schema of the arguments.
    """

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: JsonObject = Field(default_factory=dict)


class TagExchange(ContractModel):
    """One request and its response, ready to render into tag windows.

    Message positions count request messages first and response turns after them, so
    ``messages[i]`` is position ``i`` and ``response[j]`` is ``len(messages) + j``.

    Attributes:
        exchange_id: Identifier of the captured exchange, at most 512 characters.
        conversation_key: Caller or store conversation key of at most 512 characters; None
            to derive ``opening_key``.
        tools: Tools offered in the request.
        messages: Request messages, in order; at least one.
        response: Response turns, in order; at least one.
        truncated: Whether the response stopped because it reached its length limit.
        structured_output: Whether the request asked for JSON or schema-constrained output.
    """

    exchange_id: str = Field(min_length=1, max_length=512)
    conversation_key: str | None = Field(default=None, min_length=1, max_length=512)
    tools: tuple[TagTool, ...] = ()
    messages: tuple[TagMessage, ...] = Field(min_length=1)
    response: tuple[TagMessage, ...] = Field(min_length=1)
    truncated: bool = False
    structured_output: bool = False

    def all_messages(self) -> tuple[TagMessage, ...]:
        """Return request messages followed by response turns, in position order."""
        return self.messages + self.response


def first_user_index(messages: tuple[TagMessage, ...]) -> int | None:
    """Return the position of the first user message.

    Args:
        messages: Request messages.

    Returns:
        The index, or None when no message has the user role.
    """
    return next((i for i, message in enumerate(messages) if message.role == "user"), None)


def leading_instructions(messages: tuple[TagMessage, ...]) -> int:
    """Count the system and developer messages that open a request.

    Args:
        messages: Request messages.

    Returns:
        The length of the leading run of system and developer messages.
    """
    count = 0
    for message in messages:
        if message.role not in ("system", "developer"):
            break
        count += 1
    return count


def opening_end(messages: tuple[TagMessage, ...]) -> int:
    """Return one past the last message of a conversation's opening.

    The opening runs through the first user message; without one it is the leading
    instructions, or the first message when there are none.

    Args:
        messages: Request messages.

    Returns:
        The exclusive end position of the opening.
    """
    first = first_user_index(messages)
    return first + 1 if first is not None else max(leading_instructions(messages), 1)


def opening_key(exchange: TagExchange) -> Sha256:
    """Derive a conversation key from what every turn of a conversation repeats.

    The key covers every message of the opening (``opening_end``) and every tool's name,
    description and schema in a canonical order: everything the opening window shows. Later
    turns of the same conversation resend that opening, so they share the key; instructions
    inserted later in the conversation do not change it.

    Args:
        exchange: Any exchange of the conversation.

    Returns:
        A SHA-256 hex digest.
    """
    opening = exchange.messages[: opening_end(exchange.messages)]
    tools = sorted(
        (tool.model_dump(mode="json") for tool in exchange.tools),
        key=lambda tool: json.dumps(tool, sort_keys=True),
    )
    return sha256_json(
        {"opening": [message.model_dump(mode="json") for message in opening], "tools": tools}
    )


def tag_facts(exchange: TagExchange) -> TagFacts:
    """Read the exchange facts that need no model call.

    Args:
        exchange: The exchange.

    Returns:
        Tool calls in the response, user messages in the request, length truncation, request
        images and a structured output request.
    """
    return TagFacts(
        tool_calls=sum(len(message.tool_calls) for message in exchange.response),
        turns=sum(1 for message in exchange.messages if message.role == "user"),
        truncated=exchange.truncated,
        has_images=any(message.has_image() for message in exchange.messages),
        structured_output=exchange.structured_output,
    )


__all__ = [
    "MessageRole",
    "TagContentPart",
    "TagExchange",
    "TagFunction",
    "TagImagePart",
    "TagMessage",
    "TagTextPart",
    "TagTool",
    "TagToolCall",
    "first_user_index",
    "leading_instructions",
    "opening_end",
    "opening_key",
    "tag_facts",
]
