"""Decode Anthropic Messages bodies into canonical serving requests.

The decoder is strict and lossless for supported content: text, ``tool_use``,
``tool_result``, ``thinking``, and ``redacted_thinking`` blocks translate
faithfully (thinking history rides the opaque provider-reasoning carrier with
byte-exact signatures); ``cache_control`` annotations are validated and
dropped because they do not change model semantics; ``image`` and
``document`` blocks are rejected loudly because the serving surface cannot
preserve them. Unknown or unsupported fields are rejected with a
field-specific error, never silently dropped. Errors raise
:class:`OpenAIProtocolError` so the shared boundary stays single-authority;
the HTTP layer renders them in the Anthropic envelope.
"""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_core import ErrorDetails

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.anthropic_protocol.manifest import MESSAGES_MANIFEST
from exp.runtime.gateway.contracts import (
    CompatibilityDisposition,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
    ProviderReasoningBlock,
    RedactedThinkingBlock,
    ThinkingBlock,
)
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    unsupported_field,
)
from exp.runtime.openai_protocol.manifest import disposition_map
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest

_REJECTED_BLOCK_HINTS = {
    "image": "image blocks are not supported: this gateway surface is text-only",
    "document": "document blocks are not supported: this gateway surface is text-only",
    "server_tool_use": "server tools are not supported by this gateway",
    "web_search_tool_result": "server tools are not supported by this gateway",
}


class _WireModel(BaseModel):
    """Strict private Anthropic wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class _CacheControl(_WireModel):
    """Anthropic prompt-caching annotation, validated and then dropped."""

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None


class _TextBlock(_WireModel):
    """One plain text content block."""

    type: Literal["text"]
    text: str
    cache_control: _CacheControl | None = None


class _ThinkingBlock(_WireModel):
    """Extended-thinking assistant history block, carried verbatim."""

    type: Literal["thinking"]
    thinking: str = ""
    signature: str | None = None


class _RedactedThinkingBlock(_WireModel):
    """Redacted-thinking assistant history block, carried verbatim."""

    type: Literal["redacted_thinking"]
    data: str = ""


class _ToolUseBlock(_WireModel):
    """One assistant tool invocation retained in request history."""

    type: Literal["tool_use"]
    id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    input: JsonObject
    cache_control: _CacheControl | None = None


class _ToolResultBlock(_WireModel):
    """One tool result the caller returns for a prior assistant tool call.

    ``is_error`` is carried on the canonical tool message
    (``GatewayMessage.tool_is_error``) so the Anthropic upstream dialect can
    round-trip it losslessly; the OpenAI-family wire formats have no
    tool-error flag, so on those routes the error state travels in the
    result text the model reads.
    """

    type: Literal["tool_result"]
    tool_use_id: str = Field(min_length=1, max_length=256)
    content: str | tuple[_TextBlock, ...] | None = None
    is_error: bool = False
    cache_control: _CacheControl | None = None


_ContentBlock = (
    _TextBlock | _ThinkingBlock | _RedactedThinkingBlock | _ToolUseBlock | _ToolResultBlock
)


class _Message(_WireModel):
    """One Anthropic conversation turn."""

    role: Literal["user", "assistant"]
    content: str | tuple[_ContentBlock, ...]
    cache_control: _CacheControl | None = None


class _Tool(_WireModel):
    """One caller-defined custom tool with its JSON Schema declaration."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    input_schema: JsonObject
    cache_control: _CacheControl | None = None
    type: Literal["custom"] | None = None


class _ToolChoice(_WireModel):
    """Anthropic tool-choice selector."""

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = Field(default=None, min_length=1, max_length=256)
    disable_parallel_tool_use: bool | None = None


class _Metadata(_WireModel):
    """Request metadata; only ``user_id`` is defined by the public API."""

    user_id: str | None = Field(default=None, max_length=256)


class _ThinkingConfig(_WireModel):
    """Extended-thinking configuration validated closed, then forwarded verbatim."""

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_budget_only_when_enabled(self) -> _ThinkingConfig:
        """Bind the token budget to the one mode Anthropic defines it for."""
        if self.type == "enabled" and self.budget_tokens is None:
            raise ValueError("thinking.budget_tokens is required when thinking is enabled")
        if self.type != "enabled" and self.budget_tokens is not None:
            raise ValueError("thinking.budget_tokens is valid only when thinking is enabled")
        return self


class _MessagesRequest(_WireModel):
    """Closed gateway Anthropic Messages request profile."""

    model: str = Field(min_length=1, max_length=256)
    messages: tuple[_Message, ...] = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    system: str | tuple[_TextBlock, ...] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    stop_sequences: tuple[str, ...] | None = None
    stream: bool = False
    tools: tuple[_Tool, ...] = ()
    tool_choice: _ToolChoice | None = None
    metadata: _Metadata | None = None
    thinking: _ThinkingConfig | None = None
    context_management: JsonObject | None = None


def decode_messages(payload: JsonObject) -> DecodedGatewayRequest:
    """Decode one Anthropic Messages body without silently dropping fields.

    The Anthropic protocol defines no idempotency header, so this surface
    never carries a caller operation and never participates in keyed replay.

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
            The HTTP layer renders it in the Anthropic error envelope.
    """
    _validate_manifest(payload)
    request = _validate_wire(payload)
    messages: list[GatewayMessage] = []
    system_text = _system_text(request.system)
    if system_text:
        messages.append(GatewayMessage(role="system", content=system_text))
    for index, message in enumerate(request.messages):
        messages.extend(_gateway_messages(message, index))
    parallel_tool_calls: bool | None = None
    if request.tool_choice is not None and request.tool_choice.disable_parallel_tool_use:
        parallel_tool_calls = False
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=tuple(messages),
            tools=tuple(_gateway_tool(tool) for tool in request.tools),
            tool_choice=_gateway_tool_choice(request.tool_choice),
            parallel_tool_calls=parallel_tool_calls,
            maximum_output_tokens=request.max_tokens,
            maximum_output_tokens_parameter="max_tokens",
            stop=_stop_sequences(request.stop_sequences),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream=request.stream,
            include_usage=request.stream,
            metadata=_gateway_metadata(request.metadata),
            # The raw payload value, not the re-serialized wire model, so the
            # provider receives the caller's thinking config byte-for-byte.
            provider_thinking_config=(
                cast(JsonObject, payload["thinking"]) if request.thinking is not None else None
            ),
            context_management=_context_management(payload),
        )
    except ValidationError as exc:
        raise _validation_error(exc.errors(include_url=False)[0]) from exc
    return DecodedGatewayRequest(alias=request.model, request=canonical)


def _context_management(payload: JsonObject) -> JsonObject | None:
    """Validate the caller's context-editing config as an object, verbatim.

    The nested shape is an evolving Anthropic beta the gateway forwards
    byte-for-byte (with the required beta header), so validation is
    deliberately shallow: a closed model here would recreate the
    reject-what-real-clients-send incident class.

    Raises:
        OpenAIProtocolError: The field is present but not a JSON object.
    """
    value = payload.get("context_management")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise invalid_field("context_management", "context_management must be a JSON object.")
    return cast(JsonObject, value)


def _validate_manifest(payload: JsonObject) -> None:
    """Reject unsupported and unknown top-level fields before decoding."""
    decisions = disposition_map(MESSAGES_MANIFEST)
    for field in payload:
        disposition = decisions.get(field)
        if disposition is None or disposition == CompatibilityDisposition.UNSUPPORTED:
            raise unsupported_field(field)


def _validate_wire(payload: JsonObject) -> _MessagesRequest:
    """Validate the strict wire model with a field-specific public error."""
    try:
        return _MessagesRequest.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        hint = _rejected_block_hint(payload)
        if hint is not None:
            raise invalid_field("messages", hint) from exc
        raise _validation_error(first) from exc


def _validation_error(first: ErrorDetails) -> OpenAIProtocolError:
    """Convert one Pydantic error location into a stable dotted field error."""
    location = first["loc"]
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        # Union member class names in pydantic locations are noise for callers.
        if isinstance(part, str) and (part.startswith("_") or "[" in text):
            continue
        cleaned.append(text)
    return invalid_field(".".join(cleaned) or "body")


def _rejected_block_hint(payload: JsonObject) -> str | None:
    """Return a targeted message when a known-but-unsupported block is present."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in cast(list[object], message["content"]):
            if not isinstance(block, dict):
                continue
            block_object = cast(JsonObject, block)
            hint = _REJECTED_BLOCK_HINTS.get(str(block_object.get("type")))
            if hint is not None:
                return hint
            if block_object.get("type") == "tool_result" and isinstance(
                block_object.get("content"), list
            ):
                for inner in cast(list[object], block_object["content"]):
                    if (
                        isinstance(inner, dict)
                        and str(cast(JsonObject, inner).get("type")) in _REJECTED_BLOCK_HINTS
                    ):
                        return _REJECTED_BLOCK_HINTS[str(cast(JsonObject, inner).get("type"))]
    return None


def _system_text(system: str | tuple[_TextBlock, ...] | None) -> str | None:
    """Flatten the system prompt; blocks join with a blank line."""
    if system is None or isinstance(system, str):
        return system
    return "\n\n".join(block.text for block in system)


def _stop_sequences(sequences: tuple[str, ...] | None) -> tuple[str, ...]:
    """Dedupe stop sequences in caller order and reject empty entries."""
    if not sequences:
        return ()
    deduped = tuple(dict.fromkeys(sequences))
    if any(not sequence for sequence in deduped):
        raise invalid_field("stop_sequences", "stop_sequences entries must be non-empty strings.")
    return deduped


def _gateway_tool(tool: _Tool) -> GatewayToolDefinition:
    """Convert one Anthropic custom tool to the canonical tool definition."""
    return GatewayToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.input_schema,
    )


def _gateway_tool_choice(
    choice: _ToolChoice | None,
) -> Literal["auto", "none", "required"] | GatewayNamedToolChoice | None:
    """Normalize the Anthropic tool-choice selector to the canonical form."""
    if choice is None:
        return None
    if choice.type == "auto":
        return "auto"
    if choice.type == "none":
        return "none"
    if choice.type == "any":
        return "required"
    if not choice.name:
        raise invalid_field("tool_choice.name", "tool_choice of type 'tool' requires a name.")
    return GatewayNamedToolChoice(name=choice.name)


def _gateway_metadata(metadata: _Metadata | None) -> JsonObject:
    """Forward only the defined ``user_id`` metadata field."""
    if metadata is None or metadata.user_id is None:
        return {}
    return {"user_id": metadata.user_id}


def _gateway_messages(message: _Message, index: int) -> list[GatewayMessage]:
    """Translate one Anthropic turn into one or more canonical messages.

    ``tool_result`` blocks must become standalone tool-role messages (the
    canonical contract rejects ``tool_call_id`` on other roles), so a user
    turn mixing tool results and text splits into several messages, in order.

    Args:
        message: One validated Anthropic message.
        index: Zero-based message index used in public error paths.

    Returns:
        Ordered canonical gateway messages.

    Raises:
        OpenAIProtocolError: A block is invalid for the message role or the
            turn carries no gateway-visible content.
    """
    param = f"messages.{index}"
    if isinstance(message.content, str):
        if not message.content:
            raise invalid_field(
                f"{param}.content", f"{message.role} message content must not be empty."
            )
        return [GatewayMessage(role=message.role, content=message.content)]
    out: list[GatewayMessage] = []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning: list[ProviderReasoningBlock] = []

    def flush() -> None:
        """Emit the pending text, tool calls, and reasoning as one canonical message."""
        content = "".join(text_parts) if text_parts else None
        if content is None and not tool_calls and not reasoning:
            return
        out.append(
            GatewayMessage(
                role=message.role,
                content=content,
                tool_calls=tuple(tool_calls),
                provider_reasoning=tuple(reasoning),
            )
        )
        text_parts.clear()
        tool_calls.clear()
        reasoning.clear()

    for block_index, block in enumerate(message.content):
        if isinstance(block, _TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, (_ThinkingBlock, _RedactedThinkingBlock)):
            if message.role != "assistant":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "thinking blocks are only valid in assistant messages.",
                )
            reasoning.append(
                ThinkingBlock(text=block.thinking, signature=block.signature)
                if isinstance(block, _ThinkingBlock)
                else RedactedThinkingBlock(data=block.data)
            )
        elif isinstance(block, _ToolUseBlock):
            if message.role != "assistant":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "tool_use blocks are only valid in assistant messages.",
                )
            tool_calls.append(
                ToolCall(
                    call_id=block.id,
                    name=block.name,
                    arguments=block.input,
                    raw_arguments=json.dumps(
                        block.input, separators=(",", ":"), ensure_ascii=False
                    ),
                    cache_control=(
                        block.cache_control.model_dump(mode="json", exclude_none=True)
                        if block.cache_control is not None
                        else None
                    ),
                )
            )
        else:
            if message.role != "user":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "tool_result blocks are only valid in user messages.",
                )
            flush()
            out.append(
                GatewayMessage(
                    role="tool",
                    content=_tool_result_text(block),
                    tool_call_id=block.tool_use_id,
                    tool_is_error=block.is_error,
                )
            )
    flush()
    if not out:
        raise invalid_field(
            f"{param}.content",
            f"{message.role} message must contain text, thinking, tool_use, "
            "or tool_result content.",
        )
    return out


def _tool_result_text(block: _ToolResultBlock) -> str:
    """Flatten one tool result into the canonical tool-message text."""
    if block.content is None:
        return ""
    if isinstance(block.content, str):
        return block.content
    return "".join(part.text for part in block.content)
