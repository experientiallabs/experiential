"""Decode Anthropic Messages bodies into canonical serving requests.

The decoder is strict and lossless for supported content: text, ``tool_use``,
``tool_result``, ``thinking``, and ``redacted_thinking`` blocks translate
faithfully (thinking history rides the opaque provider-reasoning carrier with
byte-exact signatures); echoed server-tool output (``server_tool_use``,
``web_search_tool_result``, and citation-bearing ``text`` blocks) rides the
verbatim per-block carrier so it round-trips byte-for-byte to native
Anthropic rungs; ``cache_control`` annotations are validated
everywhere and carried on the surfaces the Anthropic wire caches natively
(text and image content blocks, tool_use blocks, tool definitions, and the
top-level automatic marker), and dropped on wires that do not cache a marked
block because a cache hint changes cost, not semantics; ``image`` and PDF
``document`` blocks are retained as canonical content parts so a route that
declares the matching input capability carries them, while a document
inside ``tool_result`` content is rejected loudly because the serving
surface cannot preserve it there.
Unknown or unsupported fields are rejected with a
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
from exp.common.models.content import (
    DOCUMENT_MEDIA_TYPES,
    IMAGE_MEDIA_TYPES,
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_DOCUMENT_NAME_CHARACTERS,
    MAXIMUM_IMAGE_BASE64_BYTES,
    DocumentContentPart,
    ImageContentPart,
    MessageContentPart,
    TextContentPart,
    image_part_from_url,
)
from exp.common.models.model import ReasoningEffort, ToolCall
from exp.runtime.anthropic_protocol.manifest import (
    MESSAGES_BETA_TOKENS_FORWARDED,
    MESSAGES_MANIFEST,
    MESSAGES_SERVER_TOOL_TYPES_ACCEPTED,
)
from exp.runtime.gateway.compatibility import CompatibilityDisposition
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
    ProviderReasoningBlock,
    RedactedThinkingBlock,
    ThinkingBlock,
)
from exp.runtime.models.providers.reasoning_compat import REASONING_EFFORTS
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    unsupported_field,
)
from exp.runtime.openai_protocol.manifest import disposition_map
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest

_REJECTED_TOOL_RESULT_BLOCK_HINTS = {
    "document": "document blocks are not supported inside tool_result content",
}


class _WireModel(BaseModel):
    """Strict private Anthropic wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class _CacheControl(_WireModel):
    """Anthropic prompt-caching annotation, validated and then dropped."""

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None


class _TextBlock(_WireModel):
    """One plain text content block.

    ``citations`` exists only as server-tool output echoed back in assistant
    history (Claude Code resends the cited answer verbatim, and the SDK
    accumulator materializes the key as null for uncited blocks); each
    citation is an evolving provider shape with a provider-issued encrypted
    index, so validation is deliberately shallow.
    """

    type: Literal["text"]
    text: str
    cache_control: _CacheControl | None = None
    citations: tuple[JsonObject, ...] | None = None


class _ImageSource(_WireModel):
    """Where one image block's bytes come from: inline base64 or a URL."""

    type: Literal["base64", "url"]
    media_type: str | None = Field(default=None, max_length=64)
    data: str | None = Field(default=None, max_length=MAXIMUM_IMAGE_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)


class _ImageBlock(_WireModel):
    """One caller image content block."""

    type: Literal["image"]
    source: _ImageSource
    cache_control: _CacheControl | None = None


class _DocumentSource(_WireModel):
    """Where one document block's bytes come from: inline base64 or a URL.

    ``file`` sources name an uploaded Files API object this gateway does not
    host and ``text``/``content`` sources carry non-PDF documents no other
    wire accepts, so only the two carriers every declared route can serve
    are accepted.
    """

    type: Literal["base64", "url"]
    media_type: str | None = Field(default=None, max_length=64)
    data: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_BASE64_BYTES)
    url: str | None = Field(default=None, max_length=8_192)


class _DocumentCitations(_WireModel):
    """Per-document citations toggle; only the disabled form is servable."""

    enabled: bool


class _DocumentBlock(_WireModel):
    """One caller PDF document content block."""

    type: Literal["document"]
    source: _DocumentSource
    title: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_NAME_CHARACTERS)
    citations: _DocumentCitations | None = None
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


class _ServerToolUseBlock(BaseModel):
    """One server-tool invocation echoed in history, carried shallowly.

    Server-tool block shapes are an evolving provider surface; a closed
    model here would recreate the reject-what-real-clients-send incident
    class, so only the discriminator is validated and the raw block forwards
    byte-for-byte on native Anthropic rungs.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["server_tool_use"]


class _WebSearchToolResultBlock(BaseModel):
    """One server-tool result echoed in history, carried shallowly."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_tool_result"]


_ContentBlock = (
    _TextBlock
    | _ImageBlock
    | _DocumentBlock
    | _ThinkingBlock
    | _RedactedThinkingBlock
    | _ToolUseBlock
    | _ToolResultBlock
    | _ServerToolUseBlock
    | _WebSearchToolResultBlock
)


class _Message(_WireModel):
    """One Anthropic conversation turn.

    ``system`` is a first-class mid-conversation role on the live API (the
    provider itself enforces its placement rules); Claude Code appends one
    system turn by default.
    """

    role: Literal["user", "assistant", "system"]
    content: str | tuple[_ContentBlock, ...]
    cache_control: _CacheControl | None = None


class _Tool(_WireModel):
    """One caller-defined custom tool with its JSON Schema declaration.

    The description bound is generous on purpose: the provider accepts
    40k-character descriptions live (verified 2026-08-30) and a real Claude
    Code toolset exceeded the earlier 8k bound; the request-body size cap
    remains the effective total limit.

    ``eager_input_streaming``, ``defer_loading``, ``allowed_callers``, and
    ``input_examples`` are provider-native tool annotations the live API
    accepts bare (each verified 2026-08-30; Claude Code sends
    ``eager_input_streaming`` conditionally). They forward verbatim on
    Anthropic rungs, which own their validity rules, and drop with
    disclosure elsewhere.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    input_schema: JsonObject
    cache_control: _CacheControl | None = None
    type: Literal["custom"] | None = None
    strict: bool = False
    eager_input_streaming: bool | None = None
    defer_loading: bool | None = None
    allowed_callers: tuple[str, ...] | None = None
    input_examples: tuple[JsonObject, ...] | None = None


class _ServerTool(BaseModel):
    """One Anthropic server tool, validated shallowly and carried verbatim.

    Server tools (``web_search_20250305``-style) execute at the provider and
    carry no ``input_schema``; their per-type configuration is an evolving
    provider surface, so only the discriminator pair is validated and the
    raw entry forwards byte-for-byte on native Anthropic rungs.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_server_type(self) -> _ServerTool:
        """Reject the custom discriminator: custom tools take the strict model."""
        if self.type == "custom":
            raise ValueError("custom tools must declare an input_schema")
        return self


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
    display: str | None = Field(default=None, max_length=64)
    """Display disposition, forwarded verbatim (Claude Code sends "omitted";
    accepted live without a beta, 2026-08-30). Bounded but deliberately not
    enumerated: the value set is an evolving provider surface."""

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
    tools: tuple[_Tool | _ServerTool, ...] = ()
    tool_choice: _ToolChoice | None = None
    metadata: _Metadata | None = None
    thinking: _ThinkingConfig | None = None
    context_management: JsonObject | None = None
    output_config: JsonObject | None = None
    diagnostics: JsonObject | None = None
    speed: str | None = Field(default=None, min_length=1, max_length=64)
    """Fast-mode selector, forwarded verbatim (Claude Code sends "fast";
    accepted live behind its beta header, 2026-08-30). Bounded but
    deliberately not enumerated: the value set is an evolving provider
    surface."""
    cache_control: _CacheControl | None = None
    """Top-level automatic prompt-caching marker (accepted live without a
    beta, 2026-08-30). Validated closed, forwarded verbatim on Anthropic
    rungs, and dropped with disclosure elsewhere: a cache hint changes
    cost, not semantics."""
    inference_geo: str | None = Field(default=None, min_length=1, max_length=64)
    """Inference-region selector, forwarded verbatim (accepted live without
    a beta, 2026-08-30). Bounded but deliberately not enumerated: the
    region set is an evolving provider surface."""


def decode_messages(
    payload: JsonObject,
    *,
    anthropic_beta: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one Anthropic Messages body without silently dropping fields.

    The Anthropic protocol defines no idempotency header, so this surface
    never carries a caller operation and never participates in keyed replay.

    Args:
        payload: Parsed JSON request body.
        anthropic_beta: Optional raw caller ``anthropic-beta`` header value.
            Allowlisted tokens are retained for Anthropic dispatch; the rest
            are dropped with a per-token disclosure.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
            The HTTP layer renders it in the Anthropic error envelope.
    """
    _validate_manifest(payload)
    request = _validate_wire(payload)
    _require_served_server_tool_types(request.tools)
    forwarded_betas, dropped_beta_disclosures = _beta_tokens(anthropic_beta)
    messages: list[GatewayMessage] = []
    system_text = _system_text(request.system)
    if system_text:
        messages.append(
            GatewayMessage(
                role="system",
                content=system_text,
                provider_text_blocks=_marked_text_blocks(request.system),
            )
        )
    for index, message in enumerate(request.messages):
        messages.extend(_gateway_messages(message, index))
    parallel_tool_calls: bool | None = None
    if request.tool_choice is not None and request.tool_choice.disable_parallel_tool_use:
        parallel_tool_calls = False
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=tuple(messages),
            tools=tuple(_gateway_tool(tool) for tool in request.tools if isinstance(tool, _Tool)),
            # Raw payload entries, mirroring thinking: the provider receives
            # each server tool byte-for-byte on Anthropic rungs.
            provider_server_tools=tuple(
                cast(JsonObject, cast(list, payload["tools"])[tool_index])
                for tool_index, tool in enumerate(request.tools)
                if isinstance(tool, _ServerTool)
            ),
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
            diagnostics=_diagnostics(payload),
            speed=request.speed,
            # Raw payload value, mirroring thinking: the provider receives the
            # caller's cache marker byte-for-byte on Anthropic rungs.
            provider_cache_control=(
                cast(JsonObject, payload["cache_control"])
                if request.cache_control is not None
                else None
            ),
            inference_geo=request.inference_geo,
            provider_beta_tokens=forwarded_betas,
            ignored_parameters=dropped_beta_disclosures,
            reasoning_effort=_output_config_effort(request.output_config),
            provider_output_config=(
                cast(JsonObject, payload["output_config"])
                if request.output_config is not None
                else None
            ),
        )
    except ValidationError as exc:
        raise _validation_error(exc.errors(include_url=False)[0]) from exc
    return DecodedGatewayRequest(alias=request.model, request=canonical)


def _require_served_server_tool_types(tools: tuple[_Tool | _ServerTool, ...]) -> None:
    """Reject any server tool type the gateway cannot serve truthfully.

    Acceptance means the data plane carries every block the tool makes the
    provider stream (see the decision tables in ``manifest.py``); an
    unclassified type stays rejected until the SDK drift gate forces its
    decision, so a new provider tool never half-works silently.

    Raises:
        OpenAIProtocolError: A tool entry names an unserved server tool type.
    """
    for tool_index, tool in enumerate(tools):
        if isinstance(tool, _ServerTool) and tool.type not in MESSAGES_SERVER_TOOL_TYPES_ACCEPTED:
            supported = ", ".join(sorted(MESSAGES_SERVER_TOOL_TYPES_ACCEPTED))
            raise invalid_field(
                f"tools.{tool_index}.type",
                f"the server tool type '{tool.type}' is not supported by this gateway. "
                f"Supported server tool types: {supported}. Remove the tool or use a "
                "supported type.",
            )


def _output_config_effort(config: JsonObject | None) -> ReasoningEffort | None:
    """Map a canonical caller ``output_config.effort`` into the shared field.

    A canonical ladder value rides ``reasoning_effort`` so route narrowing,
    the coercion policy, and non-Anthropic rungs all see it; the raw object
    still forwards verbatim on Anthropic rungs with the caller's keys
    winning, so an unrecognized future effort value stays provider-decided
    instead of gateway-rejected.
    """
    if config is None:
        return None
    effort = config.get("effort")
    if isinstance(effort, str) and effort in REASONING_EFFORTS:
        # The membership check above is the narrowing proof for this cast.
        return cast("ReasoningEffort", effort)
    return None


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


def _diagnostics(payload: JsonObject) -> JsonObject | None:
    """Validate the caller's diagnostics correlation as an object, verbatim.

    Like ``context_management``, the nested shape is an evolving Anthropic
    beta the gateway forwards byte-for-byte (with the required beta
    header), so validation is deliberately shallow.

    Raises:
        OpenAIProtocolError: The field is present but not a JSON object.
    """
    value = payload.get("diagnostics")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise invalid_field("diagnostics", "diagnostics must be a JSON object.")
    return cast(JsonObject, value)


def _beta_tokens(header: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition a caller ``anthropic-beta`` header into forward and drop sets.

    Forwarding is an exact allowlist (:data:`MESSAGES_BETA_TOKENS_FORWARDED`):
    a caller header is operator-trust surface and is never blind-forwarded.
    Dropped tokens are disclosed per token, never rejected, because the
    provider itself tolerates unknown beta tokens.

    Args:
        header: Raw comma-separated header value, or ``None``.

    Returns:
        ``(forwarded_tokens, dropped_disclosures)`` in caller order, deduped.

    Raises:
        OpenAIProtocolError: The header carries a non-display-safe value.
    """
    if header is None:
        return (), ()
    if len(header) > 4_096 or any(ord(char) < 32 for char in header):
        raise invalid_field("anthropic-beta", "anthropic-beta must be a display-safe header value.")
    forwarded: list[str] = []
    dropped: list[str] = []
    for raw_token in header.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token in MESSAGES_BETA_TOKENS_FORWARDED:
            if token not in forwarded:
                forwarded.append(token)
        else:
            disclosure = f"anthropic-beta.{token}"
            if disclosure not in dropped:
                dropped.append(disclosure)
    return tuple(forwarded), tuple(dropped)


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
    """Convert one Pydantic error location into a stable dotted field error.

    The public message keeps the expected-vs-got shape: it names the field
    and states what the decoder expected there (Pydantic's own expectation
    text, which never echoes the caller's value), so a rejected request says
    what to fix instead of only where it failed.
    """
    location = first["loc"]
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        # Union member class names in pydantic locations are noise for callers.
        if isinstance(part, str) and (part.startswith("_") or "[" in text):
            continue
        cleaned.append(text)
    param = ".".join(cleaned) or "body"
    if first["type"] == "extra_forbidden":
        return invalid_field(
            param,
            f"Unknown parameter '{param}'. Remove the field and resend the request.",
        )
    return invalid_field(param, f"Invalid value for '{param}': {first['msg']}.")


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
            if block_object.get("type") == "tool_result" and isinstance(
                block_object.get("content"), list
            ):
                for inner in cast(list[object], block_object["content"]):
                    if not isinstance(inner, dict):
                        continue
                    hint = _REJECTED_TOOL_RESULT_BLOCK_HINTS.get(
                        str(cast(JsonObject, inner).get("type"))
                    )
                    if hint is not None:
                        return hint
    return None


def _image_part(block: _ImageBlock, param: str) -> ImageContentPart:
    """Convert one Anthropic image block into the canonical image part.

    Args:
        block: Validated caller image block.
        param: Public parameter path used to report an invalid image.

    Returns:
        The canonical image part carrying the caller's bytes or URL.

    Raises:
        OpenAIProtocolError: The source is not a supported image.
    """
    source = block.source
    marker = (
        block.cache_control.model_dump(mode="json", exclude_none=True)
        if block.cache_control is not None
        else None
    )
    try:
        if source.type == "url":
            part = image_part_from_url(source.url or "")
            return part.model_copy(update={"cache_control": marker})
        return ImageContentPart(
            media_type=IMAGE_MEDIA_TYPES[source.media_type or ""],
            data=source.data,
            cache_control=marker,
        )
    except (KeyError, ValueError) as exc:
        raise invalid_field(
            f"{param}.source",
            f"'{param}.source' must carry an http(s) URL or base64 data "
            "for a PNG, JPEG, GIF, or WebP image.",
        ) from exc


def _document_part(block: _DocumentBlock, param: str) -> DocumentContentPart:
    """Convert one Anthropic document block into the canonical document part.

    Args:
        block: Validated caller document block.
        param: Public parameter path used to report an invalid document.

    Returns:
        The canonical document part carrying the caller's bytes or URL.

    Raises:
        OpenAIProtocolError: The source is not a PDF this gateway forwards,
            or the block enables citations.
    """
    if block.citations is not None and block.citations.enabled:
        raise invalid_field(
            f"{param}.citations",
            "document citations are not supported over this gateway; "
            "send citations.enabled as false or omit the field.",
        )
    source = block.source
    marker = (
        block.cache_control.model_dump(mode="json", exclude_none=True)
        if block.cache_control is not None
        else None
    )
    try:
        if source.type == "url":
            return DocumentContentPart(url=source.url, name=block.title, cache_control=marker)
        return DocumentContentPart(
            media_type=DOCUMENT_MEDIA_TYPES[source.media_type or ""],
            data=source.data,
            name=block.title,
            cache_control=marker,
        )
    except (KeyError, ValueError) as exc:
        raise invalid_field(
            f"{param}.source",
            f"'{param}.source' must carry an http(s) URL or base64 data for a PDF "
            "(media_type application/pdf).",
        ) from exc


def _system_text(system: str | tuple[_TextBlock, ...] | None) -> str | None:
    """Flatten the system prompt; blocks join with a blank line."""
    if system is None or isinstance(system, str):
        return system
    return "\n\n".join(block.text for block in system)


def _marked_text_blocks(
    blocks: str | tuple[_TextBlock, ...] | None,
) -> tuple[JsonObject, ...]:
    """Rebuild a text-block run verbatim when any block carries a cache marker.

    Claude Code marks system blocks and the last text block of recent user
    turns (captured live 2026-09-01). The flattened string stays the
    canonical content on every wire; this carrier exists so Anthropic rungs
    re-emit the caller's exact block structure with its markers, which is
    what makes the prompt cacheable at all. A markerless run carries
    nothing, keeping existing payloads byte-identical.
    """
    if blocks is None or isinstance(blocks, str):
        return ()
    if all(block.cache_control is None for block in blocks):
        return ()
    rebuilt: list[JsonObject] = []
    for block in blocks:
        entry: JsonObject = {"type": "text", "text": block.text}
        if block.cache_control is not None:
            entry["cache_control"] = block.cache_control.model_dump(mode="json", exclude_none=True)
        rebuilt.append(entry)
    return tuple(rebuilt)


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
        strict=tool.strict,
        cache_control=(
            tool.cache_control.model_dump(mode="json", exclude_none=True)
            if tool.cache_control is not None
            else None
        ),
        eager_input_streaming=tool.eager_input_streaming,
        defer_loading=tool.defer_loading,
        allowed_callers=tool.allowed_callers,
        input_examples=tool.input_examples,
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
    text_parts: list[_TextBlock] = []
    content_parts: list[MessageContentPart] = []
    tool_calls: list[ToolCall] = []
    reasoning: list[ProviderReasoningBlock] = []

    def flush() -> None:
        """Emit the pending content, tool calls, and reasoning as one message."""
        content = "".join(part.text for part in text_parts) if text_parts else None
        attachments = any(part.kind != "text" for part in content_parts)
        if content is None and not tool_calls and not reasoning and not attachments:
            return
        out.append(
            GatewayMessage(
                role=message.role,
                content=content or ("" if attachments else None),
                content_parts=tuple(content_parts) if attachments else (),
                tool_calls=tuple(tool_calls),
                provider_reasoning=tuple(reasoning),
                # The marked run is carried alongside the retained parts: its
                # blocks are the same text in the same order, so a multimodal
                # turn keeps its cache markers when it re-emits.
                provider_text_blocks=_marked_text_blocks(tuple(text_parts)),
            )
        )
        text_parts.clear()
        content_parts.clear()
        tool_calls.clear()
        reasoning.clear()

    for block_index, block in enumerate(message.content):
        if isinstance(block, _TextBlock):
            if block.citations:
                if message.role != "assistant":
                    raise invalid_field(
                        f"{param}.content.{block_index}",
                        "citations are only valid in assistant messages.",
                    )
                # A cited answer is server-tool output; the block re-emits
                # verbatim so provider-issued encrypted indexes round-trip.
                flush()
                out.append(
                    GatewayMessage(
                        role="assistant",
                        provider_anthropic_block=block.model_dump(
                            mode="json", exclude_none=True, exclude={"cache_control"}
                        ),
                    )
                )
                continue
            # An empty citations array (the SDK accumulator's uncited shape)
            # carries no information and drops; a cache marker on the block
            # survives through the marked-run carrier.
            text_parts.append(block)
            # An empty text block cannot ride a multimodal turn: Anthropic
            # rejects a standalone empty block, so it never becomes a part.
            if block.text:
                content_parts.append(TextContentPart(text=block.text))
        elif isinstance(block, _ImageBlock):
            if message.role != "user":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "image blocks are only valid in user messages.",
                )
            content_parts.append(_image_part(block, f"{param}.content.{block_index}"))
        elif isinstance(block, _DocumentBlock):
            if message.role != "user":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "document blocks are only valid in user messages.",
                )
            content_parts.append(_document_part(block, f"{param}.content.{block_index}"))
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
        elif isinstance(block, (_ServerToolUseBlock, _WebSearchToolResultBlock)):
            if message.role != "assistant":
                raise invalid_field(
                    f"{param}.content.{block_index}",
                    "server tool blocks are only valid in assistant messages.",
                )
            # The raw echoed block (extras included) carries the whole
            # message, mirroring provider_native_item on the Responses wire.
            flush()
            out.append(
                GatewayMessage(
                    role="assistant",
                    provider_anthropic_block=block.model_dump(mode="json"),
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
                    # The marker Claude Code puts on its conversation
                    # breakpoints usually lands on a tool_result block; it
                    # re-emits with the block on Anthropic rungs.
                    cache_control=(
                        block.cache_control.model_dump(mode="json", exclude_none=True)
                        if block.cache_control is not None
                        else None
                    ),
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
