"""Decode Chat Completions and Responses into canonical serving requests."""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Literal, cast

from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import ReasoningEffort, ToolCall
from exp.runtime.gateway.contracts import (
    CompatibilityDisposition,
    CompatibilityManifest,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field, unsupported_field
from exp.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    RESPONSES_MANIFEST,
    disposition_map,
)

_CHAT_OFFICIAL = TypeAdapter(CompletionCreateParams)
_RESPONSES_OFFICIAL = TypeAdapter(ResponseCreateParams)
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class DecodedGatewayRequest(ContractModel):
    """Public alias plus its canonical provider-neutral request."""

    alias: str = Field(min_length=1, max_length=256)
    request: GatewayRequest


class _WireModel(BaseModel):
    """Strict private OpenAI wire model used only after official shape validation."""

    model_config = ConfigDict(extra="forbid")


class _EphemeralCacheControl(_WireModel):
    """OpenCode/Anthropic cache breakpoint accepted only so it can be dropped.

    The object form is ``{"type": "ephemeral"}`` with an optional ``ttl`` of
    ``5m`` or ``1h``. An explicit ``ttl: null`` is not in that allowlist.
    """

    type: Literal["ephemeral"]
    ttl: Literal["5m", "1h"] | None = None

    @field_validator("ttl", mode="before")
    @classmethod
    def _reject_null_ttl(cls, value: object) -> object:
        """Reject an explicit null TTL while still allowing the key to be omitted."""
        if value is None:
            raise ValueError("ttl must be 5m or 1h when present")
        return value


class _TextPart(_WireModel):
    """One supported text-only content part."""

    type: Literal["text", "input_text", "output_text"]
    text: str


class _FunctionCall(_WireModel):
    """Function name and raw JSON argument string."""

    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)


class _AssistantToolCall(_WireModel):
    """One assistant function call retained in request history."""

    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"] = "function"
    function: _FunctionCall


class _Message(_WireModel):
    """Text-only OpenAI message with complete assistant tool history.

    Assistant messages returned by this gateway (and by official OpenAI SDK
    clients) carry `refusal`, `annotations`, `audio`, and `function_call`
    keys even when they are empty. Callers echo those messages back verbatim
    on tool-call continuations, so the empty forms are accepted here; only a
    populated value is rejected as unsupported.
    """

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | tuple[_TextPart, ...] | None = None
    tool_calls: tuple[_AssistantToolCall, ...] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    refusal: None = None
    annotations: tuple[()] | None = None
    audio: None = None
    function_call: None = None

    @property
    def history_tool_calls(self) -> tuple[_AssistantToolCall, ...]:
        """Return retained assistant tool calls, treating a null list as empty."""
        return self.tool_calls or ()

    @model_validator(mode="after")
    def _require_role_fields(self) -> _Message:
        """Require tool linkage and assistant calls on their legal roles."""
        if self.role == "assistant" and self.content is None and not self.history_tool_calls:
            raise ValueError("assistant messages need content or tool calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role != "assistant" and self.history_tool_calls:
            raise ValueError("tool_calls are valid only for assistant messages")
        return self


class _ResponseMessage(_Message):
    """Responses message item with its optional official discriminator."""

    type: Literal["message"] = "message"


class _FunctionDefinition(_WireModel):
    """One function schema offered through Chat Completions."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    parameters: JsonObject = Field(default_factory=dict)
    strict: bool = False


class _ChatTool(_WireModel):
    """Chat Completions function tool wrapper."""

    type: Literal["function"] = "function"
    function: _FunctionDefinition


class _StructuredSchema(_WireModel):
    """Named strict JSON Schema in a Chat response format."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    schema_: JsonObject = Field(alias="schema")
    strict: bool = True


class _ChatResponseFormat(_WireModel):
    """Supported Chat text or strict structured-text format."""

    type: Literal["text", "json_schema"]
    json_schema: _StructuredSchema | None = None

    @model_validator(mode="after")
    def _require_schema(self) -> _ChatResponseFormat:
        """Require schema details only for the JSON Schema format."""
        if (self.type == "json_schema") != (self.json_schema is not None):
            raise ValueError("json_schema details must match response_format.type")
        return self


class _ChatStreamOptions(_WireModel):
    """Supported Chat streaming options."""

    include_usage: bool = False


class _ChatRequest(_WireModel):
    """Closed gateway Chat Completions request profile."""

    model: str = Field(min_length=1, max_length=256)
    messages: tuple[_Message, ...] = Field(min_length=1)
    tools: tuple[_ChatTool, ...] = ()
    tool_choice: JsonValue = None
    parallel_tool_calls: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | tuple[str, ...] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning_effort: ReasoningEffort | None = None
    response_format: _ChatResponseFormat | None = None
    stream: bool = False
    stream_options: _ChatStreamOptions | None = None
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints; captured, never forwarded to the model.
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)

    @model_validator(mode="after")
    def _require_coherent_options(self) -> _ChatRequest:
        """Reject conflicting token ceilings and non-stream usage options."""
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("max_tokens and max_completion_tokens are mutually exclusive")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self


class _ResponseTool(_WireModel):
    """Responses API function tool declaration."""

    type: Literal["function"] = "function"
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    parameters: JsonObject = Field(default_factory=dict)
    strict: bool | None = None


class _ResponseFunctionCall(_WireModel):
    """Completed Responses function call included as assistant history."""

    type: Literal["function_call"]
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)


class _ResponseFunctionOutput(_WireModel):
    """Text function result included as Responses tool history."""

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1, max_length=256)
    output: str


class _ResponseFormat(_WireModel):
    """Supported Responses text format."""

    type: Literal["text", "json_schema"]
    name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    schema_: JsonObject | None = Field(default=None, alias="schema")
    strict: bool = True

    @model_validator(mode="after")
    def _require_schema(self) -> _ResponseFormat:
        """Require named schema details only for structured output."""
        structured = self.type == "json_schema"
        if structured != (self.name is not None and self.schema_ is not None):
            raise ValueError("name and schema must match text.format.type")
        return self


class _ResponseText(_WireModel):
    """Responses text-output configuration."""

    format: _ResponseFormat


class _ResponseReasoning(_WireModel):
    """Responses reasoning controls preserved until route capability shaping."""

    effort: ReasoningEffort | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None


class _ResponsesRequest(_WireModel):
    """Closed gateway Responses request profile."""

    model: str = Field(min_length=1, max_length=256)
    input: str | tuple[_ResponseMessage | _ResponseFunctionCall | _ResponseFunctionOutput, ...]
    instructions: str | None = None
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    tools: tuple[_ResponseTool, ...] = ()
    tool_choice: JsonValue = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning: _ResponseReasoning | None = None
    text: _ResponseText | None = None
    stream: bool = False
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints; captured, never forwarded to the model.
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)


def decode_chat(
    payload: JsonObject,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one Chat Completions body without silently dropping fields.

    OpenCode may attach an Anthropic-style ``cache_control`` annotation on
    Chat messages and on text content parts. Supported ephemeral forms are
    validated and removed before official OpenAI validation and before
    canonical conversion. Other unknown nested fields stay rejected.

    Args:
        payload: Parsed JSON request body.
        idempotency_key: Optional standard caller operation identity.
        client_request_id: Optional gateway client request identity.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    payload = _drop_opencode_cache_control(payload)
    _validate_manifest(payload, CHAT_MANIFEST)
    _validate_official(_CHAT_OFFICIAL, payload, extension_fields={"top_k"})
    request = _validate_wire(_ChatRequest, payload)
    operation = _caller_operation(idempotency_key, client_request_id)
    maximum = request.max_completion_tokens or request.max_tokens
    stop = (
        ()
        if request.stop is None
        else (request.stop,)
        if isinstance(request.stop, str)
        else request.stop
    )
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=_messages(request.messages, "messages"),
            tools=tuple(_chat_tool(tool) for tool in request.tools),
            tool_choice=_chat_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=_chat_structured_text(request.response_format),
            maximum_output_tokens=maximum,
            stop=stop,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
            reasoning_effort=request.reasoning_effort,
            stream=request.stream,
            include_usage=(
                request.stream_options is not None and request.stream_options.include_usage
            ),
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            idempotency_key=operation if idempotency_key is not None else None,
            client_request_id=operation if client_request_id is not None else None,
        )
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc
    return DecodedGatewayRequest(alias=request.model, request=canonical)


def decode_responses(
    payload: JsonObject,
    *,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> DecodedGatewayRequest:
    """Decode one Responses body into the distinct canonical surface.

    Args:
        payload: Parsed JSON request body.
        idempotency_key: Optional standard caller operation identity.
        client_request_id: Optional gateway client request identity.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    _validate_manifest(payload, RESPONSES_MANIFEST)
    _validate_official(_RESPONSES_OFFICIAL, payload, extension_fields={"top_k"})
    request = _validate_wire(_ResponsesRequest, payload)
    operation = _caller_operation(idempotency_key, client_request_id)
    messages = list(_response_input_messages(request.input))
    if request.instructions is not None:
        messages.insert(0, GatewayMessage(role="developer", content=request.instructions))
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=tuple(messages),
            tools=tuple(_response_tool(tool) for tool in request.tools),
            tool_choice=_responses_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=_responses_structured_text(request.text),
            maximum_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=(request.top_logprobs is not None),
            top_logprobs=request.top_logprobs,
            reasoning_effort=(request.reasoning.effort if request.reasoning is not None else None),
            stream=request.stream,
            previous_response_id=request.previous_response_id,
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            idempotency_key=operation if idempotency_key is not None else None,
            client_request_id=operation if client_request_id is not None else None,
        )
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc
    return DecodedGatewayRequest(alias=request.model, request=canonical)


def _drop_opencode_cache_control(payload: JsonObject) -> JsonObject:
    """Remove supported OpenCode ``cache_control`` annotations from Chat messages.

    Args:
        payload: Parsed Chat Completions body.

    Returns:
        The original payload, or a shallow copy whose messages no longer carry
        a supported ``cache_control`` annotation.

    Raises:
        OpenAIProtocolError: A ``cache_control`` value is malformed or unsupported.
    """
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    cleaned_messages: list[JsonValue] = []
    changed = False
    for index, raw_message in enumerate(raw_messages):
        message, message_changed = _without_message_cache_control(raw_message, index)
        cleaned_messages.append(message)
        changed = changed or message_changed
    if not changed:
        return payload
    cleaned_payload = dict(payload)
    cleaned_payload["messages"] = cleaned_messages
    return cleaned_payload


def _without_message_cache_control(raw_message: JsonValue, index: int) -> tuple[JsonValue, bool]:
    """Drop a supported ``cache_control`` annotation from one Chat message.

    Args:
        raw_message: One ``messages`` entry.
        index: Zero-based message index used in public error paths.

    Returns:
        The message (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: The annotation is present but not a supported form.
    """
    if not isinstance(raw_message, dict):
        return raw_message, False
    message = cast(JsonObject, raw_message)
    changed = False
    if "cache_control" in message:
        _require_supported_cache_control(
            message["cache_control"], f"messages.{index}.cache_control"
        )
        message = {key: value for key, value in message.items() if key != "cache_control"}
        changed = True
    content = message.get("content")
    if isinstance(content, list):
        cleaned_content, content_changed = _without_text_part_cache_control(
            cast(list[JsonValue], content), index
        )
        if content_changed:
            if not changed:
                message = dict(message)
            message["content"] = cleaned_content
            changed = True
    return message, changed


def _without_text_part_cache_control(
    parts: list[JsonValue], message_index: int
) -> tuple[list[JsonValue], bool]:
    """Drop supported ``cache_control`` from OpenCode text content parts.

    Args:
        parts: Message ``content`` array.
        message_index: Zero-based parent message index used in public error paths.

    Returns:
        The content array (copied when an annotation is removed) and whether it changed.

    Raises:
        OpenAIProtocolError: A text-part annotation is present but not a supported form.
    """
    cleaned: list[JsonValue] = []
    changed = False
    for part_index, raw_part in enumerate(parts):
        if not isinstance(raw_part, dict) or "cache_control" not in raw_part:
            cleaned.append(raw_part)
            continue
        part = cast(JsonObject, raw_part)
        if part.get("type") not in _TEXT_PART_TYPES:
            cleaned.append(raw_part)
            continue
        _require_supported_cache_control(
            part["cache_control"],
            f"messages.{message_index}.content.{part_index}.cache_control",
        )
        cleaned.append({key: value for key, value in part.items() if key != "cache_control"})
        changed = True
    return (cleaned, True) if changed else (parts, False)


def _require_supported_cache_control(value: JsonValue, param: str) -> None:
    """Accept null or a supported ephemeral ``cache_control`` object.

    Args:
        value: Raw ``cache_control`` annotation.
        param: Public dotted field path used in the error.

    Raises:
        OpenAIProtocolError: The annotation is malformed or unsupported.
    """
    if value is None:
        return
    try:
        _EphemeralCacheControl.model_validate(value)
    except ValidationError as exc:
        raise invalid_field(param) from exc


def _validate_manifest(payload: JsonObject, manifest: CompatibilityManifest) -> None:
    """Reject unsupported and unknown top-level fields before responder work."""
    decisions = disposition_map(manifest)
    for field in payload:
        disposition = decisions.get(field)
        if disposition is None or disposition == CompatibilityDisposition.UNSUPPORTED:
            raise unsupported_field(field)


def _validate_official(
    adapter: TypeAdapter[object],
    payload: JsonObject,
    *,
    extension_fields: Collection[str] = frozenset(),
) -> None:
    """Run the installed official SDK request schema before gateway narrowing."""
    try:
        official_payload = {
            key: value for key, value in payload.items() if key not in extension_fields
        }
        adapter.validate_python(official_payload)
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc


def _validate_wire[ModelT: BaseModel](model: type[ModelT], payload: JsonObject) -> ModelT:
    """Validate one strict private wire model with a field-specific public error."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise _validation_protocol_error(exc) from exc


def _validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths."""
    first = error.errors(include_url=False)[0]
    location = tuple(
        part for part in first["loc"] if part not in {"body", "non-streaming", "streaming"}
    )
    param = ".".join(str(part) for part in location) or "body"
    return invalid_field(param)


def _caller_operation(idempotency_key: str | None, client_request_id: str | None) -> str | None:
    """Require two optional caller-operation headers to agree exactly."""
    for name, value in (
        ("Idempotency-Key", idempotency_key),
        ("X-Client-Request-Id", client_request_id),
    ):
        if value is not None and (
            not value or len(value) > 512 or any(ord(char) < 32 for char in value)
        ):
            raise invalid_field(name, f"{name} must be a non-empty display-safe value.")
    if (
        idempotency_key is not None
        and client_request_id is not None
        and idempotency_key != client_request_id
    ):
        raise OpenAIProtocolError(
            status_code=400,
            code="idempotency_conflict",
            message="Idempotency-Key and X-Client-Request-Id must identify the same operation.",
            param="Idempotency-Key",
        )
    return idempotency_key or client_request_id


def _messages(messages: tuple[_Message, ...], prefix: str) -> tuple[GatewayMessage, ...]:
    """Convert ordered wire messages while retaining raw assistant arguments."""
    converted: list[GatewayMessage] = []
    for message_index, message in enumerate(messages):
        calls = tuple(
            _tool_call(call, f"{prefix}.{message_index}.tool_calls.{call_index}.function.arguments")
            for call_index, call in enumerate(message.history_tool_calls)
        )
        converted.append(
            GatewayMessage(
                role=message.role,
                content=_content(message.content),
                tool_call_id=message.tool_call_id,
                tool_calls=calls,
            )
        )
    return tuple(converted)


def _content(content: str | tuple[_TextPart, ...] | None) -> str | None:
    """Join supported text parts without accepting multimodal loss."""
    if content is None or isinstance(content, str):
        return content
    return "".join(part.text for part in content)


def _tool_call(call: _AssistantToolCall, param: str) -> ToolCall:
    """Parse one complete tool call while retaining its exact raw argument string."""
    try:
        parsed = json.loads(call.function.arguments)
    except json.JSONDecodeError as exc:
        raise invalid_field(param, f"'{param}' must encode one JSON object.") from exc
    if not isinstance(parsed, dict):
        raise invalid_field(param, f"'{param}' must encode one JSON object.")
    return ToolCall(
        call_id=call.id,
        name=call.function.name,
        arguments=cast(JsonObject, parsed),
        raw_arguments=call.function.arguments,
    )


def _chat_tool(tool: _ChatTool) -> GatewayToolDefinition:
    """Convert one Chat function tool without weakening strictness."""
    return GatewayToolDefinition(
        name=tool.function.name,
        description=tool.function.description,
        parameters=tool.function.parameters,
        strict=tool.function.strict,
    )


def _response_tool(tool: _ResponseTool) -> GatewayToolDefinition:
    """Convert one Responses function tool without weakening strictness."""
    return GatewayToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        strict=bool(tool.strict),
    )


def _chat_tool_choice(
    value: JsonValue,
) -> Literal["auto", "none", "required"] | GatewayNamedToolChoice | None:
    """Normalize Chat tool-choice strings and named-function objects."""
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict):
        function = value.get("function")
        if value.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return GatewayNamedToolChoice(name=name)
    raise invalid_field("tool_choice")


def _responses_tool_choice(
    value: JsonValue,
) -> Literal["auto", "none", "required"] | GatewayNamedToolChoice | None:
    """Normalize Responses tool-choice strings and named-function objects."""
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("name")
        if isinstance(name, str):
            return GatewayNamedToolChoice(name=name)
    raise invalid_field("tool_choice")


def _chat_structured_text(value: _ChatResponseFormat | None) -> StructuredTextFormat | None:
    """Convert the Chat JSON Schema response format when requested."""
    if value is None or value.type == "text":
        return None
    schema = value.json_schema
    if schema is None:
        raise invalid_field("response_format.json_schema")
    return StructuredTextFormat(
        name=schema.name,
        description=schema.description,
        json_schema=schema.schema_,
        strict=schema.strict,
    )


def _responses_structured_text(value: _ResponseText | None) -> StructuredTextFormat | None:
    """Convert the Responses JSON Schema text format when requested."""
    if value is None or value.format.type == "text":
        return None
    schema = value.format.schema_
    name = value.format.name
    if schema is None or name is None:
        raise invalid_field("text.format")
    return StructuredTextFormat(
        name=name,
        description=value.format.description,
        json_schema=schema,
        strict=value.format.strict,
    )


def _response_input_messages(
    value: str | tuple[_ResponseMessage | _ResponseFunctionCall | _ResponseFunctionOutput, ...],
) -> tuple[GatewayMessage, ...]:
    """Convert Responses input items into ordered canonical history."""
    if isinstance(value, str):
        return (GatewayMessage(role="user", content=value),)
    messages: list[GatewayMessage] = []
    for index, item in enumerate(value):
        if isinstance(item, _ResponseMessage):
            messages.extend(_messages((item,), f"input.{index}"))
        elif isinstance(item, _ResponseFunctionCall):
            wire_call = _AssistantToolCall(
                id=item.call_id,
                function=_FunctionCall(name=item.name, arguments=item.arguments),
            )
            messages.append(
                GatewayMessage(
                    role="assistant",
                    tool_calls=(_tool_call(wire_call, f"input.{index}.arguments"),),
                )
            )
        else:
            messages.append(
                GatewayMessage(role="tool", content=item.output, tool_call_id=item.call_id)
            )
    return tuple(messages)
