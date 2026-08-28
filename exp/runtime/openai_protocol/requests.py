"""Decode Chat Completions and Responses into canonical serving requests."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Annotated, Literal, cast

from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import ReasoningEffort, ToolCall
from exp.runtime.gateway.contracts import (
    CompatibilityDisposition,
    CompatibilityManifest,
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
)
from exp.runtime.openai_protocol.cache_control import (
    EphemeralCacheControl,
    drop_opencode_cache_control,
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
    developer_messages_param: str | None = None


class _WireModel(BaseModel):
    """Strict private OpenAI wire model used only after official shape validation."""

    model_config = ConfigDict(extra="forbid")


_EchoedItemStatus = Literal["in_progress", "completed", "incomplete"]
"""Lifecycle marker carried by echoed output items; validated and dropped."""


class _TextPart(_WireModel):
    """One supported text-only content part.

    Echoed ``output_text`` parts carry empty ``annotations`` and ``logprobs``
    arrays (this gateway emits them and callers resend prior output verbatim
    on continuations); only a populated value is rejected as unsupported.
    """

    type: Literal["text", "input_text", "output_text"]
    text: str
    annotations: tuple[()] | None = None
    logprobs: tuple[()] | None = None


class _FunctionCall(_WireModel):
    """Function name and raw JSON argument string."""

    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)


class _AssistantToolCall(_WireModel):
    """One assistant function call retained in request history.

    OpenCode-style callers attach an Anthropic ``cache_control`` to the last
    content part of recent messages; when that part is a tool call the hint
    lands inside this entry, so the supported ephemeral form is accepted and
    carried for the one wire that can honor it.
    """

    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"] = "function"
    function: _FunctionCall
    cache_control: EphemeralCacheControl | None = None


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
        call_ids = tuple(call.id for call in self.history_tool_calls)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call IDs must be unique")
        return self


class _ResponseMessage(_Message):
    """Responses message item with its optional official discriminator.

    Stateless continuations echo prior OUTPUT items verbatim into the next
    input, so the provider-issued ``id`` and lifecycle ``status`` are
    accepted and dropped; the gateway mints its own public identities.
    """

    type: Literal["message"] = "message"
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None

    @model_validator(mode="after")
    def _require_output_identity_pair(self) -> _ResponseMessage:
        """Bind replayed output-message identity to its completion status."""
        if (self.id is None) != (self.status is None):
            raise ValueError("Responses output messages require both id and status")
        if self.id is not None and self.role != "assistant":
            raise ValueError("Responses output message identity requires role assistant")
        return self


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
    top_k: int | None = Field(default=None, ge=0)
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
    """Completed Responses function call included as assistant history.

    ``id`` and ``status`` arrive on verbatim echoes of prior output items
    and are accepted and dropped; ``call_id`` is the linkage that matters.
    """

    type: Literal["function_call"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: str = Field(max_length=4_000_000)
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None


class _ResponseFunctionOutput(_WireModel):
    """Text function result included as Responses tool history.

    ``id`` and ``status`` arrive when a stored turn's input items are
    re-listed and echoed; accepted and dropped like the other echo markers.
    """

    type: Literal["function_call_output"]
    call_id: str = Field(min_length=1, max_length=256)
    output: str
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None


class _ReasoningSummaryPart(_WireModel):
    """One display-only summary part replayed inside a reasoning input item."""

    type: Literal["summary_text"]
    text: str


class _ReasoningTextPart(_WireModel):
    """One display-only reasoning text part replayed inside a reasoning item."""

    type: Literal["reasoning_text"]
    text: str


class _ResponseReasoningItem(_WireModel):
    """One opaque reasoning item a stateless caller replays with its input.

    ``encrypted_content`` is the round-trip payload; the display-only
    ``summary`` and ``content`` parts and the echoed lifecycle ``status``
    are validated and dropped because the provider derives the
    model-visible reasoning from the encrypted payload alone.
    """

    type: Literal["reasoning"]
    id: str = Field(min_length=1, max_length=256)
    encrypted_content: str = Field(min_length=1)
    summary: tuple[_ReasoningSummaryPart, ...] = ()
    content: tuple[_ReasoningTextPart, ...] = ()
    status: _EchoedItemStatus | None = None


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
    """Responses reasoning controls accepted at the public boundary.

    The deprecated ``generate_summary`` alias is normalized to the current
    ``summary`` field before route capability shaping.
    """

    effort: ReasoningEffort | None = None
    context: Literal["auto", "current_turn", "all_turns"] | None = None
    generate_summary: Literal["auto", "concise", "detailed"] | None = None
    summary: Literal["auto", "concise", "detailed"] | None = None

    @model_validator(mode="after")
    def _require_matching_summary_aliases(self) -> _ResponseReasoning:
        """Reject two aliases that request different summary behavior."""
        if (
            self.generate_summary is not None
            and self.summary is not None
            and self.generate_summary != self.summary
        ):
            raise ValueError("reasoning summary and generate_summary must match")
        return self


_ResponsesOutputItem = Annotated[
    _ResponseFunctionCall | _ResponseFunctionOutput | _ResponseReasoningItem,
    Field(discriminator="type"),
]
_ResponsesInputItem = _ResponseMessage | _ResponsesOutputItem


class _ResponsesRequest(_WireModel):
    """Closed gateway Responses request profile."""

    model: str = Field(min_length=1, max_length=256)
    input: str | tuple[_ResponsesInputItem, ...]
    instructions: str | None = None
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    store: bool | None = None
    include: tuple[str, ...] | None = None
    tools: tuple[_ResponseTool, ...] = ()
    tool_choice: JsonValue = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
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
    payload = drop_opencode_cache_control(payload)
    _validate_manifest(payload, CHAT_MANIFEST)
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    _validate_official(_CHAT_OFFICIAL, payload, extension_fields={"top_k", "reasoning_effort"})
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
            maximum_output_tokens_parameter=(
                "max_completion_tokens"
                if request.max_completion_tokens is not None
                else "max_tokens"
                if request.max_tokens is not None
                else None
            ),
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
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    request = _validate_wire(_ResponsesRequest, payload)
    _validate_official(_RESPONSES_OFFICIAL, payload, extension_fields={"top_k", "reasoning"})
    include_encrypted_reasoning = _include_encrypted_reasoning(request.include)
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
            maximum_output_tokens_parameter=(
                "max_output_tokens" if request.max_output_tokens is not None else None
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=(True if request.top_logprobs is not None else None),
            top_logprobs=request.top_logprobs,
            reasoning_effort=(request.reasoning.effort if request.reasoning is not None else None),
            reasoning_context=(
                request.reasoning.context if request.reasoning is not None else None
            ),
            reasoning_summary=(
                request.reasoning.summary or request.reasoning.generate_summary
                if request.reasoning is not None
                else None
            ),
            reasoning_summary_parameters=(
                tuple(
                    path
                    for path, value in (
                        ("reasoning.generate_summary", request.reasoning.generate_summary),
                        ("reasoning.summary", request.reasoning.summary),
                    )
                    if value is not None
                )
                if request.reasoning is not None
                else ()
            ),
            response_store=request.store,
            include_encrypted_reasoning=include_encrypted_reasoning,
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
    developer_messages_param = None
    if request.instructions is not None:
        developer_messages_param = "instructions"
    elif not isinstance(request.input, str):
        developer_index = next(
            (
                index
                for index, item in enumerate(request.input)
                if isinstance(item, _ResponseMessage) and item.role == "developer"
            ),
            None,
        )
        if developer_index is not None:
            developer_messages_param = f"input.{developer_index}.role"
    return DecodedGatewayRequest(
        alias=request.model,
        request=canonical,
        developer_messages_param=developer_messages_param,
    )


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


_LOCATION_NOISE = {"body", "non-streaming", "streaming"}
_UNION_BRANCH_TYPES = {"str", "int", "float", "bool", "list", "tuple", "dict", "NoneType"}


def _cleaned_location(location: tuple[str | int, ...]) -> tuple[str, ...]:
    """Drop pydantic union-branch labels so the path names request fields."""
    cleaned: list[str] = []
    for part in location:
        text = str(part)
        if text in _LOCATION_NOISE:
            continue
        if isinstance(part, str) and (
            part.startswith("_") or "[" in text or text in _UNION_BRANCH_TYPES
        ):
            continue
        cleaned.append(text)
    return tuple(cleaned)


def _validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths.

    Union validation reports every branch's complaints. Errors group by
    their branch (the location minus its final field segment); among the
    most field-specific groups, the branch the caller actually meant is the
    one with the fewest complaints, so its deepest cleaned location names
    the real field (an echoed item's ``input.1.caller``), never a union
    branch label such as ``input.str``.
    """
    groups: dict[tuple[str | int, ...], list[tuple[str, ...]]] = {}
    for detail in error.errors(include_url=False):
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(_cleaned_location(detail["loc"]))
    if not groups:
        return invalid_field("body")
    deepest = max(len(location) for locations in groups.values() for location in locations)
    candidates = [
        locations
        for locations in groups.values()
        if any(len(location) == deepest for location in locations)
    ]
    best = min(candidates, key=len)
    location = max(best, key=len, default=())
    param = ".".join(location) or "body"
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
    # Some SDK stacks echo a zero-argument call as an empty string; the
    # canonical empty object mirrors the streaming completion seed, since no
    # provider wire accepts empty argument bytes.
    raw_arguments = call.function.arguments or "{}"
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise invalid_field(param, f"'{param}' must encode one JSON object.") from exc
    if not isinstance(parsed, dict):
        raise invalid_field(param, f"'{param}' must encode one JSON object.")
    return ToolCall(
        call_id=call.id,
        name=call.function.name,
        arguments=cast(JsonObject, parsed),
        raw_arguments=raw_arguments,
        cache_control=(
            call.cache_control.model_dump(mode="json", exclude_none=True)
            if call.cache_control is not None
            else None
        ),
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


def _include_encrypted_reasoning(include: tuple[str, ...] | None) -> bool:
    """Validate the closed ``include`` selector list.

    Args:
        include: Raw caller include paths.

    Returns:
        Whether the caller asked for ``reasoning.encrypted_content``.

    Raises:
        OpenAIProtocolError: An include path is not supported by this gateway.
    """
    if include is None:
        return False
    for path in include:
        if path != "reasoning.encrypted_content":
            raise invalid_field(
                "include",
                f"The include path {path!r} is not supported by this gateway. "
                "Only 'reasoning.encrypted_content' is available.",
            )
    return bool(include)


def _response_input_messages(
    value: str | tuple[_ResponsesInputItem, ...],
) -> tuple[GatewayMessage, ...]:
    """Convert Responses input items into ordered canonical history.

    Provider output items retain their exact identities and indexes. Contiguous
    function calls form one assistant turn so parallel calls and their leading
    reasoning replay in the provider's original order.
    """
    if isinstance(value, str):
        return (GatewayMessage(role="user", content=value),)
    messages: list[GatewayMessage] = []
    pending_reasoning: list[EncryptedReasoningBlock] = []
    pending_calls: list[ToolCall] = []

    def take_reasoning(role: str) -> tuple[EncryptedReasoningBlock, ...]:
        """Hand pending reasoning blocks to one assistant successor."""
        if role != "assistant" or not pending_reasoning:
            return ()
        taken = tuple(pending_reasoning)
        pending_reasoning.clear()
        return taken

    def flush_orphaned_reasoning() -> None:
        """Emit reasoning that has no assistant successor as its own turn."""
        if pending_reasoning:
            messages.append(
                GatewayMessage(role="assistant", provider_reasoning=take_reasoning("assistant"))
            )

    def flush_calls() -> None:
        """Group contiguous function-call items into their one assistant turn."""
        if pending_calls:
            messages.append(
                GatewayMessage(
                    role="assistant",
                    tool_calls=tuple(pending_calls),
                    provider_reasoning=take_reasoning("assistant"),
                )
            )
            pending_calls.clear()

    for index, item in enumerate(value):
        if isinstance(item, _ResponseReasoningItem):
            flush_calls()
            pending_reasoning.append(
                EncryptedReasoningBlock(
                    id=item.id,
                    encrypted_content=item.encrypted_content,
                    output_index=index,
                    status=item.status,
                )
            )
        elif isinstance(item, _ResponseMessage):
            flush_calls()
            if item.role != "assistant":
                flush_orphaned_reasoning()
            converted = _messages((item,), f"input.{index}")
            if converted and item.role == "assistant":
                converted = (
                    converted[0].model_copy(
                        update={
                            "provider_reasoning": take_reasoning("assistant"),
                            "provider_item_id": item.id,
                            "provider_output_index": index if item.id is not None else None,
                            "provider_status": item.status,
                        }
                    ),
                    *converted[1:],
                )
            messages.extend(converted)
        elif isinstance(item, _ResponseFunctionCall):
            wire_call = _AssistantToolCall(
                id=item.call_id,
                function=_FunctionCall(name=item.name, arguments=item.arguments),
            )
            pending_calls.append(
                _tool_call(wire_call, f"input.{index}.arguments").model_copy(
                    update={
                        "provider_item_id": item.id,
                        "provider_output_index": index,
                        "provider_status": item.status,
                    }
                )
            )
        else:
            flush_calls()
            flush_orphaned_reasoning()
            messages.append(
                GatewayMessage(role="tool", content=item.output, tool_call_id=item.call_id)
            )
    flush_calls()
    flush_orphaned_reasoning()
    return tuple(messages)
