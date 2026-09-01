"""Decode Chat Completions and Responses into canonical serving requests."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import Literal, cast

from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import ErrorDetails

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityManifest,
)
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
    SealedReasoningContentBlock,
    StructuredTextFormat,
)
from exp.runtime.gateway.reasoning_carrier import (
    FIREWORKS_REASONING_CONTENT_PREFIX,
    parse_reasoning_content_carrier,
)
from exp.runtime.openai_protocol.cache_control import (
    drop_opencode_cache_control,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, invalid_field, unsupported_field
from exp.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    RESPONSES_MANIFEST,
    disposition_map,
)
from exp.runtime.openai_protocol.responses_input import (
    ReplayedFunctionCall,
    ReplayedFunctionOutput,
    ReplayedInput,
    ReplayedMessage,
    ReplayedNativeItem,
    ReplayedReasoning,
    responses_input_messages,
)
from exp.runtime.openai_protocol.wire_models import (
    _AdditionalToolsItem,
    _AssistantToolCall,
    _ChatRequest,
    _ChatResponseFormat,
    _ChatTool,
    _CustomToolCall,
    _CustomToolCallOutput,
    _FunctionCall,
    _Message,
    _ResponseFunctionCall,
    _ResponseMessage,
    _ResponseReasoningItem,
    _ResponsesInputItem,
    _ResponsesRequest,
    _ResponseText,
    _ResponseTool,
    _TextPart,
)

_CHAT_OFFICIAL = TypeAdapter(CompletionCreateParams)
_RESPONSES_OFFICIAL = TypeAdapter(ResponseCreateParams)
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class DecodedGatewayRequest(ContractModel):
    """Public alias plus its canonical provider-neutral request."""

    alias: str = Field(min_length=1, max_length=256)
    request: GatewayRequest
    developer_messages_param: str | None = None


def _without_chat_reasoning_content(payload: JsonObject) -> JsonObject:
    """Hide the authenticated Chat extension from official OpenAI validation."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    changed = False
    messages: list[JsonValue] = []
    for raw_message in raw_messages:
        if isinstance(raw_message, dict) and "reasoning_content" in raw_message:
            messages.append(
                {key: value for key, value in raw_message.items() if key != "reasoning_content"}
            )
            changed = True
        else:
            messages.append(raw_message)
    return {**payload, "messages": messages} if changed else payload


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
    _validate_official(
        _CHAT_OFFICIAL,
        _without_chat_reasoning_content(payload),
        extension_fields={"top_k", "reasoning_effort"},
    )
    request = _validate_wire(_ChatRequest, payload)
    idempotency_key, client_request_id = _validated_operation_headers(
        idempotency_key, client_request_id
    )
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
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
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
    official_probe = dict(payload)
    if isinstance(raw := payload.get("input"), list):
        # The installed SDK lags the live surface on echoed output items:
        # it has no message `phase` and requires `status` alongside `id`,
        # while real Codex echoes carry id+phase and omit status, and it
        # requires reasoning `content` to be an array while Codex echoes an
        # explicit null that the provider accepts (both captured
        # 2026-08-29). The strict wire model owns those contracts, so the
        # official probe sees a normalized item.
        adapted: list[JsonValue] = []
        for entry in cast("list[JsonValue]", raw):
            if isinstance(entry, dict) and entry.get("type") == "message":
                item = {key: value for key, value in entry.items() if key != "phase"}
                if item.get("id") is not None and "status" not in item:
                    item["status"] = "completed"
                adapted.append(item)
            elif (
                isinstance(entry, dict)
                and entry.get("type") == "reasoning"
                and "content" in entry
                and entry.get("content") is None
            ):
                adapted.append({key: value for key, value in entry.items() if key != "content"})
            else:
                adapted.append(entry)
        official_probe["input"] = adapted
    _validate_official(
        _RESPONSES_OFFICIAL,
        official_probe,
        extension_fields={"top_k", "reasoning", "client_metadata"},
    )
    include_encrypted_reasoning = _include_encrypted_reasoning(request.include)
    idempotency_key, client_request_id = _validated_operation_headers(
        idempotency_key, client_request_id
    )
    raw_input = payload.get("input")
    raw_tools = payload.get("tools")
    function_tools: list[GatewayToolDefinition] = []
    native_tools: list[GatewayProviderNativeTool] = []
    for tool_index, declared in enumerate(request.tools):
        if isinstance(declared, _ResponseTool):
            function_tools.append(_response_tool(declared))
        else:
            # The raw caller declaration, not the re-serialized wire model,
            # so the native rung receives it byte-for-byte at its position.
            assert isinstance(raw_tools, list)
            native_tools.append(
                GatewayProviderNativeTool(
                    index=tool_index,
                    tool=cast("JsonObject", raw_tools[tool_index]),
                )
            )
    messages = list(
        _response_input_messages(
            request.input,
            raw_items=cast("list[JsonObject]", raw_input) if isinstance(raw_input, list) else (),
        )
    )
    if request.instructions is not None:
        messages.insert(0, GatewayMessage(role="developer", content=request.instructions))
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=tuple(messages),
            tools=tuple(function_tools),
            provider_native_tools=tuple(native_tools),
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
            text_verbosity=(request.text.verbosity if request.text is not None else None),
            client_metadata=request.client_metadata,
            response_store=request.store,
            include_encrypted_reasoning=include_encrypted_reasoning,
            stream=request.stream,
            previous_response_id=request.previous_response_id,
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
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
_OUTPUT_ITEM_VARIANTS = {
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "additional_tools",
    "custom_tool_call",
    "custom_tool_call_output",
}


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
        if text in _OUTPUT_ITEM_VARIANTS and cleaned and cleaned[-1].isdigit():
            continue
        cleaned.append(text)
    return tuple(cleaned)


_WIRE_TYPE_NAMES = {
    "str": "a string",
    "int": "an integer",
    "float": "a number",
    "bool": "a boolean",
    "list": "an array",
    "tuple": "an array",
    "dict": "an object",
    "NoneType": "null",
}
"""JSON-shape names for python input types, used in expected/got messages."""

_EXPECTED_BY_ERROR_TYPE = {
    "string_type": "a string",
    "string_too_short": "a non-empty string",
    "int_type": "an integer",
    "int_parsing": "an integer",
    "float_type": "a number",
    "float_parsing": "a number",
    "bool_type": "a boolean",
    "list_type": "an array",
    "tuple_type": "an array",
    "dict_type": "an object",
    "model_type": "an object",
    "model_attributes_type": "an object",
    "missing": "a value",
    "none_required": "null",
}
"""Shape-level expectations for the pydantic error types worth naming."""


def _shape_message(param: str, details: list[ErrorDetails]) -> str | None:
    """Describe what shape a field expected versus what arrived.

    Only structural facts appear: expectations come from this gateway's own
    wire models and the got side is the JSON type of the caller's value,
    never the value itself and never provider prose.
    """
    expected: list[str] = []
    got: str | None = None
    for detail in details:
        phrase = _EXPECTED_BY_ERROR_TYPE.get(detail["type"])
        if detail["type"] in {"literal_error", "enum"}:
            context = detail.get("ctx") or {}
            allowed = context.get("expected")
            if isinstance(allowed, str):
                phrase = f"one of {allowed}"
        if phrase is not None and phrase not in expected:
            expected.append(phrase)
        # A missing-field complaint carries the parent object as its input,
        # so it contributes no honest "got" type.
        if detail["type"] != "missing" and "input" in detail:
            got = _WIRE_TYPE_NAMES.get(type(detail["input"]).__name__, got)
    if not expected:
        return None
    description = " or ".join(expected)
    if got is not None:
        return f"Invalid value for '{param}': expected {description}, but got {got} instead."
    return f"Invalid value for '{param}': expected {description}."


def _validation_protocol_error(error: ValidationError) -> OpenAIProtocolError:
    """Convert Pydantic locations into stable dotted OpenAI ``param`` paths.

    Union validation reports every branch's complaints. Errors group by
    their branch (the location minus its final field segment); among the
    most field-specific groups, the branch the caller actually meant is the
    one with the fewest complaints, so its deepest cleaned location names
    the real field (an echoed item's ``input.1.caller``), never a union
    branch label such as ``input.str``. The chosen field's own complaints
    then name the expected shape against the arriving JSON type.
    """
    groups: dict[tuple[str | int, ...], list[tuple[tuple[str, ...], ErrorDetails]]] = {}
    for detail in error.errors(include_url=False):
        groups.setdefault(tuple(detail["loc"][:-1]), []).append(
            (_cleaned_location(detail["loc"]), detail)
        )
    if not groups:
        return invalid_field("body")
    deepest = max(len(location) for members in groups.values() for location, _ in members)
    candidates = [
        members
        for members in groups.values()
        if any(len(location) == deepest for location, _ in members)
    ]
    best = min(candidates, key=len)
    location = max((cleaned for cleaned, _ in best), key=len, default=())
    param = ".".join(location) or "body"
    details = [detail for cleaned, detail in best if cleaned == location]
    return invalid_field(param, _shape_message(param, details))


def _validated_operation_headers(
    idempotency_key: str | None, client_request_id: str | None
) -> tuple[str | None, str | None]:
    """Validate the two caller identity headers as independent concepts.

    ``Idempotency-Key`` names one retriable operation and is the only header
    that keys replay and duplicate detection. ``X-Client-Request-Id`` is a
    caller correlation identity: Codex sends its session id there on every
    request of a session (captured live 2026-08-29), and the provider serves
    those requests without deduplication, so treating it as an operation key
    would reject the second request of every real session as a conflict. It
    is echoed on responses and scopes route affinity only, and the two
    headers may therefore carry different values.
    """
    for name, value in (
        ("Idempotency-Key", idempotency_key),
        ("X-Client-Request-Id", client_request_id),
    ):
        if value is not None and (
            not value or len(value) > 512 or any(ord(char) < 32 for char in value)
        ):
            raise invalid_field(name, f"{name} must be a non-empty display-safe value.")
    return idempotency_key, client_request_id


def _messages(messages: tuple[_Message, ...], prefix: str) -> tuple[GatewayMessage, ...]:
    """Convert ordered wire messages while retaining raw assistant arguments."""
    converted: list[GatewayMessage] = []
    for message_index, message in enumerate(messages):
        calls = tuple(
            _tool_call(call, f"{prefix}.{message_index}.tool_calls.{call_index}.function.arguments")
            for call_index, call in enumerate(message.history_tool_calls)
        )
        provider_reasoning: tuple[SealedReasoningContentBlock, ...] = ()
        if message.reasoning_content is not None:
            try:
                provider_reasoning = (parse_reasoning_content_carrier(message.reasoning_content),)
            except ValueError as exc:
                param = f"{prefix}.{message_index}.reasoning_content"
                raise invalid_field(param, f"'{param}' must be a gateway-issued carrier.") from exc
        converted.append(
            GatewayMessage(
                role=message.role,
                content=_content(message.content),
                tool_call_id=message.tool_call_id,
                tool_calls=calls,
                provider_reasoning=provider_reasoning,
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
    if value is None or value.format is None or value.format.type == "text":
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
    *,
    raw_items: Sequence[JsonObject] = (),
) -> tuple[GatewayMessage, ...]:
    """Validate replay details and reconstruct OpenAI or Fireworks history."""
    if isinstance(value, str):
        return responses_input_messages(value)
    replayed: list[ReplayedInput] = []
    for index, item in enumerate(value):
        if isinstance(item, (_AdditionalToolsItem, _CustomToolCall, _CustomToolCallOutput)):
            # The raw caller item, not the re-serialized wire model, so the
            # native rung receives the item byte-for-byte.
            if isinstance(item, _CustomToolCall):
                native_role = "assistant"
            elif isinstance(item, _CustomToolCallOutput):
                native_role = "tool"
            else:
                native_role = "developer"
            replayed.append(
                ReplayedNativeItem(index=index, role=native_role, item=raw_items[index])
            )
        elif isinstance(item, _ResponseReasoningItem):
            if item.encrypted_content.startswith(FIREWORKS_REASONING_CONTENT_PREFIX):
                try:
                    block: EncryptedReasoningBlock | SealedReasoningContentBlock = (
                        parse_reasoning_content_carrier(item.encrypted_content)
                    )
                except ValueError as exc:
                    raise invalid_field(
                        f"input.{index}.encrypted_content",
                        "Responses encrypted_content must be a gateway-issued carrier.",
                    ) from exc
            else:
                block = EncryptedReasoningBlock(
                    id=item.id,
                    encrypted_content=item.encrypted_content,
                    output_index=index,
                    status=item.status,
                )
            replayed.append(ReplayedReasoning(index=index, block=block))
        elif isinstance(item, _ResponseMessage):
            converted = _messages((item,), f"input.{index}")
            if converted and item.role == "assistant":
                converted = (
                    converted[0].model_copy(
                        update={
                            "provider_item_id": item.id,
                            "provider_output_index": index if item.id is not None else None,
                            "provider_status": item.status,
                            "provider_phase": item.phase,
                        }
                    ),
                    *converted[1:],
                )
            replayed.append(ReplayedMessage(index=index, message=converted[0]))
        elif isinstance(item, _ResponseFunctionCall):
            wire_call = _AssistantToolCall(
                id=item.call_id,
                function=_FunctionCall(name=item.name, arguments=item.arguments),
            )
            replayed.append(
                ReplayedFunctionCall(
                    index=index,
                    call=_tool_call(wire_call, f"input.{index}.arguments").model_copy(
                        update={
                            "provider_item_id": item.id,
                            "provider_output_index": index,
                            "provider_status": item.status,
                        }
                    ),
                )
            )
        else:
            replayed.append(
                ReplayedFunctionOutput(index=index, call_id=item.call_id, output=item.output)
            )
    return responses_input_messages(tuple(replayed))
