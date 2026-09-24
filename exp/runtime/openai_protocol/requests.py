"""Decode Chat Completions and Responses into canonical serving requests."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from typing import Literal, cast

from openai.types import EmbeddingCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError, field_validator

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models.model import ToolCall
from exp.runtime.gateway.compatibility import (
    CompatibilityDisposition,
    CompatibilityManifest,
)
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    ExposedReasoningContentBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.embeddings_contracts import (
    EmbeddingInputs,
    EmbeddingsRequest,
    EmbeddingTokenIds,
)
from exp.runtime.gateway.reasoning_carrier import (
    FIREWORKS_REASONING_CONTENT_PREFIX,
    parse_reasoning_content_carrier,
    scheme_for_carrier,
)
from exp.runtime.models.providers.openrouter_routing import ProviderRoutingPreferences
from exp.runtime.openai_protocol.cache_control import (
    drop_opencode_cache_control,
    restore_chat_cache_control,
)
from exp.runtime.openai_protocol.enable_thinking import translate_enable_thinking
from exp.runtime.openai_protocol.errors import invalid_field, unsupported_field
from exp.runtime.openai_protocol.manifest import (
    CHAT_MANIFEST,
    EMBEDDINGS_MANIFEST,
    RESPONSES_MANIFEST,
    disposition_map,
)
from exp.runtime.openai_protocol.media_parts import message_content
from exp.runtime.openai_protocol.prompt_cache_key_alias import fold_prompt_cache_key_alias
from exp.runtime.openai_protocol.reasoning_replay import (
    ReplayedReasoningTooLong,
    fold_replayed_reasoning,
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
from exp.runtime.openai_protocol.responses_probe import (
    official_responses_probe,
    require_responses_input,
    require_responses_text_spelling,
)
from exp.runtime.openai_protocol.structured_text import (
    chat_json_object_output,
    chat_structured_text,
    responses_structured_text,
)
from exp.runtime.openai_protocol.tool_search import chat_tool_search, responses_tool_search
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error
from exp.runtime.openai_protocol.web_search import (
    chat_web_search,
    responses_web_search,
    split_online_suffix,
)
from exp.runtime.openai_protocol.wire_models import (
    HOSTED_TOOL_ITEM_TYPES_TOOL,
    _AdditionalToolsItem,
    _AssistantToolCall,
    _ChatRequest,
    _ChatTool,
    _CustomToolCall,
    _CustomToolCallOutput,
    _FunctionCall,
    _HostedToolItemEcho,
    _Message,
    _ResponseFunctionCall,
    _ResponseMessage,
    _ResponseReasoningItem,
    _ResponsesInputItem,
    _ResponsesRequest,
    _ResponseTool,
    _WireModel,
)


class _EmbeddingsRequest(_WireModel):
    """Closed OpenAI embeddings profile with text and pre-tokenized inputs.

    Attributes:
        model: Public model alias requested by the caller.
        input: One text/token input or a homogeneous batch of either form.
        dimensions: Optional positive vector width requested from the provider.
        encoding_format: Optional float or base64 response encoding.
        user: Optional gateway-only attribution, at most 1,024 characters.
        stream: Literal false convenience, validated and omitted upstream.
    """

    model: str = Field(min_length=1, max_length=256)
    input: str | EmbeddingTokenIds | EmbeddingInputs
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = Field(default=None, max_length=1024)
    stream: Literal[False] = False

    @field_validator("stream", mode="before")
    @classmethod
    def _require_nonstreaming(cls, value: JsonValue) -> Literal[False]:
        """Accept only literal false, without coercing zero or string values."""
        if value is not False:
            raise ValueError("embeddings do not stream; omit stream or set it to false")
        return False

    @field_validator("input")
    @classmethod
    def _require_nonempty_input(
        cls, value: str | EmbeddingTokenIds | EmbeddingInputs
    ) -> str | EmbeddingTokenIds | EmbeddingInputs:
        """Reject empty texts or batches without treating token ID zero as empty."""
        if isinstance(value, str):
            if not value:
                raise ValueError("input must not be an empty string")
            return value
        if not value:
            raise ValueError("input must not be an empty array")
        if any(not item for item in value if not isinstance(item, int)):
            raise ValueError("input array must not contain empty inputs")
        return value

    def batch_inputs(self) -> EmbeddingInputs:
        """Normalize one string or flat token sequence to one logical batch item."""
        if isinstance(self.input, str):
            return (self.input,)
        if isinstance(self.input[0], int):
            # The validated union is homogeneous; a flat sequence contains only token IDs.
            return (cast(EmbeddingTokenIds, self.input),)
        return cast(EmbeddingInputs, self.input)


_CHAT_OFFICIAL = TypeAdapter(CompletionCreateParams)
_RESPONSES_OFFICIAL = TypeAdapter(ResponseCreateParams)
# object-parametrized: EmbeddingCreateParams is one TypedDict, unlike the chat/responses unions.
_EMBEDDINGS_OFFICIAL: TypeAdapter[object] = TypeAdapter[object](EmbeddingCreateParams)
_TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text"})


class DecodedGatewayRequest(ContractModel):
    """Public alias plus its canonical provider-neutral request."""

    alias: str = Field(min_length=1, max_length=256)
    request: GatewayRequest
    developer_messages_param: str | None = None


class DecodedEmbeddingsRequest(ContractModel):
    """Public alias plus its canonical embeddings request.

    Distinct from :class:`DecodedGatewayRequest` because the embeddings surface
    carries its own message-less, non-streaming request contract.
    """

    alias: str = Field(min_length=1, max_length=256)
    request: EmbeddingsRequest


_CHAT_MESSAGE_EXTENSION_KEYS = frozenset(
    {
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "provider_specific_fields",
        "thinking_blocks",
        "reasoning_items",
        "images",
    }
)
"""Message keys the strict wire model owns; hidden from official validation.

``reasoning_content`` is the authenticated Chat extension and ``reasoning`` /
``reasoning_details`` are OpenRouter's replayed-reasoning fields (folded by
``reasoning_replay``); the other four are LiteLLM's message-dump keys, which
``_Message`` admits only in their empty (or, for ``provider_specific_fields``,
dropped-and-disclosed) forms.
"""


def _without_chat_message_extensions(payload: JsonObject) -> JsonObject:
    """Hide the wire-model-owned message keys from official OpenAI validation."""
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        return payload
    changed = False
    messages: list[JsonValue] = []
    for raw_message in raw_messages:
        if isinstance(raw_message, dict) and _CHAT_MESSAGE_EXTENSION_KEYS & raw_message.keys():
            messages.append(
                {
                    key: value
                    for key, value in raw_message.items()
                    if key not in _CHAT_MESSAGE_EXTENSION_KEYS
                }
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
    validated outside official OpenAI validation and retained on canonical
    cache carriers for adapters that support them. Other unknown nested fields stay rejected. The
    Vercel AI SDK's camelCase ``promptCacheKey`` is folded onto
    ``prompt_cache_key`` first, so it decodes as the documented wire field.

    Args:
        payload: Parsed JSON request body.
        idempotency_key: Optional standard caller operation identity.
        client_request_id: Optional gateway client request identity.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    payload, alias_disclosures = fold_prompt_cache_key_alias(payload)
    cache_payload = payload
    payload = drop_opencode_cache_control(payload)
    _validate_manifest(payload, CHAT_MANIFEST)
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    _validate_official(
        _CHAT_OFFICIAL,
        _without_chat_message_extensions(payload),
        extension_fields={
            "top_k",
            "reasoning_effort",
            "enable_thinking",
            "thinking_budget",
            "max_output_tokens",
            "provider",
            "gateway",
            "plugins",
        },
    )
    request = _validate_wire(_ChatRequest, payload)
    alias, online_suffix = split_online_suffix(request.model)
    idempotency_key, client_request_id = _validated_operation_headers(
        idempotency_key, client_request_id
    )
    maximum = request.max_completion_tokens or request.max_tokens or request.max_output_tokens
    stop = (
        ()
        if request.stop is None
        else (request.stop,)
        if isinstance(request.stop, str)
        else request.stop
    )
    thinking = translate_enable_thinking(request)
    raw_chat_tools = payload.get("tools")
    chat_native_tools = tuple(
        GatewayProviderNativeTool(index=index, tool=cast("JsonObject", raw_chat_tools[index]))
        for index, tool in enumerate(request.tools)
        if tool.type != "function" and isinstance(raw_chat_tools, list)
    )
    messages, cache_disclosures = restore_chat_cache_control(
        _messages(request.messages, "messages"), cache_payload
    )
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            tools=tuple(_chat_tool(tool) for tool in request.tools if tool.type == "function"),
            provider_native_tools=chat_native_tools,
            tool_search=chat_tool_search(chat_native_tools),
            tool_choice=_chat_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=chat_structured_text(request.response_format),
            json_object_output=chat_json_object_output(request.response_format),
            ignored_parameters=(
                *alias_disclosures,
                *cache_disclosures,
                *thinking.disclosures,
                *_replayed_reasoning_disclosures(request.messages),
            ),
            zdr_requested=request.provider is not None and request.provider.demands_zdr,
            provider_preferences=_provider_preferences(payload, request.provider),
            gateway=request.gateway,
            web_search=chat_web_search(
                options=request.web_search_options,
                plugins=request.plugins,
                online_suffix=online_suffix,
            ),
            maximum_output_tokens=maximum,
            maximum_output_tokens_parameter=(
                "max_completion_tokens"
                if request.max_completion_tokens is not None
                else "max_tokens"
                if request.max_tokens is not None
                else "max_output_tokens"
                if request.max_output_tokens is not None
                else None
            ),
            thinking_budget=request.thinking_budget,
            provider_thinking_config=thinking.thinking_config,
            stop=stop,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logprobs=request.logprobs,
            top_logprobs=request.top_logprobs,
            reasoning_effort=thinking.reasoning_effort,
            thinking_default_enable=thinking.thinking_default_enable,
            stream=request.stream,
            include_usage=(
                request.stream_options is not None and request.stream_options.include_usage
            ),
            metadata=request.metadata,
            safety_identifier=request.safety_identifier,
            user=request.user,
            prompt_cache_key=request.prompt_cache_key,
            service_tier=request.service_tier,
            text_verbosity=request.verbosity,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedGatewayRequest(alias=alias, request=canonical)


def decode_embeddings(payload: JsonObject) -> DecodedEmbeddingsRequest:
    """Decode one Embeddings body into the canonical embeddings surface.

    The embeddings surface has no idempotency protocol yet: keyed replay is a
    future add, so an inbound ``Idempotency-Key`` header is ignored upstream
    rather than keying this decode (which therefore takes no header arguments).

    Args:
        payload: Parsed JSON request body.

    Returns:
        Public alias and canonical embeddings request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    _validate_manifest(payload, EMBEDDINGS_MANIFEST)
    if payload.get("stream") is True:
        raise unsupported_field("stream")
    _validate_official(_EMBEDDINGS_OFFICIAL, payload, extension_fields={"stream"})
    request = _validate_wire(_EmbeddingsRequest, payload)
    try:
        canonical = EmbeddingsRequest(
            inputs=request.batch_inputs(),
            dimensions=request.dimensions,
            encoding_format=request.encoding_format,
            user=request.user,
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    return DecodedEmbeddingsRequest(alias=request.model, request=canonical)


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
    payload, alias_disclosures = fold_prompt_cache_key_alias(payload)
    _validate_manifest(payload, RESPONSES_MANIFEST)
    # The installed SDK's effort literal lags the newest provider tier
    # ("ultra"), so the strict wire model owns reasoning validation.
    request = _validate_wire(_ResponsesRequest, payload)
    alias, online_suffix = split_online_suffix(request.model)
    require_responses_input(request)
    if not isinstance(request.input, str):
        for item_index, item in enumerate(request.input):
            if isinstance(item, _ResponseMessage):
                require_responses_text_spelling(item_index, item)
    official_probe = official_responses_probe(payload)
    _validate_official(
        _RESPONSES_OFFICIAL,
        official_probe,
        extension_fields={"top_k", "reasoning", "client_metadata", "provider", "gateway"},
    )
    include_encrypted_reasoning, include_output_text_logprobs = _responses_include_options(
        request.include
    )
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
    replayed_items = cast("list[JsonObject]", raw_input) if isinstance(raw_input, list) else ()
    try:
        messages = list(_response_input_messages(request.input, raw_items=replayed_items))
    except ValidationError as exc:
        # History reconstruction folds echoed items into canonical messages,
        # so a canonical-contract violation (such as duplicate call_ids in one
        # assistant segment) first surfaces here, past the wire models. It is
        # caller-shaped input all the same: name the rule instead of letting
        # the exception escape as an unclassified 500.
        detail = exc.errors(include_url=False)[0]
        raise invalid_field(
            "input",
            "Invalid value for 'input': " + detail["msg"].removeprefix("Value error, ") + ".",
        ) from exc
    if request.instructions is not None:
        messages.insert(0, GatewayMessage(role="developer", content=request.instructions))
    try:
        canonical = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=tuple(messages),
            tools=tuple(function_tools),
            provider_native_tools=tuple(native_tools),
            web_search=responses_web_search(native_tools, online_suffix=online_suffix),
            tool_search=responses_tool_search(native_tools),
            tool_choice=_responses_tool_choice(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            structured_text=responses_structured_text(request.text),
            ignored_parameters=(
                *alias_disclosures,
                *_replayed_reasoning_disclosures(
                    ()
                    if isinstance(request.input, str)
                    else tuple(item for item in request.input if isinstance(item, _ResponseMessage))
                ),
            ),
            zdr_requested=request.provider is not None and request.provider.demands_zdr,
            provider_preferences=_provider_preferences(payload, request.provider),
            gateway=request.gateway,
            maximum_output_tokens=request.max_output_tokens,
            maximum_output_tokens_parameter=(
                "max_output_tokens" if request.max_output_tokens is not None else None
            ),
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            logprobs=None,
            top_logprobs=request.top_logprobs,
            include_output_text_logprobs=include_output_text_logprobs,
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
            service_tier=request.service_tier,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
    except ValidationError as exc:
        error = validation_protocol_error(exc)
        if error.detail.param == "messages":
            # The canonical transcript is empty: every input item was
            # consumed without a turn (``input: []`` beside a
            # previous_response_id). Name the field the caller sent.
            raise invalid_field(
                "input",
                "Invalid value for 'input': a continuation needs at least one new input "
                "item; resend the turn you want answered.",
            ) from exc
        raise error from exc
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
        alias=alias,
        request=canonical,
        developer_messages_param=developer_messages_param,
    )


def _provider_preferences(
    payload: JsonObject, preferences: ProviderRoutingPreferences | None
) -> JsonObject | None:
    """The caller's validated ``provider`` object as sent, or None when absent."""
    if preferences is None:
        return None
    raw = payload.get("provider")
    return dict(raw) if isinstance(raw, dict) else None


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
        raise validation_protocol_error(exc) from exc


def _validate_wire[ModelT: BaseModel](model: type[ModelT], payload: JsonObject) -> ModelT:
    """Validate one strict private wire model with a field-specific public error."""
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc


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
        provider_reasoning: tuple[
            SealedReasoningContentBlock | ExposedReasoningContentBlock, ...
        ] = ()
        try:
            folded = fold_replayed_reasoning(
                reasoning_content=message.reasoning_content,
                reasoning=message.reasoning,
                reasoning_details=message.reasoning_details,
            )
        except ReplayedReasoningTooLong as exc:
            param = f"{prefix}.{message_index}.reasoning_details"
            raise invalid_field(
                param,
                f"'{param}' plaintext reasoning exceeds 8,388,608 characters. "
                "Shorten the replayed reasoning_details and retry.",
            ) from exc
        if folded.plaintext is not None:
            param = f"{prefix}.{message_index}.{folded.source_field}"
            # The scheme is fixed by the carrier's own opaque prefix. A known
            # prefix MUST parse as that provider's carrier. Text under no known
            # prefix is the plaintext an exposure-gated rung itself returned on
            # an assistant turn (Tencent/DeepSeek), or the plaintext OpenRouter
            # handed the caller: it decodes as caller-owned history and route
            # admission decides which rungs may carry it.
            scheme = scheme_for_carrier(folded.plaintext)
            if scheme is None:
                # Plaintext reasoning is caller-owned history on ANY assistant
                # turn, tool-call turns included: AI-SDK clients re-serialize a
                # reasoning part onto the same assistant message as its tool
                # calls, and exposure-gated providers themselves emit
                # reasoning_content on tool turns. The field is baked into the
                # transcript, so rejecting it wedges every session that ever
                # touched a reasoning-exposed model (the sealed-carrier bond
                # applies only to text presented AS a gateway-issued carrier,
                # which keeps its strict path below).
                try:
                    provider_reasoning = (ExposedReasoningContentBlock(content=folded.plaintext),)
                except ValidationError as exc:
                    raise invalid_field(
                        param,
                        f"'{param}' plaintext reasoning exceeds 8,388,608 characters. "
                        f"Shorten the replayed {folded.source_field} and retry.",
                    ) from exc
            else:
                try:
                    provider_reasoning = (
                        parse_reasoning_content_carrier(folded.plaintext, scheme=scheme),
                    )
                except ValueError as exc:
                    raise invalid_field(
                        param, f"'{param}' must be a gateway-issued carrier."
                    ) from exc
        content, content_parts = message_content(
            message.content, f"{prefix}.{message_index}.content"
        )
        converted.append(
            GatewayMessage(
                role=message.role,
                content=content,
                content_parts=content_parts,
                tool_call_id=message.tool_call_id,
                tool_calls=calls,
                provider_tool_name=message.name,
                provider_reasoning=provider_reasoning,
                # An empty object is the common LiteLLM stamp and carries
                # nothing to disclose; only a populated one is a dropped field.
                provider_specific_fields=message.provider_specific_fields or None,
            )
        )
    return tuple(converted)


def _replayed_reasoning_disclosures(messages: Sequence[_Message]) -> tuple[str, ...]:
    """Collect the OpenRouter replay disclosures owed across a transcript, once each."""
    disclosures: list[str] = []
    for message in messages:
        try:
            folded = fold_replayed_reasoning(
                reasoning_content=message.reasoning_content,
                reasoning=message.reasoning,
                reasoning_details=message.reasoning_details,
            )
        except ReplayedReasoningTooLong:
            # ``_messages`` names the field in its own 400 before this runs
            # on the Chat path; the Responses path folds the same message.
            continue
        for disclosure in folded.disclosures:
            if disclosure not in disclosures:
                disclosures.append(disclosure)
    return tuple(disclosures)


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
    assert tool.function is not None  # the wire validator pairs type and body
    return GatewayToolDefinition(
        name=tool.function.name,
        description=tool.function.description,
        parameters=tool.function.parameters,
        strict=tool.function.strict,
        defer_loading=tool.defer_loading,
    )


def _response_tool(tool: _ResponseTool) -> GatewayToolDefinition:
    """Convert one Responses function tool without weakening strictness."""
    return GatewayToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        strict=bool(tool.strict),
        defer_loading=tool.defer_loading,
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


def _responses_include_options(include: tuple[str, ...] | None) -> tuple[bool, bool]:
    """Validate the closed ``include`` selector list.

    Args:
        include: Raw caller include paths.

    Returns:
        A pair of encrypted-reasoning and output-text-logprobs selectors.

    Raises:
        OpenAIProtocolError: An include path is not supported by this gateway.
    """
    if include is None:
        return False, False
    for path in include:
        if path not in {"reasoning.encrypted_content", "message.output_text.logprobs"}:
            raise invalid_field(
                "include",
                f"The include path {path!r} is not supported by this gateway. "
                "Only 'reasoning.encrypted_content' and "
                "'message.output_text.logprobs' are available.",
            )
    return (
        "reasoning.encrypted_content" in include,
        "message.output_text.logprobs" in include,
    )


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
        if isinstance(
            item,
            (_AdditionalToolsItem, _CustomToolCall, _CustomToolCallOutput, _HostedToolItemEcho),
        ):
            # The raw caller item, not the re-serialized wire model, so the
            # native rung receives the item byte-for-byte.
            if isinstance(item, _HostedToolItemEcho):
                native_role = "tool" if item.type in HOSTED_TOOL_ITEM_TYPES_TOOL else "assistant"
            elif isinstance(item, _CustomToolCall):
                native_role = "assistant"
            elif isinstance(item, _CustomToolCallOutput):
                native_role = "tool"
            else:
                native_role = "developer"
            replayed.append(
                ReplayedNativeItem(index=index, role=native_role, item=raw_items[index])
            )
        elif isinstance(item, _ResponseReasoningItem):
            if item.encrypted_content is None:
                # A store=true flow replays reasoning by item id alone (the
                # SDK marks encrypted_content optional); only the issuing
                # native Responses wire can resolve the id, so the item is
                # carried verbatim like a hosted-tool item and the provider
                # judges resolvability, rather than rejecting SDK-legal
                # input the provider itself may serve.
                replayed.append(
                    ReplayedNativeItem(index=index, role="assistant", item=raw_items[index])
                )
                continue
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
                            "provider_namespace": item.namespace,
                            "provider_caller": item.caller,
                        }
                    ),
                )
            )
        else:
            if isinstance(item.output, str):
                output_text, output_parts = item.output, ()
            else:
                # The SDK list form: text and image parts map onto the
                # canonical tool message (the tool-message contract carries
                # exactly those two kinds); any other kind is a named 400
                # because a tool result has no canonical carrier for it and
                # dropping it would misstate what the tool returned.
                output_text, output_parts = message_content(item.output, f"input.{index}.output")
                unsupported = next(
                    (part for part in output_parts if part.kind not in ("text", "image")),
                    None,
                )
                if unsupported is not None:
                    raise unsupported_field(
                        f"input.{index}.output",
                        message=(
                            "function_call_output.output supports text and image parts "
                            f"only; this list carries a {unsupported.kind!r} part."
                        ),
                    )
                output_text = output_text or ""
            replayed.append(
                ReplayedFunctionOutput(
                    index=index,
                    call_id=item.call_id,
                    output=output_text,
                    name=item.name,
                    namespace=item.namespace,
                    caller=item.caller,
                    content_parts=output_parts,
                )
            )
    return responses_input_messages(tuple(replayed))
