"""Decode Gemini ``generateContent`` bodies into canonical serving requests.

The decoder is strict and lossless for supported content: text, ``inlineData``
(images, video, audio, PDF), ``functionCall`` and ``functionResponse`` parts,
``systemInstruction``, function declarations, ``toolConfig`` and the
``generationConfig`` sampling, output, structured-output and thinking controls
translate faithfully. Google's REST surface accepts the protobuf JSON mapping in
both spellings, so every field is read as ``camelCase`` or ``snake_case``
(the official SDK sends ``thinking_level`` and ``parameters_json_schema`` in
snake case inside otherwise camel-cased objects). A caller's ``thought`` parts
are display copy and drop with disclosure, except a gateway-issued sealed
reasoning carrier echoed on a model turn, which replays authenticated. Unknown
or unsupported fields are rejected with a field-specific error, never silently
dropped. Errors raise :class:`OpenAIProtocolError` so the shared boundary stays
single-authority; the data plane renders them in Google's error envelope.
"""

from __future__ import annotations

import json
from typing import Literal, cast

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.models.content import (
    AUDIO_MEDIA_TYPES,
    DOCUMENT_MEDIA_TYPES,
    IMAGE_MEDIA_TYPES,
    VIDEO_MEDIA_TYPES,
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    MessageContentPart,
    TextContentPart,
    VideoContentPart,
    image_part_from_url,
    video_part_from_url,
)
from exp.common.models.model import ReasoningEffort, ToolCall
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ProviderReasoningBlock,
    StructuredTextFormat,
)
from exp.runtime.gateway.reasoning_carrier import (
    parse_reasoning_content_carrier,
    scheme_for_carrier,
)
from exp.runtime.gateway.request_policy import GatewayRequestPolicy
from exp.runtime.gateway.tool_contracts import (
    GatewayAllowedToolsChoice,
    GatewayNamedToolChoice,
    GatewayToolChoice,
    GatewayToolDefinition,
)
from exp.runtime.google_protocol.manifest import GENERATE_CONTENT_MANIFEST
from exp.runtime.models.providers.openrouter_routing import ProviderRoutingPreferences
from exp.runtime.openai_protocol.errors import (
    OpenAIProtocolError,
    invalid_field,
    unsupported_field,
)
from exp.runtime.openai_protocol.manifest import validate_manifest
from exp.runtime.openai_protocol.requests import DecodedGatewayRequest
from exp.runtime.openai_protocol.tool_argument_repairs import replayed_tool_arguments
from exp.runtime.openai_protocol.validation_errors import validation_protocol_error

_REASONING_EFFORTS: frozenset[str] = frozenset(
    ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
)
_JSON_MIME = "application/json"
_TEXT_MIMES = frozenset({"text/plain"})
THOUGHT_DROPPED = "contents.parts.thought->dropped(display_only)"
"""Disclosure for a replayed thought summary: display copy, never a provider input."""
TEXT_SIGNATURE_DROPPED = "contents.parts.thoughtSignature->dropped(text_part)"
"""Disclosure for a thought signature on a text part (only call signatures replay)."""
SAFETY_SETTINGS_DROPPED = "safetySettings->dropped(gateway_guardrails)"
"""Disclosure for Google's per-category thresholds: the gateway's guardrails apply instead."""
LABELS_DROPPED = "labels->dropped(unsupported)"
SERVICE_TIER_DROPPED = "serviceTier->dropped(unsupported)"
STORE_DROPPED = "store->dropped(unsupported)"
SEED_DROPPED = "generationConfig.seed->dropped(unsupported)"
LOGPROBS_DROPPED = "generationConfig.responseLogprobs->dropped(unsupported)"
MEDIA_RESOLUTION_DROPPED = "generationConfig.mediaResolution->dropped(unsupported)"
VALIDATED_MODE_AS_AUTO = "toolConfig.functionCallingConfig.mode->auto(validated)"
BEHAVIOR_DROPPED = "tools.functionDeclarations.behavior->dropped(unsupported)"
RESPONSE_SCHEMA_DROPPED = "tools.functionDeclarations.response->dropped(unsupported)"
"""Disclosure for a declared function output schema; no gateway wire carries one."""
_THOUGHT_SIGNATURE_MAX = 65_536
# Gemini spells MP3 audio `audio/mp3` (and WAV sometimes `audio/x-wav`); the
# canonical audio contract knows the IANA names.
_AUDIO_MIME_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
}


def _names(camel: str, snake: str) -> AliasChoices:
    """Both JSON spellings of one protobuf field."""
    return AliasChoices(camel, snake)


class _Wire(BaseModel):
    """Strict private Gemini wire model rejecting unknown nested fields."""

    model_config = ConfigDict(extra="forbid")


class _Blob(_Wire):
    """Inline media bytes."""

    mime_type: str = Field(validation_alias=_names("mimeType", "mime_type"), max_length=256)
    data: str


class _FileData(_Wire):
    """A URI-referenced media part; only public HTTP(S) URLs can be served."""

    mime_type: str | None = Field(
        default=None, validation_alias=_names("mimeType", "mime_type"), max_length=256
    )
    file_uri: str = Field(validation_alias=_names("fileUri", "file_uri"), max_length=8_192)


class _FunctionCall(_Wire):
    """A replayed model function call."""

    id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    args: JsonObject | None = None


class _FunctionResponse(_Wire):
    """The caller's answer to one function call."""

    id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    response: JsonObject | None = None


class _Part(_Wire):
    """One Gemini content part; executable-code parts are refused by name below."""

    text: str | None = None
    inline_data: _Blob | None = Field(
        default=None, validation_alias=_names("inlineData", "inline_data")
    )
    file_data: _FileData | None = Field(
        default=None, validation_alias=_names("fileData", "file_data")
    )
    function_call: _FunctionCall | None = Field(
        default=None, validation_alias=_names("functionCall", "function_call")
    )
    function_response: _FunctionResponse | None = Field(
        default=None, validation_alias=_names("functionResponse", "function_response")
    )
    thought: bool | None = None
    thought_signature: str | None = Field(
        default=None,
        validation_alias=_names("thoughtSignature", "thought_signature"),
        max_length=_THOUGHT_SIGNATURE_MAX,
    )
    executable_code: JsonObject | None = Field(
        default=None, validation_alias=_names("executableCode", "executable_code")
    )
    code_execution_result: JsonObject | None = Field(
        default=None, validation_alias=_names("codeExecutionResult", "code_execution_result")
    )
    video_metadata: JsonObject | None = Field(
        default=None, validation_alias=_names("videoMetadata", "video_metadata")
    )


class _Content(_Wire):
    """One conversation turn."""

    role: Literal["user", "model"] | None = None
    parts: tuple[_Part, ...] = Field(min_length=1)


class _FunctionDeclaration(_Wire):
    """One caller function the model may call."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = None
    parameters: JsonObject | None = None
    parameters_json_schema: JsonObject | None = Field(
        default=None, validation_alias=_names("parametersJsonSchema", "parameters_json_schema")
    )
    response: JsonObject | None = None
    response_json_schema: JsonObject | None = Field(
        default=None, validation_alias=_names("responseJsonSchema", "response_json_schema")
    )
    behavior: str | None = Field(default=None, max_length=64)


_SERVER_TOOL_FIELDS = (
    ("googleSearch", "google_search"),
    ("googleSearchRetrieval", "google_search_retrieval"),
    ("codeExecution", "code_execution"),
    ("urlContext", "url_context"),
    ("fileSearch", "file_search"),
    ("computerUse", "computer_use"),
    ("googleMaps", "google_maps"),
    ("enterpriseWebSearch", "enterprise_web_search"),
    ("retrieval", "retrieval"),
    ("mcpServers", "mcp_servers"),
)


class _Tool(_Wire):
    """One tools entry; Google's server-executed tools are refused by name."""

    function_declarations: tuple[_FunctionDeclaration, ...] | None = Field(
        default=None, validation_alias=_names("functionDeclarations", "function_declarations")
    )
    google_search: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[0])
    )
    google_search_retrieval: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[1])
    )
    code_execution: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[2])
    )
    url_context: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[3])
    )
    file_search: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[4])
    )
    computer_use: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[5])
    )
    google_maps: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[6])
    )
    enterprise_web_search: JsonObject | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[7])
    )
    retrieval: JsonObject | None = None
    mcp_servers: JsonValue | None = Field(
        default=None, validation_alias=_names(*_SERVER_TOOL_FIELDS[9])
    )


class _FunctionCallingConfig(_Wire):
    """Google's tool-choice selector."""

    mode: Literal["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE", "VALIDATED"] | None = None
    allowed_function_names: tuple[str, ...] | None = Field(
        default=None, validation_alias=_names("allowedFunctionNames", "allowed_function_names")
    )


class _ToolConfig(_Wire):
    """The tool configuration object."""

    function_calling_config: _FunctionCallingConfig | None = Field(
        default=None, validation_alias=_names("functionCallingConfig", "function_calling_config")
    )


class _ThinkingConfig(_Wire):
    """Gemini thinking controls."""

    include_thoughts: bool | None = Field(
        default=None, validation_alias=_names("includeThoughts", "include_thoughts")
    )
    thinking_budget: int | None = Field(
        default=None, validation_alias=_names("thinkingBudget", "thinking_budget"), ge=-1
    )
    thinking_level: str | None = Field(
        default=None, validation_alias=_names("thinkingLevel", "thinking_level"), max_length=32
    )


class _GenerationConfig(_Wire):
    """Sampling, output and structured-output controls."""

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, validation_alias=_names("topP", "top_p"), ge=0, le=1)
    top_k: int | None = Field(default=None, validation_alias=_names("topK", "top_k"), ge=0)
    candidate_count: int | None = Field(
        default=None, validation_alias=_names("candidateCount", "candidate_count"), ge=1
    )
    max_output_tokens: int | None = Field(
        default=None, validation_alias=_names("maxOutputTokens", "max_output_tokens"), gt=0
    )
    stop_sequences: tuple[str, ...] | None = Field(
        default=None, validation_alias=_names("stopSequences", "stop_sequences")
    )
    response_mime_type: str | None = Field(
        default=None,
        validation_alias=_names("responseMimeType", "response_mime_type"),
        max_length=128,
    )
    response_schema: JsonObject | None = Field(
        default=None, validation_alias=_names("responseSchema", "response_schema")
    )
    response_json_schema: JsonObject | None = Field(
        default=None, validation_alias=_names("responseJsonSchema", "response_json_schema")
    )
    response_modalities: tuple[str, ...] | None = Field(
        default=None, validation_alias=_names("responseModalities", "response_modalities")
    )
    presence_penalty: float | None = Field(
        default=None, validation_alias=_names("presencePenalty", "presence_penalty"), ge=-2, le=2
    )
    frequency_penalty: float | None = Field(
        default=None, validation_alias=_names("frequencyPenalty", "frequency_penalty"), ge=-2, le=2
    )
    seed: int | None = None
    response_logprobs: bool | None = Field(
        default=None, validation_alias=_names("responseLogprobs", "response_logprobs")
    )
    logprobs: int | None = None
    thinking_config: _ThinkingConfig | None = Field(
        default=None, validation_alias=_names("thinkingConfig", "thinking_config")
    )
    media_resolution: str | None = Field(
        default=None, validation_alias=_names("mediaResolution", "media_resolution"), max_length=64
    )
    speech_config: JsonObject | None = Field(
        default=None, validation_alias=_names("speechConfig", "speech_config")
    )
    image_config: JsonObject | None = Field(
        default=None, validation_alias=_names("imageConfig", "image_config")
    )
    audio_timestamp: bool | None = Field(
        default=None, validation_alias=_names("audioTimestamp", "audio_timestamp")
    )
    enable_enhanced_civic_answers: bool | None = Field(
        default=None,
        validation_alias=_names("enableEnhancedCivicAnswers", "enable_enhanced_civic_answers"),
    )
    audio_transcription_config: JsonObject | None = Field(
        default=None,
        validation_alias=_names("audioTranscriptionConfig", "audio_transcription_config"),
    )


class _GenerateContentRequest(_Wire):
    """Closed gateway ``generateContent`` request profile."""

    contents: tuple[_Content, ...] = Field(min_length=1)
    system_instruction: _Content | None = Field(
        default=None, validation_alias=_names("systemInstruction", "system_instruction")
    )
    tools: tuple[_Tool, ...] | None = None
    tool_config: _ToolConfig | None = Field(
        default=None, validation_alias=_names("toolConfig", "tool_config")
    )
    generation_config: _GenerationConfig | None = Field(
        default=None, validation_alias=_names("generationConfig", "generation_config")
    )
    safety_settings: tuple[JsonObject, ...] | None = Field(
        default=None, validation_alias=_names("safetySettings", "safety_settings")
    )
    cached_content: str | None = Field(
        default=None, validation_alias=_names("cachedContent", "cached_content")
    )
    labels: JsonObject | None = None
    service_tier: str | None = Field(
        default=None, validation_alias=_names("serviceTier", "service_tier"), max_length=64
    )
    store: bool | None = None
    continuation_token: str | None = Field(
        default=None, validation_alias=_names("continuationToken", "continuation_token")
    )
    provider: ProviderRoutingPreferences | None = None
    gateway: GatewayRequestPolicy | None = None


class _CountTokensRequest(_Wire):
    """The ``countTokens`` body: the prompt alone, or a whole wrapped request."""

    contents: tuple[_Content, ...] | None = Field(default=None, min_length=1)
    generate_content_request: _GenerateContentRequest | None = Field(
        default=None,
        validation_alias=_names("generateContentRequest", "generate_content_request"),
    )


def decode_generate_content(
    payload: JsonObject, *, model: str, stream: bool
) -> DecodedGatewayRequest:
    """Decode one ``generateContent`` / ``streamGenerateContent`` body.

    The model travels in the URL path on this protocol, so the caller passes it
    separately; the body carries no model field. Google defines no idempotency
    header, so this surface never participates in keyed replay.

    Args:
        payload: Parsed JSON request body.
        model: The catalog alias named in the request path.
        stream: Whether the action is ``streamGenerateContent``.

    Returns:
        Public alias and lossless canonical gateway request.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
            The data plane renders it in Google's error envelope.
    """
    if not model:
        raise invalid_field(
            "model", "The request path must name a model: /v1beta/models/<model>:generateContent."
        )
    validate_manifest(payload, GENERATE_CONTENT_MANIFEST)
    request = _validate(payload, _GenerateContentRequest)
    return DecodedGatewayRequest(alias=model, request=_canonical(request, stream=stream))


def decode_generate_content_count_tokens(
    payload: JsonObject, *, model: str
) -> DecodedGatewayRequest:
    """Decode one ``countTokens`` body for prompt counting.

    Google's count request carries ``contents`` (or a whole
    ``generateContentRequest``), so the canonical request has no output budget
    unless the wrapped request named one and is never dispatched, only counted.

    Args:
        payload: Parsed JSON request body.
        model: The catalog alias named in the request path.

    Returns:
        Public alias and the canonical request to count.

    Raises:
        OpenAIProtocolError: The body is invalid, unknown, or unsupported.
    """
    if not model:
        raise invalid_field(
            "model", "The request path must name a model: /v1beta/models/<model>:countTokens."
        )
    count = _validate(payload, _CountTokensRequest)
    if count.generate_content_request is not None:
        if count.contents is not None:
            raise invalid_field("contents", "Send contents or generateContentRequest, not both.")
        request = count.generate_content_request
    elif count.contents is not None:
        request = _GenerateContentRequest(contents=count.contents)
    else:
        raise invalid_field("contents", "countTokens needs contents or a generateContentRequest.")
    return DecodedGatewayRequest(alias=model, request=_canonical(request, stream=False))


def _validate[WireT: BaseModel](payload: JsonObject, wire: type[WireT]) -> WireT:
    """Validate against the closed wire model, naming the offending field."""
    try:
        return wire.model_validate(payload)
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc


def _canonical(request: _GenerateContentRequest, *, stream: bool) -> GatewayRequest:
    """Build the canonical request from one validated wire request."""
    disclosures: list[str] = []
    if request.cached_content is not None:
        raise unsupported_field(
            "cachedContent",
            message=(
                "cachedContent names a Google cache resource the gateway cannot address. "
                "Remove it; the gateway caches prompts on Gemini lanes automatically."
            ),
        )
    if request.continuation_token is not None:
        raise unsupported_field(
            "continuationToken",
            message=(
                "continuationToken resumes a Google-side generation the gateway never holds. "
                "Resend the whole conversation instead."
            ),
        )
    if request.safety_settings:
        disclosures.append(SAFETY_SETTINGS_DROPPED)
    if request.labels:
        disclosures.append(LABELS_DROPPED)
    if request.service_tier is not None:
        disclosures.append(SERVICE_TIER_DROPPED)
    if request.store is not None:
        disclosures.append(STORE_DROPPED)
    messages: list[GatewayMessage] = []
    system = _system_text(request.system_instruction)
    if system is not None:
        messages.append(GatewayMessage(role="system", content=system))
    pending_calls: dict[str, str] = {}
    unanswered: list[tuple[str, str]] = []
    for index, content in enumerate(request.contents):
        messages.extend(
            _turn_messages(content, f"contents.{index}", pending_calls, unanswered, disclosures)
        )
    tools, tool_choice = _tools(request.tools, request.tool_config, disclosures)
    config = request.generation_config or _GenerationConfig()
    output = _output_format(config)
    thinking = _thinking(config.thinking_config)
    try:
        return GatewayRequest(
            surface=GatewayApiSurface.GENERATE_CONTENT,
            messages=tuple(messages),
            tools=tools,
            tool_choice=tool_choice,
            structured_text=output[0],
            json_object_output=output[1],
            maximum_output_tokens=config.max_output_tokens,
            maximum_output_tokens_parameter=(
                "generationConfig.maxOutputTokens" if config.max_output_tokens is not None else None
            ),
            stop=_stop_sequences(config.stop_sequences),
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            reasoning_effort=thinking[0],
            reasoning_effort_parameter=(
                "generationConfig.thinkingConfig.thinkingLevel" if thinking[0] is not None else None
            ),
            thinking_budget=thinking[1],
            include_thoughts=thinking[2],
            stream=stream,
            include_usage=stream,
            ignored_parameters=tuple(dict.fromkeys((*disclosures, *_config_disclosures(config)))),
            gateway=request.gateway,
            zdr_requested=request.provider is not None and request.provider.demands_zdr,
            provider_preferences=(
                request.provider.model_dump(mode="json", exclude_none=True)
                if request.provider is not None
                else None
            ),
        )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc


def _system_text(instruction: _Content | None) -> str | None:
    """Flatten the system instruction's text parts; blank-line joined."""
    if instruction is None:
        return None
    texts: list[str] = []
    for index, part in enumerate(instruction.parts):
        param = f"systemInstruction.parts.{index}"
        _require_one_payload(part, param)
        if part.text is None:
            raise invalid_field(param, "systemInstruction parts must be text.")
        texts.append(part.text)
    joined = "\n\n".join(text for text in texts if text)
    return joined or None


def _require_one_payload(part: _Part, param: str) -> None:
    """Refuse code-execution parts and any part not carrying exactly one payload.

    ``Part`` is a protobuf ``oneof``: JSON that sets two members (``text`` and
    ``inlineData``, a ``functionCall`` beside a ``functionResponse``) is
    malformed, and picking one would silently change the caller's content.
    """
    if part.executable_code is not None:
        raise unsupported_field(f"{param}.executableCode")
    if part.code_execution_result is not None:
        raise unsupported_field(f"{param}.codeExecutionResult")
    if part.video_metadata is not None:
        raise unsupported_field(f"{param}.videoMetadata")
    payloads = sum(
        value is not None
        for value in (
            part.text,
            part.inline_data,
            part.file_data,
            part.function_call,
            part.function_response,
        )
    )
    if payloads != 1:
        raise invalid_field(
            param,
            "A part carries exactly one of text, inlineData, fileData, functionCall or "
            "functionResponse.",
        )


def _turn_messages(
    content: _Content,
    param: str,
    pending_calls: dict[str, str],
    unanswered: list[tuple[str, str]],
    disclosures: list[str],
) -> list[GatewayMessage]:
    """Translate one Gemini content into canonical messages, in part order.

    A ``functionResponse`` part becomes a standalone tool message (the canonical
    contract keeps ``tool_call_id`` on tool messages only), so a user turn
    mixing answers and text splits into several messages. A response without
    an ``id`` answers the oldest unanswered call of the same name, which is how
    Gemini itself pairs them.
    """
    role = content.role or "user"
    out: list[GatewayMessage] = []
    segments: list[MessageContentPart] = []
    calls: list[ToolCall] = []
    reasoning: list[ProviderReasoningBlock] = []

    def flush() -> None:
        """Emit the pending segments, calls and reasoning as one message.

        Segments keep the caller's part order (an image, its caption, the next
        image); the canonical ``content`` is their text flattened in that order
        and ``content_parts`` is retained only when media is present.
        """
        if not segments and not calls and not reasoning:
            return
        text = "".join(part.text for part in segments if part.kind == "text")
        multimodal = any(part.kind != "text" for part in segments)
        retained = tuple(part for part in segments if part.kind != "text" or part.text)
        out.append(
            GatewayMessage(
                role="assistant" if role == "model" else "user",
                content=text if (segments or not calls) else None,
                content_parts=retained if multimodal else (),
                tool_calls=tuple(calls),
                provider_reasoning=tuple(reasoning),
            )
        )
        segments.clear()
        calls.clear()
        reasoning.clear()

    for index, part in enumerate(content.parts):
        part_param = f"{param}.parts.{index}"
        _require_one_payload(part, part_param)
        if part.function_response is not None:
            if role == "model":
                raise invalid_field(part_param, "functionResponse parts belong to user turns.")
            flush()
            out.append(_tool_message(part.function_response, part_param, pending_calls, unanswered))
            continue
        if part.function_call is not None:
            if role != "model":
                raise invalid_field(part_param, "functionCall parts belong to model turns.")
            calls.append(_tool_call(part, pending_calls, unanswered, len(calls)))
            continue
        if part.thought:
            scheme = scheme_for_carrier(part.text) if part.text else None
            if scheme is not None and role == "model" and part.text is not None:
                # A gateway-issued sealed carrier echoed on a model turn
                # replays authenticated; any other thought text is display
                # copy the provider never sees again.
                try:
                    reasoning.append(parse_reasoning_content_carrier(part.text, scheme=scheme))
                except ValueError as exc:
                    raise invalid_field(
                        part_param, "thought part is not a gateway-issued carrier."
                    ) from exc
            else:
                disclosures.append(THOUGHT_DROPPED)
            continue
        if part.thought_signature is not None:
            disclosures.append(TEXT_SIGNATURE_DROPPED)
        if part.inline_data is not None:
            segments.append(_inline_part(part.inline_data, part_param))
        elif part.file_data is not None:
            segments.append(_file_part(part.file_data, part_param))
        elif part.text is not None:
            segments.append(TextContentPart(text=part.text))
    flush()
    if not out:
        raise invalid_field(f"{param}.parts", "A content needs at least one gateway-visible part.")
    return out


def _tool_call(
    part: _Part, pending_calls: dict[str, str], unanswered: list[tuple[str, str]], ordinal: int
) -> ToolCall:
    """One replayed function call, keeping its exact arguments and signature."""
    call = part.function_call
    assert call is not None
    call_id = call.id or f"gemini-call-{len(pending_calls)}-{ordinal}"
    raw = json.dumps(call.args or {}, separators=(",", ":"), ensure_ascii=False)
    arguments, treatment = replayed_tool_arguments(raw)
    pending_calls[call_id] = call.name
    unanswered.append((call_id, call.name))
    return ToolCall(
        call_id=call_id,
        name=call.name,
        arguments=arguments,
        raw_arguments=raw,
        arguments_treatment=treatment,
        provider_thought_signature=part.thought_signature,
    )


def _tool_message(
    response: _FunctionResponse,
    param: str,
    pending_calls: dict[str, str],
    unanswered: list[tuple[str, str]],
) -> GatewayMessage:
    """One tool message answering a replayed call by id, else by name order."""
    call_id = response.id
    if call_id is None:
        match = next((entry for entry in unanswered if entry[1] == response.name), None)
        if match is None:
            raise invalid_field(
                f"{param}.functionResponse",
                f"functionResponse '{response.name}' answers no preceding functionCall.",
            )
        call_id = match[0]
    elif pending_calls.get(call_id) != response.name or all(
        entry[0] != call_id for entry in unanswered
    ):
        raise invalid_field(
            f"{param}.functionResponse.id",
            "functionResponse.id must name a preceding, still unanswered functionCall of the "
            "same name.",
        )
    unanswered[:] = [entry for entry in unanswered if entry[0] != call_id]
    return GatewayMessage(
        role="tool", tool_call_id=call_id, content=_response_text(response.response)
    )


def _response_text(response: JsonObject | None) -> str:
    """The tool result as text: a lone string result verbatim, else compact JSON."""
    if response is None:
        return ""
    if len(response) == 1:
        (value,) = response.values()
        key = next(iter(response))
        if key in {"result", "content", "output"} and isinstance(value, str):
            return value
    return json.dumps(response, separators=(",", ":"), ensure_ascii=False)


def _inline_part(blob: _Blob, param: str) -> MessageContentPart:
    """One inline media part by its declared media type."""
    media = _AUDIO_MIME_ALIASES.get(blob.mime_type.lower(), blob.mime_type.lower())
    try:
        if media in IMAGE_MEDIA_TYPES:
            return ImageContentPart(media_type=IMAGE_MEDIA_TYPES[media], data=blob.data.strip())
        if media in VIDEO_MEDIA_TYPES:
            return VideoContentPart(media_type=VIDEO_MEDIA_TYPES[media], data=blob.data.strip())
        if media in AUDIO_MEDIA_TYPES.values():
            return AudioContentPart(
                media_type=cast("Literal['audio/wav', 'audio/mpeg']", media), data=blob.data.strip()
            )
        if media in DOCUMENT_MEDIA_TYPES:
            return DocumentContentPart(
                media_type=DOCUMENT_MEDIA_TYPES[media], data=blob.data.strip()
            )
    except ValidationError as exc:
        raise validation_protocol_error(exc) from exc
    raise invalid_field(
        f"{param}.inlineData.mimeType",
        f"Unsupported inline media type '{blob.mime_type}'. Send PNG, JPEG, GIF, WebP, "
        "MP4/MOV/WebM video, WAV/MP3 audio or a PDF.",
    )


def _file_part(file: _FileData, param: str) -> MessageContentPart:
    """One URL media part; Google ``files/`` handles cannot cross the gateway."""
    uri = file.file_uri
    if not uri.startswith(("http://", "https://")):
        raise unsupported_field(
            f"{param}.fileData.fileUri",
            message=(
                "fileData must be a public http(s) URL; Google File API handles are not served."
            ),
        )
    media = (file.mime_type or "").lower()
    if media.startswith("audio/"):
        raise unsupported_field(
            f"{param}.fileData.fileUri",
            message=(
                "Audio is served inline only (inlineData); a fileData audio URL is not fetched."
            ),
        )
    try:
        if media.startswith("video/"):
            return video_part_from_url(uri)
        if media in DOCUMENT_MEDIA_TYPES or uri.lower().split("?")[0].endswith(".pdf"):
            return DocumentContentPart(url=uri)
        return image_part_from_url(uri)
    except (ValueError, ValidationError) as exc:
        raise invalid_field(f"{param}.fileData.fileUri", str(exc)) from exc


def _tools(
    tools: tuple[_Tool, ...] | None, config: _ToolConfig | None, disclosures: list[str]
) -> tuple[tuple[GatewayToolDefinition, ...], GatewayToolChoice]:
    """Function declarations and the tool choice; server tools are refused."""
    definitions: list[GatewayToolDefinition] = []
    for index, tool in enumerate(tools or ()):
        for camel, snake in _SERVER_TOOL_FIELDS:
            if getattr(tool, snake) is not None:
                raise unsupported_field(
                    f"tools.{index}.{camel}",
                    message=(
                        f"{camel} is a Google server-executed tool the gateway does not run. "
                        "Declare function tools, or call Gemini directly for that tool."
                    ),
                )
        for declaration in tool.function_declarations or ():
            if declaration.behavior is not None:
                disclosures.append(BEHAVIOR_DROPPED)
            if declaration.response is not None or declaration.response_json_schema is not None:
                disclosures.append(RESPONSE_SCHEMA_DROPPED)
            schema = declaration.parameters_json_schema
            if schema is None:
                schema = (
                    _json_schema(declaration.parameters)
                    if declaration.parameters
                    else {"type": "object"}
                )
            definitions.append(
                GatewayToolDefinition(
                    name=declaration.name, description=declaration.description, parameters=schema
                )
            )
    choice: GatewayToolChoice = None
    calling = config.function_calling_config if config is not None else None
    if calling is not None:
        names = tuple(calling.allowed_function_names or ())
        mode = calling.mode or "AUTO"
        if mode == "VALIDATED":
            disclosures.append(VALIDATED_MODE_AS_AUTO)
            mode = "AUTO"
        if mode == "NONE":
            choice = "none"
        elif names and (mode == "ANY") and len(names) == 1:
            choice = GatewayNamedToolChoice(name=names[0])
        elif names:
            try:
                choice = GatewayAllowedToolsChoice(
                    mode="required" if mode == "ANY" else "auto", names=names
                )
            except ValidationError as exc:
                raise validation_protocol_error(exc) from exc
        else:
            choice = "required" if mode == "ANY" else "auto"
    return tuple(definitions), choice


def _json_schema(schema: JsonObject) -> JsonObject:
    """Google's ``Schema`` (upper-case ``type`` enums) as plain JSON Schema."""
    converted: JsonObject = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            converted[key] = value.lower()
        elif key in {"properties", "defs"} and isinstance(value, dict):
            converted[key] = {
                name: _json_schema(cast("JsonObject", entry)) if isinstance(entry, dict) else entry
                for name, entry in value.items()
            }
        elif key in {"items"} and isinstance(value, dict):
            converted[key] = _json_schema(cast("JsonObject", value))
        elif key in {"anyOf", "any_of"} and isinstance(value, list):
            converted["anyOf"] = [
                _json_schema(cast("JsonObject", entry)) if isinstance(entry, dict) else entry
                for entry in value
            ]
        elif key == "nullable" and value is True:
            converted["nullable"] = True
        else:
            converted[key] = value
    return converted


def _output_format(config: _GenerationConfig) -> tuple[StructuredTextFormat | None, bool]:
    """The structured-output selection: a schema, schema-free JSON, or neither."""
    if config.candidate_count is not None and config.candidate_count != 1:
        raise unsupported_field(
            "generationConfig.candidateCount",
            message="The gateway returns one candidate; omit candidateCount or send 1.",
        )
    if config.response_modalities and any(
        modality.upper() != "TEXT" for modality in config.response_modalities
    ):
        raise unsupported_field(
            "generationConfig.responseModalities",
            message="Only TEXT output is served on this surface.",
        )
    if config.speech_config is not None:
        raise unsupported_field("generationConfig.speechConfig")
    if config.image_config is not None:
        raise unsupported_field("generationConfig.imageConfig")
    if config.audio_transcription_config is not None:
        raise unsupported_field("generationConfig.audioTranscriptionConfig")
    mime = (config.response_mime_type or "").lower()
    schema = config.response_json_schema
    if schema is None and config.response_schema is not None:
        schema = _json_schema(config.response_schema)
    if schema is not None:
        if mime and mime != _JSON_MIME:
            raise invalid_field(
                "generationConfig.responseMimeType",
                "A response schema needs responseMimeType application/json.",
            )
        try:
            return StructuredTextFormat(name="response", json_schema=schema, strict=True), False
        except ValidationError as exc:
            raise validation_protocol_error(exc) from exc
    if mime == _JSON_MIME:
        return None, True
    if mime and mime not in _TEXT_MIMES:
        raise unsupported_field(
            "generationConfig.responseMimeType",
            message=(
                f"responseMimeType '{config.response_mime_type}' is not served; "
                "use application/json or text/plain."
            ),
        )
    return None, False


def _thinking(config: _ThinkingConfig | None) -> tuple[ReasoningEffort | None, int | None, bool]:
    """The effort, budget and thought-display selection of one thinking config."""
    if config is None:
        return None, None, False
    effort: ReasoningEffort | None = None
    if config.thinking_level is not None:
        level = config.thinking_level.lower()
        if level not in _REASONING_EFFORTS:
            raise invalid_field(
                "generationConfig.thinkingConfig.thinkingLevel",
                "thinkingLevel must be MINIMAL, LOW, MEDIUM or HIGH.",
            )
        effort = cast("ReasoningEffort", level)
    if config.thinking_budget is not None and effort is not None:
        raise invalid_field(
            "generationConfig.thinkingConfig",
            "Send thinkingLevel or thinkingBudget, not both.",
        )
    return effort, config.thinking_budget, bool(config.include_thoughts)


def _config_disclosures(config: _GenerationConfig) -> tuple[str, ...]:
    """Controls accepted for compatibility and dropped with disclosure."""
    dropped: list[str] = []
    if config.seed is not None:
        dropped.append(SEED_DROPPED)
    if config.response_logprobs or config.logprobs is not None:
        dropped.append(LOGPROBS_DROPPED)
    if config.media_resolution is not None:
        dropped.append(MEDIA_RESOLUTION_DROPPED)
    if config.audio_timestamp is not None:
        dropped.append("generationConfig.audioTimestamp->dropped(unsupported)")
    if config.enable_enhanced_civic_answers is not None:
        dropped.append("generationConfig.enableEnhancedCivicAnswers->dropped(unsupported)")
    return tuple(dropped)


def _stop_sequences(sequences: tuple[str, ...] | None) -> tuple[str, ...]:
    """Dedupe stop sequences in caller order and reject empty entries."""
    if not sequences:
        return ()
    deduped = tuple(dict.fromkeys(sequences))
    if any(not sequence for sequence in deduped):
        raise invalid_field(
            "generationConfig.stopSequences", "stopSequences entries must be non-empty strings."
        )
    return deduped


__all__ = [
    "OpenAIProtocolError",
    "decode_generate_content",
    "decode_generate_content_count_tokens",
]
