"""Strict private wire models for the public Chat and Responses surfaces.

Split from ``requests`` for the module line budget: these closed pydantic
profiles own field-level validation; the decoders in ``requests`` own
manifest gating, official-SDK cross-checks, and canonical translation.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import (
    MAXIMUM_DOCUMENT_BASE64_BYTES,
    MAXIMUM_DOCUMENT_NAME_CHARACTERS,
    MAXIMUM_IMAGE_BASE64_BYTES,
    MAXIMUM_VIDEO_BASE64_BYTES,
)
from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.reasoning_carrier import MAXIMUM_REASONING_CARRIER_BYTES
from exp.runtime.openai_protocol.cache_control import EphemeralCacheControl


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


_ImageDetail = Literal["auto", "low", "high"]
"""Caller fidelity hint forwarded verbatim to the wires that accept it."""

_MAXIMUM_IMAGE_URL_CHARACTERS = MAXIMUM_IMAGE_BASE64_BYTES + 128
"""Room for the largest inline image plus its ``data:`` URL preamble."""


class _ChatImageUrl(_WireModel):
    """Chat Completions image reference: a remote URL or a base64 data URL."""

    url: str = Field(min_length=1, max_length=_MAXIMUM_IMAGE_URL_CHARACTERS)
    detail: _ImageDetail | None = None


class _ChatImagePart(_WireModel):
    """One Chat Completions ``image_url`` content part."""

    type: Literal["image_url"]
    image_url: _ChatImageUrl


class _ResponsesImagePart(_WireModel):
    """One Responses ``input_image`` content part.

    Responses carries the reference as a bare string, and ``file_id`` names
    an uploaded file this gateway does not host, so only the null form of
    that field is accepted.
    """

    type: Literal["input_image"]
    image_url: str = Field(min_length=1, max_length=_MAXIMUM_IMAGE_URL_CHARACTERS)
    detail: _ImageDetail | None = None
    file_id: None = None


_MAXIMUM_VIDEO_URL_CHARACTERS = MAXIMUM_VIDEO_BASE64_BYTES + 128
"""Room for the largest inline video plus its ``data:`` URL preamble."""


class _ChatVideoUrl(_WireModel):
    """Chat Completions video reference: a remote URL or a base64 data URL."""

    url: str = Field(min_length=1, max_length=_MAXIMUM_VIDEO_URL_CHARACTERS)


class _ChatVideoPart(_WireModel):
    """One Chat Completions ``video_url`` content part.

    This is the OpenAI-compatible shape OpenRouter and Fireworks document.
    The Responses surface defines no video part, so none is accepted there.
    """

    type: Literal["video_url"]
    video_url: _ChatVideoUrl


_MAXIMUM_FILE_DATA_CHARACTERS = MAXIMUM_DOCUMENT_BASE64_BYTES + 128
"""Room for the largest inline document plus its ``data:`` URL preamble."""


class _ChatFile(_WireModel):
    """Chat Completions ``file`` payload: inline ``file_data`` with a filename.

    ``file_id`` names an uploaded file this gateway does not host, so only
    the null form of that field is accepted.
    """

    file_data: str = Field(min_length=1, max_length=_MAXIMUM_FILE_DATA_CHARACTERS)
    filename: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_NAME_CHARACTERS)
    file_id: None = None


class _ChatFilePart(_WireModel):
    """One Chat Completions ``file`` content part."""

    type: Literal["file"]
    file: _ChatFile


class _ResponsesFilePart(_WireModel):
    """One Responses ``input_file`` content part.

    Exactly one of inline ``file_data`` or a remote ``file_url`` is present;
    ``file_id`` names an uploaded file this gateway does not host, so only
    the null form of that field is accepted.
    """

    type: Literal["input_file"]
    file_data: str | None = Field(
        default=None, min_length=1, max_length=_MAXIMUM_FILE_DATA_CHARACTERS
    )
    file_url: str | None = Field(default=None, min_length=1, max_length=8_192)
    filename: str | None = Field(default=None, max_length=MAXIMUM_DOCUMENT_NAME_CHARACTERS)
    file_id: None = None

    @model_validator(mode="after")
    def _require_one_carrier(self) -> _ResponsesFilePart:
        """Require exactly one of ``file_data`` or ``file_url``."""
        if (self.file_data is None) == (self.file_url is None):
            raise ValueError("input_file needs exactly one of file_data or file_url")
        return self


_ContentPart = Annotated[
    _TextPart
    | _ChatImagePart
    | _ResponsesImagePart
    | _ChatVideoPart
    | _ChatFilePart
    | _ResponsesFilePart,
    Field(discriminator="type"),
]
"""One accepted content part on either OpenAI-style request surface."""


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
    """One OpenAI message with its content parts and assistant tool history.

    Assistant messages returned by this gateway (and by official OpenAI SDK
    clients) carry `refusal`, `annotations`, `audio`, and `function_call`
    keys even when they are empty. Callers echo those messages back verbatim
    on tool-call continuations, so the empty forms are accepted here; only a
    populated value is rejected as unsupported.
    """

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | tuple[_ContentPart, ...] | None = None
    tool_calls: tuple[_AssistantToolCall, ...] | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    refusal: None = None
    annotations: tuple[()] | None = None
    audio: None = None
    function_call: None = None
    reasoning_content: str | None = Field(
        default=None,
        max_length=MAXIMUM_REASONING_CARRIER_BYTES,
    )

    @property
    def image_capable_parts(self) -> tuple[_ContentPart, ...]:
        """Return this message's structured content parts, empty for plain text."""
        return () if self.content is None or isinstance(self.content, str) else self.content

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
        if self.role != "assistant" and self.reasoning_content is not None:
            raise ValueError("reasoning_content is valid only for assistant messages")
        if self.role != "user" and any(
            not isinstance(part, _TextPart) for part in self.image_capable_parts
        ):
            raise ValueError("image and video parts are valid only for user messages")
        call_ids = tuple(call.id for call in self.history_tool_calls)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call IDs must be unique")
        return self


class _ResponseMessage(_Message):
    """Responses message item with its optional official discriminator.

    ``id``, ``status``, and ``phase`` are all independently optional: real
    Codex traffic mints ids on its own developer and user input messages and
    echoes assistant output items with id and phase but WITHOUT status
    (captured live 2026-08-29). Non-assistant identity is accepted and
    dropped; assistant identity is retained for replay.
    """

    type: Literal["message"] = "message"
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None
    phase: Literal["commentary", "final_answer"] | None = None

    @model_validator(mode="after")
    def _require_output_identity_pair(self) -> _ResponseMessage:
        """Bind echoed lifecycle markers to a present item identity."""
        if self.status is not None and self.id is None:
            raise ValueError("Responses output message status requires an item id")
        if self.phase is not None and self.id is None:
            raise ValueError("Responses output message phase requires output identity")
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
    n: int | None = None
    """Completion-count selector, accepted only at its no-op default of 1.

    VS Code Copilot's custom-endpoint provider hardcodes ``n: 1`` on every
    Chat request (wire-captured 2026-09-02); this gateway serves exactly one
    completion per request, so 1 is accepted as already satisfied and any
    other value stays a named rejection.
    """

    @field_validator("n")
    @classmethod
    def _require_single_completion(cls, value: int | None) -> int | None:
        """Accept the completion count only as already satisfied."""
        if value is not None and value != 1:
            raise ValueError(
                "supported only at its default of 1: this gateway serves "
                "exactly one completion per request"
            )
        return value

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


class _EmbeddingsRequest(_WireModel):
    """Closed gateway embeddings request profile.

    ``input`` narrows the official OpenAI union to text only: the token-array
    forms (``list[int]`` / ``list[list[int]]``) pass official validation but
    are rejected here with a field-specific 400, since this surface serves
    visible text, not pre-tokenized ids.
    """

    model: str = Field(min_length=1, max_length=256)
    input: str | tuple[str, ...]
    dimensions: int | None = Field(default=None, gt=0)
    encoding_format: Literal["float", "base64"] | None = None
    user: str | None = Field(default=None, max_length=1024)

    @field_validator("input")
    @classmethod
    def _require_nonempty_input(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        """Reject empty text, an empty array, or empty array members."""
        if isinstance(value, str):
            if not value:
                raise ValueError("input must not be an empty string")
            return value
        if not value:
            raise ValueError("input must not be an empty array")
        if any(not text for text in value):
            raise ValueError("input array must not contain empty strings")
        return value


class _ResponseTool(_WireModel):
    """Responses API function tool declaration."""

    type: Literal["function"] = "function"
    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    parameters: JsonObject = Field(default_factory=dict)
    strict: bool | None = None


class _NativeResponseTool(BaseModel):
    """One non-function Responses tool declaration carried opaquely.

    Codex ships ``custom`` (freeform grammar), ``namespace`` (nested tool
    tree), ``web_search``, and ``tool_search`` declarations whose shapes
    exist on no other wire. Like ``_AdditionalToolsItem``, validation is
    deliberately shallow and the raw declaration forwards byte-for-byte on
    native Responses rungs only (each type captured live from Codex 0.151.0
    and accepted by the provider with a plain API key, 2026-09-01); the
    provider stays the authority on each declaration's internal shape.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(min_length=1, max_length=64)

    @field_validator("type")
    @classmethod
    def _require_non_function(cls, value: str) -> str:
        """Keep typed function declarations on the strict model."""
        if value == "function":
            raise ValueError("function tool declarations use the typed profile")
        return value


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
    model-visible reasoning from the encrypted payload alone. Codex echoes
    reasoning output items with an explicit ``content: null`` (captured
    live 2026-08-29); the provider accepts that null while rejecting a
    null ``summary``, so exactly ``content`` is nullable here.
    """

    type: Literal["reasoning"]
    id: str = Field(min_length=1, max_length=256)
    encrypted_content: str = Field(min_length=1)
    summary: tuple[_ReasoningSummaryPart, ...] = ()
    content: tuple[_ReasoningTextPart, ...] | None = None
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

    format: _ResponseFormat | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


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


class _AdditionalToolsItem(_WireModel):
    """Codex-native input item shipping tool definitions in the input stream.

    The nested tool tree (namespaces containing custom and function tools)
    has no cross-wire representation, so validation is deliberately shallow
    and the raw item forwards byte-for-byte on native Responses rungs only
    (captured live from Codex 0.151.0 and accepted by the provider with a
    plain API key, 2026-08-29).
    """

    type: Literal["additional_tools"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    role: str | None = Field(default=None, max_length=64)
    tools: tuple[JsonValue, ...] = Field(min_length=1)


class _CustomToolCall(_WireModel):
    """One freeform (custom) tool call echoed as assistant history."""

    type: Literal["custom_tool_call"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None
    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    input: str = Field(max_length=4_000_000)


class _CustomToolCallOutput(_WireModel):
    """One freeform tool result echoed as tool history."""

    type: Literal["custom_tool_call_output"]
    id: str | None = Field(default=None, min_length=1, max_length=256)
    status: _EchoedItemStatus | None = None
    call_id: str = Field(min_length=1, max_length=256)
    output: JsonValue


_ResponsesOutputItem = Annotated[
    _ResponseFunctionCall
    | _ResponseFunctionOutput
    | _ResponseReasoningItem
    | _AdditionalToolsItem
    | _CustomToolCall
    | _CustomToolCallOutput,
    Field(discriminator="type"),
]
_ResponsesInputItem = _ResponseMessage | _ResponsesOutputItem


class _PromptCacheOptions(_WireModel):
    """Responses prompt-cache selector, accepted only in its implicit mode.

    VS Code Copilot's custom-endpoint provider hardcodes
    ``{"mode": "implicit"}`` on every Responses request (wire-captured
    2026-09-02). Implicit prefix caching is exactly what served routes
    already do, so the value is accepted as already satisfied; any other
    mode names behavior this gateway does not provide and stays a closed
    rejection.
    """

    mode: Literal["implicit"]


class _ResponsesRequest(_WireModel):
    """Closed gateway Responses request profile."""

    model: str = Field(min_length=1, max_length=256)
    input: str | tuple[_ResponsesInputItem, ...]
    instructions: str | None = None
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    store: bool | None = None
    include: tuple[str, ...] | None = None
    tools: tuple[_ResponseTool | _NativeResponseTool, ...] = ()
    tool_choice: JsonValue = None
    parallel_tool_calls: bool | None = None
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning: _ResponseReasoning | None = None
    text: _ResponseText | None = None
    truncation: str | None = Field(default=None, max_length=64)
    """Context-truncation selector, accepted only at its no-op default.

    VS Code Copilot's custom-endpoint provider hardcodes
    ``truncation: "disabled"`` on every Responses request (wire-captured
    2026-09-02). This gateway never truncates context, so "disabled" is
    accepted as already satisfied; "auto" asks for dropping context the
    gateway does not implement and stays a closed rejection.
    """

    @field_validator("truncation")
    @classmethod
    def _require_no_truncation(cls, value: str | None) -> str | None:
        """Accept the truncation selector only as already satisfied."""
        if value is not None and value != "disabled":
            raise ValueError("supported only as 'disabled': this gateway never truncates context")
        return value

    prompt_cache_options: _PromptCacheOptions | None = None
    stream: bool = False
    client_metadata: JsonObject | None = None
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints; captured, never forwarded to the model.
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)
