"""Immutable gateway request, target, event, failure, and compatibility contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256
from exp.common.models.content import (
    MAXIMUM_IMAGES_PER_REQUEST,
    ImageContentPart,
    MessageContentPart,
)
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelId,
    ExactModelPoolId,
    FailoverMode,
)
from exp.common.models.model import ReasoningEffort, ToolCall

GatewayAliasName = ArtifactId
OrganizationId = ArtifactId
IdentityId = ArtifactId
VirtualKeyId = ArtifactId
GatewayAliasRevisionId = ArtifactId
ProjectRef = ArtifactId
ActivationRef = ArtifactId
RequestId = ArtifactId
AttemptId = ArtifactId


class DirectTarget(ContractModel):
    """An alias target that resolves directly to one exact-model pool."""

    kind: Literal["direct"] = "direct"
    pool_id: ExactModelPoolId


class ProjectTarget(ContractModel):
    """An alias target that selects through one immutable EXP router activation."""

    kind: Literal["project"] = "project"
    project_ref: ProjectRef
    activation_ref: ActivationRef
    catalog_sha256: Sha256


GatewayTarget = Annotated[DirectTarget | ProjectTarget, Field(discriminator="kind")]


class GatewayApiSurface(StrEnum):
    """Public endpoint family used by one canonical request."""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    MESSAGES = "messages"
    EMBEDDINGS = "embeddings"


class GatewayToolDefinition(ContractModel):
    """One caller-defined function tool with its exact JSON Schema declaration.

    The description bound is deliberately generous: both providers accept
    40k-character tool descriptions live (verified 2026-08-30), and real
    Claude Code toolsets exceeded the earlier 8k bound. The request-body
    size cap remains the effective total limit.
    """

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    parameters: JsonObject
    strict: bool = False
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching hint attached to this tool definition.

    Forwarded onto the native Anthropic tool block and dropped with
    disclosure on other wires. Like ``ToolCall.cache_control``, a cache hint
    changes cost, not semantics, so it joins neither serialization nor
    replay identity: two requests differing only here are the same request.
    """
    eager_input_streaming: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic fine-grained tool-input streaming selector.

    Accepted bare by the provider (verified live 2026-08-30; no beta
    header). Claude Code sends it conditionally. It changes how the
    provider frames tool-input deltas, so like the other Anthropic-native
    carriers it is excluded from serialization (tool digests predate it)
    and a present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    defer_loading: bool | None = Field(default=None, exclude=True)
    """Verbatim Anthropic tool-search deferred-loading selector.

    Accepted bare by the provider, which owns the cross-tool validity rules
    (verified live 2026-08-30: ``false`` is a no-op and an all-deferred
    toolset is the provider's own 400). Excluded from serialization; a
    present value changes what the model initially sees, so it joins replay
    identity through :func:`canonical_request_sha256`.
    """
    allowed_callers: tuple[str, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic programmatic-tool-calling caller allowlist.

    Accepted bare by the provider even without a companion server tool
    (verified live 2026-08-30), so the provider stays the authority on the
    combination rules. Excluded from serialization; a present value joins
    replay identity through :func:`canonical_request_sha256`.
    """
    input_examples: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)
    """Verbatim Anthropic example tool inputs.

    Accepted bare by the provider (verified live 2026-08-30). Examples add
    provider-visible prompt content, so a present value is excluded from
    serialization and joins replay identity through
    :func:`canonical_request_sha256`; reservation counts its bytes with the
    rest of the replay envelope.
    """

    def has_anthropic_tool_carriers(self) -> bool:
        """Whether any Anthropic-native tool carrier is present on this tool."""
        return (
            self.eager_input_streaming is not None
            or self.defer_loading is not None
            or self.allowed_callers is not None
            or self.input_examples is not None
        )


class StructuredTextFormat(ContractModel):
    """A strict structured-text output schema requested by the caller."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    json_schema: JsonObject
    strict: bool = True


class GatewayProviderNativeTool(ContractModel):
    """One verbatim non-function OpenAI Responses tool declaration.

    Codex ships ``custom`` (freeform grammar), ``namespace`` (nested tool
    tree), ``web_search``, and ``tool_search`` declarations whose shapes
    exist on no other wire; each is validated shallowly at decode and
    re-emitted byte-for-byte on native Responses rungs only, with the
    provider owning the declaration's internal shape (each type captured
    live from Codex 0.151.0 and accepted with a plain API key, 2026-09-01).
    ``index`` is the declaration's position in the caller's ``tools`` array
    so re-emission preserves the caller's exact interleaving with the
    converted function tools.
    """

    index: int = Field(ge=0)
    tool: JsonObject


class GatewayNamedToolChoice(ContractModel):
    """A request to require one named caller-defined function."""

    name: str = Field(min_length=1, max_length=256)


class ThinkingBlock(ContractModel):
    """One verbatim Anthropic extended-thinking block from assistant history.

    ``signature`` is an opaque cryptographic value the provider issued with
    the block; it must round-trip byte-exact or the provider rejects the
    replayed turn, so it is never normalized or re-encoded.
    """

    kind: Literal["thinking"] = "thinking"
    text: str = ""
    signature: str | None = None


class RedactedThinkingBlock(ContractModel):
    """One opaque Anthropic redacted-thinking block from assistant history."""

    kind: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class EncryptedReasoningBlock(ContractModel):
    """One opaque OpenAI Responses reasoning item replayed with the input.

    ``encrypted_content`` is the provider-issued opaque payload a stateless
    caller (``store: false``) replays so the model can resume its own prior
    reasoning; it must reach the provider byte-exact.
    """

    kind: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    id: str = Field(min_length=1, max_length=256)
    encrypted_content: str = Field(min_length=1)
    output_index: int | None = Field(default=None, ge=0, exclude=True)
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        exclude=True,
    )


class OpaqueReasoningContentBlock(ContractModel):
    """Authenticated Fireworks reasoning retained only inside the gateway.

    The provider-issued text is never accepted directly from a caller. Public
    decoding creates a sealed block, and admission replaces it with this
    plaintext form only after authenticating the carrier against the exact
    current deployment and credential authority.
    """

    kind: Literal["reasoning_content"] = "reasoning_content"
    route_sha256: Sha256
    content: str = Field(min_length=1, max_length=8 * 1024 * 1024)
    carrier_size_bytes: int = Field(default=0, ge=0, exclude=True)


class SealedReasoningContentBlock(ContractModel):
    """One bounded, still-encrypted Fireworks continuation carrier."""

    kind: Literal["sealed_reasoning_content"] = "sealed_reasoning_content"
    carrier: str = Field(min_length=1)
    deployment_hint: str = Field(min_length=1, max_length=256)


ProviderReasoningBlock = Annotated[
    ThinkingBlock
    | RedactedThinkingBlock
    | EncryptedReasoningBlock
    | OpaqueReasoningContentBlock
    | SealedReasoningContentBlock,
    Field(discriminator="kind"),
]


class GatewayMessage(ContractModel):
    """One canonical gateway message preserving developer and tool-call identity."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_is_error: bool = Field(default=False, exclude=True)
    """Whether this tool result reports a failed tool invocation.

    Only the Anthropic Messages surface can express it (``tool_result.is_error``),
    and only the Anthropic upstream dialect can emit it back, so route
    admission requires every waterfall rung to use that dialect. OpenAI-family
    wires cannot represent the flag and are rejected instead of dropping it. Like
    ``ToolCall.raw_arguments``, the field is deliberately excluded from model
    serialization so request digests, replay identity, and immutable
    artifacts are unaffected by it.
    """
    provider_reasoning: tuple[ProviderReasoningBlock, ...] = Field(default=(), exclude=True)
    """Ordered opaque provider-reasoning blocks carried on assistant turns.

    Thinking and redacted-thinking blocks exist only on the Anthropic wire;
    encrypted reasoning items exist only on the OpenAI Responses wire. Route
    admission therefore requires every waterfall rung to speak the one
    dialect that can replay them, mirroring ``tool_is_error``. Like that
    flag, the carrier is excluded from model serialization so immutable
    artifacts and carrier-free request digests are unperturbed; requests that
    do carry it join replay identity through
    :func:`canonical_request_sha256`, so a caller operation key reused with
    different reasoning is a rejected conflict, never a silent replay.
    """
    provider_item_id: str | None = Field(default=None, min_length=1, max_length=256, exclude=True)
    provider_output_index: int | None = Field(default=None, ge=0, exclude=True)
    provider_status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        exclude=True,
    )
    provider_phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        exclude=True,
    )
    """OpenAI Responses assistant-message phase retained for exact replay."""
    provider_native_item: JsonObject | None = Field(default=None, exclude=True)
    """One verbatim OpenAI Responses input item the gateway carries opaquely.

    Codex ships tool definitions and freeform tool history as native input
    items (``additional_tools``, ``custom_tool_call``,
    ``custom_tool_call_output``) whose shapes cannot be expressed on any
    other wire; the item is validated shallowly at decode and re-emitted
    byte-for-byte at its position on native Responses rungs only. A message
    carrying it carries nothing else. Excluded from serialization like the
    other carriers so item-free digests are unperturbed; a present item
    joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_anthropic_block: JsonObject | None = Field(default=None, exclude=True)
    """One verbatim Anthropic content block the gateway carries opaquely.

    Server tools return ``server_tool_use`` and ``web_search_tool_result``
    blocks, plus citation-bearing ``text`` blocks (citations exist only as
    server-tool output), whose shapes exist on no other wire; a caller
    echoing them in history gets each carried shallowly at its position and
    re-emitted byte-for-byte on native Anthropic rungs only (route admission
    mirrors ``provider_native_item``). Decode splits the assistant turn at
    block boundaries so re-emission preserves the exact block order. A
    message carrying it carries nothing else. Excluded from serialization
    like the other carriers so block-free digests are unperturbed; a present
    block joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_text_blocks: tuple[JsonObject, ...] = Field(default=(), exclude=True)
    """This message's verbatim Anthropic text blocks when one carries a
    prompt-cache marker.

    Claude Code marks system blocks and the last text block of recent user
    turns (captured live 2026-09-01); flattening them to one plain string
    strips every marker, so nothing the caller sends is ever cacheable and
    long sessions bill full input each turn (measured ~10x). When present,
    the blocks' concatenated text equals ``content`` exactly and Anthropic
    rungs re-emit them verbatim; other wires keep the flattened string and
    disclose the dropped markers. A cache hint changes cost, not semantics,
    so like the other cache carriers this joins neither serialization nor
    replay identity.
    """
    content_parts: tuple[MessageContentPart, ...] = ()
    """Ordered caller content parts for a message that carries images.

    Empty on every text-only message, so a text-only request serializes and
    digests exactly as it did before images existed. When present, the text
    parts concatenate to ``content`` byte-for-byte and at least one image
    part is included, so a route that cannot carry images is rejected at
    admission instead of silently serving the text alone. Images change what
    the model sees, so unlike the cache carriers this field is serialized and
    joins request identity.
    """
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching marker on this tool-result message.

    Claude Code marks the last block of recent user turns, which in an agent
    loop is usually a ``tool_result``; the split tool message carries the
    marker onto the re-emitted block on Anthropic rungs. Cost, not
    semantics: never in digests or replay identity.
    """

    @model_validator(mode="after")
    def _require_role_coherence(self) -> GatewayMessage:
        """Reject payload fields that do not belong to the selected message role.

        Returns:
            The validated canonical message.

        Raises:
            ValueError: Content, tool linkage, or assistant calls are incoherent.
        """
        if self.provider_native_item is not None:
            if (
                self.content is not None
                or self.tool_calls
                or self.provider_reasoning
                or self.provider_item_id is not None
                or self.tool_call_id is not None
                or self.tool_is_error
            ):
                raise ValueError("a native provider item carries the whole message")
            return self
        if self.provider_anthropic_block is not None:
            if (
                self.content is not None
                or self.tool_calls
                or self.provider_reasoning
                or self.provider_item_id is not None
                or self.tool_call_id is not None
                or self.tool_is_error
            ):
                raise ValueError("a native Anthropic block carries the whole message")
            return self
        if (
            self.content is None
            and not self.tool_calls
            and not self.provider_reasoning
            and self.provider_item_id is None
        ):
            raise ValueError("gateway messages need content, tool calls, or reasoning blocks")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are valid only for assistant messages")
        if self.role != "assistant" and self.provider_reasoning:
            raise ValueError("provider reasoning blocks are valid only for assistant messages")
        if self.role != "assistant" and (
            self.provider_item_id is not None
            or self.provider_output_index is not None
            or self.provider_status is not None
            or self.provider_phase is not None
        ):
            raise ValueError("provider output identity is valid only for assistant messages")
        if (self.provider_item_id is None) != (self.provider_output_index is None):
            raise ValueError("provider item ID and output index must be retained together")
        if self.provider_status is not None and self.provider_item_id is None:
            raise ValueError("provider output status requires retained item identity")
        if self.provider_phase is not None and self.provider_item_id is None:
            raise ValueError("provider output phase requires retained item identity")
        call_ids = tuple(call.call_id for call in self.tool_calls)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("assistant tool call IDs must be unique")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role != "tool" and self.tool_is_error:
            raise ValueError("tool_is_error is valid only for tool messages")
        if self.cache_control is not None and self.role != "tool":
            raise ValueError("message cache_control is valid only for tool messages")
        if self.provider_text_blocks:
            if self.role == "tool":
                raise ValueError("text blocks are not valid for tool messages")
            # The carrier never changes semantics: its text must flatten to
            # this message's canonical content (message runs join adjacent
            # parts directly; system blocks join with one blank line).
            texts = [str(block.get("text", "")) for block in self.provider_text_blocks]
            if (self.content or "") not in ("".join(texts), "\n\n".join(texts)):
                raise ValueError("provider text blocks must flatten to the message content")
        if self.content_parts:
            if self.role != "user":
                raise ValueError("content parts are valid only for user messages")
            if not any(part.kind == "image" for part in self.content_parts):
                raise ValueError("content parts are retained only for multimodal messages")
            texts = [part.text for part in self.content_parts if part.kind == "text"]
            if (self.content or "") != "".join(texts):
                raise ValueError("content parts must flatten to the message content")
        return self

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return this message's retained image parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "image")


class GatewayRequest(ContractModel):
    """Lossless canonical request shared by protocol and provider implementations."""

    surface: GatewayApiSurface
    messages: tuple[GatewayMessage, ...] = Field(min_length=1)
    tools: tuple[GatewayToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    structured_text: StructuredTextFormat | None = None
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens_parameter: (
        Literal["max_tokens", "max_completion_tokens", "max_output_tokens"] | None
    ) = Field(default=None, exclude=True)
    """Exact caller field normalized into ``maximum_output_tokens``."""
    stop: tuple[str, ...] = ()
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    reasoning_summary_parameters: tuple[
        Literal["reasoning.generate_summary", "reasoning.summary"], ...
    ] = Field(default=(), exclude=True)
    """Exact caller selector paths normalized into ``reasoning_summary``."""
    provider_thinking_config: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``thinking`` configuration from the Messages surface.

    The object is opaque to the gateway: it is validated against the closed
    wire profile at decode time and then forwarded byte-for-byte to the
    Anthropic upstream, overriding the catalog's adaptive default. Excluded
    from serialization like the other Anthropic-only carriers so
    config-free digests are unperturbed; a present config joins replay
    identity through :func:`canonical_request_sha256`.
    """
    context_management: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``context_management`` from the Messages surface.

    Anthropic's native context-editing configuration (Claude Code sends it
    by default). The object is deliberately validated only as an object and
    forwarded byte-for-byte with the required beta header on Anthropic
    rungs: the shape is an evolving provider beta, and a closed model here
    would recreate the reject-what-real-clients-send incident class.
    Excluded from serialization like the other Anthropic-only carriers so
    config-free digests are unperturbed; a present value joins replay
    identity through :func:`canonical_request_sha256`.
    """
    diagnostics: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``diagnostics`` from the Messages surface.

    Anthropic's diagnostics-correlation object (Claude Code sends
    ``{"previous_message_id": ...}`` conditionally). Validated only as an
    object and forwarded byte-for-byte with the required beta header on
    Anthropic rungs, dropped with disclosure elsewhere; the shape is an
    evolving provider beta, so validation stays shallow. Excluded from
    serialization; a present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    speed: str | None = Field(default=None, max_length=64, exclude=True)
    """Verbatim caller ``speed`` selector from the Messages surface.

    Anthropic's fast-mode selector (Claude Code sends ``"fast"``; accepted
    live behind its beta header, 2026-08-30). Bounded but deliberately not
    enumerated: the value set is an evolving provider surface. Forwarded
    with the required beta header on Anthropic rungs and dropped with
    disclosure elsewhere. Fast-mode output is provider-priced at a premium,
    so a present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    provider_cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller top-level ``cache_control`` from the Messages surface.

    Anthropic's automatic prompt-caching marker for the last cacheable
    block (accepted bare, verified live 2026-08-30). Forwarded byte-for-byte
    on Anthropic rungs and dropped with disclosure elsewhere. Like the
    tool-call cache hint, it changes cost, not semantics, so it deliberately
    joins NEITHER serialization NOR replay identity: two requests differing
    only here are the same request.
    """
    inference_geo: str | None = Field(default=None, max_length=64, exclude=True)
    """Verbatim caller ``inference_geo`` selector from the Messages surface.

    Anthropic's inference-region selector (accepted bare, verified live
    2026-08-30). Bounded but deliberately not enumerated: the region set is
    an evolving provider surface. Forwarded verbatim on Anthropic rungs and
    dropped with disclosure elsewhere. Where inference runs is a
    caller-visible processing commitment, so a present value joins replay
    identity through :func:`canonical_request_sha256`.
    """
    provider_beta_tokens: tuple[str, ...] = Field(default=(), exclude=True)
    """Allowlisted caller ``anthropic-beta`` tokens from the Messages surface.

    Only tokens on the decoder's explicit forward allowlist appear here (a
    caller header is operator-trust surface and is never blind-forwarded);
    the rest are dropped at decode with an ``anthropic-beta.<token>``
    disclosure. Forwarded tokens merge with the gateway's own per-field
    injections on Anthropic rungs and are dropped with disclosure
    elsewhere. Tokens change provider behavior and pricing (the 1M context
    window rides one), so present tokens join replay identity through
    :func:`canonical_request_sha256`.
    """
    response_store: bool | None = None
    """Caller ``store`` selector from the Responses surface.

    ``False`` skips gateway-side continuation retention for the produced
    response; ``True`` and absent keep the default retention behavior.
    """
    include_encrypted_reasoning: bool = False
    """Whether the caller asked for ``include=["reasoning.encrypted_content"]``."""
    reasoning_context: Literal["auto", "current_turn", "all_turns"] | None = Field(
        default=None, exclude=True
    )
    """Caller ``reasoning.context`` selector from the Responses surface.

    Controls whether the model re-renders prior turns' reasoning. Forwarded
    verbatim to native Responses rungs. Excluded from model serialization so
    context-free request digests stay byte-identical to pre-field traffic; a
    present value joins replay identity through
    :func:`canonical_request_sha256`.
    """
    text_verbosity: Literal["low", "medium", "high"] | None = None
    """Caller ``text.verbosity`` selector from the Responses surface."""
    client_metadata: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``client_metadata`` from the Responses surface.

    Opaque client telemetry (Codex sends it by default), forwarded verbatim
    on native Responses rungs and dropped with disclosure elsewhere. It is
    semantically inert, so unlike the other carriers it deliberately joins
    NEITHER serialization nor replay identity: two requests differing only
    here are the same request.
    """
    provider_output_config: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``output_config`` from the Messages surface.

    Anthropic's native output configuration (Claude Code sends
    ``{"effort": ...}`` by default). A canonical ``effort`` value also maps
    into ``reasoning_effort`` so the shared effort machinery applies; the
    raw object forwards byte-for-byte on Anthropic rungs with caller keys
    winning over engine-derived ones. Excluded from serialization like the
    other Anthropic-only carriers; a present value joins replay identity
    through :func:`canonical_request_sha256`.
    """
    provider_native_tools: tuple[GatewayProviderNativeTool, ...] = Field(default=(), exclude=True)
    """Verbatim non-function OpenAI Responses tool declarations.

    See :class:`GatewayProviderNativeTool`. Rungs that are not native
    Responses cannot serve these, so route admission rejects by name instead
    of silently dropping a capability the caller asked for. Excluded from
    serialization like the other carriers so declaration-free digests are
    unperturbed; present entries join replay identity through
    :func:`canonical_request_sha256`.
    """
    provider_server_tools: tuple[JsonObject, ...] = Field(default=(), exclude=True)
    """Verbatim Anthropic server-tool entries from the Messages ``tools`` array.

    Server tools (``web_search_20250305``-style typed entries with no
    ``input_schema``) execute at the provider; their per-type configuration
    is an evolving provider surface, so each entry is validated shallowly at
    decode and re-emitted byte-for-byte AFTER the converted custom tools on
    native Anthropic rungs only (an accepted ordering deviation from the
    caller's interleaving). Other rungs cannot execute them, so route
    admission rejects by name instead of silently dropping a capability the
    caller asked for. Excluded from serialization like the other carriers so
    server-tool-free digests are unperturbed; present entries join replay
    identity through :func:`canonical_request_sha256`.
    """
    stream: bool = False
    include_usage: bool = False
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints from the OpenAI request. Captured for
    # gateway-side attribution and never forwarded to the model. `safety_identifier`
    # is the current stable end-user identifier; `user` its deprecated predecessor;
    # `prompt_cache_key` a same-prefix cache-routing hint (never an identity).
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)
    ignored_parameters: tuple[str, ...] = Field(default=(), exclude=True)
    """Disclosed compatibility decisions applied to this request.

    A plain field path names a control accepted but intentionally omitted
    from provider dispatch; a ``path->effective`` entry (for example
    ``reasoning_effort->high`` or ``tools.strict->false``) names a disclosed
    coercion the route applied when no deployment preserved the caller's
    exact value. Coercions are never silent: each entry here also logs and
    counts in the admission metrics.
    """
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=512)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=512)

    @property
    def attribution_label(self) -> str | None:
        """The end-user attribution label for this request, per the OpenAI spec.

        Prefers the current `safety_identifier`; falls back to the deprecated
        `user` field for older clients. `prompt_cache_key` is deliberately never
        used here — it is a cache-routing hint, not an end-user identity.

        Returns:
            The attribution label, or None when the caller sent neither field.
        """
        return self.safety_identifier or self.user

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return every image this request carries, in message and part order."""
        return tuple(image for message in self.messages for image in message.images)

    @field_validator("stop")
    @classmethod
    def _require_unique_stop_sequences(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or repeated stop sequences while preserving caller order.

        Args:
            value: Requested stop strings.

        Returns:
            The unchanged validated stop sequence.

        Raises:
            ValueError: A stop is empty or repeated.
        """
        if any(not item for item in value):
            raise ValueError("stop sequences must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("stop sequences must not repeat")
        return value

    @model_validator(mode="after")
    def _require_coherent_tools(self) -> GatewayRequest:
        """Require named and required tool choices to reference available tools.

        Returns:
            The validated canonical request.

        Raises:
            ValueError: Tool definitions or tool choice are incoherent.
        """
        names = tuple(tool.name for tool in self.tools)
        if len(set(names)) != len(names):
            raise ValueError("gateway tool names must not repeat")
        # Server tools are addressable by tool_choice too; the provider owns
        # cross-set name rules for the verbatim entries.
        server_names = tuple(
            str(entry["name"]) for entry in self.provider_server_tools if "name" in entry
        )
        if (
            isinstance(self.tool_choice, GatewayNamedToolChoice)
            and self.tool_choice.name not in names
            and self.tool_choice.name not in server_names
        ):
            raise ValueError("named gateway tool choice must name a request tool")
        if (
            self.tool_choice == "required"
            and not self.tools
            and not self.provider_server_tools
            and not self.provider_native_tools
        ):
            raise ValueError("required gateway tool choice needs at least one tool")
        if self.include_usage and not self.stream:
            raise ValueError("include_usage is valid only for streaming requests")
        if len(self.images) > MAXIMUM_IMAGES_PER_REQUEST:
            raise ValueError(f"a request carries at most {MAXIMUM_IMAGES_PER_REQUEST} images")
        if self.reasoning_summary is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_summary is valid only for Responses requests")
        if self.response_store is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("response_store is valid only for Responses requests")
        if self.include_encrypted_reasoning and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("include_encrypted_reasoning is valid only for Responses requests")
        if self.reasoning_context is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_context is valid only for Responses requests")
        if self.provider_thinking_config is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_thinking_config is valid only for Messages requests")
        if self.provider_output_config is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_output_config is valid only for Messages requests")
        if self.text_verbosity is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("text_verbosity is valid only for Responses requests")
        if self.client_metadata is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("client_metadata is valid only for Responses requests")
        if self.context_management is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("context_management is valid only for Messages requests")
        if self.diagnostics is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("diagnostics is valid only for Messages requests")
        if self.speed is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("speed is valid only for Messages requests")
        if self.provider_cache_control is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_cache_control is valid only for Messages requests")
        if self.inference_geo is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("inference_geo is valid only for Messages requests")
        if (
            any(tool.has_anthropic_tool_carriers() for tool in self.tools)
            and self.surface != GatewayApiSurface.MESSAGES
        ):
            raise ValueError("Anthropic tool carriers are valid only for Messages requests")
        if self.provider_beta_tokens and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_beta_tokens are valid only for Messages requests")
        if self.provider_server_tools and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_server_tools are valid only for Messages requests")
        if self.provider_native_tools and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("provider_native_tools are valid only for Responses requests")
        if self.provider_native_tools:
            # Positions must tile one tools array with the converted function
            # tools exactly, so native re-emission is total by construction.
            positions = tuple(entry.index for entry in self.provider_native_tools)
            declaration_count = len(self.tools) + len(positions)
            if len(set(positions)) != len(positions) or any(
                position >= declaration_count for position in positions
            ):
                raise ValueError(
                    "provider_native_tools positions must be distinct indexes "
                    "into the caller's tools array"
                )
        if self.maximum_output_tokens_parameter is not None and self.maximum_output_tokens is None:
            raise ValueError("maximum output parameter requires a maximum output value")
        if self.reasoning_summary_parameters and self.reasoning_summary is None:
            raise ValueError("reasoning summary parameter paths require a summary selector")
        if len(set(self.reasoning_summary_parameters)) != len(self.reasoning_summary_parameters):
            raise ValueError("reasoning summary parameter paths must not repeat")
        return self


class GatewayUsage(ContractModel):
    """Normalized token counts and invoked tool names from one provider attempt.

    Cached-input and reasoning counts are subsets of the total input and output counts when
    present. They identify differently priced portions of those totals and must not be added a
    second time by callers.

    A terminal event may carry only ``tool_names`` when the provider omits token usage. In that
    case both token totals remain unknown instead of being represented as zero.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    """Cache-write tokens inside the input total (Anthropic-only today),
    present only when the provider reported a nonzero count; billing keeps
    using the folded input total."""
    reasoning_tokens: int | None = Field(default=None, ge=0)
    tool_names: tuple[str, ...] = ()
    """Invoked tool names in first-use order, names only and never arguments."""

    @model_validator(mode="after")
    def _require_complete_tokens_or_tool_names(self) -> GatewayUsage:
        """Require complete token totals unless this is tool-only terminal metadata."""
        totals = (self.input_tokens, self.output_tokens)
        if (totals[0] is None) != (totals[1] is None):
            raise ValueError("input and output token counts must be reported together")
        if totals[0] is None:
            if (
                self.cached_input_tokens is not None
                or self.cache_creation_input_tokens is not None
                or self.reasoning_tokens is not None
            ):
                raise ValueError("token detail counts require input and output totals")
            if not self.tool_names:
                raise ValueError("usage requires token totals or invoked tool names")
        return self

    @property
    def has_token_counts(self) -> bool:
        """Return whether both provider token totals are known."""
        return self.input_tokens is not None and self.output_tokens is not None


class GatewayEventKind(StrEnum):
    """Provider-neutral semantic and terminal stream event categories."""

    TEXT_DELTA = "text_delta"
    REFUSAL_DELTA = "refusal_delta"
    REASONING_SUMMARY_DELTA = "reasoning_summary_delta"
    THINKING_DELTA = "thinking_delta"
    THINKING_SIGNATURE = "thinking_signature"
    REDACTED_THINKING = "redacted_thinking"
    ENCRYPTED_REASONING = "encrypted_reasoning"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_ARGUMENTS_DELTA = "tool_arguments_delta"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    USAGE = "usage"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class GatewayEvent(ContractModel):
    """One ordered provider-neutral stream event, including raw tool fragments."""

    kind: GatewayEventKind
    sequence_number: int = Field(ge=0)
    text_delta: str | None = None
    reasoning_summary_output_index: int | None = Field(default=None, ge=0)
    reasoning_summary_index: int | None = Field(default=None, ge=0)
    reasoning_item_id: str | None = Field(default=None, min_length=1, max_length=256)
    reasoning_block_index: int | None = Field(default=None, ge=0)
    """Provider content-block (or output-item) index grouping reasoning events."""
    thinking_signature: str | None = None
    redacted_thinking_data: str | None = None
    encrypted_content: str | None = None
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    tool_name: str | None = Field(default=None, min_length=1, max_length=256)
    raw_arguments_delta: str | None = None
    tool_call: ToolCall | None = None
    usage: GatewayUsage | None = None
    failure: GatewayFailure | None = None

    @model_validator(mode="after")
    def _require_event_payload(self) -> GatewayEvent:
        """Require each event kind to carry its one relevant payload.

        Returns:
            The validated stream event.

        Raises:
            ValueError: The selected event kind lacks its required payload.
        """
        if self.kind in {GatewayEventKind.TEXT_DELTA, GatewayEventKind.REFUSAL_DELTA}:
            if self.text_delta is None:
                raise ValueError("text and refusal deltas require text_delta")
        elif self.kind == GatewayEventKind.REASONING_SUMMARY_DELTA:
            if (
                self.text_delta is None
                or self.reasoning_summary_output_index is None
                or self.reasoning_summary_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("reasoning summary deltas require item, output, summary, and text")
        elif self.kind == GatewayEventKind.THINKING_DELTA:
            if self.text_delta is None or self.reasoning_block_index is None:
                raise ValueError("thinking deltas require block index and text")
        elif self.kind == GatewayEventKind.THINKING_SIGNATURE:
            if self.thinking_signature is None or self.reasoning_block_index is None:
                raise ValueError("thinking signatures require block index and signature")
        elif self.kind == GatewayEventKind.REDACTED_THINKING:
            if self.redacted_thinking_data is None or self.reasoning_block_index is None:
                raise ValueError("redacted thinking requires block index and data")
        elif self.kind == GatewayEventKind.ENCRYPTED_REASONING:
            if (
                self.encrypted_content is None
                or self.reasoning_block_index is None
                or self.reasoning_item_id is None
            ):
                raise ValueError("encrypted reasoning requires item, block index, and content")
        elif self.kind == GatewayEventKind.TOOL_CALL_STARTED:
            if self.tool_call_index is None or self.tool_call_id is None or self.tool_name is None:
                raise ValueError("tool-call start requires index, ID, and name")
        elif self.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            if self.tool_call_index is None or self.raw_arguments_delta is None:
                raise ValueError("tool argument delta requires index and raw fragment")
        elif self.kind == GatewayEventKind.TOOL_CALL_COMPLETED and self.tool_call is None:
            raise ValueError("tool-call completion requires the complete tool call")
        elif self.kind == GatewayEventKind.USAGE:
            if self.usage is None or not self.usage.has_token_counts:
                raise ValueError("usage event requires complete normalized token usage")
        elif self.kind == GatewayEventKind.FAILED and self.failure is None:
            raise ValueError("failed event requires a normalized failure")
        return self


class GatewayFailureClass(StrEnum):
    """Stable failure classes shared by provider execution and the public protocol."""

    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    QUOTA_EXCEEDED = "quota_exceeded"
    THROTTLED = "throttled"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    PROVIDER_AUTHENTICATION = "provider_authentication"
    PROVIDER_NOT_FOUND = "provider_not_found"
    REFUSAL = "refusal"
    MALFORMED_RESPONSE = "malformed_response"
    PROVIDER_INTERNAL = "provider_internal"
    CANCELLED = "cancelled"
    GUARDRAIL = "guardrail"
    INTERNAL = "internal"
    # A transient control-plane condition (a rolling deploy building the
    # authorized catalog revision) that the caller should simply retry. Unlike
    # INTERNAL it is not a bug signal and does not page; unlike a provider class
    # it never opens a deployment circuit.
    UNAVAILABLE = "unavailable"


class GatewayFailure(ContractModel):
    """Sanitized failure with retry and failover eligibility already classified."""

    failure_class: GatewayFailureClass
    safe_message: str = Field(min_length=1, max_length=2_048)
    retryable_same_deployment: bool = False
    failover_eligible: bool = False
    safe_details: JsonObject = Field(default_factory=dict)
    rejected_parameter: str | None = Field(default=None, min_length=1, max_length=128)
    """Validated provider-named parameter path; never provider prose."""
    provider_detail: str | None = Field(default=None, min_length=1, max_length=240)
    """Provider explanation of a client error, relayed only for that class."""


class ProjectSelection(ContractModel):
    """One frozen learned-router selection resolved before provider execution."""

    exact_model_id: ExactModelId
    selected_alias: ArtifactId
    activation_ref: ActivationRef
    fallback_reason: str | None = Field(default=None, max_length=512)


class AuthorizationSnapshot(ContractModel):
    """Immutable authority and alias target frozen before learned model selection."""

    request_id: RequestId
    organization_id: OrganizationId
    identity_id: IdentityId
    virtual_key_id: VirtualKeyId
    alias: GatewayAliasName
    alias_revision_id: GatewayAliasRevisionId
    target: GatewayTarget
    surface: GatewayApiSurface
    catalog_sha256: Sha256
    canonical_request_sha256: Sha256
    caller_operation_sha256: Sha256 | None = None
    refusal_failover: bool = False
    deadline_monotonic: float = Field(gt=0)
    app_referer: str | None = Field(default=None, max_length=2_048)
    """Caller-supplied ``HTTP-Referer`` app identity, content-free and never a credential."""
    app_title: str | None = Field(default=None, max_length=256)
    """Caller-supplied ``X-Title`` app label used only for content-free app attribution."""
    attribution_label: str | None = Field(default=None, max_length=1024)
    """End-user attribution from the OpenAI ``safety_identifier`` (or deprecated
    ``user``) request field: content-free and never a credential."""


class ExecutionSnapshot(ContractModel):
    """Route-bound request plan created only after exact-model selection."""

    authorization: AuthorizationSnapshot
    exact_model_id: ExactModelId
    pool_id: ExactModelPoolId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    # The pool's per-model failover policy, carried onto the route so the
    # per-attempt retry/failover decision can honor it. Defaults to the
    # historical maximize_availability.
    failover_mode: FailoverMode = "maximize_availability"
