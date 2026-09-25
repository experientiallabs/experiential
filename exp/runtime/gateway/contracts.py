"""Immutable gateway request, target, event, failure, and compatibility contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictInt, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256
from exp.common.models.content import (
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    MediaHandle,
    MessageContentPart,
    VideoContentPart,
    require_attachment_ceilings,
)
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelId,
    ExactModelPoolId,
    FailoverMode,
)
from exp.common.models.gateway_chains import ModelExecutionStage, ModelTraversalEvent
from exp.common.models.model import MAXIMUM_TOOL_CALL_ID_CHARACTERS, ReasoningEffort, ToolCall
from exp.runtime.gateway.model_chain_authority import ModelChainAuthority
from exp.runtime.gateway.reasoning_blocks import (
    EncryptedReasoningBlock as EncryptedReasoningBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ExposedReasoningContentBlock as ExposedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    OpaqueReasoningContentBlock as OpaqueReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ProviderReasoningBlock as ProviderReasoningBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    RedactedThinkingBlock as RedactedThinkingBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    SealedReasoningContentBlock as SealedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_blocks import (
    ThinkingBlock as ThinkingBlock,
)
from exp.runtime.gateway.request_policy import GatewayRequestPolicy, RequestedRouteId
from exp.runtime.gateway.stream_contracts import (
    ChoiceLogprobs as ChoiceLogprobs,
)
from exp.runtime.gateway.stream_contracts import (
    ChoiceLogprobsDelta as ChoiceLogprobsDelta,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayEvent as GatewayEvent,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayEventKind as GatewayEventKind,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayFailure as GatewayFailure,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayFailureClass as GatewayFailureClass,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayRefusalReason as GatewayRefusalReason,
)
from exp.runtime.gateway.stream_contracts import (
    GatewayUsage as GatewayUsage,
)
from exp.runtime.gateway.stream_contracts import (
    LogprobCandidate as LogprobCandidate,
)
from exp.runtime.gateway.stream_contracts import (
    TokenLogprob as TokenLogprob,
)
from exp.runtime.gateway.tool_contracts import (
    GatewayNamedToolChoice as GatewayNamedToolChoice,
)
from exp.runtime.gateway.tool_contracts import (
    GatewayProviderNativeTool as GatewayProviderNativeTool,
)
from exp.runtime.gateway.tool_contracts import (
    GatewayToolDefinition as GatewayToolDefinition,
)
from exp.runtime.gateway.tool_search.contracts import GatewayToolSearch, gateway_tool_search_name
from exp.runtime.gateway.web_search.contracts import GatewayWebSearch

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
    IMAGES = "images"
    DECISIONS = "decisions"


class StructuredTextFormat(ContractModel):
    """A strict structured-text output schema requested by the caller."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=65_536)
    json_schema: JsonObject
    strict: bool = True


class GatewayMessage(ContractModel):
    """One canonical gateway message preserving developer and tool-call identity.

    Attributes:
        capture_only_reasoning: Caller-visible copies accompanying sealed replay.
            Empty by default; excluded from serialization, provider dispatch and
            replay authority. Only content capture consumes this evidence.
    """

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = Field(
        default=None, min_length=1, max_length=MAXIMUM_TOOL_CALL_ID_CHARACTERS
    )
    tool_calls: tuple[ToolCall, ...] = ()
    capture_only_reasoning: tuple[ExposedReasoningContentBlock, ...] = Field(
        default=(), exclude=True
    )
    tool_is_error: bool = Field(default=False, exclude=True)
    """Whether this tool result reports a failed invocation; retained outside serialization."""
    provider_specific_fields: JsonObject | None = Field(default=None, exclude=True)
    """LiteLLM's echoed per-message ``provider_specific_fields``: accepted so verbatim
    replays keep working, dropped on every wire with a disclosure, excluded from
    serialization like the other carried-but-never-forwarded message fields."""
    provider_reasoning: tuple[ProviderReasoningBlock, ...] = Field(default=(), exclude=True)
    """Ordered opaque provider-reasoning blocks carried on assistant turns.

    Thinking blocks exist only on the Anthropic wire and encrypted reasoning
    items only on OpenAI Responses, so route admission requires every rung to
    speak the one dialect that can replay them (mirroring ``tool_is_error``).
    Excluded from serialization; a present carrier joins replay identity, so a
    reused operation key with different reasoning is a conflict, never a replay.
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
    provider_tool_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        exclude=True,
    )
    provider_tool_namespace: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        exclude=True,
    )
    """Tool-result attribution replayed on a Responses ``function_call_output``.

    Codex serializes an optional ``name`` and ``namespace`` on the outputs of
    namespaced tool calls; both re-emit verbatim on the rebuilt item
    (mirroring ``ToolCall.provider_namespace`` on the call side) and are
    excluded from serialization like the other replay carriers, joining
    replay identity explicitly through :func:`canonical_request_sha256`.
    ``provider_tool_name`` also carries the Chat surface's legacy
    ``role: "tool"`` ``name`` (the old ``role: "function"`` attribution many
    agent frameworks still send; the provider serves it, probed live
    2026-09-05), re-emitted on both OpenAI wires and dropped with disclosure
    elsewhere.
    """
    provider_tool_caller: JsonObject | None = Field(default=None, exclude=True)
    """Opaque SDK 3.0 ``caller`` attribution on a ``function_call_output``.

    Programmatic tool calling attributes the result to the program that
    invoked the call; the object's internal shape is an evolving provider
    surface, so it is validated only as an object and re-emitted verbatim on
    the rebuilt item (mirroring ``ToolCall.provider_caller`` on the call
    side). Excluded from serialization like the other replay carriers,
    joining replay identity explicitly through
    :func:`canonical_request_sha256`.
    """
    provider_native_item: JsonObject | None = Field(default=None, exclude=True)
    """One verbatim OpenAI Responses input item the gateway carries opaquely.

    Codex ships tool definitions and freeform tool history as native input
    items (``additional_tools``, ``custom_tool_call``,
    ``custom_tool_call_output``), and hosted-tool turns echo their
    provider-executed items (``web_search_call``, ``mcp_call``,
    ``code_interpreter_call``, their outputs, ...); every such shape exists
    on no other wire, so the item is validated shallowly at decode and
    re-emitted byte-for-byte at its position on native Responses rungs only.
    A message carrying it carries nothing else. Excluded from serialization
    like the other carriers so item-free digests are unperturbed; a present
    item joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_anthropic_blocks: tuple[JsonObject, ...] | None = Field(default=None, exclude=True)
    """The caller's assistant content blocks in their ORIGINAL order, when a
    thinking block is among them.

    The flattened fields (``content``, ``tool_calls``, ``provider_reasoning``)
    lose the order of blocks within one assistant turn; the Anthropic wire
    re-emits them as thinking, then text, then tool_use. With interleaved
    thinking a turn is [thinking, tool_use, thinking, text, tool_use ...], and
    Anthropic verifies the LATEST assistant message byte-for-byte against the
    signatures it issued: a reordered turn is refused as "thinking or
    redacted_thinking blocks in the latest assistant message cannot be
    modified" (134 requests / 48h on one Messages-surface client,
    2026-09-07). The Anthropic wire replays these verbatim when they are
    present and the flattened reasoning was not narrowed; every other wire
    keeps reading the flattened fields. Excluded from serialization like the
    other carriers.
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
    """Ordered caller content parts for a message that carries attachments.

    Empty on every text-only message, so a text-only request serializes and
    digests exactly as before attachments existed. When present, the text parts
    concatenate to ``content`` byte-for-byte and at least one attachment (image,
    video, audio, or document) is included, so a route that cannot carry it is
    rejected at admission instead of silently serving the text alone. Attachments
    change what the model sees, so the field is serialized and joins request identity.
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
        if self.role != "assistant" and (self.provider_reasoning or self.capture_only_reasoning):
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
        if self.role != "tool" and (
            self.provider_tool_name is not None
            or self.provider_tool_namespace is not None
            or self.provider_tool_caller is not None
        ):
            raise ValueError("tool-result attribution is valid only for tool messages")
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
            # Retain tool screenshots and generated assistant images as caller-owned history.
            if self.role not in ("user", "tool", "assistant"):
                raise ValueError(
                    "content parts are valid only for user, tool, and assistant messages"
                )
            if all(part.kind == "text" for part in self.content_parts):
                raise ValueError("content parts are retained only for multimodal messages")
            if self.role in ("tool", "assistant") and any(
                part.kind not in ("text", "image") for part in self.content_parts
            ):
                raise ValueError(f"{self.role} messages carry only text and image parts")
            texts = [part.text for part in self.content_parts if part.kind == "text"]
            if (self.content or "") != "".join(texts):
                raise ValueError("content parts must flatten to the message content")
        return self

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return this message's retained image parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "image")

    @property
    def videos(self) -> tuple[VideoContentPart, ...]:
        """Return this message's retained video parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "video")

    @property
    def documents(self) -> tuple[DocumentContentPart, ...]:
        """Return this message's retained document parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "document")

    def folded_tool_error_content(self) -> str:
        """Return this tool result's text with ``tool_is_error`` folded in.

        Only the Anthropic wire has a native ``tool_result.is_error`` field;
        every other wire re-states the flag in the one channel it has (the
        result text, prefixed with :data:`TOOL_ERROR_TEXT_PREFIX`) so the
        model still learns the invocation failed. The fold derives from the
        canonical flag on each request, never from previously folded text, so
        a replayed history can never accumulate prefixes.
        """
        content = self.content or ""
        if self.tool_is_error:
            return f"{TOOL_ERROR_TEXT_PREFIX}{content}"
        return content


TOOL_ERROR_TEXT_PREFIX = "[tool error] "
"""Prefix folding Anthropic's ``tool_result.is_error`` into plain result text
on wires without a native error flag (see
:meth:`GatewayMessage.folded_tool_error_content`)."""


class GatewayRequest(ContractModel):
    """Lossless canonical request shared by protocol and provider implementations.

    Attributes:
        include_output_text_logprobs: Responses probability selector (default false).
        gateway: Optional request routing/retry policy; excluded from provider serialization.
    """

    surface: GatewayApiSurface
    messages: tuple[GatewayMessage, ...] = Field(min_length=1)
    tools: tuple[GatewayToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    structured_text: StructuredTextFormat | None = None
    json_object_output: bool = Field(default=False, exclude=True)
    # Native JSON mode or disclosed instruction; enabled mode joins replay identity.
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens_parameter: (
        Literal["max_tokens", "max_completion_tokens", "max_output_tokens"] | None
    ) = Field(default=None, exclude=True)
    """Exact caller field normalized into ``maximum_output_tokens``."""
    stop: tuple[str, ...] = ()
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    logprobs: StrictBool | None = None
    top_logprobs: StrictInt | None = Field(default=None, ge=0, le=20)
    include_output_text_logprobs: bool = Field(default=False, exclude=True)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_effort_parameter: (
        Literal["reasoning_effort", "reasoning.effort", "output_config.effort"] | None
    ) = Field(default=None, exclude=True)
    """Exact caller field normalized into ``reasoning_effort``, when a surface
    offers more than one (the Messages surface takes Anthropic's
    ``output_config.effort`` and the OpenRouter ``reasoning.effort`` extension);
    an effort the route cannot serve is rejected by that name so the caller's
    own recovery finds the field it sent. ``None`` means the surface default
    (see :attr:`caller_effort_parameter`)."""
    # Level-less enable-thinking; the route seam resolves the concrete effort.
    thinking_budget: int | None = Field(default=None, ge=-1, strict=True, exclude=True)
    """Numeric Chat thinking control; provider validation owns zero and -1 semantics."""
    thinking_default_enable: bool = False
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    reasoning_summary_parameters: tuple[
        Literal["reasoning.generate_summary", "reasoning.summary"], ...
    ] = Field(default=(), exclude=True)
    """Exact caller selector paths normalized into ``reasoning_summary``."""
    provider_thinking_config: JsonObject | None = Field(default=None, exclude=True)
    """Verbatim caller ``thinking`` configuration from Messages or budgeted Chat.

    Validated at decode and route admission. Numeric values survive translation
    to qualified native wires; other config fields require verbatim support.
    Excluded from serialization; joins :func:`canonical_request_sha256`.
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
    """Caller output-length hint: Responses ``text.verbosity`` or Chat ``verbosity``.

    One canonical carrier for both spellings; the surface decides which public
    path a drop disclosure names.
    """
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

    Anthropic's native output configuration (Claude Code sends ``{"effort":
    ...}``); a canonical ``effort`` also maps into ``reasoning_effort``. The
    raw object forwards byte-for-byte on Anthropic rungs, caller keys winning.
    Excluded from serialization like the other Anthropic-only carriers; a
    present value joins replay identity through :func:`canonical_request_sha256`.
    """
    provider_native_tools: tuple[GatewayProviderNativeTool, ...] = Field(default=(), exclude=True)
    """Verbatim non-function OpenAI Responses tool declarations (see
    :class:`GatewayProviderNativeTool`); excluded, join replay identity when present."""
    native_tool_translation: dict[str, tuple[str, str | None, bool]] | None = Field(
        default=None, exclude=True
    )
    """Provider-only reverse map for translated Codex native tools; not in replay identity."""
    provider_server_tools: tuple[JsonObject, ...] = Field(default=(), exclude=True)
    """Verbatim Anthropic server-tool entries from the Messages ``tools`` array.

    Typed entries with no ``input_schema`` execute at the provider; validated
    shallowly at decode, re-emitted byte-for-byte AFTER the converted custom tools
    on native Anthropic rungs only (other rungs reject by name). Excluded from
    serialization; present entries join replay identity (``canonical_request_sha256``).
    """
    web_search: GatewayWebSearch | None = Field(default=None, exclude=True)
    """The caller's normalized pre-answer web-search request (any spelling); excluded
    from serialization, joins replay identity when present. See ``web_search.plan``."""
    tool_search: GatewayToolSearch | None = Field(default=None, exclude=True)
    """The caller's normalized tool-search declaration (any spelling); excluded from
    serialization, joins replay identity when present. See ``tool_search.plan``."""
    # `provider: {"zdr": true}`: the caller demanded ZDR routing. Tightening
    # only: the host applies its require_zdr posture filter to this request and
    # refuses with the same 403 when no rung qualifies. Part of identity.
    zdr_requested: bool = False
    # Verbatim caller `provider` object: forwarded to OpenRouter rungs (tightened
    # when the rung is constrained), dropped on every other wire.
    provider_preferences: JsonObject | None = Field(default=None, exclude=True)
    gateway: GatewayRequestPolicy | None = Field(default=None, exclude=True)
    stream: bool = False
    include_usage: bool = False
    previous_response_id: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: JsonObject = Field(default_factory=dict)
    # End-user attribution / cache hints from the OpenAI request. Captured for
    # gateway-side attribution and never forwarded verbatim. `safety_identifier`
    # is the current stable end-user identifier; `user` its deprecated predecessor;
    # `prompt_cache_key` a same-prefix cache-routing hint (never an identity) that
    # reaches the provider only as the namespaced `provider_prompt_cache_key`.
    safety_identifier: str | None = Field(default=None, max_length=1024)
    user: str | None = Field(default=None, max_length=1024)
    prompt_cache_key: str | None = Field(default=None, max_length=1024)
    provider_prompt_cache_key: str | None = Field(default=None, max_length=128, exclude=True)
    """Tenant-namespaced cache-affinity key dispatched to rungs that route by it.

    Derived at admission (``prompt_cache_affinity.provider_prompt_cache_key``)
    from the caller's ``prompt_cache_key`` or, absent one, from the
    conversation stem (the leading system/developer messages, else the first
    user turn), so every request sharing a cacheable prefix lands on the
    provider cache node that holds it while the caller's raw key never leaves
    the gateway. Excluded from serialization: it is routing state, never
    request identity.
    """
    service_tier: str | None = Field(default=None, max_length=64, exclude=True)
    """Caller provider processing tier, forwarded only on BYOK OpenAI-family
    rungs (routing and billing rules live at streaming_requests and
    capability_policy). Excluded from serialization so tier-free digests are
    unperturbed; a present value joins replay identity through
    :func:`canonical_request_sha256`: the same body at a different tier is a
    different provider price and schedule."""
    ignored_parameters: tuple[str, ...] = Field(default=(), exclude=True)
    """The caller sent ``parallel_tool_calls: false`` and at least one admitted
    rung has no such wire control: the data plane serializes those rungs' tool
    calls to one per turn instead. Disclosed through ``ignored_parameters``."""
    serialize_tool_calls: bool = Field(default=False, exclude=True)
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

    @property
    def videos(self) -> tuple[VideoContentPart, ...]:
        """Return every video this request carries, in message and part order."""
        return tuple(video for message in self.messages for video in message.videos)

    @property
    def audios(self) -> tuple[AudioContentPart, ...]:
        """Return every audio clip this request carries, in message and part order."""
        parts = (part for message in self.messages for part in message.content_parts)
        return tuple(part for part in parts if part.kind == "audio")

    @property
    def documents(self) -> tuple[DocumentContentPart, ...]:
        """Return every document this request carries, in message and part order."""
        return tuple(part for message in self.messages for part in message.documents)

    @property
    def media_handles(self) -> tuple[MediaHandle, ...]:
        """Return every provider media handle this request carries, in caller order."""
        return tuple(
            part.handle
            for message in self.messages
            for part in message.content_parts
            if isinstance(part, (ImageContentPart, VideoContentPart, DocumentContentPart))
            and part.handle is not None
        )

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

    @property
    def caller_effort_parameter(
        self,
    ) -> Literal["reasoning_effort", "reasoning.effort", "output_config.effort"]:
        """The public field an unservable effort is rejected under.

        The recorded caller field when the decoder knows it; otherwise the
        surface's one effort field. The name matters: Claude Code carries its
        effort as Messages ``output_config.effort`` and auto-recovers (drops
        the field and retries) only when the 400 names that channel, so naming
        a translated internal field wedges every turn instead.
        """
        if self.reasoning_effort_parameter is not None:
            return self.reasoning_effort_parameter
        match self.surface:
            case GatewayApiSurface.RESPONSES:
                return "reasoning.effort"
            case GatewayApiSurface.MESSAGES:
                return "output_config.effort"
            case _:
                return "reasoning_effort"

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
        if self.tool_search is not None and any(tool.defer_loading for tool in self.tools):
            server_names = (*server_names, gateway_tool_search_name(names))
        if (
            isinstance(self.tool_choice, GatewayNamedToolChoice)
            and self.tool_choice.name not in names
            and self.tool_choice.name not in server_names
        ):
            raise ValueError("named gateway tool choice must name a request tool")
        has_tools = bool(self.tools or self.provider_server_tools or self.provider_native_tools)
        if self.tool_choice == "required" and not has_tools:
            raise ValueError("required gateway tool choice needs at least one tool")
        if self.include_usage and not self.stream:
            raise ValueError("include_usage is valid only for streaming requests")
        if self.json_object_output and self.structured_text is not None:
            raise ValueError("json_object_output and structured_text are mutually exclusive")
        if self.json_object_output and self.surface != GatewayApiSurface.CHAT_COMPLETIONS:
            raise ValueError("json_object_output is valid only for Chat Completions requests")
        parts = (part for message in self.messages for part in message.content_parts)
        require_attachment_ceilings(parts)
        if len({handle.provider for handle in self.media_handles}) > 1:
            raise ValueError(
                "media handles in one request must all name the same provider; "
                "no single route can resolve handles from two providers"
            )
        if self.reasoning_summary is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_summary is valid only for Responses requests")
        if self.response_store is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("response_store is valid only for Responses requests")
        if self.include_output_text_logprobs and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("include_output_text_logprobs is valid only for Responses requests")
        if self.include_encrypted_reasoning and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("include_encrypted_reasoning is valid only for Responses requests")
        if self.reasoning_context is not None and self.surface != GatewayApiSurface.RESPONSES:
            raise ValueError("reasoning_context is valid only for Responses requests")
        if self.provider_thinking_config is not None and self.surface not in {
            GatewayApiSurface.MESSAGES,
            GatewayApiSurface.CHAT_COMPLETIONS,
        }:
            raise ValueError("provider_thinking_config is valid only for Messages or Chat requests")
        if self.provider_output_config is not None and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_output_config is valid only for Messages requests")
        if self.text_verbosity is not None and self.surface not in {
            GatewayApiSurface.RESPONSES,
            GatewayApiSurface.CHAT_COMPLETIONS,
        }:
            raise ValueError("text_verbosity is valid only for Responses and Chat requests")
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
            # defer_loading is also OpenAI's and OpenRouter's marker, so it is
            # valid on every surface; the other carriers stay Anthropic-only.
            any(
                tool.eager_input_streaming is not None
                or tool.allowed_callers is not None
                or tool.input_examples is not None
                for tool in self.tools
            )
            and self.surface != GatewayApiSurface.MESSAGES
        ):
            raise ValueError("Anthropic tool carriers are valid only for Messages requests")
        if self.provider_beta_tokens and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_beta_tokens are valid only for Messages requests")
        if self.service_tier is not None and self.surface == GatewayApiSurface.MESSAGES:
            raise ValueError("service_tier is not valid for Messages requests")
        if self.provider_server_tools and self.surface != GatewayApiSurface.MESSAGES:
            raise ValueError("provider_server_tools are valid only for Messages requests")
        if self.provider_native_tools and self.surface not in {
            GatewayApiSurface.RESPONSES,
            GatewayApiSurface.CHAT_COMPLETIONS,
        }:
            raise ValueError("provider_native_tools are valid only for Responses and Chat requests")
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


class ProjectSelection(ContractModel):
    """One frozen learned-router selection resolved before provider execution."""

    exact_model_id: ExactModelId
    selected_alias: ArtifactId
    activation_ref: ActivationRef
    fallback_reason: str | None = Field(default=None, max_length=512)


class AuthorizationSnapshot(ContractModel):
    """Immutable authority and alias target frozen before learned model selection.

    Attributes:
        model_chain_authority: Optional backend-issued binding, revalidated by
            the host at acceptance and every attempt reservation.
        fair_share_weight: Organization weight in [1, 1,000,000], default 1,
            used only on rungs authoring weighted fair-share admission.
        descendant_start_authorized: False unless the host proves root funding
            and policy gates before allowing a request to start at a child.
        zdr_requested: Caller demand for stricter ZDR filtering, default False.
        requested_route_id: Optional public selector, excluded from durable serialization.
    """

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
    model_chain_authority: ModelChainAuthority | None = None
    requested_route_id: RequestedRouteId | None = Field(default=None, exclude=True)
    refusal_failover: bool = False
    deadline_monotonic: float = Field(gt=0)
    app_referer: str | None = Field(default=None, max_length=2_048)
    """Caller-supplied ``HTTP-Referer`` app identity, content-free and never a credential."""
    app_title: str | None = Field(default=None, max_length=256)
    """Caller-supplied ``X-Title`` app label used only for content-free app attribution."""
    attribution_label: str | None = Field(default=None, max_length=1024)
    """End-user attribution from the OpenAI ``safety_identifier`` (or deprecated
    ``user``) request field: content-free and never a credential."""
    client_ip: str | None = Field(default=None, max_length=45)
    """Caller IP from the TRUSTED proxy hop (``X-Real-IP``, else the RIGHTMOST
    ``X-Forwarded-For`` entry; never the leftmost, which is client-forgeable),
    for per-key IP allow/deny enforcement by the hosted authority. Content-free
    and never a credential; ``None`` when no trusted hop yields an address (an
    allowlist then fails closed, a denylist open). 45 chars fits any IPv6 form."""
    fair_share_weight: int = Field(default=1, ge=1, le=1_000_000)
    descendant_start_authorized: bool = False
    zdr_requested: bool = False
    # Fair-share weights default to equal shares and apply only on opted-in rungs.


class ExecutionSnapshot(ContractModel):
    """Route-bound request plan created only after exact-model selection.

    Attributes:
        model_stages: Frozen ordered stage segments; empty for an unstaged exact-model route.
        traversal_events: Immutable expansion provenance, not mutable runtime visitation state.
    """

    authorization: AuthorizationSnapshot
    exact_model_id: ExactModelId
    pool_id: ExactModelPoolId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    # The pool's per-model failover policy, carried onto the route so the
    # per-attempt retry/failover decision can honor it.
    failover_mode: FailoverMode = "maximize_availability"
    # Optional cache-stakes threshold; otherwise the failover mode decides.
    throttle_cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    # The pool's backoff-and-redial schedule for throttled rungs, carried so
    # the admission can hand the data plane its frozen retry facts and the
    # per-attempt decision can honor a post-backoff redial.
    throttle_redial: GatewayThrottleRedialPolicy | None = None
    # Rungs the host flagged for OpenRouter's per-request ZDR constraint; the
    # dispatch builder tightens each and fails closed on a wire that cannot.
    zdr_constrained_deployment_ids: tuple[DeploymentId, ...] = ()
    model_stages: tuple[ModelExecutionStage, ...] = ()
    traversal_events: tuple[ModelTraversalEvent, ...] = ()

    @model_validator(mode="after")
    def _require_stage_projection(self) -> ExecutionSnapshot:
        """Require the ordered stage leaves to exactly cover the dispatch cursor."""
        if (
            self.model_stages
            and tuple(d for s in self.model_stages for d in s.deployment_ids) != self.deployment_ids
        ):
            raise ValueError("execution stages must exactly cover ordered deployment_ids")
        return self

    def stage_for_depth(self, depth: int) -> ModelExecutionStage:
        """Return the exact destination authority, never substitute the root model."""
        if not 0 <= depth < len(self.deployment_ids):
            raise ValueError("execution route depth is outside the authorized plan")
        if not self.model_stages:
            return ModelExecutionStage(
                stage_index=0,
                exact_model_id=self.exact_model_id,
                pool_id=self.pool_id,
                deployment_ids=self.deployment_ids,
                failover_mode=self.failover_mode,
                throttle_cache_threshold=self.throttle_cache_threshold,
                throttle_redial=self.throttle_redial,
            )
        cursor = 0
        for stage in self.model_stages:
            cursor += len(stage.deployment_ids)
            if depth < cursor:
                return stage
        raise ValueError("execution stage projection is incomplete")
