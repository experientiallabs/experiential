"""Immutable gateway request, target, event, failure, and compatibility contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256
from exp.common.models.gateway_catalog import (
    DeploymentId,
    ExactModelId,
    ExactModelPoolId,
)
from exp.common.models.model import ToolCall

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


class GatewayToolDefinition(ContractModel):
    """One caller-defined function tool with its exact JSON Schema declaration."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    parameters: JsonObject
    strict: bool = False


class StructuredTextFormat(ContractModel):
    """A strict structured-text output schema requested by the caller."""

    name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=8_192)
    json_schema: JsonObject
    strict: bool = True


class GatewayNamedToolChoice(ContractModel):
    """A request to require one named caller-defined function."""

    name: str = Field(min_length=1, max_length=256)


class GatewayMessage(ContractModel):
    """One canonical gateway message preserving developer and tool-call identity."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = Field(default=None, min_length=1, max_length=256)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_is_error: bool = Field(default=False, exclude=True)
    """Whether this tool result reports a failed tool invocation.

    Only the Anthropic Messages surface can express it (``tool_result.is_error``),
    and only the Anthropic upstream dialect can emit it back, so an
    Anthropic-to-Anthropic round trip is lossless. The OpenAI-family wire
    formats have no tool-error flag; their builders ignore this field and the
    error state travels in the result text the model reads. Like
    ``ToolCall.raw_arguments``, the field is deliberately excluded from model
    serialization so request digests, replay identity, and immutable
    artifacts are unaffected by it.
    """

    @model_validator(mode="after")
    def _require_role_coherence(self) -> GatewayMessage:
        """Reject payload fields that do not belong to the selected message role.

        Returns:
            The validated canonical message.

        Raises:
            ValueError: Content, tool linkage, or assistant calls are incoherent.
        """
        if self.content is None and not self.tool_calls:
            raise ValueError("gateway messages need content or assistant tool calls")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("tool_calls are valid only for assistant messages")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role != "tool" and self.tool_is_error:
            raise ValueError("tool_is_error is valid only for tool messages")
        return self


class GatewayRequest(ContractModel):
    """Lossless canonical request shared by protocol and provider implementations."""

    surface: GatewayApiSurface
    messages: tuple[GatewayMessage, ...] = Field(min_length=1)
    tools: tuple[GatewayToolDefinition, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | GatewayNamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    structured_text: StructuredTextFormat | None = None
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    stop: tuple[str, ...] = ()
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
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
        if (
            isinstance(self.tool_choice, GatewayNamedToolChoice)
            and self.tool_choice.name not in names
        ):
            raise ValueError("named gateway tool choice must name a request tool")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("required gateway tool choice needs at least one tool")
        if self.include_usage and not self.stream:
            raise ValueError("include_usage is valid only for streaming requests")
        return self


class GatewayUsage(ContractModel):
    """Normalized token counts and invoked tool names from one provider attempt.

    A terminal event may carry only ``tool_names`` when the provider omits token usage. In that
    case both token totals remain unknown instead of being represented as zero.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
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
            if self.cached_input_tokens is not None or self.reasoning_tokens is not None:
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


class GatewayFailure(ContractModel):
    """Sanitized failure with retry and failover eligibility already classified."""

    failure_class: GatewayFailureClass
    safe_message: str = Field(min_length=1, max_length=2_048)
    retryable_same_deployment: bool = False
    failover_eligible: bool = False
    safe_details: JsonObject = Field(default_factory=dict)


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


class CompatibilityDisposition(StrEnum):
    """How one installed public request field is handled by the gateway."""

    SUPPORTED = "supported"
    CONDITIONALLY_SUPPORTED = "conditionally_supported"
    METADATA_ONLY = "metadata_only"
    UNSUPPORTED = "unsupported"


class CompatibilityField(ContractModel):
    """One explicit public-field decision in a versioned compatibility manifest."""

    field_path: str = Field(min_length=1, max_length=512)
    disposition: CompatibilityDisposition
    capability: str | None = Field(default=None, min_length=1, max_length=256)


class CompatibilityManifest(ContractModel):
    """Closed field-classification contract for one public API surface."""

    schema_version: int = Field(ge=1)
    surface: GatewayApiSurface
    fields: tuple[CompatibilityField, ...]

    @model_validator(mode="after")
    def _require_unique_field_paths(self) -> CompatibilityManifest:
        """Reject duplicate field decisions that could make parsing ambiguous.

        Returns:
            The validated manifest.

        Raises:
            ValueError: A field path appears more than once.
        """
        paths = tuple(field.field_path for field in self.fields)
        if len(set(paths)) != len(paths):
            raise ValueError("compatibility manifest field paths must be unique")
        return self
