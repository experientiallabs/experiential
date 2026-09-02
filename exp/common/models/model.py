"""Canonical model identities, actions, usage, and economics."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Final, Literal

from pydantic import (
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from exp.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256, sha256_json
from exp.common.models.content import (
    DocumentContentPart,
    ImageContentPart,
    MessageContentPart,
    VideoContentPart,
)
from exp.common.tasks import ToolSchema

ModelAlias = ArtifactId
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max"]
ChatMaxTokensField = Literal["max_tokens", "max_completion_tokens"]

DEFAULT_REASONING_EFFORT: Final[ReasoningEffort] = "medium"
"""Reasoning effort pinned by default for models known to accept the parameter.

OpenAI documents ``medium`` as the balanced default effort and recommends lowering effort for
latency- and throughput-sensitive workloads, so provider setup pins ``medium`` on every
reasoning-capable model unless the user picks a different effort for that entry. Every request
through the resolved client, serving and optimization alike, uses the entry's pinned effort;
models without a pinned effort never receive the parameter.
"""


class BillingSource(StrEnum):
    """Credential owner responsible for one provider-backed model operation."""

    HOST_MANAGED = "host_managed"
    CUSTOMER_MANAGED = "customer_managed"


class ModelSnapshot(ContractModel):
    """Resolved model identity captured at an immutable artifact boundary.

    The connection digest identifies the normalized, secret-free provider endpoint used for the
    model. It never carries a credential value or credential reference.
    """

    provider: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, max_length=256)
    billing_source: BillingSource
    capabilities_sha256: Sha256
    connection_sha256: Sha256


class RoutedCandidateSnapshot(ContractModel):
    """A stable local alias paired with the model identity used at evaluation time."""

    alias: ModelAlias
    model: ModelSnapshot


class Usage(ContractModel):
    """Provider-neutral token accounting for one operation.

    Cache-read and cache-write counts are subsets of ``input_tokens`` when present. They never
    replace the total input count and must not be added a second time by callers.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)


class NumericMeasurement(ContractModel):
    """A numeric value with explicit observed versus estimated provenance."""

    value: float
    provenance: Literal["observed", "estimated"]

    @field_validator("value")
    @classmethod
    def _require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("numeric measurements must be finite")
        return value


class OperationEconomics(ContractModel):
    """Usage, cost, and latency observed for one isolated operation."""

    usage: Usage | None = None
    cost_usd: NumericMeasurement | None = None
    latency_seconds: NumericMeasurement | None = None


def combine_economics(
    records: Sequence[OperationEconomics],
    *,
    require_complete_usage: bool = True,
) -> OperationEconomics:
    """Aggregate per-operation economics without representing a partial total as complete.

    Args:
        records: Economics observed for each aggregated operation.
        require_complete_usage: When ``True``, report usage only if every record carries it.
            When ``False``, sum the records that report usage and omit usage only when all
            records lack it.

    Returns:
        One economics value. Cost and latency are summed only when every record exposes that
        measurement, preserving a clear unknown rather than a partial sum.
    """
    if not records:
        return OperationEconomics()
    usages = tuple(record.usage for record in records)
    present = tuple(item for item in usages if item is not None)
    usage: Usage | None = None
    if present and (not require_complete_usage or len(present) == len(usages)):
        usage = _sum_usage(present)
    return OperationEconomics(
        usage=usage,
        cost_usd=_sum_measurements(tuple(record.cost_usd for record in records)),
        latency_seconds=_sum_measurements(tuple(record.latency_seconds for record in records)),
    )


def _sum_usage(values: Sequence[Usage]) -> Usage:
    """Sum provider token usage without manufacturing missing cache counts.

    Args:
        values: Usage records reported by the aggregated operations.

    Returns:
        Summed input and output tokens, with cached input tokens summed only when every
        record reports them.
    """
    cached = tuple(value.cached_input_tokens for value in values)
    cached_total: int | None = None
    if all(item is not None for item in cached):
        cached_total = sum(item for item in cached if item is not None)
    return Usage(
        input_tokens=sum(value.input_tokens for value in values),
        output_tokens=sum(value.output_tokens for value in values),
        cached_input_tokens=cached_total,
    )


def _sum_measurements(
    values: Sequence[NumericMeasurement | None],
) -> NumericMeasurement | None:
    """Sum a measurement series while retaining its weakest provenance.

    Args:
        values: Optional measurements from each aggregated operation.

    Returns:
        The summed measurement, or ``None`` when any operation omitted it.
    """
    present: list[NumericMeasurement] = []
    for value in values:
        if value is None:
            return None
        present.append(value)
    return NumericMeasurement(
        value=sum(item.value for item in present),
        provenance=(
            "observed" if all(item.provenance == "observed" for item in present) else "estimated"
        ),
    )


class ToolCall(ContractModel):
    """One complete tool invocation emitted by an assistant.

    ``arguments`` retains the existing parsed-object contract used by environments and
    optimization artifacts. The excluded replay fields preserve the provider's exact arguments,
    item identity, and output position for immediate protocol replay. They remain absent from
    immutable artifacts but join gateway idempotency identity explicitly.
    """

    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: JsonObject = Field(default_factory=dict)
    raw_arguments: str | None = Field(
        default=None,
        max_length=4_000_000,
        exclude=True,
    )
    cache_control: JsonObject | None = Field(default=None, exclude=True)
    """Validated caller prompt-caching hint attached to this tool call.

    OpenAI-compatible callers (the @ai-sdk stack) attach an Anthropic-style
    ephemeral ``cache_control`` to a message's last content part, which lands
    inside the ``tool_calls`` entry when that part is a tool call. The hint
    forwards onto the native Anthropic ``tool_use`` block and is a no-op on
    other wires. Like ``raw_arguments``, it is excluded from serialization so
    a cache hint can never affect immutable artifacts or request digests.
    """
    provider_item_id: str | None = Field(default=None, min_length=1, max_length=256, exclude=True)
    provider_output_index: int | None = Field(default=None, ge=0, exclude=True)
    provider_status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        exclude=True,
    )

    @model_validator(mode="after")
    def _require_matching_raw_arguments(self) -> ToolCall:
        """Require retained raw JSON to decode to the existing parsed object.

        Returns:
            The validated tool call.

        Raises:
            ValueError: Raw arguments are invalid JSON, not an object, or change the parsed value.
        """
        if self.raw_arguments is not None:
            try:
                parsed = _JSON_OBJECT_ADAPTER.validate_json(self.raw_arguments)
            except ValidationError as exc:
                raise ValueError("raw tool arguments must encode one JSON object") from exc
            if parsed != self.arguments:
                raise ValueError("raw tool arguments must match parsed tool arguments")
        if self.provider_item_id is not None and self.provider_output_index is None:
            raise ValueError("provider tool-call item identity requires retained output order")
        if self.provider_status is not None and self.provider_output_index is None:
            raise ValueError("provider tool-call status requires retained output order")
        return self

    def arguments_json(self, *, sort_keys: bool = False, compact: bool = False) -> str:
        """Return provider-order raw JSON or encode the parsed object for one caller.

        Args:
            sort_keys: Whether fallback encoding sorts object keys.
            compact: Whether fallback encoding omits insignificant separators.

        Returns:
            Exact retained JSON when present and no canonicalization was requested. Otherwise,
            encoded parsed arguments honoring the requested canonicalization options.
        """
        if self.raw_arguments is not None and not sort_keys and not compact:
            return self.raw_arguments
        separators = (",", ":") if compact else None
        return json.dumps(self.arguments, sort_keys=sort_keys, separators=separators)


class AssistantAction(ContractModel):
    """One complete assistant output, including zero or more tool calls."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def _require_content_or_tool_call(self) -> AssistantAction:
        if self.content is None and not self.tool_calls:
            raise ValueError("an assistant action needs content or at least one tool call")
        return self


class ModelMessage(ContractModel):
    """One request-visible message exchanged with a model."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_call_id: str | None = None
    assistant_action: AssistantAction | None = None
    content_parts: tuple[MessageContentPart, ...] = Field(default=(), exclude=True)
    """Ordered caller content parts when a user message carries attachments.

    Empty on every text-only message. The text parts concatenate to
    ``content``, so selectors, simulators, and persisted artifacts keep
    seeing exactly the text they saw before media existed; provider clients
    that can carry media read the parts and emit the caller's exact
    interleaving. Excluded from serialization so identities of text-only
    requests are byte-identical to pre-media traffic.
    """

    @model_validator(mode="after")
    def _require_message_payload(self) -> ModelMessage:
        if self.content is None and self.assistant_action is None:
            raise ValueError("a model message needs text or an assistant action")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is valid only for tool messages")
        if self.role != "assistant" and self.assistant_action is not None:
            raise ValueError("assistant_action is valid only for assistant messages")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages require tool_call_id")
        if self.content_parts:
            if self.role != "user":
                raise ValueError("content parts are valid only for user messages")
            texts = [part.text for part in self.content_parts if part.kind == "text"]
            if (self.content or "") != "".join(texts):
                raise ValueError("content parts must flatten to the message content")
        return self

    @property
    def images(self) -> tuple[ImageContentPart, ...]:
        """Return this message's image parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "image")

    @property
    def videos(self) -> tuple[VideoContentPart, ...]:
        """Return this message's video parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "video")

    @property
    def documents(self) -> tuple[DocumentContentPart, ...]:
        """Return this message's document parts in caller order."""
        return tuple(part for part in self.content_parts if part.kind == "document")


class ModelFinishReason(StrEnum):
    """Terminal condition reported by a non-streaming provider completion."""

    COMPLETED = "completed"
    LENGTH = "length"


class ModelResponse(ContractModel):
    """A completed model response with resolved identity and operation accounting."""

    output: AssistantAction
    model: ModelSnapshot
    economics: OperationEconomics
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED

    @classmethod
    def completed(
        cls,
        *,
        output: AssistantAction,
        configured_model: ModelSnapshot,
        served_model_id: JsonValue | None,
        usage: Usage | None,
        latency_seconds: float,
        hit_length_limit: bool = False,
    ) -> ModelResponse:
        """Build the shared completed-response shape every provider returns.

        Args:
            output: Typed assistant action parsed from the provider payload.
            configured_model: Resolved catalog identity used for the request.
            served_model_id: Provider-reported model identifier, preferred over the
                configured identity when it is a non-empty string.
            usage: Provider-reported token accounting, when present.
            latency_seconds: Observed duration of the successful request sequence.
            hit_length_limit: Whether the provider stopped at its output-token limit.

        Returns:
            A completed response with observed latency and the served model identity.
        """
        model = (
            configured_model.model_copy(update={"model_id": served_model_id})
            if isinstance(served_model_id, str) and served_model_id
            else configured_model
        )
        return cls(
            output=output,
            model=model,
            economics=OperationEconomics(
                usage=usage,
                latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
            ),
            finish_reason=(
                ModelFinishReason.LENGTH if hit_length_limit else ModelFinishReason.COMPLETED
            ),
        )


class ModelCapabilities(ContractModel):
    """Static capabilities known before a model request is sent.

    The runtime records a digest of this object in every resolved model identity. The fields
    describe protocol support, not a claim that a provider accepts every possible prompt.

    ``supports_tools`` and ``supports_embeddings`` are tri-state: ``True`` and ``False`` are
    explicit operator declarations, while ``None`` means unknown. Unknown support is permissive
    at runtime so an undeclared or newly released model stays usable; only an explicit ``False``
    blocks the corresponding protocol feature before dispatch.

    ``supports_temperature`` and the optional sampling capability fields declare whether the
    provider accepts the corresponding generation controls for this model. ``None`` means the
    older catalog did not carry an explicit declaration; runtime clients fall back to the broad
    temperature capability for ``top_p`` and omit the less portable controls. Reasoning models
    that pin their sampling reject temperature and nucleus sampling, so clients omit both when
    the route says they are unsupported. ``supports_logprobs`` remains provider metadata, but
    public logprob requests fail locally until normalized responses can expose their values.
    ``supports_reasoning`` is an explicit wire capability, not an inference from
    ``reasoning_effort``. ``reasoning_effort`` pins an explicit reasoning-effort level only when
    that capability is true.
    """

    supports_tools: bool | None = None
    supports_embeddings: bool | None = None
    supports_structured_output: bool = False
    supports_completions: bool | None = None
    supports_temperature: bool = True
    supports_top_p: bool | None = None
    supports_top_k: bool | None = None
    supports_logprobs: bool | None = None
    supports_reasoning: bool = False
    reasoning_effort: ReasoningEffort | None = None
    sampling_requires_reasoning_none: bool = False
    """Whether temperature and top-p are valid only with ``reasoning_effort='none'``."""
    chat_max_tokens_field: ChatMaxTokensField | None = None
    minimum_temperature: float | None = Field(default=None, ge=0, le=2)
    maximum_temperature: float | None = Field(default=None, ge=0, le=2)
    minimum_top_p: float | None = Field(default=None, ge=0, le=1)
    maximum_top_p: float | None = Field(default=None, ge=0, le=1)
    minimum_top_k: int | None = Field(default=None, ge=0)
    maximum_top_k: int | None = Field(default=None, ge=0)
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    output_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    cached_input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)
    cache_write_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)

    @field_validator(
        "input_cost_per_million_tokens_usd",
        "output_cost_per_million_tokens_usd",
        "cached_input_cost_per_million_tokens_usd",
        "cache_write_cost_per_million_tokens_usd",
    )
    @classmethod
    def _require_finite_prices(cls, value: float | None) -> float | None:
        """Reject non-finite catalog prices before they enter budget arithmetic.

        Args:
            value: Optional nonnegative price declared by the operator.

        Returns:
            The unchanged finite price or ``None`` when pricing is unknown.

        Raises:
            ValueError: The declared price is infinite or NaN.
        """
        if value is not None and not math.isfinite(value):
            raise ValueError("model token prices must be finite")
        return value

    @model_validator(mode="after")
    def _require_ordered_generation_ranges(self) -> ModelCapabilities:
        """Reject inverted per-model generation-control ranges."""
        ranges = (
            ("temperature", self.minimum_temperature, self.maximum_temperature),
            ("top_p", self.minimum_top_p, self.maximum_top_p),
            ("top_k", self.minimum_top_k, self.maximum_top_k),
        )
        for name, minimum, maximum in ranges:
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"minimum_{name} cannot exceed maximum_{name}")
        if self.reasoning_effort is not None and not self.supports_reasoning:
            raise ValueError("reasoning_effort requires reasoning support")
        if self.sampling_requires_reasoning_none and not self.supports_reasoning:
            raise ValueError("conditional sampling requires reasoning support")
        return self

    def identity_sha256(self) -> Sha256:
        """Hash capabilities that identify the provider model protocol.

        Workflow-only completion, structured-output, sampling, and pricing declarations are
        excluded from provider model identity. Router evaluation freezes its exact execution
        declarations in a separate candidate capability digest and freezes prices in the pricing
        snapshot.

        Unknown tool and embedding support hashes as its own tri-state value: runtime dispatch
        treats unknown support permissively and an explicit ``False`` as a hard denial, so
        moving between them is a semantic change that must invalidate frozen identities.

        Returns:
            Stable digest of capability fields that identify the provider protocol boundary.
        """
        excluded = {
            "supports_structured_output",
            "supports_temperature",
            "supports_top_p",
            "supports_top_k",
            "supports_logprobs",
            "supports_reasoning",
            "reasoning_effort",
            "sampling_requires_reasoning_none",
            "chat_max_tokens_field",
            "minimum_temperature",
            "maximum_temperature",
            "minimum_top_p",
            "maximum_top_p",
            "minimum_top_k",
            "maximum_top_k",
            "input_cost_per_million_tokens_usd",
            "output_cost_per_million_tokens_usd",
            "cached_input_cost_per_million_tokens_usd",
            "cache_write_cost_per_million_tokens_usd",
        }
        excluded.add("supports_completions")
        return sha256_json(self.model_dump(mode="json", exclude=excluded))


class ToolChoice(ContractModel):
    """A request to require one named tool when the provider supports forced tools."""

    name: str = Field(min_length=1, max_length=256)


class ModelRequest(ContractModel):
    """A complete non-streaming model request independent of provider wire format.

    Args:
        messages: Ordered visible conversation messages.
        tools: Tool schemas available for this turn.
        tool_choice: Optional automatic, disabled, required, or named-tool selection.
        temperature: Optional sampling temperature.
        top_p: Optional nucleus-sampling probability mass in ``[0, 1]``.
        top_k: Optional maximum number of candidate tokens considered during sampling.
        logprobs: Optional request for token log probabilities. Gateway serving rejects a true
            request until normalized responses can return them losslessly.
        top_logprobs: Optional count for alternate token probabilities, subject to the same
            lossless-response requirement as ``logprobs``.
        reasoning_effort: Optional caller-selected reasoning effort, preserved only on routes that
            explicitly declare support.
        maximum_output_tokens: Optional upper bound for generated tokens.
    """

    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    reasoning_effort: ReasoningEffort | None = None
    maximum_output_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _require_coherent_tools_and_messages(self) -> ModelRequest:
        tool_names = tuple(tool.name for tool in self.tools)
        if len(set(tool_names)) != len(tool_names):
            raise ValueError("model request tool names must be unique")
        if isinstance(self.tool_choice, ToolChoice) and self.tool_choice.name not in tool_names:
            raise ValueError("named tool_choice must name a request tool")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("required tool_choice needs at least one request tool")
        for message in self.messages:
            if message.role == "tool" and message.assistant_action is not None:
                raise ValueError("tool messages cannot carry assistant actions")
        return self


class Embedding(ContractModel):
    """One normalized vector returned for a request-visible text input."""

    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _require_finite_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding values must be finite")
        norm = math.sqrt(sum(item * item for item in value))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("embedding values must have unit norm")
        return value


class RawEmbedding(ContractModel):
    """One provider-returned embedding vector preserved without renormalization.

    Unlike :class:`Embedding`, which unit-normalizes for cosine routing, the
    public ``/v1/embeddings`` surface must return the provider's exact vector,
    so this carrier keeps the raw magnitude and validates finiteness only.
    """

    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _require_finite_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("embedding values must be finite")
        return value


class RawEmbeddingBatch(ContractModel):
    """Ordered raw embeddings with the provider's input-token usage.

    The public embeddings surface bills input tokens, so the provider's
    ``prompt_tokens`` count is carried alongside the vectors rather than
    dropped. ``served_model_id`` is the exact model the provider reported, kept
    for attribution and never invented when the provider omits it.
    """

    embeddings: tuple[RawEmbedding, ...] = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    served_model_id: str | None = None
