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

from wmo.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256, sha256_json
from wmo.common.tasks import ToolSchema

ModelAlias = ArtifactId
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]

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
    optimization artifacts. ``raw_arguments`` optionally preserves the exact provider-emitted
    JSON string for protocol replay. Its absent default is omitted from serialization so existing
    immutable payload bytes and hashes do not change.
    """

    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: JsonObject = Field(default_factory=dict)
    raw_arguments: str | None = Field(
        default=None,
        max_length=4_000_000,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def _require_matching_raw_arguments(self) -> ToolCall:
        """Require retained raw JSON to decode to the existing parsed object.

        Returns:
            The validated tool call.

        Raises:
            ValueError: Raw arguments are invalid JSON, not an object, or change the parsed value.
        """
        if self.raw_arguments is None:
            return self
        try:
            parsed = _JSON_OBJECT_ADAPTER.validate_json(self.raw_arguments)
        except ValidationError as exc:
            raise ValueError("raw tool arguments must encode one JSON object") from exc
        if parsed != self.arguments:
            raise ValueError("raw tool arguments must match parsed tool arguments")
        return self

    def arguments_json(self, *, sort_keys: bool = False, compact: bool = False) -> str:
        """Return provider-order raw JSON or encode the parsed object for one caller.

        Args:
            sort_keys: Whether fallback encoding sorts object keys.
            compact: Whether fallback encoding omits insignificant separators.

        Returns:
            Exact retained JSON when present, otherwise encoded parsed arguments.
        """
        if self.raw_arguments is not None:
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
        return self


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

    ``supports_temperature`` declares whether the provider accepts an explicit sampling
    temperature for this model; reasoning models that pin their sampling reject the parameter, so
    clients omit it when this is ``False``. ``reasoning_effort`` pins an explicit reasoning-effort
    level on providers whose wire protocol accepts one.
    """

    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_structured_output: bool = False
    supports_completions: bool | None = None
    supports_temperature: bool = True
    reasoning_effort: ReasoningEffort | None = None
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

    def identity_sha256(self) -> Sha256:
        """Hash capabilities that identify the provider model protocol.

        Workflow-only completion, structured-output, sampling, and pricing declarations are
        excluded from provider model identity. Router evaluation freezes its exact execution
        declarations in a separate candidate capability digest and freezes prices in the pricing
        snapshot.

        Returns:
            Stable digest of capability fields that identify the provider protocol boundary.
        """
        excluded = {
            "supports_structured_output",
            "supports_temperature",
            "reasoning_effort",
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
        maximum_output_tokens: Optional upper bound for generated tokens.
    """

    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = ()
    tool_choice: Literal["auto", "none", "required"] | ToolChoice | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
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
