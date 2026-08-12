"""Canonical model identities, actions, usage, and economics."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256
from wmo.common.tasks import ToolSchema

ModelAlias = ArtifactId


class ModelSnapshot(ContractModel):
    """Resolved model identity captured at an immutable artifact boundary.

    The connection digest identifies the normalized, secret-free provider endpoint used for the
    model. It never carries a credential value or credential reference.
    """

    provider: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, max_length=256)
    capabilities_sha256: Sha256
    connection_sha256: Sha256


class RoutedCandidateSnapshot(ContractModel):
    """A stable local alias paired with the model identity used at evaluation time."""

    alias: ModelAlias
    model: ModelSnapshot


class Usage(ContractModel):
    """Provider-neutral token accounting for one operation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)


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


class ToolCall(ContractModel):
    """One complete tool invocation emitted by an assistant."""

    call_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    arguments: JsonObject = Field(default_factory=dict)


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


class ModelCapabilities(ContractModel):
    """Static capabilities known before a model request is sent.

    The runtime records a digest of this object in every resolved model identity. The fields
    describe protocol support, not a claim that a provider accepts every possible prompt.
    """

    supports_tools: bool = False
    supports_embeddings: bool = False
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)


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
