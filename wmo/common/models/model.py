"""Canonical model identities, actions, usage, and economics."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, JsonObject, Sha256

ModelAlias = ArtifactId


class ModelSnapshot(ContractModel):
    """Resolved model identity captured at an immutable artifact boundary."""

    provider: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    revision: str | None = Field(default=None, max_length=256)
    capabilities_sha256: Sha256


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
        return self


class ModelResponse(ContractModel):
    """A completed model response with resolved identity and operation accounting."""

    output: AssistantAction
    model: ModelSnapshot
    economics: OperationEconomics
