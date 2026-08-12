"""OpenTelemetry-aligned spans and simulator identity for rollout artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ContractModel,
    JsonObject,
    Sha256,
    SourceIdentity,
    StructuredFailure,
)
from wmo.common.models import ModelSnapshot, Usage


class RolloutEventKind(StrEnum):
    """Typed events preserved inside a completed agent rollout."""

    AGENT_MODEL_CALL = "agent_model_call"
    SIMULATOR_WORLD_MODEL_CALL = "simulator_world_model_call"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    MESSAGE = "message"
    LIFECYCLE = "lifecycle"


class RolloutSpan(ContractModel):
    """One ordered OpenTelemetry-style span emitted during an agent episode."""

    span_id: str = Field(min_length=1, max_length=256)
    parent_span_id: str | None = Field(default=None, min_length=1, max_length=256)
    kind: RolloutEventKind
    started_at: datetime
    ended_at: datetime
    payload: JsonObject = Field(default_factory=dict)
    model: ModelSnapshot | None = None
    tool_name: str | None = Field(default=None, max_length=256)
    usage: Usage | None = None
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_valid_timestamps(self) -> RolloutSpan:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("rollout span timestamps must include timezones")
        if self.ended_at < self.started_at:
            raise ValueError("rollout span ended_at cannot be before started_at")
        return self


class ProductionSimulatorSnapshot(ContractModel):
    """Simulator identity for evidence reconstructed from a production trace."""

    kind: Literal["production"] = "production"
    source: SourceIdentity


class WorldModelSimulatorSnapshot(ContractModel):
    """Simulator identity for a text world-model rollout."""

    kind: Literal["world_model"] = "world_model"
    simulator_id: str = Field(min_length=1, max_length=256)
    prompt_id: str = Field(min_length=1, max_length=256)
    prompt_version: str = Field(min_length=1, max_length=256)
    prompt_sha256: Sha256
    world_model: ModelSnapshot


class SandboxSimulatorSnapshot(ContractModel):
    """Simulator identity for an executable environment rollout."""

    kind: Literal["sandbox"] = "sandbox"
    simulator_id: str = Field(min_length=1, max_length=256)
    environment_id: str = Field(min_length=1, max_length=256)
    environment_sha256: Sha256


SimulatorSnapshot = Annotated[
    ProductionSimulatorSnapshot | WorldModelSimulatorSnapshot | SandboxSimulatorSnapshot,
    Field(discriminator="kind"),
]
