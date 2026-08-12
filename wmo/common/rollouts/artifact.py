"""Canonical immutable simulation and rollout artifact contracts."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    Sha256,
    StructuredFailure,
    validate_artifact_file_path,
)
from wmo.common.models import AssistantAction, ModelSnapshot, OperationEconomics
from wmo.common.rollouts.otel import (
    ProductionSimulatorSnapshot,
    RolloutSpan,
    SandboxSimulatorSnapshot,
    SimulatorSnapshot,
    WorldModelSimulatorSnapshot,
)


class SimulationMode(StrEnum):
    """Execution mode selected by an immutable simulation specification."""

    WORLD_MODEL = "world_model"
    SANDBOX = "sandbox"
    MIXED_REALITY = "mixed_reality"


class StopReason(StrEnum):
    """Terminal reason captured for an agent episode."""

    COMPLETED = "completed"
    AGENT_STOP = "agent_stop"
    MAXIMUM_STEPS = "maximum_steps"
    MAXIMUM_COST = "maximum_cost"
    CONTEXT_OVERFLOW = "context_overflow"
    LENGTH = "length"
    FAILURE = "failure"
    CANCELLED = "cancelled"


class SimulationArtifact(ArtifactEnvelope):
    """Shared envelope emitted by a concrete simulator for one planned cell."""

    artifact_id: ArtifactId
    simulation_id: ArtifactId
    cell_id: ArtifactId
    mode: SimulationMode


class RolloutArtifact(SimulationArtifact):
    """The v1 simulation artifact subtype that preserves one full agent episode."""

    artifact_kind: Literal["rollout"] = "rollout"
    rollout_id: ArtifactId
    trace_id: str = Field(min_length=1, max_length=512)
    evidence_source: Literal["production", "world_model", "sandbox"]
    source_run_id: str = Field(min_length=1, max_length=512)
    task_id: ArtifactId
    candidate: ModelSnapshot
    agent_id: str = Field(min_length=1, max_length=256)
    simulator: SimulatorSnapshot
    world_model: ModelSnapshot | None = None
    seed: int | None = None
    repeat: int = Field(ge=0)
    spans: tuple[RolloutSpan, ...]
    final_output: AssistantAction | None = None
    stop_reason: StopReason
    failure: StructuredFailure | None = None
    candidate_economics: OperationEconomics
    world_model_economics: OperationEconomics | None = None
    sandbox_economics: OperationEconomics | None = None
    orchestration_economics: OperationEconomics | None = None
    simulation_spec_sha256: Sha256 | None = None

    @field_validator("spans")
    @classmethod
    def _require_unique_span_ids(cls, value: tuple[RolloutSpan, ...]) -> tuple[RolloutSpan, ...]:
        if not value:
            raise ValueError("a rollout must contain at least one span")
        span_ids = tuple(span.span_id for span in value)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("rollout span IDs must be unique")
        return value

    @model_validator(mode="after")
    def _require_consistent_source_provenance(self) -> RolloutArtifact:
        if self.evidence_source == "production" and self.simulation_spec_sha256 is not None:
            raise ValueError("production rollouts must not name a simulation specification")
        if self.evidence_source != "production" and self.simulation_spec_sha256 is None:
            raise ValueError("simulated rollouts require a simulation specification digest")
        if self.stop_reason == StopReason.FAILURE and self.failure is None:
            raise ValueError("failed rollouts require a structured failure")
        if self.evidence_source == "production":
            if not isinstance(self.simulator, ProductionSimulatorSnapshot):
                raise ValueError("production rollouts require a production simulator snapshot")
            if self.world_model is not None:
                raise ValueError("production rollouts must not name a world model")
        elif self.evidence_source == "world_model":
            if self.mode != SimulationMode.WORLD_MODEL:
                raise ValueError("world-model rollouts require world_model mode")
            if not isinstance(self.simulator, WorldModelSimulatorSnapshot):
                raise ValueError("world-model rollouts require a world-model simulator snapshot")
            if self.world_model != self.simulator.world_model:
                raise ValueError("world-model rollout identity must match its simulator snapshot")
        elif self.evidence_source == "sandbox":
            if self.mode != SimulationMode.SANDBOX:
                raise ValueError("sandbox rollouts require sandbox mode")
            if not isinstance(self.simulator, SandboxSimulatorSnapshot):
                raise ValueError("sandbox rollouts require a sandbox simulator snapshot")
            if self.world_model is not None:
                raise ValueError("sandbox rollouts must not name a world model")
        return self


class SimulationArtifactSet(ArtifactEnvelope):
    """A frozen set of completed artifacts emitted for one simulation specification."""

    artifact_set_id: ArtifactId
    simulation_id: ArtifactId
    artifact_ids: tuple[ArtifactId, ...]
    artifacts_path: str = Field(min_length=1)
    artifacts_sha256: Sha256

    @field_validator("artifacts_path")
    @classmethod
    def _require_safe_artifacts_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("artifact_ids")
    @classmethod
    def _require_unique_artifact_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if not value:
            raise ValueError("a simulation artifact set must contain at least one artifact")
        if len(set(value)) != len(value):
            raise ValueError("artifact_ids must not contain duplicates")
        return value
