"""Typed, transport-neutral domain events for durable EXP Project stages."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    FailureCode,
)


class ProjectStage(StrEnum):
    """EXP-owned durable stages exposed to hosted Project orchestration."""

    PREPARING_TRACES = "preparing_traces"
    BUILDING_WORLD_MODEL = "building_world_model"
    OPTIMIZING_ROUTER = "optimizing_router"
    COMPLETING_REPORT = "completing_report"


class ProjectStageEventKind(StrEnum):
    """Supported state transitions and bounded progress observations for one stage."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ProjectStageFailure(ContractModel):
    """Customer-safe failure identity without provider text or free-form detail payloads."""

    code: FailureCode
    retryable: bool = False
    detail_code: ArtifactId | None = None


class ProjectStageEvent(ContractModel):
    """One ordered EXP domain event independent of persistence and delivery transport."""

    schema_version: Literal[1] = 1
    event_id: ArtifactId
    project_id: ArtifactId
    attempt_id: ArtifactId
    sequence: int = Field(ge=0)
    occurred_at: AwareDatetime
    stage: ProjectStage
    kind: ProjectStageEventKind
    completed_units: int | None = Field(default=None, ge=0)
    total_units: int | None = Field(default=None, gt=0)
    outputs: tuple[ArtifactInput, ...] = ()
    failure: ProjectStageFailure | None = None

    @field_validator("outputs")
    @classmethod
    def _require_sorted_unique_outputs(
        cls, value: tuple[ArtifactInput, ...]
    ) -> tuple[ArtifactInput, ...]:
        """Require canonical output ordering without conflicting or repeated IDs."""
        artifact_ids = tuple(item.artifact_id for item in value)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("stage event outputs must not repeat an artifact_id")
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise ValueError("stage event outputs must be sorted by artifact_id")
        return value

    @model_validator(mode="after")
    def _require_kind_specific_payload(self) -> ProjectStageEvent:
        """Keep progress, completion, and failure payloads disjoint and fully typed."""
        has_progress = self.completed_units is not None or self.total_units is not None
        if self.kind == ProjectStageEventKind.PROGRESS:
            if self.completed_units is None or self.total_units is None:
                raise ValueError("progress events require completed_units and total_units")
            if self.completed_units > self.total_units:
                raise ValueError("completed_units cannot exceed total_units")
        elif has_progress:
            raise ValueError("only progress events may carry completed_units or total_units")
        if self.kind == ProjectStageEventKind.COMPLETED:
            if not self.outputs:
                raise ValueError("completed events require at least one immutable output")
        elif self.outputs:
            raise ValueError("only completed events may carry immutable outputs")
        if self.kind == ProjectStageEventKind.FAILED:
            if self.failure is None:
                raise ValueError("failed events require a structured failure")
        elif self.failure is not None:
            raise ValueError("only failed events may carry a structured failure")
        return self
