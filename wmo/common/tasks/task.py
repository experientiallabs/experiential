"""Canonical representative task and task-set contracts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    validate_artifact_file_path,
)


class ToolSchema(ContractModel):
    """A task-visible tool definition available to the customer agent."""

    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1)
    input_schema: JsonObject


class TaskCase(ContractModel):
    """One selected real task with request-visible inputs and source provenance."""

    task_id: ArtifactId
    lineage_group_id: ArtifactId
    partition: Literal["fit", "held_out"]
    instruction: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    tools: tuple[ToolSchema, ...] = ()
    workload_weight: float = Field(gt=0)
    source_trace_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("workload_weight")
    @classmethod
    def _require_finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("workload_weight must be finite")
        return value

    @field_validator("source_trace_ids")
    @classmethod
    def _require_unique_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("source_trace_ids must not contain duplicates")
        return value


class TaskSet(ArtifactEnvelope):
    """A frozen task-set manifest that references hashed task and coverage files."""

    task_set_id: ArtifactId
    task_ids: tuple[ArtifactId, ...]
    tasks_path: str = Field(min_length=1)
    tasks_sha256: Sha256
    coverage_path: str | None = None
    coverage_sha256: Sha256 | None = None

    @field_validator("tasks_path")
    @classmethod
    def _require_safe_tasks_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("coverage_path")
    @classmethod
    def _require_safe_coverage_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_file_path(value).as_posix()

    @field_validator("task_ids")
    @classmethod
    def _require_unique_task_ids(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if not value:
            raise ValueError("a task set must contain at least one task")
        if len(set(value)) != len(value):
            raise ValueError("task_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _require_complete_coverage_reference(self) -> TaskSet:
        if (self.coverage_path is None) != (self.coverage_sha256 is None):
            raise ValueError("task-set coverage path and digest must be set together")
        return self
