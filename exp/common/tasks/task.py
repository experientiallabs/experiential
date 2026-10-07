"""Canonical representative task and task-set contracts."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
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
    """One evaluated task with request-visible inputs and provenance.

    A mined task is selected from captured traffic and names the traces it represents. An
    authored task is written or imported by the caller: it has no source traces, so every
    trace-sized step (cost estimates, tool-call checks, judge calibration samples) treats it
    from its declared inputs alone.

    Attributes:
        task_id: Stable task identity.
        lineage_group_id: Leakage lineage shared by duplicate requests.
        partition: Fit or held-out membership.
        instruction: Request-visible task text.
        initial_context: Request-visible context, such as the opening conversation.
        tools: Tool definitions available to the customer agent.
        workload_weight: Positive finite traffic weight.
        source_trace_ids: Captured traces behind a mined task; empty for an authored task.
        origin: ``mined`` (the default, omitted from serialized records) or ``authored``.
        reference_answer: Optional expected answer for an authored task. The judge sees it;
            the candidate and the world model never do. Omitted from records when absent.
    """

    task_id: ArtifactId
    lineage_group_id: ArtifactId
    partition: Literal["fit", "held_out"]
    instruction: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    tools: tuple[ToolSchema, ...] = ()
    workload_weight: float = Field(gt=0)
    source_trace_ids: tuple[str, ...] = ()
    origin: Literal["mined", "authored"] = Field(
        default="mined", exclude_if=lambda value: value == "mined"
    )
    reference_answer: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )

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

    @model_validator(mode="after")
    def _require_origin_provenance(self) -> TaskCase:
        """Require source traces exactly for mined tasks and a reference only on authored ones."""
        if self.origin == "mined":
            if not self.source_trace_ids:
                raise ValueError("a mined task must name at least one source trace")
            if self.reference_answer is not None:
                raise ValueError("only an authored task may carry a reference answer")
        elif self.source_trace_ids:
            raise ValueError("an authored task has no source traces")
        return self


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
