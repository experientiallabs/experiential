"""Canonical normalized production-trace contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    Sha256,
    SourceIdentity,
    StructuredFailure,
    validate_artifact_file_path,
)
from exp.common.models import ModelSnapshot, Usage
from exp.common.tasks import ToolSchema


class TraceSource(ContractModel):
    """Origin and semantic-convention version for a normalized production trace."""

    identity: SourceIdentity
    semantic_convention_version: str = Field(min_length=1, max_length=128)


class TraceOutcome(ContractModel):
    """Captured terminal outcome for one production trace."""

    status: Literal["success", "failure", "abandoned", "unknown"]
    outcome_name: str | None = Field(default=None, max_length=256)
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_failure_for_failed_outcome(self) -> TraceOutcome:
        if self.status == "failure" and self.failure is None:
            raise ValueError("failed trace outcomes require a structured failure")
        return self


class TraceSpan(ContractModel):
    """One ordered OpenTelemetry-style event in a normalized production trace."""

    span_id: str = Field(min_length=1, max_length=256)
    parent_span_id: str | None = Field(default=None, min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    started_at: AwareDatetime
    ended_at: AwareDatetime
    attributes: JsonObject = Field(default_factory=dict)
    model: ModelSnapshot | None = None
    usage: Usage | None = None
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_ordered_timestamps(self) -> TraceSpan:
        if self.ended_at < self.started_at:
            raise ValueError("trace span ended_at cannot be before started_at")
        return self


class Trace(ContractModel):
    """One normalized customer agent trace with source provenance."""

    trace_id: str = Field(min_length=1, max_length=512)
    conversation_id: str | None = Field(default=None, max_length=512)
    task: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    tools: tuple[ToolSchema, ...] = ()
    spans: tuple[TraceSpan, ...]
    outcome: TraceOutcome | None = None
    source: TraceSource

    @field_validator("spans")
    @classmethod
    def _require_unique_span_ids(cls, value: tuple[TraceSpan, ...]) -> tuple[TraceSpan, ...]:
        if not value:
            raise ValueError("a trace must contain at least one span")
        span_ids = tuple(span.span_id for span in value)
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("trace span IDs must be unique")
        return value


class TraceDataset(ArtifactEnvelope):
    """A frozen normalized trace-dataset manifest with auditable exclusions."""

    schema_version: Literal[1, 2] = 2
    dataset_id: ArtifactId
    semantic_convention_version: str = Field(min_length=1, max_length=128)
    traces_path: str = Field(min_length=1)
    traces_sha256: Sha256
    issues_path: str | None = None
    issues_sha256: Sha256 | None = None
    invalid_trace_count: int = Field(default=0, ge=0)
    trace_ids: tuple[str, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_integer_schema_version(cls, value: object) -> object:
        """Reject boolean and floating-point lookalikes at the version boundary."""
        if type(value) is not int:
            raise ValueError("trace dataset schema_version must be an integer")
        return value

    @field_validator("traces_path")
    @classmethod
    def _require_safe_traces_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("issues_path")
    @classmethod
    def _require_safe_issues_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_artifact_file_path(value).as_posix()

    @field_validator("trace_ids")
    @classmethod
    def _require_unique_trace_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a trace dataset must contain at least one trace")
        if len(set(value)) != len(value):
            raise ValueError("trace_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _require_complete_issue_reference(self) -> TraceDataset:
        if (self.issues_path is None) != (self.issues_sha256 is None):
            raise ValueError("trace-dataset issues path and digest must be set together")
        if self.invalid_trace_count and self.issues_path is None:
            raise ValueError("invalid trace count requires an immutable issues report")
        return self
