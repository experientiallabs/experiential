"""Immutable contracts for retrieval grounded only in observed production transitions."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    SourceIdentity,
    validate_artifact_file_path,
)
from wmo.common.models import ModelSnapshot

RAG_ARTIFACT_TYPE = "trace-rag-index"
RAG_KEY_SCHEMA_VERSION = "observed-transition-key-v1"
RAG_TRANSITIONS_PATH = "transitions.jsonl"
RAG_VECTORS_PATH = "vectors.jsonl"
RAG_INDEX_PATH = "rag-index.json"


class RAGSourceRef(ContractModel):
    """Exact verified real-trace artifact consumed by one RAG index."""

    kind: Literal["trace_dataset", "runtime_trace_snapshot"]
    artifact_input: ArtifactInput
    source: SourceIdentity
    records_sha256: Sha256
    trace_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("trace_ids")
    @classmethod
    def _require_ordered_unique_trace_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("RAG source trace IDs must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("RAG source trace IDs must be sorted")
        return value


class RAGLineageBinding(ContractModel):
    """Frozen fit or held-out assignment for one real source trace."""

    trace_id: str = Field(min_length=1, max_length=512)
    lineage_id: ArtifactId
    partition: Literal["fit", "held_out"]


class RAGAction(ContractModel):
    """One visible agent action captured in a real trace."""

    kind: Literal["message", "tool_call"]
    content: str | None = None
    tool_name: str | None = Field(default=None, min_length=1, max_length=256)
    tool_arguments: JsonValue = None

    @model_validator(mode="after")
    def _require_kind_payload(self) -> RAGAction:
        if self.kind == "message":
            if not self.content:
                raise ValueError("message RAG actions need visible content")
            if self.tool_name is not None or self.tool_arguments is not None:
                raise ValueError("message RAG actions cannot carry tool fields")
        else:
            if self.tool_name is None:
                raise ValueError("tool-call RAG actions need a tool name")
            if self.content is not None:
                raise ValueError("tool-call RAG actions cannot carry message content")
        return self


class RAGObservation(ContractModel):
    """One real user or environment observation following an agent action."""

    kind: Literal["message", "tool_result"]
    content: str = Field(min_length=1)


class RAGTransition(ContractModel):
    """One immutable real action-to-subsequent-observation demonstration."""

    transition_id: ArtifactId
    trace_id: str = Field(min_length=1, max_length=512)
    conversation_id: str | None = Field(default=None, max_length=512)
    lineage_id: ArtifactId
    action_span_id: str = Field(min_length=1, max_length=256)
    observation_span_id: str = Field(min_length=1, max_length=256)
    task: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    action: RAGAction
    observation: RAGObservation
    key_text: str = Field(min_length=1)
    key_sha256: Sha256


class RAGVector(ContractModel):
    """Persisted unit vector bound to one exact observed transition."""

    transition_id: ArtifactId
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _require_finite_unit_vector(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("RAG vectors must be finite")
        norm = math.sqrt(sum(item * item for item in value))
        if not math.isclose(norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("RAG vectors must have unit norm")
        return value


class RAGIndex(ArtifactEnvelope):
    """Completed immutable observed-transition index and its exact embedding identity."""

    rag_id: ArtifactId
    key_schema_version: Literal["observed-transition-key-v1"] = RAG_KEY_SCHEMA_VERSION
    sources: tuple[RAGSourceRef, ...] = Field(min_length=1)
    embedder: ModelSnapshot
    transitions_path: str = RAG_TRANSITIONS_PATH
    transitions_sha256: Sha256
    vectors_path: str = RAG_VECTORS_PATH
    vectors_sha256: Sha256
    transition_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    fit_lineage_ids: tuple[ArtifactId, ...] = Field(min_length=1)
    included_lineage_ids: tuple[ArtifactId, ...] = ()
    included_partitions: tuple[Literal["fit", "held_out"], ...] = ("fit",)
    embedding_dimension: int = Field(gt=0)
    transition_count: int = Field(gt=0)
    default_top_k: int = Field(default=5, gt=0)

    @field_validator("transitions_path", "vectors_path")
    @classmethod
    def _require_safe_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @field_validator("sources")
    @classmethod
    def _require_sorted_unique_sources(
        cls, value: tuple[RAGSourceRef, ...]
    ) -> tuple[RAGSourceRef, ...]:
        ids = tuple(item.artifact_input.artifact_id for item in value)
        if len(set(ids)) != len(ids):
            raise ValueError("RAG sources must not repeat artifacts")
        if ids != tuple(sorted(ids)):
            raise ValueError("RAG sources must be sorted by artifact ID")
        return value

    @field_validator("transition_ids", "fit_lineage_ids", "included_lineage_ids")
    @classmethod
    def _require_sorted_unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("RAG artifact IDs must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("RAG artifact IDs must be sorted")
        return value

    @field_validator("included_partitions")
    @classmethod
    def _require_ordered_unique_partitions(
        cls, value: tuple[Literal["fit", "held_out"], ...]
    ) -> tuple[Literal["fit", "held_out"], ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("RAG included partitions must be non-empty and unique")
        if value != tuple(sorted(value)):
            raise ValueError("RAG included partitions must be sorted")
        return value

    @model_validator(mode="after")
    def _require_consistent_counts_and_inputs(self) -> RAGIndex:
        if self.transition_count != len(self.transition_ids):
            raise ValueError("RAG transition count must match transition IDs")
        source_inputs = tuple(source.artifact_input for source in self.sources)
        if self.inputs != source_inputs:
            raise ValueError("RAG envelope inputs must exactly match source inputs")
        if self.included_lineage_ids and not set(self.fit_lineage_ids).issubset(
            self.included_lineage_ids
        ):
            raise ValueError("RAG included lineages must contain every frozen fit lineage")
        return self


class RAGQuery(ContractModel):
    """Read-only retrieval query with explicit leakage exclusions."""

    task: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    action: RAGAction
    excluded_lineage_ids: tuple[ArtifactId, ...] = ()
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("excluded_lineage_ids")
    @classmethod
    def _require_unique_exclusions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("excluded RAG lineage IDs must be unique")
        return value


class RAGMatch(ContractModel):
    """One retrieved real transition and its cosine-similarity score."""

    transition: RAGTransition
    score: float

    @field_validator("score")
    @classmethod
    def _require_finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("RAG match scores must be finite")
        return value
