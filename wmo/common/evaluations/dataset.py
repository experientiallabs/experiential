"""Canonical judged evaluation rows, datasets, and fidelity reports."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    StructuredFailure,
)
from wmo.common.models import ModelAlias, ModelSnapshot, NumericMeasurement, RoutedCandidateSnapshot


class EvaluationProtocol(ContractModel):
    """Exact identity and pricing provenance for one class of evidence rows."""

    protocol_id: ArtifactId
    evidence_source: Literal["world_model", "sandbox", "production"]
    agent_id: str = Field(min_length=1, max_length=256)
    simulator_id: str = Field(min_length=1, max_length=256)
    world_model: ModelSnapshot | None = None
    simulator_prompt_id: str | None = Field(default=None, max_length=256)
    rubric_id: ArtifactId
    judge_calibration_id: ArtifactId
    pricing_snapshot_id: ArtifactId
    fidelity_report_id: ArtifactId | None = None


class EvaluationRow(ContractModel):
    """One sparse, explicit task, candidate, and repeat measurement."""

    cell_id: ArtifactId
    task_id: ArtifactId
    candidate_alias: ModelAlias
    repeat: int = Field(ge=0)
    protocol_id: ArtifactId
    source_run_id: str | None = Field(default=None, max_length=512)
    purpose: Literal["fit", "held_out", "fidelity"]
    status: Literal["observed", "completed", "failed", "not_run"]
    rollout_id: ArtifactId | None = None
    judgment_id: ArtifactId | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    candidate_cost_usd: NumericMeasurement | None = None
    candidate_latency_seconds: NumericMeasurement | None = None
    world_model_cost_usd: NumericMeasurement | None = None
    sandbox_cost_usd: NumericMeasurement | None = None
    orchestration_cost_usd: NumericMeasurement | None = None
    judge_cost_usd: NumericMeasurement | None = None
    error: StructuredFailure | None = None

    @field_validator("score")
    @classmethod
    def _require_finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("evaluation scores must be finite")
        return value

    @model_validator(mode="after")
    def _require_status_consistency(self) -> EvaluationRow:
        if self.status == "failed" and self.error is None:
            raise ValueError("failed evaluation rows require a structured error")
        if self.status in {"completed", "observed"} and self.rollout_id is None:
            raise ValueError("completed and observed rows require a rollout_id")
        if self.status == "not_run" and self.rollout_id is not None:
            raise ValueError("not-run rows must not reference a rollout")
        return self


class FidelityFailure(ContractModel):
    """A planned overlap that could not provide a usable fidelity pair."""

    cell_id: ArtifactId
    failure: StructuredFailure


class FidelityReport(ArtifactEnvelope):
    """Measured world-model agreement against precommitted observed overlap cells."""

    fidelity_report_id: ArtifactId
    protocol_sha256: Sha256
    overlap_cell_ids: tuple[ArtifactId, ...]
    planned_overlap_count: int = Field(ge=0)
    usable_overlap_count: int = Field(ge=0)
    failed_overlap_count: int = Field(ge=0)
    score_mae: float | None = Field(default=None, ge=0)
    failures: tuple[FidelityFailure, ...] = ()
    gate_id: ArtifactId
    gate_sha256: Sha256
    status: Literal["approved", "rejected", "insufficient"]
    approved_at: datetime | None = None

    @model_validator(mode="after")
    def _require_consistent_fidelity_counts(self) -> FidelityReport:
        if self.usable_overlap_count + self.failed_overlap_count > self.planned_overlap_count:
            raise ValueError("usable and failed overlap counts exceed the planned overlap count")
        if self.status == "approved" and self.approved_at is None:
            raise ValueError("approved fidelity reports require approved_at")
        if self.status != "approved" and self.approved_at is not None:
            raise ValueError("only approved fidelity reports may set approved_at")
        return self


class EvaluationDatasetManifest(ArtifactEnvelope):
    """A frozen manifest for sparse JSONL evaluation rows."""

    evaluation_id: ArtifactId
    evaluation_plan_id: ArtifactId
    task_set_id: ArtifactId
    fit_task_ids: tuple[ArtifactId, ...]
    held_out_task_ids: tuple[ArtifactId, ...]
    candidate_snapshots: tuple[RoutedCandidateSnapshot, ...]
    protocols: tuple[EvaluationProtocol, ...]
    fidelity_report_ids: tuple[ArtifactId, ...] = ()
    rows_path: str = Field(min_length=1)
    rows_sha256: Sha256


class EvaluationDataset(ContractModel):
    """A materialized sparse dataset with a frozen manifest and explicit missing cells."""

    manifest: EvaluationDatasetManifest
    rows: tuple[EvaluationRow, ...]

    @field_validator("rows")
    @classmethod
    def _require_unique_rows(cls, value: tuple[EvaluationRow, ...]) -> tuple[EvaluationRow, ...]:
        cell_ids = tuple(row.cell_id for row in value)
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("evaluation datasets must not repeat a cell ID")
        return value
