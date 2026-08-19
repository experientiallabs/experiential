"""Canonical judged evaluation rows, datasets, and fidelity reports."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    StructuredFailure,
    validate_artifact_file_path,
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

    @model_validator(mode="after")
    def _require_source_specific_identity(self) -> EvaluationProtocol:
        if self.evidence_source == "world_model":
            if self.world_model is None or self.simulator_prompt_id is None:
                raise ValueError("world-model protocols require world-model and prompt identities")
        elif self.world_model is not None or self.simulator_prompt_id is not None:
            raise ValueError("sandbox and production protocols must not name world-model identity")
        return self


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
    retrieval_cost_usd: NumericMeasurement | None = None
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
        """Require status-specific evidence and economics fields to remain consistent.

        Returns:
            The validated row when its status agrees with every execution field.

        Raises:
            ValueError: Required evidence is absent or forbidden evidence is present.
        """
        if self.status == "failed" and self.error is None:
            raise ValueError("failed evaluation rows require a structured error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("only failed evaluation rows may contain a structured error")
        if self.status in {"completed", "observed"} and self.rollout_id is None:
            raise ValueError("completed and observed rows require a rollout_id")
        if self.status != "not_run" and self.source_run_id is None:
            raise ValueError("started evaluation rows require a source_run_id")
        if (self.judgment_id is None) != (self.score is None):
            raise ValueError("evaluation scores and judgment IDs must be set together")
        if self.status == "failed" and (self.judgment_id is not None or self.score is not None):
            raise ValueError("failed evaluation rows must not contain a judgment or score")
        if self.status == "not_run":
            mutable_fields = (
                self.rollout_id,
                self.judgment_id,
                self.score,
                self.error,
                self.source_run_id,
                self.candidate_cost_usd,
                self.candidate_latency_seconds,
                self.world_model_cost_usd,
                self.retrieval_cost_usd,
                self.sandbox_cost_usd,
                self.orchestration_cost_usd,
                self.judge_cost_usd,
            )
            if any(value is not None for value in mutable_fields):
                raise ValueError("not-run rows must not contain execution evidence")
        return self


class FidelityFailure(ContractModel):
    """A planned overlap that could not provide a usable fidelity pair."""

    cell_id: ArtifactId
    failure: StructuredFailure


class FidelityPair(ContractModel):
    """One planned observed versus simulated overlap with exact rollout identities."""

    fidelity_cell_id: ArtifactId
    observed_cell_id: ArtifactId
    observed_rollout_id: ArtifactId | None = None
    simulated_rollout_id: ArtifactId | None = None
    observed_score: float | None = Field(default=None, ge=0, le=1)
    simulated_score: float | None = Field(default=None, ge=0, le=1)
    absolute_error: float | None = Field(default=None, ge=0)
    status: Literal["usable", "failed", "not_run"]
    error: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_pair_status(self) -> FidelityPair:
        metrics = (self.observed_score, self.simulated_score, self.absolute_error)
        if self.status == "usable":
            if self.observed_rollout_id is None or self.simulated_rollout_id is None:
                raise ValueError("usable fidelity pairs require both rollout IDs")
            if any(value is None for value in metrics) or self.error is not None:
                raise ValueError("usable fidelity pairs require metrics and no error")
        elif self.error is None or any(value is not None for value in metrics):
            raise ValueError("unusable fidelity pairs require one error and no metrics")
        return self


class FidelityReport(ArtifactEnvelope):
    """Measured world-model agreement against frozen observed overlap cells."""

    fidelity_report_id: ArtifactId
    evaluation_plan_id: ArtifactId
    evaluation_plan_sha256: Sha256
    protocol_sha256: Sha256
    overlap_cell_ids: tuple[ArtifactId, ...]
    planned_overlap_count: int = Field(ge=0)
    usable_overlap_count: int = Field(ge=0)
    failed_overlap_count: int = Field(ge=0)
    score_mae: float | None = Field(default=None, ge=0)
    failures: tuple[FidelityFailure, ...] = ()
    pairs: tuple[FidelityPair, ...]

    @field_validator("overlap_cell_ids")
    @classmethod
    def _require_unique_overlap_cells(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        if len(set(value)) != len(value):
            raise ValueError("fidelity reports must not repeat overlap cells")
        return value

    @field_validator("score_mae")
    @classmethod
    def _require_finite_score_mae(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("fidelity score MAE must be finite")
        return value

    @model_validator(mode="after")
    def _require_consistent_fidelity_counts(self) -> FidelityReport:
        if len(self.overlap_cell_ids) != self.planned_overlap_count:
            raise ValueError("fidelity overlap cells must match the planned overlap count")
        pair_cell_ids = tuple(pair.fidelity_cell_id for pair in self.pairs)
        if pair_cell_ids != self.overlap_cell_ids or len(set(pair_cell_ids)) != len(pair_cell_ids):
            raise ValueError("fidelity pairs must account for every planned overlap exactly once")
        usable_pairs = sum(pair.status == "usable" for pair in self.pairs)
        if usable_pairs != self.usable_overlap_count:
            raise ValueError("usable fidelity pair records must match the usable count")
        usable_errors = tuple(pair.absolute_error for pair in self.pairs if pair.status == "usable")
        measured_mae = (
            sum(error for error in usable_errors if error is not None) / len(usable_errors)
            if usable_errors
            else None
        )
        if measured_mae is None:
            if self.score_mae is not None:
                raise ValueError("fidelity score MAE requires at least one usable pair")
        elif self.score_mae is None or not math.isclose(
            self.score_mae,
            measured_mae,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("fidelity score MAE must equal the mean usable pair error")
        if self.usable_overlap_count + self.failed_overlap_count != self.planned_overlap_count:
            raise ValueError(
                "usable and failed overlap counts must match the planned overlap count"
            )
        failure_cell_ids = tuple(failure.cell_id for failure in self.failures)
        if len(set(failure_cell_ids)) != len(failure_cell_ids):
            raise ValueError("fidelity failures must not repeat an overlap cell")
        if not set(failure_cell_ids).issubset(self.overlap_cell_ids):
            raise ValueError("fidelity failures must name planned overlap cells")
        if len(self.failures) != self.failed_overlap_count:
            raise ValueError("fidelity failure records must match the failed overlap count")
        return self


class EvaluationDatasetManifest(ArtifactEnvelope):
    """A frozen manifest for sparse JSONL evaluation rows."""

    evaluation_id: ArtifactId
    evaluation_plan_id: ArtifactId
    evaluation_plan_sha256: Sha256
    task_set_id: ArtifactId
    fit_task_ids: tuple[ArtifactId, ...]
    held_out_task_ids: tuple[ArtifactId, ...]
    candidate_snapshots: tuple[RoutedCandidateSnapshot, ...]
    protocols: tuple[EvaluationProtocol, ...]
    rows_path: str = Field(min_length=1)
    rows_sha256: Sha256

    @field_validator("rows_path")
    @classmethod
    def _require_safe_rows_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @model_validator(mode="after")
    def _require_consistent_dataset_scope(self) -> EvaluationDatasetManifest:
        fit_task_ids = set(self.fit_task_ids)
        held_out_task_ids = set(self.held_out_task_ids)
        if len(fit_task_ids) != len(self.fit_task_ids):
            raise ValueError("evaluation manifest fit task IDs must not repeat")
        if len(held_out_task_ids) != len(self.held_out_task_ids):
            raise ValueError("evaluation manifest held-out task IDs must not repeat")
        if fit_task_ids.intersection(held_out_task_ids):
            raise ValueError("evaluation manifest fit and held-out task IDs must be disjoint")
        candidate_aliases = tuple(candidate.alias for candidate in self.candidate_snapshots)
        if not candidate_aliases or len(set(candidate_aliases)) != len(candidate_aliases):
            raise ValueError("evaluation manifest candidate aliases must be non-empty and unique")
        protocol_ids = tuple(protocol.protocol_id for protocol in self.protocols)
        if not protocol_ids or len(set(protocol_ids)) != len(protocol_ids):
            raise ValueError("evaluation manifest protocol IDs must be non-empty and unique")
        return self


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

    @model_validator(mode="after")
    def _require_rows_to_match_manifest_scope(self) -> EvaluationDataset:
        fit_task_ids = set(self.manifest.fit_task_ids)
        held_out_task_ids = set(self.manifest.held_out_task_ids)
        candidate_aliases = {candidate.alias for candidate in self.manifest.candidate_snapshots}
        protocol_ids = {protocol.protocol_id for protocol in self.manifest.protocols}
        for row in self.rows:
            if row.task_id not in fit_task_ids.union(held_out_task_ids):
                raise ValueError(f"evaluation row {row.cell_id} names a task outside the manifest")
            if row.candidate_alias not in candidate_aliases:
                raise ValueError(
                    f"evaluation row {row.cell_id} names a candidate outside the manifest"
                )
            if row.protocol_id not in protocol_ids:
                raise ValueError(
                    f"evaluation row {row.cell_id} names a protocol outside the manifest"
                )
            if row.purpose == "held_out" and row.task_id not in held_out_task_ids:
                raise ValueError("held-out evaluation rows must name held-out tasks")
            if row.purpose in {"fit", "fidelity"} and row.task_id not in fit_task_ids:
                raise ValueError("fit and fidelity evaluation rows must name fit tasks")
        return self
