"""Canonical sparse evaluation-plan and fidelity-gate contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from wmo.common.models import ModelAlias, RoutedCandidateSnapshot


class EvaluationCell(ContractModel):
    """One explicit task, candidate, and repeat cell in a frozen evaluation plan."""

    cell_id: ArtifactId
    task_id: ArtifactId
    candidate_alias: ModelAlias
    repeat: int = Field(ge=0)
    purpose: Literal["fit", "held_out", "fidelity"]
    execution: Literal["observed", "simulate"]
    observed_rollout_id: ArtifactId | None = None
    comparison_observed_cell_id: ArtifactId | None = None

    @model_validator(mode="after")
    def _require_explicit_evidence_shape(self) -> EvaluationCell:
        if self.execution == "observed" and self.observed_rollout_id is None:
            raise ValueError("observed evaluation cells require observed_rollout_id")
        if self.execution == "simulate" and self.observed_rollout_id is not None:
            raise ValueError("simulated evaluation cells must not name an observed rollout")
        if self.purpose == "fidelity":
            if self.execution != "simulate" or self.comparison_observed_cell_id is None:
                raise ValueError("fidelity cells must simulate against an observed comparison cell")
        elif self.comparison_observed_cell_id is not None:
            raise ValueError("only fidelity cells may name a comparison observed cell")
        return self


class FidelityGate(ArtifactEnvelope):
    """Frozen default threshold for accepting world-model overlap evidence."""

    fidelity_gate_id: ArtifactId
    planned_overlaps: Literal[10] = 10
    minimum_usable_overlaps: Literal[8] = 8
    maximum_score_mae: float = Field(default=0.10, ge=0)


class EvaluationPlan(ArtifactEnvelope):
    """A frozen sparse plan that names every observed and simulated evaluation cell."""

    plan_id: ArtifactId
    task_set_id: ArtifactId
    candidate_snapshots: tuple[RoutedCandidateSnapshot, ...]
    fidelity_gate_id: ArtifactId
    fidelity_gate_sha256: Sha256
    cells: tuple[EvaluationCell, ...]

    @field_validator("candidate_snapshots")
    @classmethod
    def _require_unique_candidates(
        cls, value: tuple[RoutedCandidateSnapshot, ...]
    ) -> tuple[RoutedCandidateSnapshot, ...]:
        aliases = tuple(candidate.alias for candidate in value)
        if not aliases:
            raise ValueError("an evaluation plan needs at least one candidate")
        if len(set(aliases)) != len(aliases):
            raise ValueError("evaluation plan candidate aliases must be unique")
        return value

    @field_validator("cells")
    @classmethod
    def _require_unique_cells(cls, value: tuple[EvaluationCell, ...]) -> tuple[EvaluationCell, ...]:
        cell_ids = tuple(cell.cell_id for cell in value)
        if not cell_ids:
            raise ValueError("an evaluation plan needs at least one cell")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("evaluation plan cell IDs must be unique")
        return value
