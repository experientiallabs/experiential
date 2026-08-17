"""Canonical sparse evaluation-plan contracts."""

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
        """Validate the evidence references required by this cell's execution mode.

        Returns:
            The validated evaluation cell.

        Raises:
            ValueError: The execution mode and evidence references disagree.
        """
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


class EvaluationPlan(ArtifactEnvelope):
    """A frozen sparse plan that names every observed and simulated evaluation cell."""

    plan_id: ArtifactId
    task_set_id: ArtifactId
    candidate_snapshots: tuple[RoutedCandidateSnapshot, ...]
    pricing_snapshot_id: ArtifactId
    pricing_snapshot_sha256: Sha256
    fidelity_protocol_sha256: Sha256 | None = None
    cells: tuple[EvaluationCell, ...]

    @field_validator("candidate_snapshots")
    @classmethod
    def _require_unique_candidates(
        cls, value: tuple[RoutedCandidateSnapshot, ...]
    ) -> tuple[RoutedCandidateSnapshot, ...]:
        """Require a nonempty candidate sequence with unique aliases.

        Args:
            value: Candidate snapshots in plan order.

        Returns:
            The validated candidate snapshots.

        Raises:
            ValueError: The sequence is empty or repeats an alias.
        """
        aliases = tuple(candidate.alias for candidate in value)
        if not aliases:
            raise ValueError("an evaluation plan needs at least one candidate")
        if len(set(aliases)) != len(aliases):
            raise ValueError("evaluation plan candidate aliases must be unique")
        return value

    @field_validator("cells")
    @classmethod
    def _require_unique_cells(cls, value: tuple[EvaluationCell, ...]) -> tuple[EvaluationCell, ...]:
        """Require a nonempty cell sequence with unique cell identities.

        Args:
            value: Evaluation cells in plan order.

        Returns:
            The validated evaluation cells.

        Raises:
            ValueError: The sequence is empty or repeats a cell identity.
        """
        cell_ids = tuple(cell.cell_id for cell in value)
        if not cell_ids:
            raise ValueError("an evaluation plan needs at least one cell")
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("evaluation plan cell IDs must be unique")
        return value

    @model_validator(mode="after")
    def _require_consistent_cell_references(self) -> EvaluationPlan:
        """Validate candidate scope, unique coordinates, and fidelity comparisons.

        Returns:
            The validated evaluation plan.

        Raises:
            ValueError: A cell falls outside the plan scope or has invalid references.
        """
        fidelity_cells = tuple(cell for cell in self.cells if cell.purpose == "fidelity")
        if bool(fidelity_cells) != (self.fidelity_protocol_sha256 is not None):
            raise ValueError("fidelity cells and protocol identity must be present together")
        candidate_aliases = {candidate.alias for candidate in self.candidate_snapshots}
        cells_by_id = {cell.cell_id: cell for cell in self.cells}
        planned_cell_keys: set[tuple[ArtifactId, ModelAlias, int, str]] = set()
        for cell in self.cells:
            if cell.candidate_alias not in candidate_aliases:
                raise ValueError(
                    f"evaluation cell {cell.cell_id} names a candidate outside the plan snapshots"
                )
            cell_key = (cell.task_id, cell.candidate_alias, cell.repeat, cell.purpose)
            if cell_key in planned_cell_keys:
                raise ValueError(
                    "evaluation plan must not repeat a task, candidate, repeat, and purpose cell"
                )
            planned_cell_keys.add(cell_key)
            if cell.purpose != "fidelity":
                continue
            comparison = cells_by_id.get(cell.comparison_observed_cell_id)
            if comparison is None or comparison.execution != "observed":
                raise ValueError("fidelity cells must reference an observed cell in the same plan")
            if comparison.purpose != "fit":
                raise ValueError("fidelity cells must compare against observed fit evidence")
            if (comparison.task_id, comparison.candidate_alias, comparison.repeat) != (
                cell.task_id,
                cell.candidate_alias,
                cell.repeat,
            ):
                raise ValueError(
                    "fidelity cells must preserve the compared task, candidate, and repeat"
                )
        return self
