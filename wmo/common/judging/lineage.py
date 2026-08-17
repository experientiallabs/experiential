"""Frozen router-lineage split evidence used by judging calibration."""

from __future__ import annotations

from pydantic import field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    canonical_json_bytes,
)
from wmo.common.judging.provenance import JudgingProvenanceError, resolve_artifact
from wmo.common.project import ArtifactCorruptionError, ProjectStore


class RouterLineageAssignment(ContractModel):
    """Binds one source rollout to its frozen router leakage lineage."""

    rollout_id: ArtifactId
    lineage_id: ArtifactId


class RouterLineageSplit(ArtifactEnvelope):
    """Immutable fit and held-out router partitions with rollout-to-lineage evidence."""

    split_id: ArtifactId
    source_task_set_id: ArtifactId
    fit_lineage_ids: tuple[ArtifactId, ...]
    held_out_lineage_ids: tuple[ArtifactId, ...]
    assignments: tuple[RouterLineageAssignment, ...]

    def lineage_for_rollout(self, rollout_id: ArtifactId) -> ArtifactId:
        """Return the frozen lineage assigned to one rollout.

        Args:
            rollout_id: Source rollout whose calibration lineage is required.

        Returns:
            The lineage identifier recorded by this immutable split.

        Raises:
            ValueError: The split has no assignment for the source rollout.
        """
        for assignment in self.assignments:
            if assignment.rollout_id == rollout_id:
                return assignment.lineage_id
        raise ValueError(f"router lineage split has no assignment for rollout {rollout_id}")

    @field_validator("fit_lineage_ids", "held_out_lineage_ids")
    @classmethod
    def _require_unique_lineages(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        """Reject duplicate lineage IDs in one frozen router partition."""
        if len(set(value)) != len(value):
            raise ValueError("router lineage IDs must not repeat")
        return value

    @field_validator("fit_lineage_ids", "held_out_lineage_ids")
    @classmethod
    def _require_sorted_lineages(cls, value: tuple[ArtifactId, ...]) -> tuple[ArtifactId, ...]:
        """Require deterministic partition ordering in persisted lineage evidence."""
        if value != tuple(sorted(value)):
            raise ValueError("router lineage IDs must be sorted")
        return value

    @field_validator("assignments")
    @classmethod
    def _require_sorted_assignments(
        cls, value: tuple[RouterLineageAssignment, ...]
    ) -> tuple[RouterLineageAssignment, ...]:
        """Require one deterministically ordered assignment for each source rollout."""
        rollout_ids = tuple(item.rollout_id for item in value)
        if not value:
            raise ValueError("router lineage splits require at least one rollout assignment")
        if len(set(rollout_ids)) != len(rollout_ids):
            raise ValueError("router lineage splits must not repeat rollout assignments")
        if rollout_ids != tuple(sorted(rollout_ids)):
            raise ValueError("router lineage assignments must be sorted by rollout ID")
        return value

    @model_validator(mode="after")
    def _require_coherent_assignments(self) -> RouterLineageSplit:
        """Require every rollout assignment to belong to exactly one frozen partition."""
        fit = set(self.fit_lineage_ids)
        held_out = set(self.held_out_lineage_ids)
        if not fit:
            raise ValueError("router lineage splits require at least one fit lineage")
        if fit.intersection(held_out):
            raise ValueError("router fit and held-out lineage IDs must be disjoint")
        known = fit | held_out
        if any(assignment.lineage_id not in known for assignment in self.assignments):
            raise ValueError("router lineage assignments must use a frozen partition lineage")
        if tuple(item.artifact_id for item in self.inputs) != (self.source_task_set_id,):
            raise ValueError("router lineage splits must hash exactly their source task set")
        return self


def write_router_lineage_split(
    store: ProjectStore, split: RouterLineageSplit
) -> RouterLineageSplit:
    """Persist or verify one immutable router-lineage split artifact.

    Args:
        store: Project-local store that owns the completed split artifact.
        split: Frozen partition and rollout lineage assignments to persist.

    Returns:
        The stored split, including a safe idempotent retry result.

    Raises:
        JudgingProvenanceError: An existing artifact conflicts with the supplied split.
    """
    if len(split.inputs) != 1:
        raise JudgingProvenanceError(
            "router lineage splits must provide one source task-set manifest input"
        )
    try:
        _task_set, verified_task_set_input = resolve_artifact(
            store,
            artifact_id=split.source_task_set_id,
            expected_artifact_type="task-set",
            expected_input=split.inputs[0],
        )
    except JudgingProvenanceError as exc:
        raise JudgingProvenanceError(
            "router lineage splits require a completed immutable source task set"
        ) from exc
    if split.inputs != (verified_task_set_input,):
        raise JudgingProvenanceError(
            "router lineage split inputs are not the verified source task-set manifest"
        )
    try:
        existing, _ = store.artifacts.write_or_replay(
            artifact_id=split.split_id,
            artifact_type="router-lineage-split",
            envelope=split,
            envelope_path="split.json",
            envelope_type=RouterLineageSplit,
            files={"split.json": canonical_json_bytes(split)},
        )
    except ArtifactCorruptionError as exc:
        raise JudgingProvenanceError(
            "existing router-lineage split artifact cannot be resumed safely"
        ) from exc
    except ValueError as exc:
        raise JudgingProvenanceError(
            "existing router-lineage split artifact conflicts with this split"
        ) from exc
    return existing
