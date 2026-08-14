"""Deterministic sparse evaluation planning over immutable production evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import field_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from wmo.common.evaluations.evidence import (
    EvaluationEvidenceError,
    read_evaluation_plan,
    read_fidelity_gate,
    read_fidelity_thresholds,
    read_rollout,
    sorted_evaluation_inputs,
)
from wmo.common.evaluations.plan import (
    EvaluationCell,
    EvaluationPlan,
    FidelityGate,
    FidelityThresholds,
)
from wmo.common.models import RoutedCandidateSnapshot, load_pricing_snapshot
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from wmo.common.rollouts import RolloutArtifact
from wmo.common.routing import RouterFeatureExtractor
from wmo.common.tasks import TaskCase, load_task_set


class ObservedProductionCell(ContractModel):
    """One production rollout assigned to an explicit task, candidate, and repeat cell."""

    task_id: ArtifactId
    candidate_alias: ArtifactId
    repeat: int
    rollout_artifact_id: ArtifactId

    @field_validator("repeat")
    @classmethod
    def _require_nonnegative_repeat(cls, value: int) -> int:
        if value < 0:
            raise ValueError("observed production repeat must be nonnegative")
        return value


def default_fidelity_thresholds(
    *,
    created_at: datetime,
    code_revision: str,
    planned_overlaps: int = 10,
    minimum_usable_overlaps: int = 8,
) -> FidelityThresholds:
    """Create a frozen world-model fidelity gate for an exact real-evidence denominator.

    Args:
        created_at: Time the immutable threshold record is created.
        code_revision: Exact WMO revision creating the gate.
        planned_overlaps: Positive exact number of real fit overlaps frozen into the plan.
        minimum_usable_overlaps: Positive numerical gate no larger than the denominator.

    Returns:
        Exact-denominator 0.10-MAE fidelity thresholds.
    """
    version = (
        "world-model-fidelity-v1"
        if planned_overlaps == 10 and minimum_usable_overlaps == 8
        else "world-model-fidelity-v2"
    )
    thresholds_id = stable_id(
        "fidelity-thresholds",
        {
            "version": version,
            "planned_overlaps": planned_overlaps,
            "minimum_usable_overlaps": minimum_usable_overlaps,
            "maximum_score_mae": 0.10,
        },
    )
    return FidelityThresholds(
        schema_version=1,
        created_at=created_at,
        inputs=(),
        code_revision=code_revision,
        fidelity_thresholds_id=thresholds_id,
        planned_overlaps=planned_overlaps,
        minimum_usable_overlaps=minimum_usable_overlaps,
    )


def persist_fidelity_thresholds(store: ArtifactStore, thresholds: FidelityThresholds) -> None:
    """Persist one immutable fidelity gate in the common project store.

    Args:
        store: Project-local immutable artifact store.
        thresholds: Frozen fidelity thresholds to persist.

    Raises:
        EvaluationEvidenceError: An existing artifact differs from deterministic replay.
    """
    destination = store.project_directory / "artifacts" / thresholds.fidelity_thresholds_id
    if destination.exists():
        existing, _input = read_fidelity_thresholds(store, thresholds.fidelity_thresholds_id)
        replay = thresholds.model_copy(update={"created_at": existing.created_at})
        if existing != replay:
            raise EvaluationEvidenceError(
                "existing fidelity thresholds differ from deterministic replay"
            )
        return
    store.write_json(
        artifact_id=thresholds.fidelity_thresholds_id,
        artifact_type="fidelity-thresholds",
        envelope=thresholds,
        files={"thresholds.json": thresholds},
    )


def build_evaluation_plan(
    store: ArtifactStore,
    *,
    task_set_id: ArtifactId,
    candidate_snapshots: Sequence[RoutedCandidateSnapshot],
    pricing_snapshot_id: ArtifactId,
    observed_cells: Sequence[ObservedProductionCell],
    fidelity_thresholds_id: ArtifactId,
    fidelity_protocol_sha256: Sha256,
    additional_inputs: Sequence[ArtifactInput] = (),
    repeats: Sequence[int] = (0,),
    created_at: datetime,
    code_revision: str,
) -> EvaluationPlan:
    """Build and persist a complete sparse plan from observed and missing cells.

    Observed production evidence takes the main fit or held-out cell. Every other requested
    candidate cell is explicitly marked for simulation. The frozen threshold denominator selects
    that many distinct observed fit lineages for separate fidelity-only simulation cells.

    Args:
        store: Project-local immutable artifact store containing tasks, gate, and rollouts.
        task_set_id: Frozen representative task-set artifact.
        candidate_snapshots: Exact aliases and model identities to evaluate.
        pricing_snapshot_id: Exact candidate-pricing artifact bound to the plan.
        observed_cells: Production rollout assignments that already fill matrix cells.
        fidelity_thresholds_id: Precommitted world-model fidelity thresholds artifact.
        additional_inputs: Optional manifest-bound workflow contracts to freeze into the plan.
        repeats: Nonnegative repeat indexes planned for every task and candidate.
        created_at: Time the plan is completed.
        code_revision: Exact WMO revision creating the plan.

    Returns:
        The persisted immutable evaluation plan.

    Raises:
        EvaluationEvidenceError: Inputs conflict, held-out state leaks, or the exact planned fit
            overlap denominator is unavailable.
    """
    loaded_tasks = load_task_set(store, task_set_id)
    task_manifest = store.read(task_set_id).manifest
    task_input = artifact_input(task_manifest)
    thresholds, thresholds_input = read_fidelity_thresholds(store, fidelity_thresholds_id)
    candidates = tuple(sorted(candidate_snapshots, key=lambda item: item.alias))
    _require_unique_candidates(candidates)
    for item in additional_inputs:
        try:
            current = artifact_input(store.read(item.artifact_id).manifest)
        except (ArtifactCorruptionError, ValueError) as exc:
            raise EvaluationEvidenceError(
                f"additional plan input is unavailable or invalid: {item.artifact_id}"
            ) from exc
        if current != item:
            raise EvaluationEvidenceError(
                f"additional plan input manifest changed: {item.artifact_id}"
            )
    try:
        pricing, pricing_sha256 = load_pricing_snapshot(store, pricing_snapshot_id)
        pricing_input = artifact_input(store.read(pricing_snapshot_id).manifest)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise EvaluationEvidenceError(
            f"required pricing snapshot is unavailable or invalid: {pricing_snapshot_id}"
        ) from exc
    if tuple(item.candidate_alias for item in pricing.candidate_prices) != tuple(
        item.alias for item in candidates
    ):
        raise EvaluationEvidenceError("pricing snapshot candidates differ from the evaluation plan")
    repeat_ids = _normalized_repeats(repeats)
    tasks_by_id = {task.task_id: task for task in loaded_tasks.tasks}
    if len(tasks_by_id) != len(loaded_tasks.tasks):
        raise EvaluationEvidenceError("task set repeats a task ID")
    _require_sealed_lineages(loaded_tasks.tasks)
    candidates_by_alias = {candidate.alias: candidate for candidate in candidates}

    observed_by_key: dict[tuple[str, str, int], RolloutArtifact] = {}
    observed_inputs = []
    for observed in observed_cells:
        key = (observed.task_id, observed.candidate_alias, observed.repeat)
        if key in observed_by_key:
            raise EvaluationEvidenceError(
                "observed production evidence repeats a task, candidate, and repeat cell"
            )
        task = tasks_by_id.get(observed.task_id)
        candidate = candidates_by_alias.get(observed.candidate_alias)
        if task is None or candidate is None or observed.repeat not in repeat_ids:
            raise EvaluationEvidenceError(
                "observed production evidence lies outside the requested task, candidate, or "
                "repeat scope"
            )
        rollout, rollout_input = read_rollout(store, observed.rollout_artifact_id)
        _require_observed_rollout(observed, task, candidate, rollout)
        observed_by_key[key] = rollout
        observed_inputs.append(rollout_input)

    main_cells = tuple(
        _main_cell(
            task,
            candidate,
            repeat,
            observed_by_key.get((task.task_id, candidate.alias, repeat)),
            task_set_input=task_input,
        )
        for task in loaded_tasks.tasks
        for candidate in candidates
        for repeat in repeat_ids
    )
    fidelity_cells = _fidelity_cells(
        main_cells,
        tasks_by_id,
        task_set_input=task_input,
        planned_overlaps=thresholds.planned_overlaps,
    )
    cells = (*main_cells, *fidelity_cells)
    plan_inputs = sorted_evaluation_inputs(
        (
            task_input,
            pricing_input,
            thresholds_input,
            *observed_inputs,
            *additional_inputs,
        )
    )
    plan_id = stable_id(
        "evaluation-plan",
        {
            "version": "sparse-evaluation-plan-v2",
            "inputs": [item.model_dump(mode="json") for item in plan_inputs],
            "task_set": task_input.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "pricing_snapshot": pricing_input.model_dump(mode="json"),
            "fidelity_thresholds": thresholds_input.model_dump(mode="json"),
            "fidelity_protocol_sha256": fidelity_protocol_sha256,
            "cells": [cell.model_dump(mode="json") for cell in cells],
        },
    )
    plan = EvaluationPlan(
        schema_version=2,
        created_at=created_at,
        inputs=plan_inputs,
        code_revision=code_revision,
        plan_id=plan_id,
        task_set_id=task_set_id,
        candidate_snapshots=candidates,
        pricing_snapshot_id=pricing_snapshot_id,
        pricing_snapshot_sha256=pricing_sha256,
        fidelity_thresholds_id=thresholds.fidelity_thresholds_id,
        fidelity_thresholds_sha256=thresholds_input.sha256,
        fidelity_protocol_sha256=fidelity_protocol_sha256,
        cells=cells,
    )
    plan_destination = store.project_directory / "artifacts" / plan.plan_id
    if plan_destination.exists():
        existing, _input = read_evaluation_plan(store, plan.plan_id)
        replay = plan.model_copy(update={"created_at": existing.created_at})
        if existing != replay:
            raise EvaluationEvidenceError(
                "existing evaluation plan differs from deterministic replay"
            )
        plan = existing
    else:
        store.write_json(
            artifact_id=plan.plan_id,
            artifact_type="evaluation-plan",
            envelope=plan,
            files={"evaluation-plan.json": plan, "plan.json": plan},
        )
    plan_input = artifact_input(store.read(plan.plan_id).manifest)
    overlap_ids = tuple(cell.cell_id for cell in fidelity_cells)
    scope_sha256 = sha256_json(
        {
            "task_set_id": plan.task_set_id,
            "candidates": [item.model_dump(mode="json") for item in plan.candidate_snapshots],
            "overlap_cell_ids": overlap_ids,
        }
    )
    gate_id = plan_bound_fidelity_gate_id(plan_input.sha256, fidelity_protocol_sha256)
    gate = FidelityGate(
        schema_version=1,
        created_at=created_at,
        inputs=sorted_evaluation_inputs((plan_input, thresholds_input)),
        code_revision=code_revision,
        fidelity_gate_id=gate_id,
        fidelity_thresholds_id=thresholds.fidelity_thresholds_id,
        fidelity_thresholds_sha256=thresholds_input.sha256,
        evaluation_plan_id=plan.plan_id,
        evaluation_plan_sha256=plan_input.sha256,
        protocol_sha256=fidelity_protocol_sha256,
        task_model_scope_sha256=scope_sha256,
        overlap_cell_ids=overlap_ids,
        planned_overlaps=thresholds.planned_overlaps,
        minimum_usable_overlaps=thresholds.minimum_usable_overlaps,
        maximum_score_mae=thresholds.maximum_score_mae,
    )
    gate_destination = store.project_directory / "artifacts" / gate_id
    if gate_destination.exists():
        existing_gate, _input = read_fidelity_gate(store, gate_id)
        replay_gate = gate.model_copy(update={"created_at": existing_gate.created_at})
        if existing_gate != replay_gate:
            raise EvaluationEvidenceError(
                "existing fidelity gate differs from deterministic replay"
            )
    else:
        store.write_json(
            artifact_id=gate_id,
            artifact_type="fidelity-gate",
            envelope=gate,
            files={"gate.json": gate},
        )
    return plan


def plan_bound_fidelity_gate_id(plan_sha256: Sha256, protocol_sha256: Sha256) -> ArtifactId:
    """Return the non-replayable gate ID for one exact plan and protocol."""
    return stable_id(
        "fidelity-gate",
        {"evaluation_plan_sha256": plan_sha256, "protocol_sha256": protocol_sha256},
    )


def _require_unique_candidates(candidates: tuple[RoutedCandidateSnapshot, ...]) -> None:
    """Reject empty or alias-conflicting candidate snapshots before cell expansion."""
    aliases = tuple(candidate.alias for candidate in candidates)
    if not aliases:
        raise EvaluationEvidenceError("evaluation planning needs at least one candidate")
    if len(set(aliases)) != len(aliases):
        raise EvaluationEvidenceError("evaluation planning candidate aliases must be unique")


def _require_sealed_lineages(tasks: Sequence[TaskCase]) -> None:
    """Reject a task lineage that straddles router fit and held-out partitions."""
    fit_lineages = {task.lineage_group_id for task in tasks if task.partition == "fit"}
    held_out_lineages = {task.lineage_group_id for task in tasks if task.partition == "held_out"}
    leaked = sorted(fit_lineages.intersection(held_out_lineages))
    if leaked:
        raise EvaluationEvidenceError(
            f"router fit and held-out tasks share sealed lineages: {leaked[:3]}"
        )
    extractor = RouterFeatureExtractor()
    fit_fingerprints = {extractor.from_task(task) for task in tasks if task.partition == "fit"}
    held_out_fingerprints = {
        extractor.from_task(task) for task in tasks if task.partition == "held_out"
    }
    if fit_fingerprints.intersection(held_out_fingerprints):
        raise EvaluationEvidenceError(
            "identical normalized request-visible tasks straddle fit and held-out partitions"
        )


def _normalized_repeats(repeats: Sequence[int]) -> tuple[int, ...]:
    """Return canonical nonnegative repeat indexes."""
    normalized = tuple(sorted(repeats))
    if not normalized or normalized[0] < 0 or len(set(normalized)) != len(normalized):
        raise EvaluationEvidenceError("evaluation repeats must be unique nonnegative integers")
    return normalized


def _require_observed_rollout(
    observed: ObservedProductionCell,
    task: TaskCase,
    candidate: RoutedCandidateSnapshot,
    rollout: RolloutArtifact,
) -> None:
    """Require one production rollout to match its explicit observed cell exactly."""
    if rollout.artifact_id != observed.rollout_artifact_id:
        raise EvaluationEvidenceError("stored rollout record does not match its artifact ID")
    if rollout.rollout_id != observed.rollout_artifact_id:
        raise EvaluationEvidenceError("production rollout ID must match its immutable artifact ID")
    if rollout.evidence_source != "production":
        raise EvaluationEvidenceError("observed evaluation cells require production rollouts")
    if rollout.task_id != task.task_id or rollout.repeat != observed.repeat:
        raise EvaluationEvidenceError("production rollout task or repeat does not match its cell")
    if rollout.trace_id not in task.source_trace_ids:
        raise EvaluationEvidenceError(
            "production rollout trace is not bound to the task source lineage"
        )
    if rollout.candidate != candidate.model:
        raise EvaluationEvidenceError(
            "production rollout model identity does not match the candidate snapshot"
        )


def _main_cell(
    task: TaskCase,
    candidate: RoutedCandidateSnapshot,
    repeat: int,
    observed: RolloutArtifact | None,
    *,
    task_set_input: ArtifactInput,
) -> EvaluationCell:
    """Create one observed or missing main matrix cell."""
    purpose = task.partition
    execution = "observed" if observed is not None else "simulate"
    material = {
        "version": "evaluation-cell-v1",
        "task_set": task_set_input.model_dump(mode="json"),
        "task_id": task.task_id,
        "candidate_alias": candidate.alias,
        "candidate": candidate.model.model_dump(mode="json"),
        "repeat": repeat,
        "purpose": purpose,
        "execution": execution,
        "observed_rollout_id": observed.rollout_id if observed is not None else None,
    }
    return EvaluationCell(
        cell_id=stable_id("cell", material),
        task_id=task.task_id,
        candidate_alias=candidate.alias,
        repeat=repeat,
        purpose=purpose,
        execution=execution,
        observed_rollout_id=observed.rollout_id if observed is not None else None,
    )


def _fidelity_cells(
    main_cells: Sequence[EvaluationCell],
    tasks_by_id: dict[str, TaskCase],
    *,
    task_set_input: ArtifactInput,
    planned_overlaps: int,
) -> tuple[EvaluationCell, ...]:
    """Select deterministic observed fit cells from distinct lineages for fidelity only."""
    eligible = [
        cell for cell in main_cells if cell.purpose == "fit" and cell.execution == "observed"
    ]
    ranked = sorted(
        eligible,
        key=lambda cell: stable_id(
            "rank",
            {
                "task_id": cell.task_id,
                "candidate_alias": cell.candidate_alias,
                "repeat": cell.repeat,
                "observed_rollout_id": cell.observed_rollout_id,
            },
        ),
    )
    selected = []
    selected_lineages: set[str] = set()
    selected_tasks: set[str] = set()
    for observed in ranked:
        task = tasks_by_id[observed.task_id]
        if task.task_id in selected_tasks or task.lineage_group_id in selected_lineages:
            continue
        selected.append(observed)
        selected_tasks.add(task.task_id)
        selected_lineages.add(task.lineage_group_id)
        if len(selected) == planned_overlaps:
            break
    if len(selected) != planned_overlaps:
        raise EvaluationEvidenceError(
            f"fidelity planning requires {planned_overlaps} observed fit cells from distinct "
            f"lineages, found {len(selected)}"
        )
    return tuple(
        EvaluationCell(
            cell_id=stable_id(
                "cell",
                {
                    "version": "evaluation-cell-v1",
                    "task_set": task_set_input.model_dump(mode="json"),
                    "task_id": observed.task_id,
                    "candidate_alias": observed.candidate_alias,
                    "repeat": observed.repeat,
                    "purpose": "fidelity",
                    "comparison_observed_cell_id": observed.cell_id,
                },
            ),
            task_id=observed.task_id,
            candidate_alias=observed.candidate_alias,
            repeat=observed.repeat,
            purpose="fidelity",
            execution="simulate",
            comparison_observed_cell_id=observed.cell_id,
        )
        for observed in selected
    )
