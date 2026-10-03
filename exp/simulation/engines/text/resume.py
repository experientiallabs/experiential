"""Immutable rollout loading and strict resume validation for text simulation cells."""

from __future__ import annotations

from dataclasses import dataclass

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    sorted_unique_inputs,
)
from exp.common.evaluations import EvaluationCell, EvaluationPlan
from exp.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from exp.common.rollouts import (
    RolloutArtifact,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    retryable_dispatch_failure,
)
from exp.runtime.models.budget import RequestBudget
from exp.simulation.engines.text.bindings import rollout_id_for_binding
from exp.simulation.engines.text.errors import (
    SimulationConfigurationError,
    SimulationResumeError,
)

ROLLOUT_FILE = "rollout.json"

MAXIMUM_CELL_ATTEMPTS = 3
"""Hard ceiling on immutable re-execution generations per bound simulation cell."""


def interrupted_cell_failure(rollout: RolloutArtifact) -> bool:
    """Recognize exact stale-lease outcomes without changing their saved classification.

    Args:
        rollout: Immutable cell evidence, including legacy budget-classified interruption.

    Returns:
        True only for bound, lease-identified interrupted text-cell outcomes.
    """
    failure = rollout.failure
    if (
        failure is None
        or rollout.simulation_binding is None
        or failure.details.get("phase") != "paid_cell_stale_lease"
        or not isinstance(failure.details.get("lease_id"), str)
        or not failure.details["lease_id"]
    ):
        return False
    return (
        failure.code == FailureCode.CANCELLED
        and failure.attribution == FailureAttribution.ENVIRONMENT
        and failure.retryable
        and rollout.stop_reason == StopReason.FAILURE
    ) or (
        failure.code == FailureCode.BUDGET
        and failure.attribution == FailureAttribution.MODEL
        and not failure.retryable
        and rollout.stop_reason == StopReason.MAXIMUM_COST
    )


def reexecutable_dispatch_failure(
    rollout: RolloutArtifact, *, retry_interrupted: bool = False
) -> bool:
    """Return whether resume would supersede this rollout with another attempt.

    A retryable-class dispatch failure is re-executed only while the cell has generations
    left under ``MAXIMUM_CELL_ATTEMPTS``; the final permitted attempt replays exactly so a
    persistently failing provider converges instead of accumulating artifacts forever.

    Args:
        rollout: Persisted final-attempt rollout for one bound cell.
        retry_interrupted: Current uncapped authority may supersede legacy interruption evidence.

    Returns:
        ``True`` when the rollout is a superseded retryable failure below the attempt cap.
    """
    return (
        retryable_dispatch_failure(rollout.failure)
        or (
            interrupted_cell_failure(rollout)
            and rollout.failure is not None
            and (rollout.failure.retryable or retry_interrupted)
        )
    ) and rollout.retry_attempt + 1 < MAXIMUM_CELL_ATTEMPTS


@dataclass(frozen=True)
class ResumePins:
    """Exact immutable manifest pointers every replayed rollout must carry."""

    plan_input: ArtifactInput
    task_set_input: ArtifactInput
    fit_rag_input: ArtifactInput
    resolution_input: ArtifactInput


def verify_persisted_evaluation_plan(
    store: ArtifactStore,
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_set_id: str,
) -> None:
    """Reject a caller-provided plan object that differs from its immutable manifest input.

    Args:
        store: Project artifact store owning the persisted plan.
        plan: Caller-supplied plan proposed for simulation.
        plan_input: Manifest pointer the plan must exactly match.
        task_set_id: Identity of the immutable task set supplied alongside the plan.

    Raises:
        SimulationConfigurationError: The plan, manifest, or task-set identity disagrees.
    """
    try:
        stored = store.read(plan_input.artifact_id)
        persisted = EvaluationPlan.model_validate_json(
            store.read_bytes(plan_input.artifact_id, "evaluation-plan.json")
        )
    except (ArtifactCorruptionError, ValueError) as exc:
        raise SimulationConfigurationError(
            f"simulation evaluation plan {plan_input.artifact_id!r} cannot be read safely"
        ) from exc
    if stored.manifest.artifact_type != "evaluation-plan" or (
        artifact_input(stored.manifest) != plan_input
    ):
        raise SimulationConfigurationError(
            "evaluation_plan_input must name the exact persisted evaluation-plan manifest"
        )
    if persisted != plan:
        raise SimulationConfigurationError(
            "supplied evaluation plan differs from its immutable persisted manifest"
        )
    if plan.task_set_id != task_set_id:
        raise SimulationConfigurationError(
            "evaluation plan task_set_id does not match the supplied immutable task set"
        )


def load_rollout(store: ArtifactStore, rollout_id: ArtifactId) -> RolloutArtifact:
    """Load a verified rollout or surface malformed immutable data to the caller."""
    stored = store.read(rollout_id)
    if stored.manifest.artifact_type != "rollout":
        raise ArtifactCorruptionError(f"artifact {rollout_id!r} is not a rollout")
    try:
        return RolloutArtifact.model_validate_json(store.read_bytes(rollout_id, ROLLOUT_FILE))
    except (ArtifactCorruptionError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"rollout {rollout_id!r} is not valid canonical evidence"
        ) from exc


def load_optional_rollout(store: ArtifactStore, rollout_id: ArtifactId) -> RolloutArtifact | None:
    """Load an existing rollout while distinguishing absence from immutable corruption."""
    if not store.exists(rollout_id):
        return None
    return load_rollout(store, rollout_id)


def validate_resume_rollout(
    rollout: RolloutArtifact,
    cell: EvaluationCell,
    binding: SimulationCellBinding,
    pins: ResumePins,
    *,
    attempt: int = 0,
) -> None:
    """Validate an existing rollout against every requested immutable pin.

    Args:
        rollout: Previously persisted rollout proposed for exact replay.
        cell: Requested evaluation cell.
        binding: Newly derived complete cell binding.
        pins: Exact manifest pointers required by the replay.
        attempt: Zero-based re-execution generation named by the requested identity.

    Raises:
        SimulationResumeError: Any stable ID, task, mode, RAG, input, or binding differs.
    """
    expected_inputs = sorted_unique_inputs(
        pins.plan_input,
        pins.task_set_input,
        pins.fit_rag_input,
        binding.grounded_world_model_input,
        binding.simulation_spec_input,
        pins.resolution_input,
        *((rollout.continuation_of,) if rollout.continuation_of is not None else ()),
    )
    if (
        rollout.cell_id != cell.cell_id
        or rollout.task_id != cell.task_id
        or rollout.mode != SimulationMode.WORLD_MODEL
        or rollout.rollout_id != rollout_id_for_binding(binding, attempt=attempt)
        or rollout.retry_attempt != attempt
        or rollout.simulation_binding != binding
        or rollout.inputs != expected_inputs
    ):
        raise SimulationResumeError(
            f"stored rollout {rollout.artifact_id!r} does not match the requested simulation cell"
        )


def resolve_cell_attempt(
    store: ArtifactStore,
    cell: EvaluationCell,
    binding: SimulationCellBinding,
    pins: ResumePins,
    *,
    request_budget: RequestBudget | None,
) -> tuple[int, RolloutArtifact | None]:
    """Return the active attempt for one cell and its final rollout, if any.

    Every persisted attempt is validated against the exact immutable binding. A persisted
    retryable-class dispatch failure is superseded evidence, not a final result: the next
    attempt number becomes the active identity so resume can deliberately re-execute the
    cell as a new immutable artifact under fresh budget. Completed rollouts, non-retryable
    failures, and the last permitted generation under ``MAXIMUM_CELL_ATTEMPTS`` replay
    exactly.

    Args:
        store: Project artifact store owning persisted rollouts.
        cell: Requested evaluation cell.
        binding: Complete immutable cell binding.
        pins: Exact manifest pointers required for replay.
        request_budget: Current operator authorization, rechecked before a fresh pricing retry.

    Returns:
        The first attempt without final evidence and ``None``, or an attempt paired with
        its final replayable rollout.

    Raises:
        SimulationResumeError: A persisted attempt mismatches its pins, or a fresh pricing
            retry lacks current uncapped authorization. Existing evidence remains unchanged.
    """
    attempt = 0
    requires_uncapped_budget = False
    previous: RolloutArtifact | None = None
    while True:
        rollout = load_optional_rollout(store, rollout_id_for_binding(binding, attempt=attempt))
        if rollout is None:
            if requires_uncapped_budget and (
                request_budget is None or not request_budget.is_uncapped
            ):
                if (
                    previous is not None
                    and interrupted_cell_failure(previous)
                    and previous.failure is not None
                    and not previous.failure.retryable
                ):
                    return previous.retry_attempt, previous
                raise SimulationResumeError(
                    "retrying unpriceable or truncated responses or interrupted cells requires an "
                    "explicitly uncapped request budget; saved response and rollout are unchanged"
                )
            return attempt, None
        validate_resume_rollout(rollout, cell, binding, pins, attempt=attempt)
        if not reexecutable_dispatch_failure(rollout, retry_interrupted=True):
            return attempt, rollout
        requires_uncapped_budget = (
            requires_uncapped_budget
            or interrupted_cell_failure(rollout)
            or (
                rollout.failure is not None
                and rollout.failure.details.get("retry_classification")
                in {"unpriceable_completed_response", "truncated_completed_response"}
            )
        )
        previous = rollout
        attempt += 1


def persisted_cell_attempts(
    store: ArtifactStore,
    cell: EvaluationCell,
    binding: SimulationCellBinding,
    pins: ResumePins,
) -> tuple[RolloutArtifact, ...]:
    """Return every validated persisted attempt for one bound cell, oldest first.

    Args:
        store: Project artifact store owning persisted rollouts.
        cell: Evaluation cell owning the binding.
        binding: Complete immutable cell binding.
        pins: Exact manifest pointers required for replay.

    Returns:
        All persisted attempts, including superseded retryable failures.

    Raises:
        SimulationResumeError: A persisted attempt does not match its immutable pins.
    """
    attempts: list[RolloutArtifact] = []
    attempt = 0
    while True:
        rollout = load_optional_rollout(store, rollout_id_for_binding(binding, attempt=attempt))
        if rollout is None:
            return tuple(attempts)
        validate_resume_rollout(rollout, cell, binding, pins, attempt=attempt)
        attempts.append(rollout)
        attempt += 1
