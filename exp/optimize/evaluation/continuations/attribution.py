"""Who owns a continuation branch's ending: the actor, the simulator, a provider or the store.

Simulator, provider and store faults are not actor verdicts. A branch that ended on a refused
world prediction, a provider deadline or an artifact-store error says nothing about the
actor's decisions, and a protected value the simulator rewrote under an unrelated action is
not the actor's violation. Every saved branch names the owner in ``failure_detail.attribution``
so mining and qualification can keep those endings out of actor evidence while the raw finding
stays in the record.
"""

from __future__ import annotations

from exp.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from exp.simulation.evaluation.tasks import VerifierGrade

NON_ACTOR = frozenset(
    {FailureAttribution.SIMULATOR, FailureAttribution.PROVIDER, FailureAttribution.STORE}
)
"""Endings that are not the actor's and are masked from actor evidence."""

_SIMULATOR_PHASES = frozenset(
    {
        "initial_state_probe",
        "tool_dispatch_failed_known_effects",
        "tool_dispatch",
        "backend_execution",
        "final_state_capture",
        "world_cleanup",
    }
)
"""Execution phases owned by the simulated world."""


def ending_owner(exception: str | None, phase: str | None) -> FailureAttribution | None:
    """The owner of an execution exception, by its type and then its phase; None if unknown.

    Args:
        exception: The sanitized exception type name.
        phase: The execution phase the branch stopped in.
    """
    if exception is None:
        return None
    if exception == "InvisibleToolCallError":
        return FailureAttribution.ACTOR
    if exception.startswith("Provider"):
        return FailureAttribution.PROVIDER
    if "Store" in exception or exception in ("UnfinishedDispatchIntent", "ArtifactAlreadyExists"):
        return FailureAttribution.STORE
    if exception in ("InvalidPredictionError",) or "Prediction" in exception:
        return FailureAttribution.SIMULATOR
    if phase == "actor_model_dispatch":
        return FailureAttribution.PROVIDER
    if phase in _SIMULATOR_PHASES:
        return FailureAttribution.SIMULATOR
    return None


def branch_attribution(
    assessment: VerifierGrade | None, exception: str | None, phase: str | None
) -> FailureAttribution | None:
    """The owner of a branch's non-pass result; None for a pass.

    A known failure the grader found (the actor's own prohibited call or protected-state
    change, or a goal it left unmet) is the actor's even in an interrupted episode. Otherwise
    the execution ending's owner decides, then simulator drift the grader masked.
    """
    if assessment is not None and assessment.status == "pass":
        return None
    if assessment is not None and assessment.status == "fail":
        return FailureAttribution.ACTOR
    owner = ending_owner(exception, phase)
    if owner is not None:
        return owner
    if assessment is not None and assessment.attribution == "simulator":
        return FailureAttribution.SIMULATOR
    return None


def attributed(
    failure: StructuredFailure | None,
    assessment: VerifierGrade | None,
    exception: str | None,
    phase: str | None,
) -> StructuredFailure | None:
    """The branch's failure detail with its owner named; a graded non-pass gets one if absent.

    Args:
        failure: The executor's failure detail, when the branch stopped on an exception.
        assessment: The branch's grade.
        exception: The sanitized exception type name.
        phase: The execution phase the branch stopped in.

    Returns:
        The detail with ``attribution`` set, None for a passing branch with no failure, or a
        new detail for a completed branch whose grade is not a pass.
    """
    owner = branch_attribution(assessment, exception, phase)
    if failure is not None:
        return failure.model_copy(update={"attribution": owner})
    if assessment is None or assessment.status == "pass":
        return None
    return StructuredFailure(
        code=FailureCode.VALIDATION,
        message=f"Graded {assessment.status}: {assessment.reason}"[:2000],
        attribution=owner,
        details={"phase": phase or "verifier_grade"},
    )
