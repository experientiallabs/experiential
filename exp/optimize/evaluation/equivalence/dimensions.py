"""Mechanical observation of every admission dimension in one saved trajectory.

Observations read only the saved branch, its frozen verifier and the task's declared tools.
They apply no policy and consult no calibration: a simulated reply is reported as such, and
whether it can be trusted is decided later under a registered policy.

Applicability rules (when a dimension can be assessed at all):

- ``goal_success``: the verifier declares an outcome predicate beyond shape and tool policy.
- ``policy_compliance``: the verifier declares an allowed-tool or argument policy.
- ``forbidden_effects``: the verifier declares protected state (``preserve_state`` or
  ``preserved_paths``).
- ``state_fidelity``: the trajectory executed at least one observed tool call. Native world
  receipts on every reply admit it; a simulated reply needs calibration; a reply without a
  receipt binding its exact action is unassessable.
- ``tool_validity``: the trajectory attempted at least one tool call.
- ``recovery``: at least one call failed or returned an error reply.

An inapplicable dimension is ``unassessable`` with ``applicable`` false, never a pass.
"""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, sha256_json
from exp.common.models import ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.policy import DIMENSIONS, Dimension, Verdict
from exp.runtime.environments.mixed import MixedCheckpoint
from exp.runtime.environments.world_outcomes import WorldExecutionReceipt
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.evaluation.admission import has_outcome_constraints
from exp.simulation.evaluation.tasks import Assessment, TaskVerifier, grade

StepOwner = Literal["native", "simulated", "unknown"]
PairAgreement = Literal["match", "different", "unassessable", "inapplicable"]
_VERDICT: dict[str, Verdict] = {"pass": "admitted", "fail": "refused"}


class DimensionObservation(ContractModel):
    """One dimension as observed in one trajectory, before any policy or calibration.

    Attributes:
        dimension: Observed dimension.
        applicable: Whether the dimension's applicability rule holds for this trajectory.
        status: Observed verdict; always ``unassessable`` when not applicable.
        reason: Concrete evidence for the status.
    """

    dimension: Dimension
    applicable: bool
    status: Verdict
    reason: str

    @model_validator(mode="after")
    def inapplicable_is_unassessable(self) -> Self:
        """Keep an inapplicable dimension from ever reading as a pass.

        Returns:
            This observation, unchanged.

        Raises:
            ValueError: An inapplicable dimension carries an admitted or refused status.
        """
        if not self.applicable and self.status != "unassessable":
            raise ValueError("an inapplicable dimension is unassessable, never passed or failed")
        return self


class PairDimension(ContractModel):
    """One dimension observed on both sides of a paired reference/other execution.

    Attributes:
        dimension: Compared dimension.
        reference: Observation on the reference execution.
        other: Observation on the candidate or repeated-reference execution.
        agreement: ``inapplicable`` when neither side applies, ``unassessable`` when an
            applicable side is unassessable, else whether the two labels match. State fidelity
            is ``inapplicable`` when neither side executed a tool, ``unassessable`` when the
            reference's replies are not all native, and otherwise the pair's declared
            material-state agreement instead of the two labels.
        other_only_admitted: Whether the other side admitted what the reference refused.
    """

    dimension: Dimension
    reference: DimensionObservation
    other: DimensionObservation
    agreement: PairAgreement
    other_only_admitted: bool


def step_owner(step: JsonValue) -> StepOwner:
    """Read the authority of one executed step from its exact world receipt.

    Args:
        step: Saved step with its action and the backend's execution receipt.

    Returns:
        The receipt's owner, or unknown when no receipt binds this exact action.
    """
    if not isinstance(step, dict):
        return "unknown"
    raw = step.get("execution")
    if not isinstance(raw, dict):
        return "unknown"
    receipt = WorldExecutionReceipt.model_validate(raw)
    if receipt.action_sha256 != sha256_json(ToolCall.model_validate(step.get("action"))):
        return "unknown"
    return receipt.owner


def world_identity(checkpoint: ContinuationCheckpoint) -> JsonObject:
    """Identify the world a checkpoint runs in by manifest, backend and policy digests.

    Every checkpoint of one backend shares this identity, so a calibration measured on one
    checkpoint applies to continuations of the same world and to no other world.

    Args:
        checkpoint: Continuation checkpoint.

    Returns:
        Secret-free world identity.
    """
    world = checkpoint.world
    if isinstance(world, MixedCheckpoint):
        worlds = {"native": world.native, "simulated": world.candidate}
        return {
            "kind": "mixed",
            "policy_sha256": world.policy_sha256,
            "backends": {
                owner: {
                    "manifest_sha256": point.manifest_sha256,
                    "backend_sha256": point.backend_sha256,
                }
                for owner, point in worlds.items()
                if point is not None
            },
        }
    return {
        "kind": "executable",
        "manifest_sha256": world.manifest_sha256,
        "backend_sha256": world.backend_sha256,
        "owner": world.owner,
    }


def executed_owners(branch: ContinuationBranch) -> tuple[StepOwner, ...]:
    """Return the receipt owner of every observed tool call the branch executed."""
    steps = branch.outcome.get("steps")
    if not isinstance(steps, list):
        return ()
    return tuple(
        step_owner(step)
        for step in steps
        if isinstance(step, dict) and step.get("status") == "valid"
    )


def observe_trajectory(
    branch: ContinuationBranch, verifier: TaskVerifier, tools: Sequence[ToolSchema]
) -> tuple[DimensionObservation, ...]:
    """Observe every dimension of one saved trajectory, in ``DIMENSIONS`` order.

    Args:
        branch: Saved branch, including interrupted or failed executions.
        verifier: Frozen task verifier the study registered.
        tools: Tools the task declared to the actor.

    Returns:
        One observation per dimension.
    """
    outcome = branch.grading_outcome()
    checks = grade(verifier, outcome)
    goal = _goal(verifier, checks.goal)
    observed = {
        "goal_success": goal,
        "policy_compliance": _policy(verifier, outcome),
        "forbidden_effects": _effects(verifier, outcome),
        "state_fidelity": _fidelity(branch),
        "tool_validity": _tools(branch, tools),
        "recovery": _recovery(branch, goal),
    }
    return tuple(observed[dimension] for dimension in DIMENSIONS)


def pair_dimensions(
    reference: ContinuationBranch,
    other: ContinuationBranch,
    *,
    verifier: TaskVerifier,
    tools: Sequence[ToolSchema],
    material_agreement: Literal["match", "different", "unassessable"],
) -> tuple[PairDimension, ...]:
    """Compare every dimension across one paired execution.

    Args:
        reference: Reference execution.
        other: Candidate or repeated-reference execution of the same case.
        verifier: Frozen verifier shared by both sides.
        tools: Tools the task declared.
        material_agreement: Declared material-state agreement of the pair.

    Returns:
        One comparison per dimension, in ``DIMENSIONS`` order.
    """
    rows = []
    pairs = zip(
        observe_trajectory(reference, verifier, tools),
        observe_trajectory(other, verifier, tools),
        strict=True,
    )
    for left, right in pairs:
        agreement: PairAgreement
        if left.dimension == "state_fidelity":
            agreement = _fidelity_agreement(left, right, material_agreement)
        elif not left.applicable and not right.applicable:
            agreement = "inapplicable"
        elif any(side.applicable and side.status == "unassessable" for side in (left, right)):
            agreement = "unassessable"
        else:
            agreement = "match" if _label(left) == _label(right) else "different"
        rows.append(
            PairDimension(
                dimension=left.dimension,
                reference=left,
                other=right,
                agreement=agreement,
                other_only_admitted=agreement == "different"
                and left.dimension != "state_fidelity"
                and left.status == "refused"
                and right.status == "admitted",
            )
        )
    return tuple(rows)


def _fidelity_agreement(
    reference: DimensionObservation,
    other: DimensionObservation,
    material_agreement: Literal["match", "different", "unassessable"],
) -> PairAgreement:
    """Compare state fidelity only where a tool ran and the reference's replies are native.

    A pair in which neither side executed a tool never exercised the world, so its agreement says
    nothing about fidelity. A reference reply without a native receipt for its exact action is
    not truth, so the pair cannot be assessed against it.
    """
    if not reference.applicable and not other.applicable:
        return "inapplicable"
    if reference.applicable and reference.status != "admitted":
        return "unassessable"
    return material_agreement


def _label(observation: DimensionObservation) -> str:
    """Name an assessable side for agreement: its status, or inapplicable."""
    return observation.status if observation.applicable else "inapplicable"


def _observation(
    dimension: Dimension, assessment: Assessment | None, reason: str | None = None
) -> DimensionObservation:
    """Map a verifier assessment, or an inapplicable rule with its reason, to an observation."""
    if assessment is None:
        assert reason is not None
        return DimensionObservation(
            dimension=dimension, applicable=False, status="unassessable", reason=reason
        )
    return DimensionObservation(
        dimension=dimension,
        applicable=True,
        status=_VERDICT.get(assessment.status, "unassessable"),
        reason=assessment.reason,
    )


def _goal(verifier: TaskVerifier, goal: Assessment) -> DimensionObservation:
    """Observe the declared goal, which applies only with a real outcome predicate."""
    if not has_outcome_constraints(verifier):
        return _observation(
            "goal_success",
            None,
            "not applicable: the verifier declares no outcome predicate beyond shape and tools",
        )
    return _observation("goal_success", goal)


def _policy(verifier: TaskVerifier, outcome: JsonObject) -> DimensionObservation:
    """Observe the declared tool and argument policy, separately from protected state."""
    if (
        verifier.allowed_tools is None
        and not verifier.argument_rules
        and not verifier.argument_schemas
    ):
        return _observation(
            "policy_compliance",
            None,
            "not applicable: the verifier declares no allowed-tool or argument policy",
        )
    actions_only = verifier.model_copy(update={"preserve_state": False, "preserved_paths": ()})
    return _observation("policy_compliance", grade(actions_only, outcome).policy)


def _effects(verifier: TaskVerifier, outcome: JsonObject) -> DimensionObservation:
    """Observe protected state across every observed transition, separately from tool policy."""
    if not verifier.preserve_state and not verifier.preserved_paths:
        return _observation(
            "forbidden_effects",
            None,
            "not applicable: the verifier declares no protected state",
        )
    state_only = verifier.model_copy(
        update={"allowed_tools": None, "argument_rules": {}, "argument_schemas": {}}
    )
    return _observation("forbidden_effects", grade(state_only, outcome).policy)


def _fidelity(branch: ContinuationBranch) -> DimensionObservation:
    """Observe which world authority answered each executed call."""
    owners = executed_owners(branch)
    if not owners:
        return _observation(
            "state_fidelity", None, "not applicable: no tool call was executed and observed"
        )
    if "unknown" in owners:
        reason = (
            "a reply has no world receipt binding its exact action, so its authority is unproven"
        )
    elif "simulated" in owners:
        reason = (
            f"{owners.count('simulated')} of {len(owners)} replies came from a simulated owner; "
            "their fidelity is known only from calibration"
        )
    else:
        return DimensionObservation(
            dimension="state_fidelity",
            applicable=True,
            status="admitted",
            reason=f"all {len(owners)} replies carry a native world receipt",
        )
    return DimensionObservation(
        dimension="state_fidelity", applicable=True, status="unassessable", reason=reason
    )


def _tools(branch: ContinuationBranch, tools: Sequence[ToolSchema]) -> DimensionObservation:
    """Observe whether every attempted call named a declared tool and executed."""
    steps = branch.outcome.get("steps")
    if not isinstance(steps, list):
        return DimensionObservation(
            dimension="tool_validity",
            applicable=True,
            status="unassessable",
            reason="the branch has no saved step inventory",
        )
    if not steps:
        return _observation("tool_validity", None, "not applicable: no tool call was attempted")
    declared = {tool.name for tool in tools}
    refusals, unknown, errors = [], [], 0
    for index, step in enumerate(steps):
        action = step.get("action") if isinstance(step, dict) else None
        name = action.get("name") if isinstance(action, dict) else None
        status = step.get("status") if isinstance(step, dict) else None
        if not isinstance(name, str):
            unknown.append(f"step {index} has no action identity")
        elif name not in declared:
            refusals.append(f"step {index} called undeclared tool {name}")
        elif status == "blocked":
            refusals.append(f"step {index} was blocked before dispatch")
        elif status == "failure":
            if _known_rollback(step):
                refusals.append(f"step {index} failed without committing effects")
            else:
                unknown.append(f"step {index} failed with unknown effects")
        elif status != "valid":
            unknown.append(f"step {index} has unobserved status {status}")
        elif _is_error(step):
            errors += 1
    if refusals:
        reason, verdict = "; ".join(refusals), "refused"
    elif unknown:
        reason, verdict = "; ".join(unknown), "unassessable"
    elif branch.status != "completed":
        reason, verdict = (
            "execution did not complete, so later calls are unobserved",
            "unassessable",
        )
    else:
        reason = f"{len(steps)} calls to declared tools executed"
        if errors:
            reason += f"; {errors} error replies are assessed under recovery"
        verdict = "admitted"
    return DimensionObservation(
        dimension="tool_validity", applicable=True, status=verdict, reason=reason
    )


def _recovery(branch: ContinuationBranch, goal: DimensionObservation) -> DimensionObservation:
    """Observe whether the actor continued to a passing goal after its last tool error."""
    steps = branch.outcome.get("steps")
    rows = steps if isinstance(steps, list) else []
    errors = [
        index
        for index, step in enumerate(rows)
        if isinstance(step, dict)
        and (_is_error(step) or (step.get("status") == "failure" and _known_rollback(step)))
    ]
    if not errors:
        return _observation("recovery", None, "not applicable: no tool call failed or errored")
    last = errors[-1]
    retried = any(
        isinstance(step, dict) and step.get("status") == "valid" and not _is_error(step)
        for step in rows[last + 1 :]
    )
    if goal.status == "refused":
        verdict: Verdict = "refused"
        reason = f"the goal failed after the error at step {last}"
    elif goal.status == "unassessable":
        verdict, reason = "unassessable", "recovery needs an assessable goal: " + goal.reason
    elif not retried:
        verdict = "refused"
        reason = f"no successful call followed the last tool error at step {last}"
    else:
        verdict = "admitted"
        reason = (
            f"a successful call followed the last tool error at step {last} and the goal passed"
        )
    return DimensionObservation(
        dimension="recovery", applicable=True, status=verdict, reason=reason
    )


def _is_error(step: JsonObject) -> bool:
    """Whether a step's recorded reply reported an error."""
    reply = step.get("reply")
    return isinstance(reply, dict) and reply.get("is_error") is True


def _known_rollback(step: JsonObject) -> bool:
    """Whether a failed step's receipt proves no effect was committed."""
    raw = step.get("execution")
    if not isinstance(raw, dict):
        return False
    return WorldExecutionReceipt.model_validate(raw).effects in ("not_started", "rolled_back")
