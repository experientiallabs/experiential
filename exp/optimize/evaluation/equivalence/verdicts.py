"""Admit one saved trajectory under a registered policy and any in-scope calibration.

Each dimension starts from its mechanical observation (``dimensions.py``). A trajectory with a
reply that no native world receipt proves is a simulated observation: a dimension declared
``calibrated`` is admitted only when a calibration under the same policy covers the
trajectory's unit, world and actor and admits that dimension. Calibration can withhold an
admission, never rescue a refusal. ``state_fidelity`` of a simulated trajectory is decided by
calibration alone, and a calibration that refuses it refuses the trajectory's fidelity. Any
covering refusal wins over covering admissions, whatever the order calibrations are supplied in.

The trajectory verdict follows the gates: refused when a gating dimension is refused,
unassessable when one is unassessable, admitted only when every gating dimension is admitted.
"""

from collections.abc import Sequence

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json
from exp.common.models import ModelSnapshot
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.equivalence.calibration import PolicyCalibration
from exp.optimize.evaluation.equivalence.dimensions import (
    DimensionObservation,
    executed_owners,
    observe_trajectory,
    world_identity,
)
from exp.optimize.evaluation.equivalence.policy import (
    AdmissionPolicy,
    Dimension,
    Gate,
    Verdict,
)
from exp.optimize.evaluation.equivalence.runner import load_equivalence_report
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.evaluation.tasks import TaskVerifier


class CalibrationEvidence(ContractModel):
    """A saved comparison report and its calibration under one policy.

    Attributes:
        report: Exact saved ``actor-equivalence-report`` reference.
        calibration: The report's policy calibration.
    """

    report: ArtifactInput
    calibration: PolicyCalibration


class DimensionVerdict(ContractModel):
    """One dimension's admission verdict for one trajectory.

    Attributes:
        dimension: Judged dimension.
        gate: The policy's gate for this dimension.
        applicable: Whether the dimension's applicability rule held.
        observed: Mechanical observation before calibration.
        status: Verdict after the policy's calibration requirement.
        reason: Concrete evidence for the verdict.
        calibration: Comparison report consulted, when calibration decided the verdict.
    """

    dimension: Dimension
    gate: Gate
    applicable: bool
    observed: Verdict
    status: Verdict
    reason: str
    calibration: str | None = None

    def gates(self) -> bool:
        """Whether this verdict decides the trajectory's admission under its gate."""
        return self.gate == "required" or (self.gate == "when_applicable" and self.applicable)


class TrajectoryAdmission(ContractModel):
    """Per-dimension and overall admission of one trajectory under a registered policy.

    Attributes:
        policy: Registered policy applied.
        status: Admitted only when every gating dimension is admitted.
        simulated: Whether a reply lacked a native world receipt.
        dimensions: One verdict per dimension, in policy order.
        reasons: Why each gating dimension was not admitted.
    """

    policy: ArtifactInput
    status: Verdict
    simulated: bool
    dimensions: tuple[DimensionVerdict, ...]
    reasons: tuple[str, ...] = ()


def policy_calibrations(
    project: ProjectStore, reports: Sequence[ArtifactInput], policy: ArtifactInput
) -> tuple[CalibrationEvidence, ...]:
    """Load explicitly supplied comparison reports and require their calibration under a policy.

    Args:
        project: Project holding the reports.
        reports: Saved ``compare-environments`` report references.
        policy: Registered policy the consumer applies.

    Returns:
        One calibration per report, in the given order.

    Raises:
        ValueError: A report does not reproduce or was not registered under this policy.
    """
    found = []
    for reference in reports:
        report = load_equivalence_report(project, reference)
        if report.calibration is None or report.calibration.policy != policy:
            raise ValueError(
                f"comparison {reference.artifact_id} was not registered under admission policy "
                f"{policy.artifact_id}; register a comparison with this policy before its "
                "executions"
            )
        found.append(CalibrationEvidence(report=reference, calibration=report.calibration))
    return tuple(found)


def admit_trajectory(
    policy_ref: ArtifactInput,
    policy: AdmissionPolicy,
    branch: ContinuationBranch,
    *,
    checkpoint: ContinuationCheckpoint,
    verifier: TaskVerifier,
    actor_model: ModelSnapshot,
    calibrations: Sequence[CalibrationEvidence] = (),
) -> TrajectoryAdmission:
    """Judge every dimension of one saved trajectory and decide its admission.

    Args:
        policy_ref: Registered policy reference.
        policy: The registered policy.
        branch: Saved branch, including failed and interrupted ones.
        checkpoint: Checkpoint the branch resumed from: its task, tools and world.
        verifier: Frozen verifier the study registered.
        actor_model: Actor model snapshot the study registered.
        calibrations: Calibrations under this policy that may cover the trajectory.

    Returns:
        Per-dimension verdicts and the gated trajectory verdict.
    """
    owners = executed_owners(branch)
    simulated = any(owner != "native" for owner in owners)
    world = sha256_json(world_identity(checkpoint))
    actor = sha256_json(actor_model)
    lineage = checkpoint.task.lineage_group_id
    verdicts = []
    for observed in observe_trajectory(branch, verifier, checkpoint.task.tools):
        rule = policy.rule(observed.dimension)
        status, reason, used = observed.status, observed.reason, None
        fidelity = observed.dimension == "state_fidelity"
        if (
            simulated
            and observed.applicable
            and rule.simulated == "calibrated"
            and (observed.status == "admitted" or (fidelity and "unknown" not in owners))
        ):
            status, reason, used = _calibrated(
                observed,
                policy.unit(observed.dimension, lineage),
                world,
                actor,
                calibrations,
            )
        verdicts.append(
            DimensionVerdict(
                dimension=observed.dimension,
                gate=rule.gate,
                applicable=observed.applicable,
                observed=observed.status,
                status=status,
                reason=reason,
                calibration=used,
            )
        )
    gating = [row for row in verdicts if row.gates() and row.status != "admitted"]
    overall: Verdict = (
        "refused"
        if any(row.status == "refused" for row in gating)
        else "unassessable"
        if gating
        else "admitted"
    )
    return TrajectoryAdmission(
        policy=policy_ref,
        status=overall,
        simulated=simulated,
        dimensions=tuple(verdicts),
        reasons=tuple(f"{row.dimension} {row.status}: {row.reason}" for row in gating),
    )


def _calibrated(
    observed: DimensionObservation,
    unit: str,
    world: str,
    actor: str,
    calibrations: Sequence[CalibrationEvidence],
) -> tuple[Verdict, str, str | None]:
    """Decide a simulated observation from the calibrations that cover it.

    Args:
        observed: Mechanical observation: admitted, or the simulated state-fidelity placeholder.
        unit: The trajectory's independent unit under the dimension's rule.
        world: Digest of the trajectory's world identity.
        actor: Digest of the trajectory's actor model.
        calibrations: Supplied calibrations under the applied policy.

    Returns:
        Status, reason and the deciding report ID, if any.
    """
    dimension = observed.dimension
    if not calibrations:
        return (
            "unassessable",
            "simulated observation needs a calibration under this policy; none was supplied",
            None,
        )
    covering = [
        row for row in calibrations if row.calibration.covers(dimension, unit, world, actor)
    ]
    if not covering:
        return (
            "unassessable",
            f"no supplied calibration covers {unit} with this world and actor; new families, "
            "worlds and actors start outside the calibrated scope",
            None,
        )
    # A covering refusal wins over any covering admission, whatever the order supplied: one
    # favorable report never masks a demonstrated disagreement.
    for row in covering:
        measured = row.calibration.dimension(dimension)
        if measured.status == "refused":
            status: Verdict = "refused" if dimension == "state_fidelity" else "unassessable"
            return status, f"calibration refused: {measured.reason}", row.report.artifact_id
    for row in covering:
        measured = row.calibration.dimension(dimension)
        if measured.status == "admitted":
            prefix = "" if dimension == "state_fidelity" else observed.reason + "; "
            return "admitted", f"{prefix}calibrated: {measured.reason}", row.report.artifact_id
    first = covering[0]
    return (
        "unassessable",
        f"calibration {first.calibration.dimension(dimension).status}: "
        f"{first.calibration.dimension(dimension).reason}",
        first.report.artifact_id,
    )
