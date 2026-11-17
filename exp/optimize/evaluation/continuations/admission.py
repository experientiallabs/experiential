"""Independent, pre-registered acceptance of saved development continuations for SFT.

Every branch is judged twice: against the independent training specification frozen before
sampling, and dimension by dimension under a registered admission policy the study pinned
before sampling. A branch is admitted only when both pass; its per-dimension verdicts stay in
the report.
"""

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.evidence import (
    TrainingEvidenceAssessment,
    verify_training_evidence,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.equivalence.policy import (
    load_admission_policy,
    require_policy_precedes,
)
from exp.optimize.evaluation.equivalence.runner import load_equivalence_report
from exp.optimize.evaluation.equivalence.verdicts import (
    CalibrationEvidence,
    TrajectoryAdmission,
    admit_trajectory,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    has_outcome_constraints,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.tasks import VerifierChallenge, grade


class BranchAdmission(ContractModel):
    """One planned branch's acceptance or explicit exclusion, retaining its denominator.

    Attributes:
        index: Original branch position.
        admitted: Whether the saved execution passed both independent and study criteria.
        reason: Concrete refusal or scope of acceptance.
        witness: Re-derived specification witness, when available.
        policy: Per-dimension verdicts under the registered admission policy.
    """

    index: int = Field(ge=0)
    admitted: bool
    reason: str
    policy: TrajectoryAdmission
    witness: ArtifactInput | None = None


class ContinuationAdmission(ContractModel):
    """An acceptance report over every branch of an immutable development study.

    Attributes:
        study: Exact saved study result, including all failures.
        policy: Registered admission policy, fixed before the study was registered.
        branches: Decisions in the original registration order.
        scope: Checked success under the declared observation and policy contract only.
        evidence: Re-derived task qualification and comparison scope, absent for outcome-only
            admission. Positive observed outcomes alone do not qualify an environment.
    """

    study: ArtifactInput
    policy: ArtifactInput
    branches: tuple[BranchAdmission, ...]
    scope: str = "development; independently specified observed outcome, not native fidelity"
    evidence: TrainingEvidenceAssessment | None = None


def require_pinned_policy(
    store: ArtifactStore, study_ref: ArtifactInput, study: ContinuationStudy, policy: ArtifactInput
) -> None:
    """Refuse any admission policy other than the one the study pinned before running.

    Citation alone is not enough: the study registration also cites its training evidence,
    whose comparisons cite their own policies, so a policy reached only that way was chosen for
    other results and would let the gate be picked after the study's outcomes were seen.

    Args:
        store: Project artifact store.
        study_ref: Exact saved study result.
        study: That study, as loaded and verified.
        policy: Registered policy asked to judge it.

    Raises:
        ValueError: The study pinned another policy or none, or its registration does not
            cite the pinned one.
    """
    if study.spec.admission_policy != policy:
        pinned = study.spec.admission_policy
        raise ValueError(
            f"continuation study {study_ref.artifact_id} "
            + ("pins no admission policy" if pinned is None else f"pins {pinned.artifact_id}")
            + f", not {policy.artifact_id}; only the policy a study pins as admission_policy "
            "before running can judge it, so register a fresh study that pins this policy"
        )
    require_policy_precedes(
        store, policy, study_ref.artifact_id.removesuffix("-result"), subject="continuation study"
    )


def admit_continuations(
    store: ArtifactStore,
    study_ref: ArtifactInput,
    *,
    policy: ArtifactInput,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Re-grade all saved branches against the independent specification frozen before sampling.

    Args:
        store: Trusted artifact owner; no caller-supplied outcomes are accepted.
        study_ref: Complete saved development study.
        policy: Registered admission policy, pinned in the study registration.
        created_at: Decision publication time.
        code_revision: Producing implementation identity.

    Returns:
        Immutable report including rejected and unknown branches. No calls are dispatched.

    Raises:
        ValueError: The policy is not registered or is not the one the study pinned.
    """
    study = load_study(store, study_ref)
    rules = load_admission_policy(store, policy)
    require_pinned_policy(store, study_ref, study, policy)
    evidence = verify_training_evidence(
        store,
        study.spec,
        execution_revision=store.read(study_ref.artifact_id).manifest.code_revision,
    )
    checkpoint = load_continuation_checkpoint(store, study.spec.checkpoint)
    require_development_task(store, checkpoint.task)
    if checkpoint.task.partition != "fit":
        raise ValueError("reserved evaluation continuations cannot become training data")
    specification = study.spec.training_specification
    predicate = None
    if specification is not None:
        manifest = store.read(specification.artifact_id).manifest
        if (
            artifact_input(manifest) != specification
            or manifest.artifact_type != "evaluation-specification"
        ):
            raise ValueError("training specification differs from the pre-outcome registration")
        predicate = EvaluationSpecification.model_validate_json(
            store.read_bytes(specification.artifact_id, "specification.json")
        )
    calibrations = _calibrations(store, evidence, policy)
    decisions = []
    for branch in study.branches:
        reason = None
        witness = None
        verdicts = admit_trajectory(
            policy,
            rules,
            branch,
            checkpoint=checkpoint,
            verifier=study.spec.verifier,
            actor_model=study.spec.actor_model,
            calibrations=calibrations,
        )
        assessment = grade(study.spec.verifier, branch.grading_outcome())
        if branch.status != "completed" or not assessment.complete:
            reason = "execution incomplete or outcome unknown"
        elif assessment.status != "pass" or assessment.policy.status != "pass":
            reason = "declared goal or policy did not pass"
        elif predicate is None or specification is None:
            reason = "no independent training specification registered before sampling"
        elif predicate.subject != "outcome" or predicate.predicate.allowed_tools is None:
            reason = (
                "training specification must cover the observed outcome and explicit tool policy"
            )
        elif not has_outcome_constraints(predicate.predicate):
            reason = (
                "training specification needs an outcome predicate beyond shape and tool policy"
            )
        else:
            witness = record_label_witness(
                store,
                specification=specification,
                outcome=branch.outcome,
                created_at=created_at,
                code_revision=code_revision,
            )
            challenge = VerifierChallenge(
                id=f"branch-{branch.index}",
                outcome=branch.outcome,
                expected="pass",
                authority="synthetic_specification",
                witness=witness,
                rationale="Independent specification pinned before continuation sampling.",
            )
            refusal = verify_label_witness(
                store, challenge, label="pass", compared=(study.spec.verifier,)
            )
            if refusal is not None:
                reason = f"{refusal.reason}: {refusal.detail}"
        if reason is None and verdicts.status != "admitted":
            reason = f"admission policy {verdicts.status}: " + "; ".join(verdicts.reasons)
        decisions.append(
            BranchAdmission(
                index=branch.index,
                admitted=reason is None,
                reason=reason
                or "goal, policy, independent pre-registered specification and every gating "
                "admission dimension passed",
                policy=verdicts,
                witness=witness,
            )
        )
    result = ContinuationAdmission(
        study=study_ref, policy=policy, branches=tuple(decisions), evidence=evidence
    )
    identity = stable_id(
        "continuation-admission", {"result": sha256_json(result), "code": code_revision}
    )
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    inputs = {
        study_ref,
        policy,
        *(row.witness for row in decisions if row.witness is not None),
    }
    if study.spec.training_evidence is not None:
        inputs.update(study.spec.training_evidence.sources())
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type="continuation-admission",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=created_at,
            code_revision=code_revision,
            inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
        ),
        files={"admission.json": result},
    )
    return artifact_input(manifest)


def _calibrations(
    store: ArtifactStore, evidence: TrainingEvidenceAssessment | None, policy: ArtifactInput
) -> tuple[CalibrationEvidence, ...]:
    """Use the comparison pinned before generation, when it was calibrated under this policy.

    Args:
        store: Project artifact store.
        evidence: Re-derived training evidence, possibly with a pinned comparison.
        policy: Applied admission policy.

    Returns:
        The pinned comparison's calibration, or nothing; simulated observations then stay
        unassessable rather than borrowing another policy's margins.
    """
    if evidence is None or evidence.comparison is None:
        return ()
    project = ProjectStore(store.paths.root, store.paths.project_id)
    report = load_equivalence_report(project, evidence.comparison)
    if report.calibration is None or report.calibration.policy != policy:
        return ()
    return (CalibrationEvidence(report=evidence.comparison, calibration=report.calibration),)
