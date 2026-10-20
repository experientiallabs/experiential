"""Independent, pre-registered acceptance of saved development continuations for SFT."""

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.optimize.evaluation.continuation_evidence import (
    TrainingEvidenceAssessment,
    verify_training_evidence,
)
from exp.optimize.evaluation.continuation_records import load_study
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.development_scope import require_development_task
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
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
    """

    index: int = Field(ge=0)
    admitted: bool
    reason: str
    witness: ArtifactInput | None = None


class ContinuationAdmission(ContractModel):
    """An acceptance report over every branch of an immutable development study.

    Attributes:
        study: Exact saved study result, including all failures.
        branches: Decisions in the original registration order.
        scope: Checked success under the declared observation and policy contract only.
        evidence: Re-derived task qualification and comparison scope, absent for outcome-only
            admission. Positive observed outcomes alone do not qualify an environment.
    """

    study: ArtifactInput
    branches: tuple[BranchAdmission, ...]
    scope: str = "development; independently specified observed outcome, not native fidelity"
    evidence: TrainingEvidenceAssessment | None = None


def admit_continuations(
    store: ArtifactStore,
    study_ref: ArtifactInput,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Re-grade all saved branches against the independent specification frozen before sampling.

    Args:
        store: Trusted artifact owner; no caller-supplied outcomes are accepted.
        study_ref: Complete saved development study.
        created_at: Decision publication time.
        code_revision: Producing implementation identity.

    Returns:
        Immutable report including rejected and unknown branches. No calls are dispatched.
    """
    study = load_study(store, study_ref)
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
    decisions = []
    for branch in study.branches:
        reason = None
        witness = None
        assessment = grade(study.spec.verifier, branch.outcome)
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
        decisions.append(
            BranchAdmission(
                index=branch.index,
                admitted=reason is None,
                reason=reason or "goal, policy and independent pre-registered specification passed",
                witness=witness,
            )
        )
    result = ContinuationAdmission(study=study_ref, branches=tuple(decisions), evidence=evidence)
    identity = stable_id(
        "continuation-admission", {"result": sha256_json(result), "code": code_revision}
    )
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    inputs = {study_ref, *(row.witness for row in decisions if row.witness is not None)}
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
