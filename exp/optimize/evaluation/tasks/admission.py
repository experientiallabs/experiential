"""Store-verified evidence admission for task calibration and task refinement.

Labels are admitted from saved witnesses, and contamination is decided with the existing
owners: the simulator's pinned-knowledge exposure check and the root held-out exposure ledger.
Nothing here calls a provider, so every refusal lands before paid work.
"""

import json
from collections.abc import Iterable, Sequence

from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ArtifactStore, ArtifactStoreError, ProjectStore
from exp.common.tasks import TaskCase
from exp.optimize.evaluation.audit.exposure import held_out_exposure
from exp.optimize.evaluation.fidelity.knowledge_exposure import require_unexposed_task
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.evaluation.admission import (
    AdmissionRefusal,
    EvidenceAdmission,
    adequacy_refusals,
    require_admitted,
    task_lineage_challenges,
    verify_label_witness,
)
from exp.simulation.evaluation.calibration import CalibrationCase, CalibrationPack
from exp.simulation.evaluation.tasks import (
    WITNESSED_AUTHORITIES,
    EvaluationTask,
    RuleVerifierDraft,
    TaskVerifier,
    Verdict,
    VerifierChallenge,
    label_admitted,
    load_evaluation_task,
)
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


def admit_calibration_evidence(project: ProjectStore, pack: CalibrationPack) -> EvidenceAdmission:
    """Verify every witness and decide contamination for one calibration pack.

    Args:
        project: Project holding the task, witnesses, build and exposure ledger.
        pack: Frozen calibration pack.

    Returns:
        Verified witnessed labels and every named refusal; adequacy is judged by calibration.
    """
    task = load_evaluation_task(project.artifacts, pack.task)
    verified, refusals = _verify(
        project.artifacts,
        ((case.challenge, case.challenge.expected) for case in pack.cases),
        (pack.parent, pack.candidate, *generated_predicates(project.artifacts)),
    )
    refusals += knowledge_refusals(project, task.world_model, task.task)
    if task.capability_brief is not None:
        brief = load_capability_brief(project.artifacts, task.capability_brief)
        refusals += tuple(
            AdmissionRefusal(
                reason="inadequate_evidence", subject="capability brief", detail=detail
            )
            for detail in brief.blockers()
        )
    refusals += _authoring_refusals(project.artifacts, task, pack.cases)
    ledger_refusals, ledger = _ledger_refusals(
        project,
        task.task,
        tuple((case.challenge.id, case.lineage, case.partition) for case in pack.cases),
    )
    return EvidenceAdmission(verified=verified, refusals=refusals + ledger_refusals, ledger=ledger)


def require_refinement_admission(
    project: ProjectStore,
    *,
    verifier: TaskVerifier,
    development: Sequence[VerifierChallenge],
    confirmation: Sequence[VerifierChallenge],
    world_model: ArtifactInput,
    task: TaskCase,
) -> frozenset[str]:
    """Refuse a refinement cycle that could not qualify, before any role or actor call.

    Args:
        project: Project holding witnesses and the simulator.
        verifier: Parent predicate under refinement.
        development: Labels shown to proposing roles.
        confirmation: Labels withheld until development passes.
        world_model: Simulator the cycle evaluates on.
        task: Public task being refined.

    Returns:
        IDs of witnessed labels whose saved witness re-verified.

    Raises:
        EvidenceAdmissionError: Any named refusal, including too few admitted labels, a
            development label on a held-out task, or a held-out task the exposure ledger
            cannot show independent.
    """
    challenges = (*development, *confirmation)
    verified, refusals = _verify(
        project.artifacts,
        ((challenge, challenge.expected) for challenge in challenges),
        (verifier, *generated_predicates(project.artifacts)),
    )
    admission = EvidenceAdmission(verified=verified)
    ids = admission.verified_ids(challenges)
    refusals += knowledge_refusals(project, world_model, task)
    refusals += adequacy_refusals(
        (
            *(
                ("development", row.expected, label_admitted(row, ids), frozenset[str]())
                for row in development
            ),
            *(
                ("confirmation", row.expected, label_admitted(row, ids), frozenset[str]())
                for row in confirmation
            ),
        ),
        partitions=("development", "confirmation"),
    )
    refusals += _ledger_refusals(
        project,
        task,
        tuple(
            (row.id, task.lineage_group_id, partition)
            for partition, rows in (("development", development), ("confirmation", confirmation))
            for row in rows
        ),
    )[0]
    require_admitted(admission.model_copy(update={"refusals": refusals}))
    return ids


def generated_predicates(store: ArtifactStore) -> tuple[TaskVerifier, ...]:
    """Every verifier a refinement role proposed in this project.

    Args:
        store: Project artifact owner.

    Returns:
        Compiled generated predicates; a witness specification may not be any of them.
    """
    found = []
    for identity in store.list_ids(artifact_type="task-refinement-proposal"):
        value = json.loads(store.read_bytes(identity, "proposal.json"))
        draft = value.get("verifier") if isinstance(value, dict) else None
        try:
            found.append(RuleVerifierDraft.model_validate(draft).compile())
        except ValidationError:
            continue
    return tuple(found)


def knowledge_refusals(
    project: ProjectStore, world_model: ArtifactInput, task: TaskCase
) -> tuple[AdmissionRefusal, ...]:
    """Refuse a task whose own evidence reaches the simulator through pinned knowledge.

    Args:
        project: Project resolving the simulator and its knowledge sources.
        world_model: Exact grounded simulator.
        task: Public task.

    Returns:
        A ``contaminated_lineage`` refusal, or nothing.
    """
    try:
        artifact = load_grounded_world_model_artifact(project.artifacts, world_model)
        require_unexposed_task(project, artifact, task)
    except (ArtifactStoreError, OSError, ValueError) as error:
        return (
            AdmissionRefusal(
                reason="contaminated_lineage", subject=f"task {task.task_id}", detail=str(error)
            ),
        )
    return ()


def _verify(
    store: ArtifactStore,
    labels: Iterable[tuple[VerifierChallenge, Verdict]],
    compared: Sequence[TaskVerifier],
) -> tuple[dict[str, str], tuple[AdmissionRefusal, ...]]:
    """Re-derive every witnessed label; attested and generated labels need no witness check."""
    verified: dict[str, str] = {}
    refusals: list[AdmissionRefusal] = []
    for challenge, label in labels:
        if challenge.authority not in WITNESSED_AUTHORITIES or challenge.witness is None:
            continue
        refusal = verify_label_witness(store, challenge, label=label, compared=compared)
        if refusal is None:
            verified[challenge.id] = sha256_json(challenge)
        else:
            refusals.append(refusal)
    return verified, tuple(refusals)


def _authoring_refusals(
    store: ArtifactStore, task: EvaluationTask, cases: Sequence[CalibrationCase]
) -> tuple[AdmissionRefusal, ...]:
    """Refuse selection or confirmation labels already shown to task authoring."""
    shown = {sha256_json(row.outcome) for row in task_lineage_challenges(store, task)}
    return tuple(
        AdmissionRefusal(
            reason="sealed_partition_breach",
            subject=case.challenge.id,
            detail=(
                f"this {case.partition} outcome was a development challenge of the task or an "
                "ancestor revision, so authoring already saw it; label a fresh outcome"
            ),
        )
        for case in cases
        if case.partition != "development" and sha256_json(case.challenge.outcome) in shown
    )


def _ledger_refusals(
    project: ProjectStore, task: TaskCase, labels: Sequence[tuple[str, str, str]]
) -> tuple[tuple[AdmissionRefusal, ...], str | None]:
    """Decide sealed-partition overlap and held-out exposure with the root exposure ledger.

    Args:
        project: Project holding the build and the exposure ledger.
        task: Public task the labels evaluate.
        labels: Label ID, source lineage and partition of every label.

    Returns:
        Every named refusal, and the ledger consulted when a held-out lineage is involved.
    """
    build = project.load_project().build
    if build is None:
        if task.partition != "held_out":
            return (), None
        return (
            AdmissionRefusal(
                reason="contaminated_lineage",
                subject=f"task {task.task_id}",
                detail=(
                    "a held-out task needs the project's completed build so the exposure ledger "
                    "can decide its independence; run exp build PROJECT first"
                ),
            ),
        ), None
    bindings = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id)
    held_out = {row.lineage_id for row in bindings.bindings if row.partition == "held_out"}
    if task.partition == "held_out":
        held_out.add(task.lineage_group_id)
    refusals = [
        AdmissionRefusal(
            reason="sealed_partition_breach",
            subject=subject,
            detail=(
                f"this {partition} label uses held-out evaluation lineage {lineage}, "
                "which only confirmation may touch; relabel it from a fit lineage"
            ),
        )
        for subject, lineage, partition in labels
        if lineage in held_out and partition != "confirmation"
    ]
    sealed = {lineage for _, lineage, _ in labels if lineage in held_out}
    if task.partition == "held_out":
        sealed.add(task.lineage_group_id)
    if not sealed:
        return tuple(refusals), None
    try:
        exposure = held_out_exposure(project)
    except ValueError as error:
        refusals.append(
            AdmissionRefusal(
                reason="contaminated_lineage", subject="exposure ledger", detail=str(error)
            )
        )
        return tuple(refusals), None
    if exposure.unknown:
        refusals.append(
            AdmissionRefusal(
                reason="contaminated_lineage",
                subject="exposure ledger",
                detail=(
                    "the ledger holds fingerprints this machine cannot compute, so held-out "
                    "independence is undecidable; merge or resolve the other machine's ledger"
                ),
            )
        )
    for row in exposure.exposed:
        if row.lineage in sealed:
            refusals.append(
                AdmissionRefusal(
                    reason="contaminated_lineage",
                    subject=row.lineage,
                    detail=(
                        f"held-out lineage {row.lineage} is {row.kind} in the exposure ledger; "
                        "label an independent lineage"
                    ),
                )
            )
    return tuple(refusals), exposure.ledger
