"""Witnessed qualification of an executable task's observable mechanical contract.

This narrower reward profile uses no LLM judgment. It requires a saved feasible execution,
independently witnessed positive, negative and unknown controls, and preservation checks.
It does not qualify rubric judgments, model-tier separation, native fidelity or containment.
"""

from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import WorkflowEvidence
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import load_continuation_runtime
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.executable.task import load_executable_task, preflight_executable_task
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.capability_briefs import load_capability_brief
from exp.simulation.evaluation.admission import verify_label_witness
from exp.simulation.evaluation.calibration import CalibrationCase, CalibrationPack
from exp.simulation.evaluation.record_execution import RecordReplayRuntime, require_record_runtime
from exp.simulation.evaluation.tasks import TaskVerifier, grade, normalized_outcome
from exp.simulation.reconstruction.models import path_parts


class ExecutableQualificationSpec(ContractModel):
    """Frozen objective controls and an actually completed feasible branch.

    Attributes:
        task: Exact executable task, including runtime and probe definitions.
        feasibility: Complete continuation result under precisely this world and verifier.
        cases: Independent witness-backed controls, with disjoint development/confirmation
            lineages. Per partition: two valid outcomes, a goal miss, collateral policy damage,
            and an unavailable-evidence case. Labels cannot be supplied by the candidate verifier.
        replay_native_records: Explicit permission to load the task's pinned runtime and
            re-execute isolated native record witnesses during qualification and re-verification.
    """

    task: ArtifactInput
    feasibility: ArtifactInput
    cases: tuple[CalibrationCase, ...] = Field(min_length=1)
    replay_native_records: bool = False


class ExecutableQualification(ContractModel):
    """Finite observable-contract qualification, explicitly excluding broader certificates.

    Attributes:
        task: Exact registered executable task.
        spec: Feasibility and independent control evidence.
        profile_sha256: Frozen world, probe, verifier and execution-allowance identity.
        qualified: Whether every declared observable-contract check passed.
        refusals: Concrete unmet requirements, retained on rejection.
        scope: Explicit limits; this is not a broader environment certificate.
    """

    task: ArtifactInput
    spec: ExecutableQualificationSpec
    profile_sha256: str
    qualified: bool
    refusals: tuple[str, ...]
    scope: str = (
        "development witnessed observable contract only; no independent audit, "
        "rubric-judge, tier-separation, "
        "native-fidelity, unobserved-effect or host-containment qualification"
    )


def preflight_executable_qualification(
    project: ProjectStore, spec: ExecutableQualificationSpec
) -> None:
    """Verify declared evidence without importing a runtime or re-executing native witnesses."""
    task = load_executable_task(project, spec.task)
    if spec.replay_native_records:
        preflight_executable_task(project, task)
    load_study(project.artifacts, spec.feasibility)
    predicate = task.job.study.verifier
    CalibrationPack(task=spec.task, parent=predicate, candidate=predicate, cases=spec.cases)
    for case in spec.cases:
        reference = case.challenge.witness
        if (
            reference is not None
            and artifact_input(project.artifacts.read(reference.artifact_id).manifest) != reference
        ):
            raise ValueError("qualification witness differs from its declared reference")


def assess_executable_qualification(
    project: ProjectStore,
    spec: ExecutableQualificationSpec,
) -> ExecutableQualification:
    """Re-derive every label and the feasible run from trusted immutable records."""
    preflight_executable_qualification(project, spec)
    task = load_executable_task(project, spec.task)
    record_runtime = None
    if spec.replay_native_records:
        runtime = load_continuation_runtime(project, task.job)
        if not isinstance(runtime.world, ExecutableWorldRuntime):
            raise ValueError("native record qualification requires a wholly native task runtime")
        require_record_runtime(runtime.world)
        record_runtime = RecordReplayRuntime(runtime.world, runtime.probe, runtime.probe_sha256)
    predicate = task.job.study.verifier
    # Reuse the established identity, duplicate and grouped-partition checks.
    CalibrationPack(task=spec.task, parent=predicate, candidate=predicate, cases=spec.cases)
    refusals = []
    # Argument rules constrain only the calls they name, so an unrequested write elsewhere
    # needs a check over observed state: preservation or a pin on the complete output.
    if predicate.allowed_tools is None or not (
        predicate.preserved_paths or predicate.preserve_state or _pins_full_output(predicate)
    ):
        refusals.append(
            "explicit permitted tools and a state-preservation or full-output check are required"
        )
    feasible = load_study(project.artifacts, spec.feasibility)
    task_revision = project.artifacts.read(spec.task.artifact_id).manifest.code_revision
    if project.artifacts.read(spec.feasibility.artifact_id).manifest.code_revision != (
        task.job.implementation(task_revision)
    ):
        refusals.append(
            "feasibility was not executed under this task's exact runtime, catalog, budget and code"
        )
    if feasible.spec != task.job.study or not any(
        row.status == "completed" and grade(predicate, row.outcome).status == "pass"
        for row in feasible.branches
    ):
        refusals.append("no successful saved execution under the exact task world and verifier")
    if task.capability_brief is not None:
        refusals.extend(load_capability_brief(project.artifacts, task.capability_brief).blockers())
    partitions = {"development", "confirmation"}
    if any(case.partition == "selection" for case in spec.cases):
        partitions.add("selection")
    seen_controls: set[str] = set()
    for partition in sorted(partitions):
        rows = [case for case in spec.cases if case.partition == partition]
        admitted = []
        for case in rows:
            challenge = case.challenge
            if challenge.witness is None or challenge.authority not in (
                "synthetic_specification",
                "native_execution",
            ):
                refusals.append(f"{challenge.id}: an independently re-derived witness is required")
                continue
            refusal = verify_label_witness(
                project.artifacts,
                challenge,
                label=challenge.expected,
                compared=(predicate,),
                record_runtime=record_runtime,
            )
            if refusal is not None:
                refusals.append(f"{challenge.id}: {refusal.reason}: {refusal.detail}")
                continue
            digest = _control_identity(challenge.outcome)
            if digest in seen_controls:
                refusals.append(f"{challenge.id}: duplicated control does not add coverage")
            seen_controls.add(digest)
            assessed = grade(predicate, challenge.outcome)
            if assessed.status != challenge.expected:
                refusals.append(f"{challenge.id}: verifier disagrees with the independent witness")
            admitted.append(assessed)
        if sum(row.status == "pass" for row in admitted) < 2:
            refusals.append(f"{partition}: two distinct valid outcomes are required")
        if not any(row.goal.status == "fail" and row.policy.status == "pass" for row in admitted):
            refusals.append(f"{partition}: missing permitted-action goal failure control")
        if not any(row.goal.status == "pass" and row.policy.status == "fail" for row in admitted):
            refusals.append(f"{partition}: missing goal-achieved-with-prohibited-effect control")
        if not any(row.status == "unassessable" for row in admitted):
            refusals.append(f"{partition}: missing unavailable-evidence control")
    return ExecutableQualification(
        task=spec.task,
        spec=spec,
        profile_sha256=task.profile(),
        qualified=not refusals,
        refusals=tuple(refusals),
    )


def _pins_full_output(predicate: TaskVerifier) -> bool:
    """Whether an equality rule fixes the whole final state or one whole observed file.

    Args:
        predicate: The task verifier under qualification.

    Returns:
        True when a rule requires the whole outcome, ``/final_state`` or one complete
        ``/final_state/files/<name>`` entry to equal an authored value, which covers every
        effect on that output.
    """
    for rule in predicate.rules:
        parts = path_parts(rule.path)
        whole_file = len(parts) == 3 and parts[:2] == ("final_state", "files")
        if rule.operator == "equals" and (parts in ((), ("final_state",)) or whole_file):
            return True
    return False


def qualify_executable_task(
    project: ProjectStore,
    spec: ExecutableQualificationSpec,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Save successful or rejected qualification without mutating the task or verifier."""
    report = assess_executable_qualification(project, spec)
    identity = stable_id(
        "executable-qualification", {"report": sha256_json(report), "code": code_revision}
    )
    store = project.artifacts
    inputs = {
        spec.task,
        spec.feasibility,
        *(case.challenge.witness for case in spec.cases if case.challenge.witness),
    }
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="executable-qualification",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=created_at,
                code_revision=code_revision,
                inputs=tuple(sorted(inputs, key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report},
        )
    return artifact_input(store.read(identity).manifest)


def load_executable_qualification(
    project: ProjectStore, reference: ArtifactInput, *, allow_native_replay: bool = True
) -> ExecutableQualification:
    """Recompute qualification, optionally refusing evidence that requires native replay."""
    report = _read_qualification(project, reference)
    if report.spec.replay_native_records and not allow_native_replay:
        raise ValueError(
            "qualification requires native replay; use a qualification with independently "
            "checkable saved witnesses for credential-free training admission"
        )
    if assess_executable_qualification(project, report.spec) != report:
        raise ValueError("executable qualification no longer reproduces its evidence")
    return report


def preflight_qualification_reference(
    project: ProjectStore, reference: ArtifactInput, task: ArtifactInput, *, diagnostic: bool
) -> None:
    """Check saved qualification sources for export without claiming native replay passed."""
    report = _read_qualification(project, reference)
    if report.task != task or report.spec.task != task:
        raise ValueError("qualification belongs to another executable task")
    if report.profile_sha256 != load_executable_task(project, task).profile():
        raise ValueError("qualification differs from the frozen executable task profile")
    preflight_executable_qualification(project, report.spec)
    if not report.qualified and not diagnostic:
        raise ValueError("observable contract is unqualified; use an explicit diagnostic export")


def _read_qualification(project: ProjectStore, reference: ArtifactInput) -> ExecutableQualification:
    """Read an exact receipt; this alone never establishes qualification."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != "executable-qualification"
    ):
        raise ValueError("executable qualification reference differs")
    report = ExecutableQualification.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )
    expected_id = stable_id(
        "executable-qualification", {"report": sha256_json(report), "code": manifest.code_revision}
    )
    expected_inputs = {
        report.spec.task,
        report.spec.feasibility,
        *(case.challenge.witness for case in report.spec.cases if case.challenge.witness),
    }
    if reference.artifact_id != expected_id or set(manifest.inputs) != expected_inputs:
        raise ValueError("executable qualification lost its frozen identity or inputs")
    return report


def _control_identity(outcome: WorkflowEvidence | JsonObject) -> str:
    """Match substantive normalized observations independently of call IDs or model metadata.

    This deliberately coarse match can reject redundant controls. It cannot certify independence
    of paraphrased cases or externally shared sources; qualification remains development evidence.
    """
    value = normalized_outcome(outcome) if isinstance(outcome, WorkflowEvidence) else outcome
    steps = value.get("steps")
    projected = []
    for step in steps if isinstance(steps, list) else []:
        if not isinstance(step, dict):
            projected.append(step)
            continue
        action = step.get("action")
        projected.append(
            {
                "action": {key: action.get(key) for key in ("name", "arguments")}
                if isinstance(action, dict)
                else action,
                "status": step.get("status"),
                "after": step.get("after"),
            }
        )
    return sha256_json(
        {
            "initial_state": value.get("initial_state"),
            "final_state": value.get("final_state"),
            "final_answer": value.get("final_answer"),
            "stop_reason": value.get("stop_reason"),
            "steps": projected if isinstance(steps, list) else None,
        }
    )
