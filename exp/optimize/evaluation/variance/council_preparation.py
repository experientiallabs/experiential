"""Native preparation and task-start QA shared by conditional council registrations."""

from __future__ import annotations

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    Sha256,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.rules.agent_loop import TRANSCRIPT_TYPE, AgentTranscript
from exp.optimize.evaluation.variance.bootstrap import frozen_report, registration_entry
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    REGISTRATION_TYPE,
    BootstrapBinding,
    BootstrapRegistration,
    BootstrapSettings,
    RevealAnswer,
    ReviewAnswer,
    StartDraft,
)
from exp.optimize.evaluation.variance.bootstrap_history import (
    PreparationSourceType,
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_plan import (
    PREPARATION_TYPE,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.bootstrap_selection import (
    read_historical_bootstrap_families,
)
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.family_simulators import (
    FamilySimulator,
    load_family_simulator,
)
from exp.simulation.continuations import load_continuation_checkpoint


class CouncilPreparationContract(ContractModel):
    """Exact native preparation inputs shared by prelude and full-study registration.

    Attributes:
        preparation_producer: True historical producer of the native preparation.
        preparation_source_sha256: Native source digest under its declared provenance type.
        preparation_provenance: Actual native source-authority artifact reference.
        preparation_provenance_type: Native digest owner, preserving its algorithm and scope.
        preparation_spec: Original frozen preparation settings.
        bootstrap: Exact selected native task-start binding.
        exposure: Saved preparation build exposure plan.
        family_simulator: Frozen prepared family world, rules and retrieval scope.
    """

    preparation_producer: str = Field(min_length=1)
    preparation_source_sha256: Sha256
    preparation_provenance: ArtifactInput
    preparation_provenance_type: PreparationSourceType
    preparation_spec: BootstrapPreparationSpec
    bootstrap: BootstrapBinding
    exposure: ArtifactInput
    family_simulator: ArtifactInput


class PreparedCouncilTask(ContractModel):
    """Read-only native preparation proof, without provider or ledger authority.

    Attributes:
        contract: Exact frozen preparation contract.
        preparation: Genuine native completed preparation reference.
        resume_binding: Original operation and physical-ledger ancestry.
        build_report: Exact genuine prepared family-build report.
        task_start: Exact native task-start registration with its QA receipts.
        start: Independently checked native start and job.
        family: Checked shared family runtime, rules and fit-only retrieval scope.
        producer_revision: Actual current council verifier and executor revision.
    """

    contract: CouncilPreparationContract
    preparation: ArtifactInput
    resume_binding: ArtifactInput
    build_report: ArtifactInput
    task_start: ArtifactInput
    start: BootstrapRegistration
    family: FamilySimulator
    producer_revision: str


def _pinned(project: ProjectStore, reference: ArtifactInput, kind: str) -> None:
    """Require exact immutable bytes and the actual native owner's artifact type."""
    if not project.artifacts.exists(reference.artifact_id):
        raise ValueError(f"completed native {kind} is missing: {reference.artifact_id}")
    manifest = project.artifacts.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != kind:
        raise ValueError(f"{reference.artifact_id} is not the exact native {kind}")


def _native_start(
    project: ProjectStore,
    binding: BootstrapBinding,
    report: BootstrapPreparationReport,
    settings: BootstrapSettings,
) -> tuple[BootstrapRegistration, ArtifactInput]:
    """Independently verify the selected committed start and its provenance-required QA."""
    if binding.task_id in report.refused_tasks:
        raise ValueError("native preparation selected task was refused")
    if binding not in report.bindings or binding.batch not in report.batches:
        raise ValueError("selected task has no binding in the completed preparation")
    entry = registration_entry(project, binding)
    assert entry.outcome is not None
    _pinned(project, entry.outcome, REGISTRATION_TYPE)
    start = bootstrapped_registration(project, binding)
    validation = start.validation
    if (
        start.task_id != binding.task_id
        or start.checkpoint != start.job.study.checkpoint
        or validation.provenance != start.provenance
        or not validation.passed
        or not start.rounds
        or start.rounds[-1].verdict != "accept"
    ):
        raise ValueError("native task-start acceptance and provenance QA are incomplete")
    draws = [start.rounds[-1].author, *start.rounds[-1].reviews]
    if tuple(draw.variant for draw in start.rounds[-1].reviews) != settings.review_variants:
        raise ValueError("native task-start has no complete mandatory reviewer receipts")
    draws.extend(validation.checks)
    manifest = project.artifacts.read(entry.outcome.artifact_id).manifest
    if manifest.code_revision != report.plan.code_revision:
        raise ValueError("native task-start producer differs from completed preparation")
    if StartDraft.model_validate(start.rounds[-1].author.answer) != start.draft:
        raise ValueError("native final author receipt describes another task-start draft")
    if any(
        ReviewAnswer.model_validate(draw.answer).verdict != "accept"
        for draw in start.rounds[-1].reviews
    ):
        raise ValueError("native mandatory reviewer did not accept the task start")
    if any(not RevealAnswer.model_validate(draw.answer).consistent for draw in validation.checks):
        raise ValueError("native first-reveal reviewer found a contradiction")
    for draw in draws:
        if draw.transcript is None or draw.failure is not None:
            raise ValueError("native task-start author/reviewer/reveal receipt is unavailable")
        _pinned(project, draw.transcript, TRANSCRIPT_TYPE)
        transcript = AgentTranscript.model_validate_json(
            project.artifacts.read_bytes(draw.transcript.artifact_id, "transcript.json")
        )
        if (
            transcript.answer != draw.answer
            or transcript.stop != "answered"
            or transcript.failure is not None
            or transcript.role != draw.role
            or transcript.variant != draw.variant
        ):
            raise ValueError("native task-start role receipt differs from its completed transcript")
        stored = project.artifacts.read(draw.transcript.artifact_id).manifest
        if artifact_input(stored) != draw.transcript or draw.transcript not in manifest.inputs:
            raise ValueError("native task-start lost its exact role receipt ancestry")
    methods = {
        "observed": "exact_restore",
        "reconstructed": "prefix_replay",
        "authored": "coherence_and_first_reveal",
    }
    if validation.method != methods[start.provenance]:
        raise ValueError("native task-start used another provenance QA method")
    if start.provenance == "reconstructed" and (
        len(validation.replay) != start.draft.prefix_steps
        or any(not row.agrees for row in validation.replay)
    ):
        raise ValueError("native prefix replay QA is incomplete")
    if start.provenance == "authored" and (
        tuple(draw.variant for draw in validation.checks) != settings.reveal_variants
        or len(validation.reveals) != len(start.draft.reveal_probes)
    ):
        raise ValueError("native authored start lacks complete first-reveal QA receipts")
    return start, entry.outcome


def qualify_prepared_council(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume_binding: ArtifactInput,
    spec: CouncilPreparationContract,
    *,
    code_revision: str,
) -> PreparedCouncilTask:
    """Verify genuine native preparation and selected task QA before any study role exists."""
    _pinned(project, prepared, PREPARATION_TYPE)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    provenance = read_bootstrap_preparation_provenance(
        project,
        resume_binding,
        spec=spec.preparation_spec,
    )
    if (
        spec.preparation_producer,
        spec.preparation_source_sha256,
        spec.preparation_provenance,
        spec.preparation_provenance_type,
    ) != (
        provenance.producer_revision,
        provenance.source_sha256,
        provenance.source_ref,
        provenance.source_type,
    ):
        raise ValueError("declared preparation provenance differs from the actual native owner")
    binding = provenance.binding
    if (
        prepared.artifact_id != binding.original_operation + "-result"
        or report.identity != binding.original_operation
        or report.plan.identity != report.identity
        or report.plan.resume_binding != resume_binding
        or report.plan.code_revision != provenance.producer_revision
        or report.plan.build_identity != binding.build_identity
        or report.failed_tasks
        or spec.bootstrap.task_id in report.refused_tasks
        or not report.bindings
    ):
        raise ValueError("native preparation is not complete under the exact original resume")
    manifest = project.artifacts.read(prepared.artifact_id).manifest
    expected = {report.build_report, *report.batches, *report.simulators, resume_binding}
    if manifest.code_revision != provenance.producer_revision or set(manifest.inputs) != expected:
        raise ValueError("native completed preparation producer or ancestry differs")
    _pinned(project, report.build_report, REPORT_TYPE)
    build = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(report.build_report.artifact_id, "report.json")
    )
    if build.identity != binding.build_identity or build.build != binding.source_build:
        raise ValueError("native preparation changed its original source build")
    execution = provenance.execution_spec
    selected = read_historical_bootstrap_families(
        project,
        build,
        execution.bootstrap,
        execution.limits,
        task_ids=execution.task_ids,
        simulators=execution.simulators,
        embedding_revision=binding.embedding_revision,
    )
    start, task_start = _native_start(project, spec.bootstrap, report, execution.bootstrap)
    families = [row for row in selected if row.simulator.family_id == start.family_id]
    if len(families) != 1 or spec.family_simulator not in report.simulators:
        raise ValueError("selected start has no exact prepared family simulator")
    family = load_family_simulator(project.artifacts, spec.family_simulator)
    unchanged = {"rule_book", "rule_memory", "rules_report", "runtime", "tasks"}
    if family.model_dump(exclude=unchanged) != families[0].simulator.model_dump(exclude=unchanged):
        raise ValueError("prepared family compatibility or scoped source inventory differs")
    if family.family_id != start.family_id or family.runtime is None:
        raise ValueError("prepared family runtime is unavailable or belongs to another family")
    if (
        family.world_model != families[0].scope.world_model
        or family.index != families[0].scope.index
        or family.corpus != families[0].scope.retrieval_scope
        or family.rule_book != start.rulebook
        or family.rule_memory != start.rule_memory
        or family.rules_report != start.rules_report
        or start.rules_report is None
        or spec.exposure != build.exposure
    ):
        raise ValueError("family world, fit-only RAG scope, exposure or frozen rules differ")
    exposure = load_exposure_plan(project, spec.exposure)
    exposure.require(family.members, "world_learning")
    exposure.require(family.rules_traces, "world_learning")
    checkpoint = load_continuation_checkpoint(project.artifacts, start.checkpoint)
    exposure.require_lineage(checkpoint.task.lineage_group_id, "mining_probes")
    runtime = GroundedRuntimeDefinition.model_validate(start.job.configuration)
    excluded = {"task", "manifest", "initial_state", "probe_paths", "facts"}
    if runtime.model_dump(exclude=excluded) != family.runtime.model_dump(exclude=excluded):
        raise ValueError("task runtime changed its frozen family definition")
    preflight_grounded_runtime(project, runtime)
    if frozen_report(project, start.rules_report).rulebook != start.rulebook:
        raise ValueError("native rules report binds another rule book")
    return PreparedCouncilTask(
        contract=spec,
        preparation=prepared,
        resume_binding=resume_binding,
        build_report=report.build_report,
        task_start=task_start,
        start=start,
        family=family,
        producer_revision=code_revision,
    )
