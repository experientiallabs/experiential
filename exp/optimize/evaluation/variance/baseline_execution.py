"""Fresh variance baselines retain original preparation authority and historical reserves.

Propose exact settings, approve their digest, register, then execute. Registration writes only
an immutable binding. Execution uses the actual native build, physical clients and original
operation ledger; only a committed native report can produce a completion receipt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import load_model_catalog
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_history import (
    BootstrapPreparationProvenance,
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_pilot import load_pilot_build
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapPlan,
    BootstrapPreparationSpec,
    _bound_bootstrap_plan,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    BootstrapResumeBinding,
    BootstrapResumeInputs,
)
from exp.optimize.evaluation.variance.bootstrap_source import (
    BootstrapExecutionSource,
    current_bootstrap_execution_source,
)
from exp.optimize.evaluation.variance.build import (
    _prepare,
    run_variance_build,
    variance_build_costs,
)
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    VarianceBuildPlan,
    VarianceBuildReport,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings, CostLine, cost_lines
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope

BASELINE_TYPE = "variance-baseline-execution"
BASELINE_RESULT_TYPE = "variance-baseline-execution-result"
BASELINE_PROTOCOL = "variance-baseline-execution-v1"


class BaselineExecutionProposal(ContractModel):
    """Exact fresh scope and native cost inventory proposed under one original operation.

    Attributes:
        protocol: Interpretation of fresh scope and original reserve checks.
        original_resume: Genuine immutable historical preparation resume.
        original_source: Native historical source witness, not current execution authority.
        original_source_sha256: Actual historical source digest.
        original_operation: Original shared physical ledger and lock identity.
        ledger_sha256: Original ledger's canonical identity.
        source_build: Unchanged trace universe, canonical task set and completed source build.
        historical_build: Original completed variance report with actual producer provenance.
        historical_exposure: Original exposure preserving held-out and learning roles.
        used_task_ids: Original selected probes that must remain development probes.
        build: Entire fresh native settings, including operational concurrency and reviewer caps.
        plan: Exact full native proposal quote, before any fresh execution.
        exposure: Exact fresh lineage roles and bindings.
        catalog_sha256: Complete credential-free model catalog.
        execution_source: Actual current executing package source bytes.
        preparation_floor: Genuine pending original preparation inventory at proposal time.
        permission: Explicit scope of fresh user consent, never a source-only resume.
    """

    protocol: Literal["variance-baseline-execution-v1"] = BASELINE_PROTOCOL
    original_resume: ArtifactInput
    original_source: ArtifactInput
    original_source_sha256: Sha256
    original_operation: str
    ledger_sha256: Sha256
    source_build: ProjectBuildArtifacts
    historical_build: ArtifactInput
    historical_exposure: ArtifactInput
    used_task_ids: tuple[str, ...]
    build: BuildSettings
    plan: VarianceBuildPlan
    exposure: ExposurePlan
    catalog_sha256: Sha256
    execution_source: BootstrapExecutionSource
    preparation_floor: BootstrapPlan
    permission: Literal["multiple_tag_tasksets_and_bases"] = "multiple_tag_tasksets_and_bases"


class BaselineExecutionApproval(ContractModel):
    """Consent to one exact proposed fresh baseline, without granting any additional budget.

    Attributes:
        proposal_sha256: Exact credential-free proposal reviewed by the caller.
    """

    proposal_sha256: Sha256


class BaselineExecutionPlan(ContractModel):
    """Current cumulative liabilities and original future preparation reserve.

    Attributes:
        binding: Exact registered fresh scope.
        native: Actual full native build plan, with current decision-cache replay inventory.
        costs: Entire current native build physical upper bound, including compatibility review.
        preparation_floor: Native historical pending work, never a copied scalar.
        accounted_usd: Original settled and retained physical dollar liabilities.
        accounted_unpriced_attempts: All original retained unpriced liabilities.
        known_usd: Cumulative known liability, current build and original preparation reserve.
        unpriced_attempts: Cumulative unpriced liability, build and original preparation reserve.
        complete: An actual pinned committed native report already exists.
    """

    binding: ArtifactInput
    native: VarianceBuildPlan
    costs: tuple[CostLine, ...]
    preparation_floor: BootstrapPlan
    accounted_usd: float = Field(ge=0)
    accounted_unpriced_attempts: int = Field(ge=0)
    known_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(ge=0)
    complete: bool


def _history(
    project: ProjectStore, reference: ArtifactInput
) -> tuple[BootstrapPreparationSpec, BootstrapPreparationProvenance]:
    """Resolve original settings only from native immutable ancestry and verify their owner."""
    raw = BootstrapResumeBinding.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "binding.json")
    )
    inputs = BootstrapResumeInputs.model_validate_json(
        project.artifacts.read_bytes(raw.original_inputs.artifact_id, "inputs.json")
    )
    spec = BootstrapPreparationSpec.model_validate(inputs.spec)
    return spec, read_bootstrap_preparation_provenance(project, reference, spec=spec)


def _native(
    project: ProjectStore, settings: BuildSettings, revision: str
) -> tuple[VarianceBuildPlan, ExposurePlan, tuple[CostLine, ...]]:
    """Resolve exact banks, inputs, catalog identities, exposure and full native costs."""
    if (
        settings.saved_report is not None
        or settings.spec.construction is not None
        or settings.spec.cluster_naming is not None
        or settings.reviewer is None
        or settings.spec.environment_families is None
    ):
        raise ValueError(
            "fresh baseline needs native family review without saved or injected stages"
        )
    prepared = _prepare(
        project,
        settings.spec,
        revision,
        None,
        settings.reviewer.model,
        sha256_json(settings.reviewer),
    )
    if prepared.summary.embedding_inputs != prepared.summary.embedding_inputs_saved:
        raise ValueError("fresh baseline requires every original scoped embedding input saved")
    components = (
        *variance_build_costs(project, prepared.summary, settings.spec.decision_alias),
        *review_costs(project, settings, prepared.summary),
    )
    return prepared.summary, prepared.plan, cost_lines(components)


def _preserve_roles(
    project: ProjectStore,
    previous: ExposurePlan,
    fresh: ExposurePlan,
    used_task_ids: tuple[str, ...],
) -> None:
    """Keep the canonical universe, all old learning/held-out lineages and used mining probes."""
    if (
        previous.bindings != fresh.bindings
        or previous.traces != fresh.traces
        or set(previous.roles) != set(fresh.roles)
    ):
        raise ValueError("fresh baseline changed the canonical trace/task lineage universe")
    for lineage, role in previous.roles.items():
        if role in ("world_learning", "final_evaluation") and fresh.roles[lineage] != role:
            raise ValueError("fresh baseline moved historical learning or held-out evidence")
    tasks = load_task_set(project.artifacts, fresh.bindings.artifact_id)
    by_id = {task.task_id: task for task in tasks.tasks}
    for task_id in used_task_ids:
        if task_id not in by_id:
            raise ValueError("historically selected probe is missing from the canonical task set")
        lineage = by_id[task_id].lineage_group_id
        previous.require_lineage(lineage, "mining_probes")
        fresh.require_lineage(lineage, "mining_probes")


def propose_baseline_execution(
    project: ProjectStore,
    *,
    original_resume: ArtifactInput,
    build: BuildSettings,
    code_revision: str,
) -> BaselineExecutionProposal:
    """Freeze a concrete fresh scope for consent, without registering, granting or dispatching."""
    spec, proof = _history(project, original_resume)
    binding = proof.binding
    pilot = binding.pilot_revision
    if pilot is None:
        raise ValueError("fresh baseline requires the genuine original completed pilot build")
    if build.reviewer is None or build.reviewer.deadline > binding.deadline:
        raise ValueError("fresh baseline reviewer must stop at the original absolute deadline")
    historical = load_pilot_build(project, pilot)
    native, exposure, _ = _native(project, build, code_revision)
    used = tuple(sorted({pilot.task_id, *proof.execution_spec.task_ids}))
    old_exposure = load_exposure_plan(project, historical.exposure)
    _preserve_roles(project, old_exposure, exposure, used)
    return BaselineExecutionProposal(
        original_resume=original_resume,
        original_source=proof.source_ref,
        original_source_sha256=proof.source_sha256,
        original_operation=binding.original_operation,
        ledger_sha256=binding.ledger_sha256,
        source_build=binding.source_build,
        historical_build=pilot.saved_build,
        historical_exposure=historical.exposure,
        used_task_ids=used,
        build=build,
        plan=native,
        exposure=exposure,
        catalog_sha256=sha256_json(load_model_catalog(project.model_catalog_path)),
        execution_source=current_bootstrap_execution_source(code_revision),
        preparation_floor=_bound_bootstrap_plan(
            project, spec, proof.producer_revision, original_resume, binding, historical=True
        ),
    )


def _verified(
    project: ProjectStore, proposal: BaselineExecutionProposal, revision: str
) -> tuple[BootstrapResumeBinding, BootstrapPlan, VarianceBuildPlan, tuple[CostLine, ...]]:
    """Check all immutable pins against their actual native owners and current source."""
    spec, proof = _history(project, proposal.original_resume)
    parent = proof.binding
    pilot = parent.pilot_revision
    if pilot is None:
        raise ValueError("fresh baseline lost original completed pilot provenance")
    historical = load_pilot_build(project, pilot)
    native, exposure, costs = _native(project, proposal.build, revision)
    expected_used = tuple(sorted({pilot.task_id, *proof.execution_spec.task_ids}))
    if (
        proposal.protocol != BASELINE_PROTOCOL
        or proposal.build.reviewer is None
        or proposal.build.reviewer.deadline > parent.deadline
        or proposal.execution_source != current_bootstrap_execution_source(revision)
        or proposal.catalog_sha256 != sha256_json(load_model_catalog(project.model_catalog_path))
        or proposal.original_source != proof.source_ref
        or proposal.original_source_sha256 != proof.source_sha256
        or proposal.original_operation != parent.original_operation
        or proposal.ledger_sha256 != parent.ledger_sha256
        or proposal.source_build != parent.source_build
        or proposal.historical_build != pilot.saved_build
        or proposal.historical_exposure != historical.exposure
        or proposal.used_task_ids != expected_used
        or proposal.exposure != exposure
        or proposal.plan.model_copy(
            update={"saved": native.saved, "cached_decision_calls": native.cached_decision_calls}
        )
        != native
    ):
        raise ValueError(
            "fresh baseline source, settings, catalog, bank, input or provenance changed"
        )
    _preserve_roles(
        project, load_exposure_plan(project, historical.exposure), exposure, expected_used
    )
    floor = _bound_bootstrap_plan(
        project, spec, proof.producer_revision, proposal.original_resume, parent, historical=True
    )
    return parent, floor, native, costs


def _admit(
    parent: BootstrapResumeBinding,
    floor: BootstrapPlan,
    native: VarianceBuildPlan,
    costs: tuple[CostLine, ...],
) -> None:
    """Preserve original preparation and whole-objective caps plus the original absolute stop."""
    if native.saved:
        return
    if datetime.now(UTC) >= parent.deadline:
        raise ValueError("fresh baseline original absolute execution deadline expired")
    dollars = floor.known_usd + sum(row.usd for row in costs if row.usd is not None)
    attempts = floor.unpriced_attempts + sum(row.attempts for row in costs if row.usd is None)
    if dollars > min(parent.preparation_usd, parent.ceiling_usd) or attempts > min(
        parent.preparation_unpriced_attempts, parent.unpriced_attempts
    ):
        raise ValueError(
            "fresh baseline full native inventory plus original preparation reserve "
            "exceeds original caps"
        )


def _references(proposal: BaselineExecutionProposal) -> tuple[ArtifactInput, ...]:
    """Return exact genuine artifact ancestry without a manufactured completed parent."""
    return tuple(
        sorted(
            {
                proposal.original_resume,
                proposal.original_source,
                proposal.historical_build,
                proposal.historical_exposure,
                proposal.source_build.trace_dataset,
                proposal.source_build.task_set,
                proposal.source_build.serving_rag,
                proposal.source_build.fit_rag,
                proposal.source_build.world_model,
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def register_baseline_execution(
    project: ProjectStore,
    proposal: BaselineExecutionProposal,
    *,
    approval: BaselineExecutionApproval,
    code_revision: str,
) -> ArtifactInput:
    """Append one exact consented fresh binding only when full native work and reserve fit."""
    if approval.proposal_sha256 != sha256_json(proposal):
        raise ValueError("fresh baseline approval does not name the exact proposal")
    lock = project.artifacts.project_directory / (proposal.original_operation + ".lock")
    with CoordinationLock(lock, timeout=0):
        parent, floor, native, costs = _verified(project, proposal, code_revision)
        _admit(parent, floor, native, costs)
        identity = stable_id(BASELINE_TYPE, proposal)
        if not project.artifacts.exists(identity):
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type=BASELINE_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=_references(proposal),
                ),
                files={"binding.json": proposal, "approval.json": approval},
            )
        reference = artifact_input(project.artifacts.read(identity).manifest)
        _load(project, reference, code_revision)
        return reference


def _load(
    project: ProjectStore, reference: ArtifactInput, revision: str
) -> BaselineExecutionProposal:
    """Validate the immutable approval, exact registration identity and actual source producer."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    proposal = BaselineExecutionProposal.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "binding.json")
    )
    approval = BaselineExecutionApproval.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "approval.json")
    )
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != BASELINE_TYPE
        or manifest.code_revision != revision
        or reference.artifact_id != stable_id(BASELINE_TYPE, proposal)
        or manifest.inputs != _references(proposal)
        or approval.proposal_sha256 != sha256_json(proposal)
    ):
        raise ValueError("fresh baseline binding identity, approval, producer or ancestry changed")
    return proposal


def plan_baseline_execution(
    project: ProjectStore, reference: ArtifactInput, *, code_revision: str
) -> BaselineExecutionPlan:
    """Price actual current liabilities and native historical reserve without any execution."""
    proposal = _load(project, reference, code_revision)
    parent, floor, native, costs = _verified(project, proposal, code_revision)
    _admit(parent, floor, native, costs)
    return BaselineExecutionPlan(
        binding=reference,
        native=native,
        costs=costs,
        preparation_floor=floor,
        accounted_usd=floor.accounted_usd,
        accounted_unpriced_attempts=floor.accounted_unpriced_attempts,
        known_usd=floor.known_usd + sum(row.usd for row in costs if row.usd is not None),
        unpriced_attempts=floor.unpriced_attempts
        + sum(row.attempts for row in costs if row.usd is None),
        complete=native.saved,
    )


def _report(
    project: ProjectStore, proposal: BaselineExecutionProposal, revision: str
) -> ArtifactInput:
    """Accept only the actual committed native result with exact fresh settings and producer."""
    identity = proposal.plan.identity + "-result"
    manifest = project.artifacts.read(identity).manifest
    report = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(identity, "report.json")
    )
    if (
        manifest.artifact_type != REPORT_TYPE
        or manifest.code_revision != revision
        or report.identity != proposal.plan.identity
        or report.plan.identity != proposal.plan.identity
        or report.spec != proposal.build.spec
        or report.build != proposal.source_build
        or load_exposure_plan(project, report.exposure) != proposal.exposure
    ):
        raise ValueError("fresh baseline result is not the actual pinned native build report")
    return artifact_input(manifest)


def execute_baseline_execution(
    project: ProjectStore, reference: ArtifactInput, *, code_revision: str
) -> ArtifactInput:
    """Execute actual native clients under the original lock, ledger, reserve and deadline.

    Replays return the identical receipt with zero calls. A crash before report publication
    retains native physical liabilities and unfinished request intents, never a completion.
    """
    proposal = _load(project, reference, code_revision)
    lock = project.artifacts.project_directory / (proposal.original_operation + ".lock")
    with CoordinationLock(lock, timeout=0):
        parent, floor, native, costs = _verified(project, proposal, code_revision)
        _admit(parent, floor, native, costs)
        result_id = stable_id(BASELINE_RESULT_TYPE, reference)
        if project.artifacts.exists(result_id):
            report_ref = _report(project, proposal, code_revision)
            manifest = project.artifacts.read(result_id).manifest
            saved = ArtifactInput.model_validate_json(
                project.artifacts.read_bytes(result_id, "report.json")
            )
            if (
                manifest.artifact_type != BASELINE_RESULT_TYPE
                or manifest.code_revision != code_revision
                or manifest.inputs
                != tuple(sorted((reference, report_ref), key=lambda ref: ref.artifact_id))
                or saved != report_ref
            ):
                raise ValueError("fresh baseline completion receipt lost its actual native result")
            return artifact_input(manifest)

        ceiling = operation_spend_ceiling(project, parent.original_operation)
        authorizations = ceiling.summary().authorizations
        reserve_usd = floor.known_usd - floor.accounted_usd
        reserve_attempts = floor.unpriced_attempts - floor.accounted_unpriced_attempts
        request_usd = max(
            (row.usd / row.attempts for row in costs if row.usd is not None and row.attempts),
            default=0.0,
        )
        request_attempts = int(any(row.usd is None and row.attempts for row in costs))

        def before_claim() -> None:
            """Recheck physical liabilities and the frozen native reserve without replanning.

            The exclusive original operation lock prevents original preparation from changing
            its immutable work inventory during this phase. Native per-intent ledgers bound
            this phase's full admitted request inventory, including every physical retry.
            """
            spent = ceiling.summary()
            if (
                datetime.now(UTC) >= parent.deadline
                or spent.authorizations != authorizations
                or spent.accounted_usd + reserve_usd + request_usd
                > min(parent.preparation_usd, parent.ceiling_usd)
                or spent.unpriced_attempts + reserve_attempts + request_attempts
                > min(parent.preparation_unpriced_attempts, parent.unpriced_attempts)
            ):
                raise ValueError("fresh baseline dispatch exceeds original stop, caps or reserve")

        if not native.saved:
            with operation_spend_scope():
                install_spend_ceiling(ceiling)
                reviewer = family_reviewer(
                    project, proposal.build, code_revision=code_revision, before_claim=before_claim
                )
                run_variance_build(
                    project,
                    proposal.build.spec,
                    code_revision=code_revision,
                    reviewer=reviewer,
                    execution_deadline=parent.deadline,
                    before_claim=before_claim,
                )
        report_ref = _report(project, proposal, code_revision)
        manifest = project.artifacts.write_json(
            artifact_id=result_id,
            artifact_type=BASELINE_RESULT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted((reference, report_ref), key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": report_ref},
        )
        return artifact_input(manifest)
