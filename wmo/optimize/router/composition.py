"""End-to-end dependency-injected composition of one frozen customer router."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field, field_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_json,
    sorted_unique_inputs,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationCell,
    EvaluationCellEvidence,
    EvaluationPlan,
    EvaluationProtocol,
    FidelityReport,
    ObservedProductionCell,
    build_evaluation_plan,
    build_fidelity_report,
    default_fidelity_thresholds,
    persist_fidelity_thresholds,
)
from wmo.common.evaluations.evidence import (
    read_evaluation_plan,
    read_fidelity_gate,
    read_fidelity_report,
    read_judgment,
    read_rollout,
)
from wmo.common.evaluations.planning import plan_bound_fidelity_gate_id
from wmo.common.judging import Judge, Judgment
from wmo.common.models import (
    ModelCatalog,
    ProviderModelSelection,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
)
from wmo.common.observability.telemetry import capture_completion_once
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ProjectStore,
    artifact_input,
)
from wmo.common.rollouts import SimulationArtifactSet
from wmo.common.routing import KnnGuard, KnnRouterPolicy
from wmo.common.routing.bank import KnnBankManifest
from wmo.optimize.router.activation import load_project_router
from wmo.optimize.router.errors import RouterCompositionError
from wmo.optimize.router.evaluation.build import (
    completed_project_build,
    reconstruct_completed_project_build,
)
from wmo.optimize.router.evaluation.setup import verify_router_evaluation_setup
from wmo.optimize.router.evaluation.simulation_spec import build_router_simulation_spec
from wmo.optimize.router.evaluation.spend import observed_rollout_spend
from wmo.optimize.router.fit.spec import RouterFitResult
from wmo.optimize.router.fit.workflow import (
    EvaluationInputs,
    RouterFitConfig,
    RouterFitWorkflowResult,
    RouterReportConfig,
    RouterWorkflowResult,
    fit_router,
    report_router,
)
from wmo.optimize.router.judgment_budget import (
    JudgmentBudgetError,
    find_verified_judgment,
    find_verified_judgments,
    persist_dispatch_reservation,
    read_dispatch_reservation,
)
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router import RouterRuntime
from wmo.simulation.build import ProjectBuild
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.orchestration import Simulator
from wmo.simulation.specs import SimulationSpec, WorldModelSettings, simulation_spec_digest


class RouterCompositionBudget(ContractModel):
    """Finite dispatch ceilings required by the composed customer workflow."""

    maximum_simulation_cost_usd: float = Field(gt=0)
    maximum_judgments: int = Field(gt=0)

    @field_validator("maximum_simulation_cost_usd")
    @classmethod
    def _require_finite_cost(cls, value: float) -> float:
        """Return a finite simulation ceiling or reject it."""
        if not math.isfinite(value):
            raise ValueError("simulation budget must be finite")
        return value


@dataclass(frozen=True)
class RouterCandidateSetupPlan:
    """Confirmed candidate roles paired with the catalog state shown to the operator."""

    selection: RouterCandidateSelection
    candidate_models: tuple[ProviderModelSelection, ...]
    prospective_catalog: ModelCatalog
    expected_catalog_sha256: str


class ApprovedRouterReview(ContractModel):
    """Explicit approved rubric and eligible calibration artifact identities."""

    rubric_id: ArtifactId
    calibration_id: ArtifactId


class RouterEvaluationSetup(ContractModel):
    """Reviewed completed inputs and bounded simulation controls for router evaluation."""

    candidates: tuple[RoutedCandidateSnapshot, ...]
    observed_cells: tuple[ObservedProductionCell, ...]
    production_protocol: EvaluationProtocol
    simulation_protocol: EvaluationProtocol
    embedding_set_id: ArtifactId
    fit_rag_input: ArtifactInput
    pricing_snapshot_id: ArtifactId
    guard: KnnGuard
    incumbent_alias: ArtifactId | None = None
    judgment_status: Literal["provisional", "human_calibrated"]
    world_model_settings: WorldModelSettings
    simulation_completion_input: ArtifactInput | None = None
    fidelity_planned_overlaps: int = Field(default=10, gt=0)
    fidelity_minimum_usable_overlaps: int = Field(default=8, gt=0)
    agent_id: str = Field(min_length=1, max_length=256)
    seed: int
    maximum_steps: int = Field(gt=0)
    maximum_concurrency: int = Field(gt=0)


class ReviewSupplier(Protocol):
    """Obtains approved rubric and calibration artifacts under an explicit budget."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        budget: RouterCompositionBudget,
    ) -> ApprovedRouterReview:
        """Return manifest-persisted review artifacts."""


class EvaluationSetupSupplier(Protocol):
    """Provides reviewed planning inputs and already completed production evidence."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Return explicit immutable planning inputs and finite simulation controls."""


class SimulatorFactory(Protocol):
    """Binds an injected simulator only after WMO has frozen its evaluation plan."""

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Return a simulator bound to the exact persisted plan."""


class FidelityApproval(Protocol):
    """Owns the explicit human or application approval boundary for passing fidelity."""

    def __call__(
        self,
        project: ProjectStore,
        plan: EvaluationPlan,
        evidence: tuple[EvaluationCellEvidence, ...],
        budget: RouterCompositionBudget,
    ) -> FidelityApprovalDecision:
        """Return explicit approval actor, evidence, and time for reviewed fidelity evidence."""


class FidelityApprovalDecision(ContractModel):
    """Non-secret actor evidence returned by one explicit fidelity approval callback."""

    actor_id: str = Field(min_length=1, max_length=256)
    evidence: str = Field(min_length=1, max_length=2_000)
    approved_at: datetime


class FidelityApprovalReceipt(ArtifactEnvelope):
    """Immutable exact plan, gate, report, actor, and approval-evidence binding."""

    approval_id: ArtifactId
    plan: ArtifactInput
    gate: ArtifactInput
    report: ArtifactInput
    protocol_sha256: Sha256
    actor_id: str = Field(min_length=1, max_length=256)
    evidence: str = Field(min_length=1, max_length=2_000)
    approved_at: datetime


class RouterPolicyLock(ArtifactEnvelope):
    """Immutable proof that fit-only evaluation froze a verified bank and policy."""

    lock_id: ArtifactId
    plan: ArtifactInput
    fit_evaluation: ArtifactInput
    bank: ArtifactInput
    policy: ArtifactInput
    fit_config_sha256: Sha256


@dataclass(frozen=True)
class RouterWorkflowServices:
    """Every service capable of dispatching work in the customer workflow."""

    review_supplier: ReviewSupplier
    setup_supplier: EvaluationSetupSupplier
    simulator_factory: SimulatorFactory
    judge: Judge
    fidelity_approval: FidelityApproval
    runtime_catalog: RuntimeModelCatalog
    evaluation_plan_inputs: tuple[ArtifactInput, ...] = ()


@dataclass(frozen=True)
class RouterCompositionResult:
    """Complete local artifact chain and loaded frozen online runtime."""

    build: ProjectBuild
    review: ApprovedRouterReview
    plan: EvaluationPlan
    simulation_spec: SimulationSpec
    held_out_simulation_spec: SimulationSpec
    fidelity_approval_id: ArtifactId
    policy_lock_id: ArtifactId
    fidelity_report_id: ArtifactId
    phase_a_simulation_spend_usd: float
    held_out_simulation_spend_usd: float
    total_simulation_spend_usd: float
    optimization: RouterWorkflowResult
    runtime: RouterRuntime


def compose_router(
    project: ProjectStore,
    trace_source: TraceNormalizationResult,
    *,
    services: RouterWorkflowServices,
    budget: RouterCompositionBudget,
    created_at: datetime,
    code_revision: str,
    phase_hook: Callable[[str], None] | None = None,
) -> RouterCompositionResult:
    """Build evidence, evaluate, freeze, report, and load one router with explicit services.

    Args:
        project: Initialized local project store.
        trace_source: Canonical normalized traces used to build task evidence.
        services: Review, simulation, judging, and runtime dependencies. None are auto-resolved.
        budget: Finite simulation spend and judgment-call ceilings.
        created_at: Timezone-aware artifact completion time.
        code_revision: Exact code revision for every new artifact.
        phase_hook: Optional local observer used to audit phase ordering.

    Returns:
        The complete immutable artifact chain and W11 frozen runtime.

    Raises:
        RouterCompositionError: A dependency, budget, artifact, or resume binding is invalid.
    """
    started = time.monotonic()
    _preflight(project, services, budget, code_revision)
    completed_build = completed_project_build(project)
    built = reconstruct_completed_project_build(
        project,
        trace_source,
        created_at=created_at,
    )
    _phase(phase_hook, "review")
    review = services.review_supplier(project, built, budget)
    _verify_review(project, review)
    setup = services.setup_supplier(project, built, review, budget)
    verify_router_evaluation_setup(
        completed=completed_build,
        fit_rag_input=setup.fit_rag_input,
        grounded_world_model_input=setup.world_model_settings.grounded_world_model_input,
        production_protocol=setup.production_protocol,
        simulation_protocol=setup.simulation_protocol,
        rubric_id=review.rubric_id,
        calibration_id=review.calibration_id,
    )

    thresholds = default_fidelity_thresholds(
        created_at=created_at,
        code_revision=code_revision,
        planned_overlaps=setup.fidelity_planned_overlaps,
        minimum_usable_overlaps=setup.fidelity_minimum_usable_overlaps,
    )
    persist_fidelity_thresholds(project.artifacts, thresholds)
    plan = build_evaluation_plan(
        project.artifacts,
        task_set_id=built.artifacts.task_set.task_set_id,
        candidate_snapshots=setup.candidates,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        observed_cells=setup.observed_cells,
        fidelity_thresholds_id=thresholds.fidelity_thresholds_id,
        fidelity_protocol_sha256=_protocol_digest(setup.simulation_protocol),
        additional_inputs=services.evaluation_plan_inputs,
        created_at=created_at,
        code_revision=code_revision,
    )
    plan_input = artifact_input(project.artifacts.read(plan.plan_id).manifest)
    task_input = built.review.task_set
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}

    _phase(phase_hook, "fidelity_fit_started")
    phase_a_cells = tuple(cell for cell in plan.cells if cell.purpose in {"fit", "fidelity"})
    spec = build_router_simulation_spec(
        plan,
        plan_input,
        task_input,
        setup,
        budget.maximum_simulation_cost_usd,
        created_at,
        code_revision,
        phase_a_cells,
        phase="fidelity-fit",
    )
    phase_a_set = _run_or_load_simulation(project, plan, spec, services.simulator_factory)
    phase_a_spend = _verified_simulation_spend(project, phase_a_set)
    if phase_a_spend > budget.maximum_simulation_cost_usd:
        raise RouterCompositionError("verified Phase A simulation spend exceeds the total budget")
    remaining_cost_usd = max(0.0, budget.maximum_simulation_cost_usd - phase_a_spend)
    if remaining_cost_usd <= 0 and any(
        cell.purpose == "held_out" and cell.execution == "simulate" for cell in plan.cells
    ):
        raise RouterCompositionError(
            "Phase A consumed the total simulation budget; held-out dispatch is blocked"
        )
    phase_a_evidence, phase_a_consumed = _complete_cell_evidence(
        project,
        plan_input,
        phase_a_cells,
        phase_a_set.artifact_ids,
        setup,
        review,
        services.judge,
        budget.maximum_judgments,
    )
    fidelity_evidence = tuple(
        item for item in phase_a_evidence if cells_by_id[item.cell_id].purpose == "fidelity"
    )
    fidelity, approval = _approve_fidelity_once(
        project,
        plan,
        setup.simulation_protocol,
        phase_a_evidence,
        fidelity_evidence,
        services.fidelity_approval,
        budget,
        created_at,
        code_revision,
    )
    approved_protocol = setup.simulation_protocol.model_copy(
        update={"fidelity_report_id": fidelity.fidelity_report_id}
    )
    fit_config = RouterFitConfig(
        fit=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            rollout_set_ids=(phase_a_set.artifact_set_id,),
            protocols=(setup.production_protocol, approved_protocol),
            cell_evidence=phase_a_evidence,
            fidelity_report_ids=(fidelity.fidelity_report_id,),
        ),
        embedding_set_id=setup.embedding_set_id,
        incumbent_alias=setup.incumbent_alias,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        guard=setup.guard,
        judgment_status=setup.judgment_status,
        created_at=created_at,
        code_revision=code_revision,
    )
    fit, policy_lock = _fit_and_lock_once(
        project,
        plan_input,
        fit_config,
        created_at,
        code_revision,
    )
    _phase(phase_hook, "policy_locked")

    held_cells = tuple(cell for cell in plan.cells if cell.purpose == "held_out")
    _phase(phase_hook, "heldout_opened")
    held_spec = build_router_simulation_spec(
        plan,
        plan_input,
        task_input,
        setup,
        remaining_cost_usd,
        created_at,
        code_revision,
        held_cells,
        phase="heldout",
    )
    held_set = _run_or_load_simulation(project, plan, held_spec, services.simulator_factory)
    held_out_spend = _verified_simulation_spend(project, held_set)
    if math.fsum((phase_a_spend, held_out_spend)) > budget.maximum_simulation_cost_usd:
        raise RouterCompositionError("verified composed simulation spend exceeds the total budget")
    held_evidence, _held_dispatched = _complete_cell_evidence(
        project,
        plan_input,
        held_cells,
        held_set.artifact_ids,
        setup,
        review,
        services.judge,
        budget.maximum_judgments - phase_a_consumed,
    )
    optimized = report_router(
        project.artifacts,
        fit,
        RouterReportConfig(
            held_out=EvaluationInputs(
                evaluation_plan_id=plan.plan_id,
                rollout_set_ids=(held_set.artifact_set_id,),
                protocols=(setup.production_protocol, approved_protocol),
                cell_evidence=held_evidence,
                fidelity_report_ids=(fidelity.fidelity_report_id,),
            ),
            embedding_set_id=setup.embedding_set_id,
            created_at=created_at,
            code_revision=code_revision,
        ),
    )
    total_spend = math.fsum((phase_a_spend, held_out_spend))
    completion_id = optimized.optimization.report.report_id
    capture_completion_once(
        "wmo simulation completed",
        completion_id,
        {
            "success": True,
            "rollout_count": len(phase_a_set.artifact_ids) + len(held_set.artifact_ids),
            "duration_seconds": max(time.monotonic() - started, 0.0),
            "cost_usd": total_spend,
        },
        root=project.paths.root,
    )
    _phase(phase_hook, "report_complete")
    runtime = load_project_router(
        project.paths.project_id,
        project.paths.root,
        policy_id=optimized.optimization.policy.policy_id,
        runtime_catalog=services.runtime_catalog,
    )
    return RouterCompositionResult(
        build=built,
        review=review,
        plan=plan,
        simulation_spec=spec,
        held_out_simulation_spec=held_spec,
        fidelity_approval_id=approval.approval_id,
        policy_lock_id=policy_lock.lock_id,
        fidelity_report_id=fidelity.fidelity_report_id,
        phase_a_simulation_spend_usd=phase_a_spend,
        held_out_simulation_spend_usd=held_out_spend,
        total_simulation_spend_usd=total_spend,
        optimization=optimized,
        runtime=runtime,
    )


def _run_or_load_simulation(
    project: ProjectStore,
    plan: EvaluationPlan,
    spec: SimulationSpec,
    simulator_factory: SimulatorFactory,
) -> SimulationArtifactSet:
    """Load an exactly completed simulation set without invoking its simulator again."""
    matches = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "simulation-artifact-set":
            continue
        artifact_set = SimulationArtifactSet.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "artifact-set.json")
        )
        if artifact_set.simulation_id != spec.simulation_id:
            continue
        index_payload = project.artifacts.read_bytes(artifact_id, artifact_set.artifacts_path)
        if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
            raise RouterCompositionError("simulation artifact-set index digest has drifted")
        rollouts = tuple(
            read_rollout(project.artifacts, rollout_id)[0]
            for rollout_id in artifact_set.artifact_ids
        )
        expected_digest = simulation_spec_digest(spec)
        rollout_cell_ids = tuple(rollout.cell_id for rollout in rollouts)
        if (
            artifact_set.artifact_set_id != artifact_id
            or any(cell_id is None for cell_id in rollout_cell_ids)
            or tuple(sorted(cell_id for cell_id in rollout_cell_ids if cell_id is not None))
            != spec.cell_ids
            or any(
                rollout.source_run_id != spec.simulation_id
                or rollout.simulation_id != spec.simulation_id
                or rollout.simulation_spec_sha256 != expected_digest
                for rollout in rollouts
            )
        ):
            raise RouterCompositionError(
                "completed simulation artifact set differs from phase spec"
            )
        matches.append(artifact_set)
    if len(matches) > 1:
        raise RouterCompositionError("multiple completed artifact sets name one simulation phase")
    if matches:
        return matches[0]
    return simulator_factory(project, plan).run(spec)


def _verified_simulation_spend(
    project: ProjectStore,
    expected: SimulationArtifactSet,
) -> float:
    """Recompute one phase's spend from verified immutable rollouts.

    Args:
        project: Project store containing the completed simulation artifacts.
        expected: Exact artifact set returned for the simulation phase.

    Returns:
        Finite total of candidate, world-model, and retrieval dispatch spend.

    Raises:
        RouterCompositionError: The set, index, rollout, or economics cannot be verified.
    """
    stored = project.artifacts.read(expected.artifact_set_id)
    if stored.manifest.artifact_type != "simulation-artifact-set":
        raise RouterCompositionError("simulation spend source has the wrong artifact type")
    artifact_set = SimulationArtifactSet.model_validate_json(
        project.artifacts.read_bytes(expected.artifact_set_id, "artifact-set.json")
    )
    if artifact_set != expected:
        raise RouterCompositionError("simulation spend source differs from its completed set")
    index_payload = project.artifacts.read_bytes(
        expected.artifact_set_id, artifact_set.artifacts_path
    )
    if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
        raise RouterCompositionError("simulation spend index digest has drifted")
    values = tuple(
        observed_rollout_spend(read_rollout(project.artifacts, rollout_id)[0])
        for rollout_id in artifact_set.artifact_ids
    )
    return math.fsum(values)


def _approve_fidelity_once(
    project: ProjectStore,
    plan: EvaluationPlan,
    protocol: EvaluationProtocol,
    phase_a_evidence: tuple[EvaluationCellEvidence, ...],
    fidelity_evidence: tuple[EvaluationCellEvidence, ...],
    approval_service: FidelityApproval,
    budget: RouterCompositionBudget,
    created_at: datetime,
    code_revision: str,
) -> tuple[FidelityReport, FidelityApprovalReceipt]:
    """Call approval exactly once, persist its exact receipt, and verify every replay."""
    plan_value, plan_input = read_evaluation_plan(project.artifacts, plan.plan_id)
    protocol_sha256 = _protocol_digest(protocol)
    gate_id = plan_bound_fidelity_gate_id(plan_input.sha256, protocol_sha256)
    _gate, gate_input = read_fidelity_gate(project.artifacts, gate_id)
    approval_id = stable_id(
        "fidelity-approval",
        {
            "plan": plan_input.model_dump(mode="json"),
            "gate": gate_input.model_dump(mode="json"),
            "protocol_sha256": protocol_sha256,
        },
    )
    destination = project.artifacts.project_directory / "artifacts" / approval_id
    if destination.exists():
        if project.artifacts.read(approval_id).manifest.artifact_type != "fidelity-approval":
            raise RouterCompositionError("fidelity approval identity has the wrong artifact type")
        receipt = FidelityApprovalReceipt.model_validate_json(
            project.artifacts.read_bytes(approval_id, "approval.json")
        )
        report, report_input = read_fidelity_report(project.artifacts, receipt.report.artifact_id)
        if (
            plan_value != plan
            or receipt.plan != plan_input
            or receipt.gate != gate_input
            or receipt.report != report_input
            or receipt.protocol_sha256 != protocol_sha256
            or sorted_unique_inputs(*receipt.inputs)
            != sorted_unique_inputs(plan_input, gate_input, report_input)
        ):
            raise RouterCompositionError("fidelity approval receipt binding has drifted")
        replay = build_fidelity_report(
            project.artifacts,
            evaluation_plan_id=plan.plan_id,
            protocol=protocol,
            cell_evidence=phase_a_evidence,
            created_at=created_at,
            code_revision=code_revision,
            approved_at=receipt.approved_at,
        )
        if replay != report:
            raise RouterCompositionError("approved fidelity report differs from exact replay")
        return report, receipt
    decision = approval_service(project, plan, fidelity_evidence, budget)
    report = build_fidelity_report(
        project.artifacts,
        evaluation_plan_id=plan.plan_id,
        protocol=protocol,
        cell_evidence=phase_a_evidence,
        created_at=created_at,
        code_revision=code_revision,
        approved_at=decision.approved_at,
    )
    report_input = artifact_input(project.artifacts.read(report.fidelity_report_id).manifest)
    receipt = FidelityApprovalReceipt(
        schema_version=1,
        created_at=created_at,
        inputs=sorted_unique_inputs(plan_input, gate_input, report_input),
        code_revision=code_revision,
        approval_id=approval_id,
        plan=plan_input,
        gate=gate_input,
        report=report_input,
        protocol_sha256=protocol_sha256,
        actor_id=decision.actor_id,
        evidence=decision.evidence,
        approved_at=decision.approved_at,
    )
    project.artifacts.write_json(
        artifact_id=approval_id,
        artifact_type="fidelity-approval",
        envelope=receipt,
        files={"approval.json": receipt},
    )
    return report, receipt


def _fit_and_lock_once(
    project: ProjectStore,
    plan_input: ArtifactInput,
    config: RouterFitConfig,
    created_at: datetime,
    code_revision: str,
) -> tuple[RouterFitWorkflowResult, RouterPolicyLock]:
    """Freeze fit once, persist an exact lock, and load it without refitting on replay."""
    config_sha256 = sha256_json(config)
    lock_id = stable_id(
        "router-policy-lock",
        {"plan": plan_input.model_dump(mode="json"), "fit_config_sha256": config_sha256},
    )
    destination = project.artifacts.project_directory / "artifacts" / lock_id
    if destination.exists():
        if project.artifacts.read(lock_id).manifest.artifact_type != "router-policy-lock":
            raise RouterCompositionError("router policy lock identity has the wrong artifact type")
        lock = RouterPolicyLock.model_validate_json(
            project.artifacts.read_bytes(lock_id, "lock.json")
        )
        fit_input = artifact_input(project.artifacts.read(lock.fit_evaluation.artifact_id).manifest)
        bank_input = artifact_input(project.artifacts.read(lock.bank.artifact_id).manifest)
        policy_input = artifact_input(project.artifacts.read(lock.policy.artifact_id).manifest)
        if (
            lock.plan != plan_input
            or lock.fit_config_sha256 != config_sha256
            or (lock.fit_evaluation, lock.bank, lock.policy)
            != (fit_input, bank_input, policy_input)
            or sorted_unique_inputs(*lock.inputs)
            != sorted_unique_inputs(plan_input, fit_input, bank_input, policy_input)
        ):
            raise RouterCompositionError("router policy lock binding has drifted")
        policy = KnnRouterPolicy.model_validate_json(
            project.artifacts.read_bytes(policy_input.artifact_id, "policy.json")
        )
        bank = KnnBankManifest.model_validate_json(
            project.artifacts.read_bytes(bank_input.artifact_id, "bank.json")
        )
        if policy.bank_artifact_id != bank.bank_artifact_id:
            raise RouterCompositionError("locked policy and bank identities differ")
        return (
            RouterFitWorkflowResult(
                fit_evaluation_id=fit_input.artifact_id,
                locked=RouterFitResult(policy=policy, bank=bank),
            ),
            lock,
        )
    fit = fit_router(project.artifacts, config)
    fit_input = artifact_input(project.artifacts.read(fit.fit_evaluation_id).manifest)
    bank_input = artifact_input(project.artifacts.read(fit.locked.bank.bank_artifact_id).manifest)
    policy_input = artifact_input(project.artifacts.read(fit.locked.policy.policy_id).manifest)
    inputs = sorted_unique_inputs(plan_input, fit_input, bank_input, policy_input)
    lock = RouterPolicyLock(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        lock_id=lock_id,
        plan=plan_input,
        fit_evaluation=fit_input,
        bank=bank_input,
        policy=policy_input,
        fit_config_sha256=config_sha256,
    )
    project.artifacts.write_json(
        artifact_id=lock_id,
        artifact_type="router-policy-lock",
        envelope=lock,
        files={"lock.json": lock},
    )
    return fit, lock


def _preflight(
    project: ProjectStore,
    services: RouterWorkflowServices,
    budget: RouterCompositionBudget,
    code_revision: str,
) -> None:
    """Fail before dispatch when explicit configuration or finite budgets are absent."""
    project.load_project()
    if not code_revision.strip():
        raise RouterCompositionError("code_revision is required before workflow dispatch")
    if budget.maximum_judgments <= 0 or not math.isfinite(budget.maximum_simulation_cost_usd):
        raise RouterCompositionError("finite simulation and judging budgets are required")
    if any(
        service is None
        for service in (
            services.review_supplier,
            services.setup_supplier,
            services.simulator_factory,
            services.judge,
            services.fidelity_approval,
            services.runtime_catalog,
        )
    ):
        raise RouterCompositionError("all workflow services must be injected explicitly")


def _verify_review(project: ProjectStore, review: ApprovedRouterReview) -> None:
    """Require exact persisted rubric and calibration artifact kinds before simulation."""
    expected = ((review.rubric_id, "rubric"), (review.calibration_id, "judge-calibration"))
    for artifact_id, artifact_type in expected:
        if project.artifacts.read(artifact_id).manifest.artifact_type != artifact_type:
            raise RouterCompositionError(f"{artifact_id} is not a completed {artifact_type}")


def _complete_cell_evidence(
    project: ProjectStore,
    plan_input: ArtifactInput,
    cells: tuple[EvaluationCell, ...],
    simulated_rollout_ids: tuple[str, ...],
    setup: RouterEvaluationSetup,
    review: ApprovedRouterReview,
    judge: Judge,
    maximum_judgments: int,
) -> tuple[tuple[EvaluationCellEvidence, ...], int]:
    """Verify evidence and reserve each bounded judgment dispatch durably before calling it."""
    rollouts_by_cell = {}
    for rollout_id in simulated_rollout_ids:
        rollout, _input = read_rollout(project.artifacts, rollout_id)
        if rollout.cell_id is None or rollout.cell_id in rollouts_by_cell:
            raise RouterCompositionError("simulator output lacks unique evaluation cell bindings")
        rollouts_by_cell[rollout.cell_id] = rollout
    observed = {
        (item.task_id, item.candidate_alias, item.repeat): item.rollout_artifact_id
        for item in setup.observed_cells
    }
    bound_cells = []
    protocols_by_rollout: dict[str, EvaluationProtocol] = {}
    for cell in cells:
        rollout_id = (
            cell.observed_rollout_id
            if cell.execution == "observed"
            else getattr(rollouts_by_cell.get(cell.cell_id), "rollout_id", None)
        )
        if rollout_id is None:
            raise RouterCompositionError(
                f"no completed rollout exists for planned cell {cell.cell_id}"
            )
        if (
            cell.execution == "observed"
            and observed.get((cell.task_id, cell.candidate_alias, cell.repeat)) != rollout_id
        ):
            raise RouterCompositionError("observed rollout binding changed after planning")
        protocol = (
            setup.production_protocol if cell.execution == "observed" else setup.simulation_protocol
        )
        existing_protocol = protocols_by_rollout.setdefault(rollout_id, protocol)
        if existing_protocol != protocol:
            raise RouterCompositionError("one rollout is bound to conflicting evaluation protocols")
        bound_cells.append((cell, rollout_id, protocol))
    try:
        judgments_by_rollout = find_verified_judgments(
            project,
            protocols_by_rollout=protocols_by_rollout,
            rubric_id=review.rubric_id,
            calibration_id=review.calibration_id,
        )
    except JudgmentBudgetError as exc:
        raise RouterCompositionError(str(exc)) from exc

    evidence = []
    consumed = 0
    for cell, rollout_id, protocol in bound_cells:
        try:
            judgment = judgments_by_rollout.get(rollout_id)
            receipt = read_dispatch_reservation(
                project,
                plan_input,
                cell,
                rollout_id,
                review.rubric_id,
                review.calibration_id,
                protocol,
            )
            if judgment is None and receipt is not None:
                judgment = find_verified_judgment(
                    project,
                    rollout_id,
                    review.rubric_id,
                    review.calibration_id,
                    protocol,
                )
                if judgment is not None:
                    judgments_by_rollout[rollout_id] = judgment
        except JudgmentBudgetError as exc:
            raise RouterCompositionError(str(exc)) from exc
        if judgment is not None or receipt is not None:
            consumed += 1
        if consumed > maximum_judgments:
            raise RouterCompositionError("judgment dispatch budget exhausted")
        if judgment is None:
            if receipt is not None:
                raise RouterCompositionError(
                    "reserved judgment dispatch has no completed judgment; retry is blocked"
                )
            if consumed >= maximum_judgments:
                raise RouterCompositionError("judgment dispatch budget exhausted")
            try:
                persist_dispatch_reservation(
                    project,
                    plan_input,
                    cell,
                    rollout_id,
                    review.rubric_id,
                    review.calibration_id,
                    protocol,
                )
            except JudgmentBudgetError as exc:
                raise RouterCompositionError(str(exc)) from exc
            consumed += 1
            judgment = judge.judge_persisted(
                project,
                rollout_artifact_id=rollout_id,
                rubric_artifact_id=review.rubric_id,
                calibration_artifact_id=review.calibration_id,
            )
            _persist_judgment(project, judgment)
            judgments_by_rollout[rollout_id] = judgment
        rollout, _input = read_rollout(project.artifacts, rollout_id)
        evidence.append(
            EvaluationCellEvidence(
                cell_id=cell.cell_id,
                protocol_id=protocol.protocol_id,
                rollout_artifact_id=rollout_id,
                judgment_artifact_id=judgment.judgment_id,
                source_run_id=rollout.source_run_id,
            )
        )
    return tuple(evidence), consumed


def _persist_judgment(project: ProjectStore, judgment: Judgment) -> None:
    """Persist or exactly verify one deterministic injected-judge result."""
    try:
        project.artifacts.write_json(
            artifact_id=judgment.judgment_id,
            artifact_type="judgment",
            envelope=judgment,
            files={"judgment.json": judgment},
        )
    except ArtifactAlreadyExistsError:
        existing, _input = read_judgment(project.artifacts, judgment.judgment_id)
        if existing != judgment:
            raise RouterCompositionError(
                "existing judgment differs from injected judge result"
            ) from None


def _protocol_digest(protocol: EvaluationProtocol) -> str:
    """Return the canonical fidelity digest without a circular report identity."""
    from wmo.common.evaluations.evidence import evaluation_protocol_digest

    return evaluation_protocol_digest(protocol)


def _phase(hook: Callable[[str], None] | None, phase: str) -> None:
    """Emit one local ordering marker without changing workflow behavior."""
    if hook is not None:
        hook(phase)
