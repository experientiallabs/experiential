"""End-to-end dependency-injected composition of one frozen customer router."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Protocol

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
    EvaluationPlan,
    EvaluationProtocol,
    ObservedProductionCell,
    build_evaluation_plan,
)
from wmo.common.evaluations.evidence import (
    read_rollout,
)
from wmo.common.judging import Judge, verify_persisted_calibration
from wmo.common.models import (
    ModelCatalog,
    ProviderConnection,
    ProviderModelSelection,
    RoutedCandidateSnapshot,
    RouterCandidateSelection,
)
from wmo.common.observability.telemetry import capture_completion_once
from wmo.common.progress import ProgressHook, report
from wmo.common.project import (
    ProjectStore,
    artifact_input,
)
from wmo.common.rollouts import (
    SimulationArtifactSet,
)
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
from wmo.optimize.router.evaluation.spend import verified_simulation_spend
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
from wmo.optimize.router.judgment_budget import complete_cell_evidence
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router import RouterRuntime
from wmo.simulation.build import ProjectBuild
from wmo.simulation.engines.text.resume import reexecutable_dispatch_failure
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.orchestration import Simulator
from wmo.simulation.specs import SimulationSpec, WorldModelSettings, simulation_spec_digest

logger = logging.getLogger(__name__)


class RouterCompositionBudget(ContractModel):
    """Finite dispatch ceilings required by the composed customer workflow.

    ``maximum_simulation_cost_usd`` is one shared provider-spend pool: simulation and judging
    both draw from it as reconciled actual cost, with no estimate-based carve-outs. By default it
    authorizes the whole run, so spend crossing it logs a warning and the run completes;
    ``stop_on_overspend`` instead blocks the next dispatch once reconciled spend reaches it.
    """

    maximum_simulation_cost_usd: float = Field(gt=0)
    maximum_judgments: int = Field(gt=0)
    stop_on_overspend: bool = False

    @field_validator("maximum_simulation_cost_usd")
    @classmethod
    def _require_finite_cost(cls, value: float) -> float:
        """Return a finite simulation ceiling or reject it."""
        if not math.isfinite(value):
            raise ValueError("simulation budget must be finite")
        return value


def _spend_ceiling_crossed(stop_on_overspend: bool, error: str, detail: str) -> None:
    """Fail closed in stop mode or log that the authorized run continues past its ceiling."""
    if stop_on_overspend:
        raise RouterCompositionError(error)
    logger.warning("%s; continuing because the run is already authorized", detail)


@dataclass(frozen=True)
class RouterCandidateSetupPlan:
    """Confirmed candidate roles paired with the catalog state shown to the operator."""

    selection: RouterCandidateSelection
    candidate_models: tuple[ProviderModelSelection, ...]
    prospective_catalog: ModelCatalog
    expected_catalog_sha256: str
    candidate_connections: tuple[ProviderConnection, ...] = ()


class ApprovedRouterReview(ContractModel):
    """Explicit approved rubric and eligible calibration artifact identities."""

    judgment_status: Literal["human_calibrated"] = "human_calibrated"
    rubric_id: ArtifactId
    calibration_id: ArtifactId
    calibration_input: ArtifactInput | None = None
    audit_input: ArtifactInput | None = None

    @property
    def artifact_inputs(self) -> tuple[ArtifactInput, ...]:
        """Return exact approved provenance inputs available to composition."""
        return tuple(
            item for item in (self.audit_input, self.calibration_input) if item is not None
        )


class ProvisionalRouterReview(ContractModel):
    """Explicit zero-label calibration provenance eligible only for provisional routing."""

    judgment_status: Literal["provisional"] = "provisional"
    rubric_id: ArtifactId
    calibration_id: ArtifactId
    calibration_input: ArtifactInput

    @property
    def artifact_inputs(self) -> tuple[ArtifactInput, ...]:
        """Return the exact provisional calibration input bound into evaluation evidence."""
        return (self.calibration_input,)


RouterReviewProvenance = Annotated[
    ApprovedRouterReview | ProvisionalRouterReview,
    Field(discriminator="judgment_status"),
]


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
    ) -> RouterReviewProvenance:
        """Return manifest-persisted review artifacts."""


class EvaluationSetupSupplier(Protocol):
    """Provides reviewed planning inputs and already completed production evidence."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: RouterReviewProvenance,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        """Return explicit immutable planning inputs and finite simulation controls."""


class SimulatorFactory(Protocol):
    """Binds an injected simulator only after WMO has frozen its evaluation plan."""

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        """Return a simulator bound to the exact persisted plan."""


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
    runtime_catalog: RuntimeModelCatalog
    evaluation_plan_inputs: tuple[ArtifactInput, ...] = ()


@dataclass(frozen=True)
class RouterCompositionResult:
    """Complete local artifact chain and loaded frozen online runtime."""

    build: ProjectBuild
    review: RouterReviewProvenance
    plan: EvaluationPlan
    simulation_spec: SimulationSpec
    held_out_simulation_spec: SimulationSpec
    policy_lock_id: ArtifactId
    fit_simulation_spend_usd: float
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
    progress: ProgressHook | None = None,
    policy_lock_hook: Callable[[RouterPolicyLock, RouterFitWorkflowResult], None] | None = None,
) -> RouterCompositionResult:
    """Build evidence, evaluate, freeze, report, and load one router with explicit services.

    Args:
        project: Initialized local project store.
        trace_source: Canonical normalized traces used to build task evidence.
        services: Review, simulation, judging, and runtime dependencies. None are auto-resolved.
        budget: Finite simulation spend and judgment-call ceilings.
        created_at: Timezone-aware artifact completion time. Once the evaluation plan is
            persisted, every later phase artifact adopts the plan's completion time so a rerun
            replays completed simulation, fit, and report work deterministically.
        code_revision: Exact code revision for every new artifact.
        phase_hook: Optional local observer used to audit phase ordering.
        progress: Optional observer of truthful stage names and exact unit counts.
        policy_lock_hook: Optional durable-stage observer invoked after the immutable fit lock and
            before any held-out simulation opens.

    Returns:
        The complete immutable artifact chain and W11 frozen runtime.

    Raises:
        RouterCompositionError: A dependency, budget, artifact, or resume binding is invalid.
    """
    started = time.monotonic()
    report(progress, "preflight")
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
    if review.calibration_input is not None and setup.judgment_status != review.judgment_status:
        raise RouterCompositionError(
            "router evaluation judgment status differs from its calibration provenance"
        )
    verify_router_evaluation_setup(
        completed=completed_build,
        fit_rag_input=setup.fit_rag_input,
        grounded_world_model_input=setup.world_model_settings.grounded_world_model_input,
        production_protocol=setup.production_protocol,
        simulation_protocol=setup.simulation_protocol,
        rubric_id=review.rubric_id,
        calibration_id=review.calibration_id,
    )

    plan = build_evaluation_plan(
        project.artifacts,
        task_set_id=built.artifacts.task_set.task_set_id,
        candidate_snapshots=setup.candidates,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        observed_cells=setup.observed_cells,
        additional_inputs=sorted_unique_inputs(
            *services.evaluation_plan_inputs,
            *review.artifact_inputs,
        ),
        created_at=created_at,
        code_revision=code_revision,
    )
    plan_input = artifact_input(project.artifacts.read(plan.plan_id).manifest)
    task_input = built.review.task_set
    artifact_time = plan.created_at
    _phase(phase_hook, "fit_started")
    fit_cells = tuple(cell for cell in plan.cells if cell.purpose == "fit")
    spec = build_router_simulation_spec(
        plan,
        plan_input,
        task_input,
        setup,
        budget.maximum_simulation_cost_usd,
        artifact_time,
        code_revision,
        fit_cells,
        phase="fit",
        stop_on_overspend=budget.stop_on_overspend,
    )
    fit_set = _run_or_load_simulation(
        project,
        plan,
        spec,
        services.simulator_factory,
        progress=progress,
        progress_detail="fit",
    )
    fit_spend = verified_simulation_spend(project, fit_set, setup.simulation_completion_input)
    if fit_spend > budget.maximum_simulation_cost_usd:
        _spend_ceiling_crossed(
            budget.stop_on_overspend,
            "verified fit simulation spend exceeds the total budget",
            f"verified fit simulation spend ${fit_spend:.4f} exceeds the authorized "
            f"${budget.maximum_simulation_cost_usd:.4f}",
        )
    fit_evidence, fit_consumed, fit_judge_spend = complete_cell_evidence(
        project,
        plan_input,
        fit_cells,
        fit_set.artifact_ids,
        setup,
        review,
        services.judge,
        budget.maximum_judgments,
        remaining_cost_usd=budget.maximum_simulation_cost_usd - fit_spend,
        stop_on_overspend=budget.stop_on_overspend,
        spend_ceiling_crossed=_spend_ceiling_crossed,
        progress=progress,
        progress_detail="fit",
    )
    remaining_cost_usd = max(
        0.0, budget.maximum_simulation_cost_usd - math.fsum((fit_spend, fit_judge_spend))
    )
    if remaining_cost_usd <= 0 and any(
        cell.purpose == "held_out" and cell.execution == "simulate" for cell in plan.cells
    ):
        _spend_ceiling_crossed(
            budget.stop_on_overspend,
            "fit simulation and judging consumed the total budget; held-out dispatch is blocked",
            f"fit simulation and judging consumed the authorized "
            f"${budget.maximum_simulation_cost_usd:.4f} before held-out dispatch",
        )
    fit_config = RouterFitConfig(
        fit=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            rollout_set_ids=(fit_set.artifact_set_id,),
            protocols=(setup.production_protocol, setup.simulation_protocol),
            cell_evidence=fit_evidence,
        ),
        embedding_set_id=setup.embedding_set_id,
        incumbent_alias=setup.incumbent_alias,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        guard=setup.guard,
        judgment_status=setup.judgment_status,
        created_at=artifact_time,
        code_revision=code_revision,
    )
    report(progress, "fitting")
    fit, policy_lock = _fit_and_lock_once(
        project,
        plan_input,
        fit_config,
        artifact_time,
        code_revision,
    )
    _phase(phase_hook, "policy_locked")
    if policy_lock_hook is not None:
        policy_lock_hook(policy_lock, fit)

    held_cells = tuple(cell for cell in plan.cells if cell.purpose == "held_out")
    _phase(phase_hook, "heldout_opened")
    held_spec = build_router_simulation_spec(
        plan,
        plan_input,
        task_input,
        setup,
        remaining_cost_usd if budget.stop_on_overspend else budget.maximum_simulation_cost_usd,
        artifact_time,
        code_revision,
        held_cells,
        phase="heldout",
        stop_on_overspend=budget.stop_on_overspend,
    )
    held_set = _run_or_load_simulation(
        project,
        plan,
        held_spec,
        services.simulator_factory,
        progress=progress,
        progress_detail="held-out",
    )
    held_out_spend = verified_simulation_spend(project, held_set, setup.simulation_completion_input)
    if math.fsum((fit_spend, held_out_spend)) > budget.maximum_simulation_cost_usd:
        _spend_ceiling_crossed(
            budget.stop_on_overspend,
            "verified composed simulation spend exceeds the total budget",
            f"verified composed simulation spend ${math.fsum((fit_spend, held_out_spend)):.4f} "
            f"exceeds the authorized ${budget.maximum_simulation_cost_usd:.4f}",
        )
    held_evidence, _held_dispatched, _held_judge_spend = complete_cell_evidence(
        project,
        plan_input,
        held_cells,
        held_set.artifact_ids,
        setup,
        review,
        services.judge,
        budget.maximum_judgments - fit_consumed,
        remaining_cost_usd=budget.maximum_simulation_cost_usd
        - math.fsum((fit_spend, fit_judge_spend, held_out_spend)),
        stop_on_overspend=budget.stop_on_overspend,
        spend_ceiling_crossed=_spend_ceiling_crossed,
        progress=progress,
        progress_detail="held-out",
    )
    report(progress, "artifact publication")
    optimized = report_router(
        project.artifacts,
        fit,
        RouterReportConfig(
            held_out=EvaluationInputs(
                evaluation_plan_id=plan.plan_id,
                rollout_set_ids=(held_set.artifact_set_id,),
                protocols=(setup.production_protocol, setup.simulation_protocol),
                cell_evidence=held_evidence,
            ),
            embedding_set_id=setup.embedding_set_id,
            created_at=artifact_time,
            code_revision=code_revision,
        ),
    )
    total_spend = math.fsum((fit_spend, held_out_spend))
    completion_id = optimized.optimization.report.report_id
    capture_completion_once(
        "wmo simulation completed",
        completion_id,
        {
            "success": True,
            "rollout_count": len(fit_set.artifact_ids) + len(held_set.artifact_ids),
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
        policy_lock_id=policy_lock.lock_id,
        fit_simulation_spend_usd=fit_spend,
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
    *,
    progress: ProgressHook | None = None,
    progress_detail: str | None = None,
) -> SimulationArtifactSet:
    """Load an exactly completed simulation set without invoking its simulator again.

    Args:
        project: Project store holding completed simulation artifacts.
        plan: Frozen evaluation plan bound to the injected simulator.
        spec: Phase-scoped simulation specification to load or run.
        simulator_factory: Injected constructor invoked only when no completed set exists.
        progress: Optional observer of exact replayed evaluation-cell counts.
        progress_detail: Phase qualifier attached to replayed evaluation-cell counts.

    Returns:
        Immutable index of one rollout artifact for every selected cell.

    Raises:
        RouterCompositionError: A stored artifact set is ambiguous, drifted, or mismatched.
    """
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
        if any(reexecutable_dispatch_failure(rollout) for rollout in rollouts):
            continue
        matches.append(artifact_set)
    if len(matches) > 1:
        raise RouterCompositionError("multiple completed artifact sets name one simulation phase")
    if matches:
        cell_count = len(matches[0].artifact_ids)
        report(
            progress,
            "evaluation cells",
            completed=cell_count,
            total=cell_count,
            detail=progress_detail,
        )
        return matches[0]
    return simulator_factory(project, plan).run(spec)


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
            services.runtime_catalog,
        )
    ):
        raise RouterCompositionError("all workflow services must be injected explicitly")


def _verify_review(project: ProjectStore, review: RouterReviewProvenance) -> None:
    """Require persisted artifacts and exact typed calibration eligibility before simulation."""
    expected = ((review.rubric_id, "rubric"), (review.calibration_id, "judge-calibration"))
    for artifact_id, artifact_type in expected:
        if project.artifacts.read(artifact_id).manifest.artifact_type != artifact_type:
            raise RouterCompositionError(f"{artifact_id} is not a completed {artifact_type}")
    if review.calibration_input is None:
        if review.judgment_status == "provisional":
            raise RouterCompositionError("provisional review requires an exact calibration input")
        return
    calibration, calibration_input = verify_persisted_calibration(project, review.calibration_id)
    if (
        calibration_input != review.calibration_input
        or calibration.status != review.judgment_status
        or calibration.rubric_id != review.rubric_id
    ):
        raise RouterCompositionError(
            "router review differs from its persisted calibration status or rubric"
        )


def _phase(hook: Callable[[str], None] | None, phase: str) -> None:
    """Emit one local ordering marker without changing workflow behavior."""
    if hook is not None:
        hook(phase)
