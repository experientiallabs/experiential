"""End-to-end dependency-injected composition of one frozen customer router."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, stable_id
from wmo.common.evaluations import (
    EvaluationCellEvidence,
    EvaluationPlan,
    EvaluationProtocol,
    ObservedProductionCell,
    build_evaluation_plan,
    build_fidelity_report,
    default_fidelity_thresholds,
    persist_fidelity_thresholds,
)
from wmo.common.evaluations.evidence import read_judgment, read_rollout
from wmo.common.judging import Judge, Judgment
from wmo.common.models import RoutedCandidateSnapshot
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore, artifact_input
from wmo.common.rollouts import SimulationMode
from wmo.common.routing import KnnGuard
from wmo.optimize.router import EvaluationInputs, RouterOptimizationConfig, RouterWorkflowResult
from wmo.optimize.router.workflow import optimize_router
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router import RouterRuntime
from wmo.runtime.router.application import load_project_router
from wmo.simulation.build import ProjectBuild, build_project
from wmo.simulation.ingest.otlp import TraceNormalizationResult, load_otlp_file
from wmo.simulation.ingest.posthog import load_posthog_file
from wmo.simulation.orchestration import Simulator
from wmo.simulation.specs import SimulationSpec, WorldModelSettings


class RouterCompositionError(ValueError):
    """Explicit workflow inputs cannot safely produce a frozen router."""


class RouterCompositionBudget(ContractModel):
    """Finite dispatch ceilings required by the composed customer workflow."""

    maximum_simulation_cost_usd: float = Field(gt=0)
    maximum_judgments: int = Field(gt=0)

    @field_validator("maximum_simulation_cost_usd")
    @classmethod
    def _require_finite_cost(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("simulation budget must be finite")
        return value


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
    pricing_snapshot_id: ArtifactId
    guard: KnnGuard
    incumbent_alias: ArtifactId | None = None
    judgment_status: Literal["provisional", "human_calibrated"]
    world_model_settings: WorldModelSettings
    agent_id: str = Field(min_length=1, max_length=256)
    seed: int
    maximum_steps: int = Field(gt=0)
    maximum_concurrency: int = Field(gt=0)


class TraceSource(Protocol):
    """Loads one explicit local source into canonical normalized traces."""

    def load(self) -> TraceNormalizationResult:
        """Return canonical normalized traces after one explicit source read."""


@dataclass(frozen=True)
class LocalTraceSource:
    """One explicit canonical local trace export selected by path and format."""

    path: Path
    source: Literal["otlp", "posthog"] = "otlp"

    def load(self) -> TraceNormalizationResult:
        """Read and normalize this local file through its selected canonical loader."""
        if self.source == "otlp":
            return load_otlp_file(self.path)
        return load_posthog_file(self.path)


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
    ) -> datetime:
        """Return the explicit timezone-aware approval time for reviewed fidelity evidence."""


@dataclass(frozen=True)
class RouterWorkflowServices:
    """Every service capable of dispatching work in the customer workflow."""

    review_supplier: ReviewSupplier
    setup_supplier: EvaluationSetupSupplier
    simulator_factory: SimulatorFactory
    judge: Judge
    fidelity_approval: FidelityApproval
    runtime_catalog: RuntimeModelCatalog


@dataclass(frozen=True)
class RouterCompositionResult:
    """Complete local artifact chain and loaded frozen online runtime."""

    build: ProjectBuild
    review: ApprovedRouterReview
    plan: EvaluationPlan
    simulation_spec: SimulationSpec
    fidelity_report_id: ArtifactId
    optimization: RouterWorkflowResult
    runtime: RouterRuntime


def compose_router(
    project: ProjectStore,
    trace_source: TraceSource | TraceNormalizationResult,
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
        trace_source: Explicit normalized input or a loader that performs the source read.
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
    _preflight(project, services, budget, code_revision)
    normalized = (
        trace_source if isinstance(trace_source, TraceNormalizationResult) else trace_source.load()
    )
    built = build_project(
        normalized,
        project,
        created_at=created_at,
        code_revision=code_revision,
    )
    _phase(phase_hook, "review")
    review = services.review_supplier(project, built, budget)
    _verify_review(project, review)
    setup = services.setup_supplier(project, built, review, budget)
    _verify_setup(setup, review)

    thresholds = default_fidelity_thresholds(created_at=created_at, code_revision=code_revision)
    persist_fidelity_thresholds(project.artifacts, thresholds)
    plan = build_evaluation_plan(
        project.artifacts,
        task_set_id=built.artifacts.task_set.task_set_id,
        candidate_snapshots=setup.candidates,
        observed_cells=setup.observed_cells,
        fidelity_thresholds_id=thresholds.fidelity_thresholds_id,
        fidelity_protocol_sha256=_protocol_digest(setup.simulation_protocol),
        created_at=created_at,
        code_revision=code_revision,
    )
    simulated_cells = tuple(
        sorted(cell.cell_id for cell in plan.cells if cell.execution == "simulate")
    )
    plan_input = artifact_input(project.artifacts.read(plan.plan_id).manifest)
    task_input = built.review.task_set
    spec_binding = {
        "plan": plan_input.model_dump(mode="json"),
        "task_set": task_input.model_dump(mode="json"),
        "cells": simulated_cells,
        "settings": setup.world_model_settings.model_dump(mode="json"),
        "agent_id": setup.agent_id,
        "seed": setup.seed,
        "maximum_steps": setup.maximum_steps,
        "maximum_concurrency": setup.maximum_concurrency,
        "maximum_cost_usd": budget.maximum_simulation_cost_usd,
        "code_revision": code_revision,
    }
    spec = SimulationSpec(
        schema_version=1,
        created_at=created_at,
        inputs=(plan_input, task_input),
        code_revision=code_revision,
        simulation_id=stable_id("simulation", spec_binding),
        evaluation_plan_id=plan.plan_id,
        cell_ids=simulated_cells,
        agent_id=setup.agent_id,
        mode=SimulationMode.WORLD_MODEL,
        world_model=setup.world_model_settings,
        seed=setup.seed,
        maximum_steps=setup.maximum_steps,
        maximum_concurrency=setup.maximum_concurrency,
        maximum_cost_usd=budget.maximum_simulation_cost_usd,
    )
    _phase(phase_hook, "simulate")
    artifact_set = services.simulator_factory(project, plan).run(spec)
    evidence = _complete_cell_evidence(
        project,
        plan,
        artifact_set.artifact_ids,
        setup,
        review,
        services.judge,
        budget,
        phase_hook,
    )
    _phase(phase_hook, "fidelity")
    approved_at = services.fidelity_approval(project, plan, evidence, budget)
    fidelity = build_fidelity_report(
        project.artifacts,
        evaluation_plan_id=plan.plan_id,
        protocol=setup.simulation_protocol,
        cell_evidence=evidence,
        created_at=created_at,
        code_revision=code_revision,
        approved_at=approved_at,
    )
    approved_protocol = setup.simulation_protocol.model_copy(
        update={"fidelity_report_id": fidelity.fidelity_report_id}
    )
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}
    fit_evidence = tuple(
        item for item in evidence if cells_by_id[item.cell_id].purpose in {"fit", "fidelity"}
    )
    held_evidence = tuple(
        item for item in evidence if cells_by_id[item.cell_id].purpose == "held_out"
    )
    config = RouterOptimizationConfig(
        fit=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            rollout_set_ids=(artifact_set.artifact_set_id,),
            protocols=(setup.production_protocol, approved_protocol),
            cell_evidence=fit_evidence,
            fidelity_report_ids=(fidelity.fidelity_report_id,),
        ),
        held_out=EvaluationInputs(
            evaluation_plan_id=plan.plan_id,
            rollout_set_ids=(artifact_set.artifact_set_id,),
            protocols=(setup.production_protocol, approved_protocol),
            cell_evidence=held_evidence,
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
    _phase(phase_hook, "fit_then_heldout")
    optimized = optimize_router(project.artifacts, config)
    _phase(phase_hook, "runtime")
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
        fidelity_report_id=fidelity.fidelity_report_id,
        optimization=optimized,
        runtime=runtime,
    )


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


def _verify_setup(setup: RouterEvaluationSetup, review: ApprovedRouterReview) -> None:
    """Bind both protocols to the approved review and require source-specific roles."""
    protocols = (setup.production_protocol, setup.simulation_protocol)
    if any(
        protocol.rubric_id != review.rubric_id
        or protocol.judge_calibration_id != review.calibration_id
        for protocol in protocols
    ):
        raise RouterCompositionError("evaluation protocols differ from approved review artifacts")
    if setup.production_protocol.evidence_source != "production":
        raise RouterCompositionError("production_protocol must name production evidence")
    if setup.simulation_protocol.evidence_source != "world_model":
        raise RouterCompositionError("simulation_protocol must name world-model evidence")
    if setup.simulation_protocol.fidelity_report_id is not None:
        raise RouterCompositionError("simulation protocol cannot preclaim a fidelity report")


def _complete_cell_evidence(
    project: ProjectStore,
    plan: EvaluationPlan,
    simulated_rollout_ids: tuple[str, ...],
    setup: RouterEvaluationSetup,
    review: ApprovedRouterReview,
    judge: Judge,
    budget: RouterCompositionBudget,
    phase_hook: Callable[[str], None] | None,
) -> tuple[EvaluationCellEvidence, ...]:
    """Verify rollout membership, resume judgments, and return evidence for every plan cell."""
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
    evidence = []
    dispatched = 0
    _phase(phase_hook, "judge")
    for cell in plan.cells:
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
        judgment = _find_judgment(project, rollout_id, review)
        if judgment is None:
            if dispatched >= budget.maximum_judgments:
                raise RouterCompositionError("judgment dispatch budget exhausted")
            judgment = judge.judge_persisted(
                project,
                rollout_artifact_id=rollout_id,
                rubric_artifact_id=review.rubric_id,
                calibration_artifact_id=review.calibration_id,
            )
            dispatched += 1
            _persist_judgment(project, judgment)
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
    return tuple(evidence)


def _find_judgment(
    project: ProjectStore,
    rollout_id: str,
    review: ApprovedRouterReview,
) -> Judgment | None:
    """Find one exact completed judgment so resume never repeats a paid judge call."""
    matches = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judgment":
            continue
        judgment, _input = read_judgment(project.artifacts, artifact_id)
        if (
            judgment.rollout_id == rollout_id
            and judgment.rubric_id == review.rubric_id
            and judgment.calibration_id == review.calibration_id
        ):
            matches.append(judgment)
    if len(matches) > 1:
        raise RouterCompositionError("multiple judgments bind the same rollout and review")
    return matches[0] if matches else None


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
