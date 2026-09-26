"""Evaluate worker models through simulation and judging without producing a router."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    canonical_json_bytes,
    sorted_unique_inputs,
    stable_id,
)
from exp.common.evaluations import (
    EvaluationPlan,
    EvaluationProtocol,
    build_evaluation_dataset,
    build_evaluation_plan,
)
from exp.common.evaluations.evidence import evaluation_protocol_digest
from exp.common.evaluations.model_report import ModelEvaluationReport, build_model_evaluation_report
from exp.common.judging import verify_persisted_calibration
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.contracts import (
    EvaluationBudget,
    EvaluationExecutionContract,
    EvaluationJudge,
    EvaluationServices,
    EvaluationSetup,
)
from exp.optimize.evaluation.simulation import run_or_load_simulation
from exp.optimize.router.errors import RouterCompositionError
from exp.optimize.router.evaluation.build import completed_project_build
from exp.optimize.router.evaluation.setup import verify_router_evaluation_setup
from exp.optimize.router.evaluation.simulation_spec import build_router_simulation_spec
from exp.optimize.router.evaluation.spend import verified_simulation_spend
from exp.optimize.router.judgment_budget import (
    JudgmentExclusionRecord,
    complete_cell_evidence,
)
from exp.simulation.specs import SimulationSpec


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Completed immutable evaluation artifacts and reconciled simulation/judging spend.

    Attributes:
        plan: Frozen worker and scenario matrix.
        simulation_spec: Exact executed simulation settings.
        evaluation_id: Persisted evaluation dataset identity.
        report: Shared-cohort model quality and operating-cost comparison.
        simulation_cost_usd: Reconciled worker, world-model and retrieval spend.
        judge_cost_usd: Reconciled durable judgment spend.
    """

    plan: EvaluationPlan
    simulation_spec: SimulationSpec
    evaluation_id: ArtifactId
    report: ModelEvaluationReport
    simulation_cost_usd: float
    judge_cost_usd: float

    @property
    def cost_usd(self) -> float:
        """Return this evaluation's cost, excluding separately accounted trace/build preparation."""
        return math.fsum((self.simulation_cost_usd, self.judge_cost_usd))


def evaluate_models(
    project: ProjectStore,
    setup: EvaluationSetup,
    *,
    services: EvaluationServices,
    budget: EvaluationBudget,
    created_at: datetime,
    code_revision: str,
    progress: ProgressHook | None = None,
) -> ModelEvaluationResult:
    """Run fresh paired worker evaluations against a completed simulation project.

    Args:
        project: Project with immutable mined tasks and a grounded world model.
        setup: Frozen workers, simulation protocol and persisted judge evidence.
        services: Runtime simulator and judge, using the same engines as router optimization.
        budget: Authorized finite simulation/judgment spend and dispatch-count ceilings.
        created_at: Stable run timestamp. Replay adopts the persisted plan's timestamp.
        code_revision: Exact producer revision.
        progress: Optional observer for real simulation, judgment and report progress.

    Returns:
        A persisted model comparison and reconciled execution spend. Exact replay dispatches
        neither simulations nor judgments again. No policy is fitted, locked, loaded or served.

    Raises:
        ValueError: Inputs drift, historical cells are supplied, or evidence/budget gates fail.
    """
    spending_limit = (
        budget.maximum_cost_usd
        if services.spending_limit_usd is None
        else services.spending_limit_usd
    )
    if not math.isfinite(spending_limit) or spending_limit <= 0:
        raise ValueError("evaluation spending limit must be finite and positive")
    if (services.judging_protocol is None) != (services.judging_input is None):
        raise ValueError("a judging revision requires both its protocol and immutable input")
    if services.judging_protocol is not None and (
        services.judging_protocol.model_dump(exclude={"protocol_id"})
        != setup.simulation_protocol.model_dump(exclude={"protocol_id"})
    ):
        raise ValueError("judging retry must preserve the frozen evaluation protocol")
    report(progress, "preflight")
    completed = completed_project_build(project)
    if setup.observed_cells:
        raise ValueError("model evaluation requires fresh rollouts for every selected worker")
    protocol = setup.simulation_protocol
    calibration, calibration_input = verify_persisted_calibration(
        project, protocol.judge_calibration_id
    )
    if calibration.status != setup.judgment_status or calibration.rubric_id != protocol.rubric_id:
        raise ValueError("evaluation judge status or rubric differs from persisted evidence")
    if services.judge.model != calibration.judge_model:
        raise ValueError("configured evaluation judge differs from the persisted judge model")
    verify_router_evaluation_setup(
        completed=completed,
        fit_rag_input=setup.fit_rag_input,
        grounded_world_model_input=setup.world_model_settings.grounded_world_model_input,
        production_protocol=setup.production_protocol,
        simulation_protocol=protocol,
        rubric_id=calibration.rubric_id,
        calibration_id=calibration.calibration_id,
    )
    tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
    required_judgments = len(tasks) * len(setup.candidates) * setup.repeats
    if required_judgments > budget.maximum_judgments:
        raise ValueError("evaluation judgment ceiling is below scenarios times worker models")
    execution_input = _execution_contract(project, setup, budget, created_at, code_revision)
    plan = build_evaluation_plan(
        project.artifacts,
        task_set_id=completed.task_set.artifact_id,
        candidate_snapshots=setup.candidates,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        observed_cells=(),
        repeats=tuple(range(setup.repeats)),
        additional_inputs=sorted_unique_inputs(
            calibration_input, execution_input, *services.plan_inputs
        ),
        created_at=created_at,
        code_revision=code_revision,
    )
    plan_input = artifact_input(project.artifacts.read(plan.plan_id).manifest)
    spec = build_router_simulation_spec(
        plan,
        plan_input,
        completed.task_set,
        setup,
        budget.maximum_cost_usd,
        plan.created_at,
        code_revision,
        plan.cells,
        phase="evaluation",
        stop_on_overspend=True,
    )
    report(progress, "simulation")
    simulated = run_or_load_simulation(
        project,
        plan,
        spec,
        services.simulator_factory,
        progress=progress,
        progress_detail="worker models",
    )
    simulation_cost = verified_simulation_spend(
        project, simulated, setup.simulation_completion_input
    )
    if simulation_cost > spending_limit:
        raise ValueError("simulation exceeded the authorized budget before judging")
    report(progress, "judging")
    judging_setup = setup
    prior_judge_cost = 0.0
    judge_inputs: tuple[ArtifactInput, ...] = ()
    if services.judging_protocol is not None:
        if services.judging_input is None:
            raise ValueError("a fresh judging protocol requires its reviewed revision")
        protocol = services.judging_protocol
        judging_setup = setup.model_copy(update={"simulation_protocol": protocol})
        judge_inputs = (services.judging_input,)
        if services.judge_spend is None:
            prior_judge_cost = _prior_judging_cost(
                project, plan_input, simulated.artifact_ids, protocol
            )
    authoritative_judge_spend = services.judge_spend
    evidence, _, judge_cost = complete_cell_evidence(
        project,
        plan_input,
        plan.cells,
        simulated.artifact_ids,
        judging_setup,
        EvaluationJudge(calibration.rubric_id, calibration.calibration_id),
        services.judge,
        budget.maximum_judgments,
        remaining_cost_usd=spending_limit - simulation_cost - prior_judge_cost,
        stop_on_overspend=True,
        spend_ceiling_crossed=_reject_overspend,
        reconciled_spend=(
            (lambda: authoritative_judge_spend(simulated.artifact_ids))
            if authoritative_judge_spend is not None
            else None
        ),
        progress=progress,
    )
    judge_cost = math.fsum((judge_cost, prior_judge_cost))
    if math.fsum((simulation_cost, judge_cost)) > spending_limit:
        raise ValueError("reconciled evaluation spend exceeds its authorized ceiling")
    dataset = build_evaluation_dataset(
        project.artifacts,
        evaluation_plan_id=plan.plan_id,
        pricing_snapshot_id=setup.pricing_snapshot_id,
        protocols=(protocol,),
        cell_evidence=evidence,
        additional_inputs=judge_inputs,
        purposes=("fit", "held_out"),
        created_at=plan.created_at,
        code_revision=code_revision,
    )
    report(progress, "report")
    result = build_model_evaluation_report(
        project.artifacts,
        dataset.manifest.evaluation_id,
        created_at=plan.created_at,
        code_revision=code_revision,
    )
    report(progress, "completed", completed=len(plan.cells), total=len(plan.cells))
    return ModelEvaluationResult(
        plan=plan,
        simulation_spec=spec,
        evaluation_id=dataset.manifest.evaluation_id,
        report=result,
        simulation_cost_usd=simulation_cost,
        judge_cost_usd=judge_cost,
    )


def _reject_overspend(stop: bool, error: str, detail: str) -> None:
    """Keep the shared executor's evaluation branch unconditionally fail-closed."""
    del stop, detail
    raise RouterCompositionError(error)


def _execution_contract(
    project: ProjectStore,
    setup: EvaluationSetup,
    budget: EvaluationBudget,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Persist settings before dispatch so changed authorization cannot reuse an old plan."""
    inputs = sorted_unique_inputs(
        setup.fit_rag_input,
        setup.world_model_settings.grounded_world_model_input,
        *((setup.continuation_of,) if setup.continuation_of is not None else ()),
        *(
            ()
            if setup.simulation_completion_input is None
            else (setup.simulation_completion_input,)
        ),
    )
    contract = EvaluationExecutionContract(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=inputs,
        contract_id=stable_id(
            "evaluation-execution",
            {
                "version": 1,
                "setup": setup.model_dump(mode="json"),
                "budget": budget.model_dump(mode="json"),
                "code_revision": code_revision,
            },
        ),
        setup=setup,
        budget=budget,
    )
    _, manifest = project.artifacts.write_or_replay(
        artifact_id=contract.contract_id,
        artifact_type="evaluation-execution-contract",
        envelope=contract,
        envelope_path="execution.json",
        envelope_type=EvaluationExecutionContract,
        files={"execution.json": canonical_json_bytes(contract)},
    )
    return artifact_input(manifest)


def _prior_judging_cost(
    project: ProjectStore,
    plan: ArtifactInput,
    rollout_ids: tuple[str, ...],
    protocol: EvaluationProtocol,
) -> float:
    """Retain costs from earlier excluded judging attempts when a new pass succeeds."""
    costs = []
    for artifact_id in project.artifacts.list_ids():
        stored = project.artifacts.read(artifact_id)
        if stored.manifest.artifact_type != "judgment-exclusion":
            continue
        exclusion = JudgmentExclusionRecord.model_validate_json(
            project.artifacts.read_bytes(artifact_id, "exclusion.json")
        )
        if (
            exclusion.plan == plan
            and exclusion.rollout.artifact_id in rollout_ids
            and exclusion.calibration.artifact_id == protocol.judge_calibration_id
            and exclusion.protocol_sha256 != evaluation_protocol_digest(protocol)
        ):
            costs.append(exclusion.conservative_cost_usd)
    return math.fsum(costs)
