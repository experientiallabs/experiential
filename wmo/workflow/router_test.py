"""Deterministic full-artifact coverage for public router composition."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import stable_id
from wmo.common.evaluations import (
    EvaluationCellEvidence,
    EvaluationPlan,
    EvaluationProtocol,
    ObservedProductionCell,
)
from wmo.common.evaluations.build_test import (
    _candidate,
    _persist_calibration,
    _persist_rollout,
    _production_rollout,
    _snapshot,
)
from wmo.common.judging import (
    DimensionJudgment,
    Judgment,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from wmo.common.models import ModelClient, OperationEconomics
from wmo.common.project import ProjectConfig, ProjectStore, artifact_input
from wmo.common.rollouts import SimulationArtifactSet
from wmo.common.routing import KnnGuard
from wmo.common.tasks import load_task_set
from wmo.optimize.router.workflow_test import _persist_embeddings, _persist_pricing
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.runtime_test import _Catalog, _Client, _request
from wmo.simulation.build import ProjectBuild
from wmo.simulation.build_test import _trace
from wmo.simulation.engines.text.simulator import WorldModelSimulator
from wmo.simulation.engines.text.simulator_test import (
    _OneTurnAgent,
    _resolved,
    _response,
    _ScriptedClient,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.orchestration import Simulator
from wmo.simulation.specs import SimulationSpec, WorldModelSettings
from wmo.workflow.router import (
    ApprovedRouterReview,
    FidelityApprovalDecision,
    RouterCompositionBudget,
    RouterEvaluationSetup,
    RouterWorkflowServices,
    compose_router,
)

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


class _ReviewSupplier:
    """Persist one explicit approved rubric and sealed calibration, idempotently."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        budget: RouterCompositionBudget,
    ) -> ApprovedRouterReview:
        del budget
        if "rubric-a" not in project.artifacts.list_ids():
            task_input = build.review.task_set
            rubric = Rubric(
                schema_version=1,
                created_at=_TIME,
                inputs=(task_input,),
                code_revision="test-revision",
                rubric_id="rubric-a",
                dimensions=(
                    RubricDimension(
                        dimension_id="dimension-a",
                        name="Task success",
                        description="Whether the task was completed.",
                        anchors=(
                            ScoreAnchor(score=0, description="Score 0 outcome."),
                            ScoreAnchor(score=1, description="Score 1 outcome."),
                            ScoreAnchor(score=2, description="Score 2 outcome."),
                            ScoreAnchor(score=3, description="Score 3 outcome."),
                            ScoreAnchor(score=4, description="Score 4 outcome."),
                            ScoreAnchor(score=5, description="Score 5 outcome."),
                        ),
                    ),
                ),
                source_task_set_id=task_input.artifact_id,
                status="human_approved",
                approved_at=_TIME,
            )
            project.artifacts.write_json(
                artifact_id=rubric.rubric_id,
                artifact_type="rubric",
                envelope=rubric,
                files={"rubric.json": rubric},
            )
            tasks = load_task_set(project.artifacts, task_input.artifact_id).tasks
            _persist_calibration(
                project.artifacts,
                fit_lineages=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "fit"
                ),
                held_out_lineages=tuple(
                    task.lineage_group_id for task in tasks if task.partition == "held_out"
                ),
            )
        return ApprovedRouterReview(rubric_id="rubric-a", calibration_id="calibration-a")


class _SetupSupplier:
    """Persist explicit production overlaps, frozen embeddings, and pricing."""

    def __call__(
        self,
        project: ProjectStore,
        build: ProjectBuild,
        review: ApprovedRouterReview,
        budget: RouterCompositionBudget,
    ) -> RouterEvaluationSetup:
        del review, budget
        tasks = load_task_set(project.artifacts, build.artifacts.task_set.task_set_id).tasks
        candidate = _candidate("candidate-a")
        observed = []
        fit = tuple(task for task in tasks if task.partition == "fit")
        for index, task in enumerate(fit[:10]):
            rollout_id = f"production-{index:02d}"
            if rollout_id not in project.artifacts.list_ids():
                rollout = _production_rollout(
                    rollout_id,
                    cell_id=f"import-{index}",
                    task=task,
                    candidate=candidate.model,
                ).model_copy(update={"trace_id": task.source_trace_ids[0]})
                _persist_rollout(project.artifacts, rollout)
            observed.append(
                ObservedProductionCell(
                    task_id=task.task_id,
                    candidate_alias=candidate.alias,
                    repeat=0,
                    rollout_artifact_id=rollout_id,
                )
            )
        if "pricing-a" not in project.artifacts.list_ids():
            _persist_pricing(project.artifacts)
            _persist_embeddings(project.artifacts, tasks)
        production = EvaluationProtocol(
            protocol_id="protocol-production",
            evidence_source="production",
            agent_id="agent-a",
            simulator_id="production-import-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="pricing-a",
        )
        world = EvaluationProtocol(
            protocol_id="protocol-world",
            evidence_source="world_model",
            agent_id="agent-a",
            simulator_id="text-world-model-v1",
            world_model=_snapshot("world-model-a"),
            simulator_prompt_id="world-model-text-v1",
            rubric_id="rubric-a",
            judge_calibration_id="calibration-a",
            pricing_snapshot_id="pricing-a",
        )
        return RouterEvaluationSetup(
            candidates=(candidate,),
            observed_cells=tuple(observed),
            production_protocol=production,
            simulation_protocol=world,
            embedding_set_id="embeddings-a",
            pricing_snapshot_id="pricing-a",
            incumbent_alias="candidate-a",
            guard=KnnGuard(
                maximum_neighbors=10,
                minimum_paired_observations=1,
                relative_similarity_threshold=0.0,
                uncertainty_multiplier=0.0,
                quality_tolerance=0.0,
            ),
            judgment_status="provisional",
            world_model_settings=WorldModelSettings(
                world_model_alias="world-model-a",
                prompt_version="text-world-model-v1",
            ),
            agent_id="agent-a",
            seed=7,
            maximum_steps=2,
            maximum_concurrency=1,
        )


class _Judge:
    """Return one stable score while counting actual dispatches."""

    def __init__(self) -> None:
        self.calls = 0
        self.log: list[tuple[str, bool]] = []

    def judge_persisted(
        self,
        store: ProjectStore,
        *,
        rollout_artifact_id: str,
        rubric_artifact_id: str,
        calibration_artifact_id: str,
    ) -> Judgment:
        self.calls += 1
        rollout_value = read_rollout(store.artifacts, rollout_artifact_id)[0]
        locked = any(
            store.artifacts.read(artifact_id).manifest.artifact_type == "router-policy-lock"
            for artifact_id in store.artifacts.list_ids()
        )
        self.log.append((rollout_value.cell_id or "observed", locked))
        rollout = store.artifacts.read(rollout_artifact_id)
        rubric = store.artifacts.read(rubric_artifact_id)
        calibration = store.artifacts.read(calibration_artifact_id)
        inputs = tuple(
            sorted(
                map(artifact_input, (rollout.manifest, rubric.manifest, calibration.manifest)),
                key=lambda item: item.artifact_id,
            )
        )
        return Judgment(
            schema_version=1,
            created_at=_TIME,
            inputs=inputs,
            code_revision="test-revision",
            judgment_id=stable_id(
                "judgment",
                {"rollout": rollout_artifact_id, "rubric": rubric_artifact_id},
            ),
            rollout_id=rollout_artifact_id,
            rubric_id=rubric_artifact_id,
            calibration_id=calibration_artifact_id,
            judge_model=_snapshot("judge-model"),
            judge_prompt_id="judge-prompt-v1",
            judge_prompt_sha256=_DIGEST,
            dimensions=(
                DimensionJudgment(
                    dimension_id="dimension-a",
                    raw_score=4,
                    calibrated_score=4.0,
                    evidence_span_ids=(
                        read_rollout(store.artifacts, rollout_artifact_id)[0].spans[0].span_id,
                    ),
                    feedback="Deterministic workflow score.",
                ),
            ),
            overall_score=0.8,
            judge_economics=OperationEconomics(),
        )


def read_rollout(store, rollout_id):  # noqa: ANN001, ANN201
    """Import the verified rollout helper lazily for the focused fake judge."""
    from wmo.common.evaluations.evidence import read_rollout as read

    return read(store, rollout_id)


class _SimulatorFactory:
    """Bind the real text simulator to deterministic explicit fake model clients."""

    def __init__(self) -> None:
        self.log: list[tuple[tuple[str, ...], bool]] = []
        self.candidate = _ScriptedClient(
            [_response("Resolved.", snapshot=_snapshot("candidate-a"), cost=0.01)] * 80
        )
        self.world = _ScriptedClient(
            [
                _response(
                    '{"message":"Done.","terminal":true}',
                    snapshot=_snapshot("world-model-a"),
                    cost=0.01,
                )
            ]
            * 80
        )

    def __call__(self, project: ProjectStore, plan: EvaluationPlan) -> Simulator:
        simulator = WorldModelSimulator(
            store=project.artifacts,
            evaluation_plan=plan,
            evaluation_plan_input=artifact_input(project.artifacts.read(plan.plan_id).manifest),
            task_set_input=artifact_input(project.artifacts.read(plan.task_set_id).manifest),
            candidate_models={
                "candidate-a": _resolved("candidate-a", cast(ModelClient, self.candidate))
            },
            world_models={
                "world-model-a": _resolved("world-model-a", cast(ModelClient, self.world))
            },
            agent_factory=_OneTurnAgent,
            clock=lambda: _TIME,
            monotonic=lambda: 1.0,
        )
        return _LoggingSimulator(project, simulator, self.log)


class _LoggingSimulator:
    """Record phase cell IDs and lock state before delegating to the real simulator."""

    def __init__(
        self,
        project: ProjectStore,
        simulator: WorldModelSimulator,
        log: list[tuple[tuple[str, ...], bool]],
    ) -> None:
        self.project = project
        self.simulator = simulator
        self.log = log

    def run(self, spec: SimulationSpec) -> SimulationArtifactSet:
        locked = any(
            self.project.artifacts.read(artifact_id).manifest.artifact_type == "router-policy-lock"
            for artifact_id in self.project.artifacts.list_ids()
        )
        self.log.append((spec.cell_ids, locked))
        return self.simulator.run(spec)


class _FidelityApproval:
    """Assert the callback sees only fidelity cells and return auditable actor evidence."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        project: ProjectStore,
        plan: EvaluationPlan,
        evidence: tuple[EvaluationCellEvidence, ...],
        budget: RouterCompositionBudget,
    ) -> FidelityApprovalDecision:
        del project, budget
        self.calls += 1
        cells = {cell.cell_id: cell for cell in plan.cells}
        assert evidence
        assert {cells[item.cell_id].purpose for item in evidence} == {"fidelity"}
        return FidelityApprovalDecision(
            actor_id="reviewer-fixture",
            evidence="Reviewed all ten plan-bound fidelity pairs.",
            approved_at=_TIME,
        )


def test_public_composition_runs_and_resumes_complete_frozen_router(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One explicit workflow reaches sticky runtime without duplicate dispatch on resume."""
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    simulator = _SimulatorFactory()
    judge = _Judge()
    approval = _FidelityApproval()
    runtime_client = _Client()
    runtime_catalog = cast(
        RuntimeModelCatalog,
        _Catalog(
            {
                "candidate-a": _snapshot("candidate-a"),
                "embedder": _snapshot("embedder"),
            },
            runtime_client,
        ),
    )
    phases = []
    services = RouterWorkflowServices(
        review_supplier=_ReviewSupplier(),
        setup_supplier=_SetupSupplier(),
        simulator_factory=simulator,
        judge=judge,
        fidelity_approval=approval,
        runtime_catalog=runtime_catalog,
    )
    budget = RouterCompositionBudget(
        maximum_simulation_cost_usd=10.0,
        maximum_judgments=100,
    )

    crash = True

    def crash_after_lock(phase: str) -> None:
        nonlocal crash
        phases.append(phase)
        if phase == "policy_locked" and crash:
            crash = False
            raise RuntimeError("simulated crash after policy lock")

    with pytest.raises(RuntimeError, match="after policy lock"):
        compose_router(
            project,
            normalized,
            services=services,
            budget=budget,
            created_at=_TIME,
            code_revision="test-revision",
            phase_hook=crash_after_lock,
        )
    assert approval.calls == 1
    from wmo.workflow import router as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "fit_router",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit repeated")),
    )
    first = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
        phase_hook=crash_after_lock,
    )
    dispatched = (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
        approval.calls,
    )
    second = compose_router(
        project,
        normalized,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert second.optimization == first.optimization
    assert (
        len(simulator.log),
        len(simulator.candidate.requests),
        len(simulator.world.requests),
        judge.calls,
        approval.calls,
    ) == dispatched
    assert phases.index("fidelity_fit_started") < phases.index("policy_locked")
    assert phases.index("policy_locked") < phases.index("heldout_opened")
    assert phases.index("heldout_opened") < phases.index("report_complete")
    purposes = {cell.cell_id: cell.purpose for cell in first.plan.cells}
    assert all(
        locked or all(purposes[cell_id] in {"fit", "fidelity"} for cell_id in cell_ids)
        for cell_ids, locked in simulator.log
    )
    assert all(locked or purposes.get(cell_id) != "held_out" for cell_id, locked in judge.log)
    decision = first.runtime.select(_request(), episode_id="customer-episode")
    assert first.runtime.select(_request(), episode_id="customer-episode") == decision
    assert first.plan.cells
