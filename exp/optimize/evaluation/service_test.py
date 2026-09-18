"""Real simulator and immutable-artifact regressions for standalone worker evaluations."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from exp.common.evaluations.model_report import ModelEvaluationReport
from exp.common.judging import HumanScoreReview, JudgeCalibrationService, PromptDefinition
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.common.project import ProjectConfig, ProjectStore
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.contracts import EvaluationBudget, EvaluationServices, EvaluationSetup
from exp.optimize.evaluation.service import evaluate_models
from exp.optimize.router.automatic.provisional import _persist_lineage_split
from exp.optimize.router.composition import RouterCompositionBudget
from exp.optimize.router.composition_test import (
    _COMPACT_MINING_SPEC,
    _TIME,
    _bind_completed_build,
    _CapExhaustedSimulatorFactory,
    _compact_normalized_traces,
    _completion_reservation,
    _ReservedSetupSupplier,
    _ReviewSupplier,
    _SetupSupplier,
    _SimulatorFactory,
    _snapshot,
    _TargetedTransportFailureClient,
)
from exp.optimize.router.composition_test import (
    _Judge as _RouterTestJudge,
)
from exp.optimize.router.evaluation.build import reconstruct_completed_project_build
from exp.runtime.models.providers.errors import ProviderTransportError


class _Judge(_RouterTestJudge):
    """Expose the configured test model before any simulation or judgment dispatch."""

    model: ModelSnapshot = _snapshot("judge-model")


class _UnavailableClient(_TargetedTransportFailureClient):
    """Record an admitted provider call that fails without returning usage evidence."""

    def __init__(self) -> None:
        """Retain requests without any scripted successful responses."""
        super().__init__([], "unused")

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail after dispatch, leaving the real simulator to reserve unknown spend."""
        self.requests.append(request)
        raise ProviderTransportError("connection reset by provider")


def _prepared(root: Path, *, multiple: bool = False) -> tuple[ProjectStore, EvaluationSetup]:
    """Reuse the router's real grounded simulator fixture, not its fit or policy path."""
    project = ProjectStore(root, "evaluation-a")
    project.initialize(ProjectConfig(project_id="evaluation-a"))
    normalized = _compact_normalized_traces()
    _bind_completed_build(
        project, normalized, revision="test-revision", mining_spec=_COMPACT_MINING_SPEC
    )
    build = reconstruct_completed_project_build(project, normalized, created_at=_TIME)
    budget = RouterCompositionBudget(maximum_simulation_cost_usd=10, maximum_judgments=100)
    review = _ReviewSupplier()(project, build, budget)
    supplier = _ReservedSetupSupplier() if multiple else _SetupSupplier()
    router_setup = supplier(project, build, review, budget)
    completed = project.load_project().build
    assert completed is not None
    task = next(
        item
        for item in load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
        if item.task_id == router_setup.observed_cells[0].task_id
    )
    split = _persist_lineage_split(
        project,
        completed.task_set,
        rollout_id="production-00",
        rollout_lineage_id=task.lineage_group_id,
        created_at=_TIME,
        code_revision="test-revision",
    )
    labels = HumanScoreReview.open(project).finalize(
        rubric_id="rubric-a",
        created_at=_TIME,
        code_revision="test-revision",
    )
    calibration = JudgeCalibrationService().bootstrap_provisional(
        project,
        rubric_id="rubric-a",
        label_set_id=labels.label_set_id,
        router_lineage_split_id=split.split_id,
        judge_model=_snapshot("judge-model"),
        judge_prompt=PromptDefinition.from_text("judge-prompt-v1", "Judge this task."),
        created_at=_TIME,
        code_revision="test-revision",
    )
    setup = EvaluationSetup.model_validate(
        {
            key: value
            for key, value in router_setup.model_dump().items()
            if key in EvaluationSetup.model_fields
        }
    )
    return project, setup.model_copy(
        update={
            "observed_cells": (),
            "production_protocol": setup.production_protocol.model_copy(
                update={
                    "judge_calibration_id": calibration.calibration_id,
                }
            ),
            "simulation_protocol": setup.simulation_protocol.model_copy(
                update={
                    "judge_calibration_id": calibration.calibration_id,
                }
            ),
        }
    )


def test_evaluation_runs_and_replays_without_fitting_or_activating_a_router(tmp_path: Path) -> None:
    """The actual text simulation engine produces a report and exact replay makes no calls."""
    project, setup = _prepared(tmp_path)
    simulator = _SimulatorFactory()
    judge = _Judge()
    services = EvaluationServices(simulator_factory=simulator, judge=judge)
    budget = EvaluationBudget(maximum_cost_usd=10, maximum_judgments=100)
    result = evaluate_models(
        project,
        setup,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert result.report.models[0].quality == pytest.approx(0.8)
    assert result.report.frontier_aliases == ("candidate-a",)
    assert result.report.compared_cells == len(result.plan.cells)
    assert result.report.excluded_cells == 0
    assert all(not locked for _, locked in simulator.log)
    assert not any(
        project.artifacts.read(identity).manifest.artifact_type
        in {"router-policy", "router-policy-lock", "knn-bank", "held-out-router-report"}
        for identity in project.artifacts.list_ids()
    )
    calls = judge.calls
    simulations = len(simulator.log)
    replay = evaluate_models(
        project,
        setup,
        services=services,
        budget=budget,
        created_at=_TIME + timedelta(hours=1),
        code_revision="test-revision",
    )
    assert replay == result
    assert judge.calls == calls
    assert len(simulator.log) == simulations
    persisted = ModelEvaluationReport.model_validate_json(
        project.artifacts.read_bytes(result.report.report_id, "report.json")
    )
    assert persisted == result.report


def test_too_few_judgments_fail_before_any_model_or_simulation_dispatch(tmp_path: Path) -> None:
    """A plan cannot start if the count ceiling cannot cover its worker/scenario matrix."""
    project, setup = _prepared(tmp_path)
    simulator = _SimulatorFactory()
    judge = _Judge()
    before = project.artifacts.list_ids()
    with pytest.raises(ValueError, match="scenarios times worker"):
        evaluate_models(
            project,
            setup,
            services=EvaluationServices(simulator, judge),
            budget=EvaluationBudget(maximum_cost_usd=10, maximum_judgments=1),
            created_at=_TIME,
            code_revision="test-revision",
        )
    assert judge.calls == 0
    assert not simulator.log
    assert project.artifacts.list_ids() == before


def test_wrong_judge_fails_before_writes_or_dispatch_on_every_retry(tmp_path: Path) -> None:
    """A mismatched runtime cannot repeatedly spend money before detecting model drift."""
    project, setup = _prepared(tmp_path)
    simulator = _SimulatorFactory()
    judge = _Judge()
    judge.model = _snapshot("wrong-model")
    before = project.artifacts.list_ids()
    for _ in range(2):
        with pytest.raises(ValueError, match="persisted judge model"):
            evaluate_models(
                project,
                setup,
                services=EvaluationServices(simulator, judge),
                budget=EvaluationBudget(maximum_cost_usd=10, maximum_judgments=100),
                created_at=_TIME,
                code_revision="test-revision",
            )
    assert judge.calls == 0
    assert not simulator.log
    assert project.artifacts.list_ids() == before


def test_exactly_exhausted_failure_report_replays_without_paid_judging(tmp_path: Path) -> None:
    """An unknown-spend terminal failure can consume the full cap and still be reported."""
    project, setup = _prepared(tmp_path, multiple=True)
    setup = setup.model_copy(update={"maximum_concurrency": 1})
    simulator = _CapExhaustedSimulatorFactory("unused")
    simulator.candidate = _UnavailableClient()
    simulator.failing = _UnavailableClient()
    judge = _Judge()
    services = EvaluationServices(simulator, judge)
    budget = EvaluationBudget(
        maximum_cost_usd=_completion_reservation("candidate-a").expected_maximum_call_cost_usd(),
        maximum_judgments=100,
    )
    first = evaluate_models(
        project,
        setup,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert first.cost_usd == budget.maximum_cost_usd
    assert first.report.compared_cells == 0
    assert first.report.excluded_cells > 0
    assert first.report.frontier_aliases == ()
    assert judge.calls == 0
    calls = len(simulator.log)
    replay = evaluate_models(
        project,
        setup,
        services=services,
        budget=budget,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert replay == first
    assert len(simulator.log) == calls
    assert judge.calls == 0


@pytest.mark.parametrize("fail_one", [False, True])
def test_multiple_workers_share_the_same_denominator_even_when_one_fails(
    tmp_path: Path, fail_one: bool
) -> None:
    """One failed worker/scenario excludes that coordinate for all compared workers."""
    project, setup = _prepared(tmp_path, multiple=True)
    completed = project.load_project().build
    assert completed is not None
    tasks = load_task_set(project.artifacts, completed.task_set.artifact_id).tasks
    simulator = _CapExhaustedSimulatorFactory(
        tasks[0].instruction if fail_one else "No task matches this instruction."
    )
    result = evaluate_models(
        project,
        setup,
        services=EvaluationServices(simulator, _Judge()),
        budget=EvaluationBudget(maximum_cost_usd=10, maximum_judgments=100),
        created_at=_TIME,
        code_revision="test-revision",
    )
    expected = len(tasks) - int(fail_one)
    assert result.report.compared_cells == expected
    assert result.report.excluded_cells == int(fail_one)
    assert all(model.compared_cells == expected for model in result.report.models)
    assert [model.failed_cells for model in result.report.models] == [0, int(fail_one)]
    assert all(model.quality == pytest.approx(0.8) for model in result.report.models)
    assert result.report.frontier_aliases == ("candidate-a", "candidate-b")


def test_changed_authorization_has_a_distinct_execution_identity(tmp_path: Path) -> None:
    """Increasing a ceiling cannot silently mutate a persisted run's authorization."""
    from exp.optimize.evaluation.service import _execution_contract

    project, setup = _prepared(tmp_path)
    first = _execution_contract(
        project,
        setup,
        EvaluationBudget(maximum_cost_usd=10, maximum_judgments=100),
        _TIME,
        "test-revision",
    )
    second = _execution_contract(
        project,
        setup,
        EvaluationBudget(maximum_cost_usd=20, maximum_judgments=100),
        _TIME,
        "test-revision",
    )
    assert first.artifact_id != second.artifact_id


def test_evaluation_api_is_public() -> None:
    """Hosted consumers import the canonical engine entry point, not a platform copy."""
    import exp

    assert exp.evaluate_models is evaluate_models
    assert exp.EvaluationSetup is EvaluationSetup
