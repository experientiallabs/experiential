"""Real project, simulation, judgment, export, and resume coverage for local eval runs."""

from pathlib import Path
from typing import cast

import pytest

from exp.cli.build.app import _build_grounded_artifacts
from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import ModelCatalog
from exp.common.progress import ProgressEvent
from exp.common.project import ProjectStore, write_project_config
from exp.common.project.records import ProjectRecords
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.optimize.evaluation.contracts import EvaluationBudget
from exp.optimize.evaluation.export import export_report, load_report_evidence
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    EvaluationPreparationOutdated,
    evaluation_tasks,
    execute_run,
    list_runs,
    load_run,
    prepare_run,
    save_run,
)
from exp.optimize.evaluation.runtime import run_prepared_model_evaluation
from exp.optimize.router.automatic.service_test import (
    _REVISION,
    _TIME,
    _completed_project,
    _ProviderState,
    _RuntimeCatalog,
    _trace,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.build import build_project, select_completed_build
from exp.simulation.mining.service import MiningSpec


def _twenty_scenarios(root: Path) -> tuple[ProjectStore, ModelCatalog, _ProviderState]:
    """Build 20 distinct scenarios through real mining and grounding with scripted transport."""
    project, catalog, state = _completed_project(root)
    runtime = _RuntimeCatalog(catalog, state)
    model, _ = runtime.snapshot("candidate-a")
    built = build_project(
        TraceNormalizationResult(
            traces=tuple(_trace(index, model) for index in range(40)), issues=()
        ),
        project,
        created_at=_TIME,
        code_revision=_REVISION,
        mining_spec=MiningSpec(
            fit_task_budget=15, held_out_task_budget=5, semantic_duplicate_threshold=1.0
        ),
    )
    world, _ = runtime.snapshot("world")
    artifacts = _build_grounded_artifacts(
        project,
        built,
        world_alias="world",
        world_snapshot=world,
        resolved_embedder=runtime.resolve("embedder"),
        top_k=2,
    )
    select_completed_build(project, artifacts, built.review)
    assert len(evaluation_tasks(project)) == 20
    return project, catalog, state


def test_twenty_scenarios_repeats_persist_report_and_exact_resume(tmp_path: Path) -> None:
    """An 80-cell run yields portable evidence and replay invokes no model again."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    defaults = EvaluationDefaults(
        models=("candidate-a", "candidate-b"),
        options=ModelEvaluationOptions(maximum_steps=1, repeats=2, maximum_concurrency=4),
    )
    progress: list[ProgressEvent] = []
    before_prepare = len(state.completion_calls), len(state.embedding_calls)
    run = prepare_run(project, catalog, defaults, code_revision=_REVISION, progress=progress.append)
    assert before_prepare == (len(state.completion_calls), len(state.embedding_calls))
    assert progress[0].stage == "Loading scenarios"
    assert progress[-1].stage == "Saving evaluation"
    tools_progress = [event for event in progress if event.stage == "Checking tool schemas"]
    assert [event.completed for event in tools_progress] == list(range(21))
    assert {event.total for event in tools_progress} == {20}
    assert run.prepared.cost.judgment_count == 80
    assert run.prepared.cost.scenario_count == 20
    assert run.spending_limit_usd is None

    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    result = execute_run(project, run, runtime, provider_spend_consented=True)
    saved = load_run(project, run.run_id)
    assert saved.status == "completed"
    assert result.report.compared_cells == 40
    assert all(
        model.operating_cost_usd == pytest.approx(0.000016) for model in result.report.models
    )
    assert all(model.latency_seconds is not None for model in result.report.models)
    evidence = load_report_evidence(project, saved)
    assert len(evidence.tasks) == 20
    assert len(evidence.rollouts) == 80
    assert {row.repeat for row in evidence.rows} == {0, 1}
    json_path, html_path = export_report(project, saved)
    assert json_path.exists() and html_path.exists()
    assert "__DATA__" not in html_path.read_text()
    assert list_runs(project)[0] == saved
    before = len(state.completion_calls), len(state.embedding_calls)
    current = project.load_project()
    write_project_config(project.paths, current.model_copy(update={"build": None, "models": None}))
    replay = execute_run(project, saved, runtime, provider_spend_consented=True)
    assert project.load_project().build is None
    write_project_config(project.paths, current)
    assert replay.report == result.report
    assert before == (len(state.completion_calls), len(state.embedding_calls))
    fresh = prepare_run(project, catalog, defaults, code_revision=_REVISION)
    assert fresh.run_id != saved.run_id
    fresh = fresh.model_copy(update={"spending_limit_usd": 100.0})
    save_run(project, fresh)
    rerun = execute_run(project, fresh, runtime, provider_spend_consented=True)
    assert rerun.simulation_spec.simulation_id != result.simulation_spec.simulation_id
    assert len(state.completion_calls) > before[0]
    fresh_evidence = load_report_evidence(project, load_run(project, fresh.run_id))
    assert {item.rollout_id for item in evidence.rollouts}.isdisjoint(
        item.rollout_id for item in fresh_evidence.rollouts
    )


def test_underfilled_project_never_dispatches_or_creates_a_run(tmp_path: Path) -> None:
    """A tiny scenario set cannot accidentally become a model report card."""
    project, catalog, state = _completed_project(tmp_path)
    before = len(state.completion_calls), len(state.embedding_calls)
    with pytest.raises(ValueError, match="requires 20"):
        prepare_run(
            project,
            catalog,
            EvaluationDefaults(models=("candidate-a", "candidate-b")),
            code_revision=_REVISION,
        )
    assert before == (len(state.completion_calls), len(state.embedding_calls))
    assert list_runs(project) == ()


def test_interrupted_parallel_run_resumes_without_repeating_paid_cells(tmp_path: Path) -> None:
    """Stopping at a progress boundary drains active cells and resumes only missing work."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(maximum_steps=1, maximum_concurrency=4),
        ),
        code_revision=_REVISION,
    )
    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))

    def interrupt(event: ProgressEvent) -> None:
        """Stop once at least one rollout has been durably published."""
        if event.stage == "evaluation cells" and event.completed:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_run(project, run, runtime, provider_spend_consented=True, progress=interrupt)
    saved = load_run(project, run.run_id)
    assert saved.status == "interrupted"
    candidate_calls = sum(alias.startswith("candidate") for alias, _ in state.completion_calls)
    assert 0 < candidate_calls < 40
    write_project_config(
        project.paths, project.load_project().model_copy(update={"build": None, "models": None})
    )
    result = execute_run(project, saved, runtime, provider_spend_consented=True)
    assert result.report.compared_cells == 20
    assert sum(alias.startswith("candidate") for alias, _ in state.completion_calls) == 40
    assert load_run(project, run.run_id).status == "completed"


def test_saved_run_requires_explicit_nullable_spending_limit(tmp_path: Path) -> None:
    """Null is an intentional no-limit setting; an omitted persisted field is outdated."""
    project, catalog, _ = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    records = ProjectRecords(project.paths.root, project.paths.project_id, "evaluation-runs")
    payload = run.model_dump(mode="json")
    assert "spending_limit_usd" in payload and payload["spending_limit_usd"] is None
    assert load_run(project, run.run_id) == run
    del payload["spending_limit_usd"]
    records.write(run.run_id, canonical_json_bytes(payload))
    with pytest.raises(EvaluationPreparationOutdated, match="new cost plan"):
        load_run(project, run.run_id)
    payload["spending_limit_usd"] = None
    records.write(run.run_id, canonical_json_bytes(payload))
    assert load_run(project, run.run_id) == run


@pytest.mark.parametrize("pause_stage", ["evaluation cells", "judging", "judgments"])
def test_lowered_cap_pauses_only_new_dispatch_and_preserves_paid_lineage(
    tmp_path: Path, pause_stage: str
) -> None:
    """Lowering a cap replays paid work, pauses new calls, and never starts a new judge pass."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(maximum_steps=1, maximum_concurrency=1),
        ),
        code_revision=_REVISION,
    )
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    initial_calls = len(state.completion_calls)

    def interrupt(event: ProgressEvent) -> None:
        """Interrupt after paid rollouts or a persisted judgment, before all work completes."""
        if event.stage == pause_stage and (pause_stage == "judging" or event.completed):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_run(project, run, runtime, provider_spend_consented=True, progress=interrupt)
    interrupted = load_run(project, run.run_id)
    assert interrupted.status == "interrupted"
    paid_calls = tuple(state.completion_calls[initial_calls:])
    assert any(alias.startswith("candidate") for alias, _ in paid_calls)
    assert any(alias == "judge" for alias, _ in paid_calls) is (pause_stage == "judgments")
    limited = interrupted.model_copy(update={"spending_limit_usd": 1e-12})
    save_run(project, limited)
    before = tuple(state.completion_calls), tuple(state.embedding_calls)
    with pytest.raises(SpendLimitReached):
        execute_run(project, limited, runtime, provider_spend_consented=True)
    paused = load_run(project, run.run_id)
    assert paused.status == "paused"
    assert paused.stage == "Spending limit reached"
    assert paused.judging_revision == interrupted.judging_revision
    assert paused.required_spending_limit_usd is not None
    assert before == (tuple(state.completion_calls), tuple(state.embedding_calls))

    resumed = paused.model_copy(update={"spending_limit_usd": None})
    save_run(project, resumed)
    result = execute_run(project, resumed, runtime, provider_spend_consented=True)
    completed = load_run(project, run.run_id)
    assert completed.status == "completed"
    assert completed.judging_revision == interrupted.judging_revision
    assert result.report.compared_cells == 20
    final_calls = state.completion_calls[initial_calls:]
    assert sum(alias.startswith("candidate") for alias, _ in final_calls) == 40
    assert sum(alias == "judge" for alias, _ in final_calls) == 40
    assert result.simulation_cost_usd > 1e-12
    assert result.judge_cost_usd > 1e-12

    before_replay = tuple(state.completion_calls), tuple(state.embedding_calls)
    replay = run_prepared_model_evaluation(
        project.snapshot(completed.project_config_sha256),
        completed.prepared,
        runtime,
        budget=EvaluationBudget(maximum_cost_usd=1e-12, maximum_judgments=40),
        provider_spend_consented=True,
        created_at=completed.created_at,
        code_revision=completed.code_revision,
        judging_revision=completed.judging_revision,
    )
    assert replay == result
    assert before_replay == (tuple(state.completion_calls), tuple(state.embedding_calls))
