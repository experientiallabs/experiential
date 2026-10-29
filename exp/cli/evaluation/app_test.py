"""Command and terminal presentation regressions for evaluation workflows."""

from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation import flow
from exp.cli.shared import consent
from exp.cli.shared.picker import PickerResult
from exp.common.config.settings import set_maximum_command_cost_usd
from exp.common.models import ModelCatalog, ModelRequest, ModelResponse
from exp.common.progress import ProgressEvent
from exp.optimize.evaluation.prepare import ModelEvaluationOptions
from exp.optimize.evaluation.runs import (
    EvaluationDefaults,
    execute_run,
    load_run,
    prepare_run,
    save_run,
)
from exp.optimize.evaluation.runs_test import _twenty_scenarios
from exp.optimize.router.automatic.service_test import _REVISION, _CompletionClient, _RuntimeCatalog
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import SpendLimitReached
from exp.simulation.engines.text.leases import (
    TextCellLeaseClaim,
    TextCellLeaseState,
    TextCellLeaseStore,
)


def test_cli_review_and_resume_preserve_exact_preparation(tmp_path: Path) -> None:
    """Dry-run review needs no provider credentials and prints a directly usable resume command."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"), options=ModelEvaluationOptions(maximum_steps=1)
        ),
        code_revision=_REVISION,
    )
    before = len(state.completion_calls), len(state.embedding_calls)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "support",
            "--root",
            str(project.paths.root),
            "--resume",
            run.run_id,
            "--dry-run",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "20 scenarios" in result.output
    assert "Spending limit off" in result.output
    assert "Maximum $" not in result.output
    assert (
        "Models:" in result.output and "World model" in result.output and "Judge" in result.output
    )
    assert run.run_id in result.output
    assert load_run(project, run.run_id).status == "prepared"
    assert before == (len(state.completion_calls), len(state.embedding_calls))


@pytest.mark.parametrize("args", [[], ["support", "--resume", "run-a", "--models", "a,b"]])
def test_noninteractive_missing_or_conflicting_inputs_fail_clearly(args: list[str]) -> None:
    """Automation never hangs for missing choices or silently changes frozen settings."""
    result = CliRunner().invoke(app, ["eval", *args, "--non-interactive"])
    assert result.exit_code != 0
    assert "PROJECT" in result.output or "frozen settings" in result.output


def test_interactive_cancel_before_launch_never_constructs_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an inexpensive prepared evaluation requires the explicit Start action."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    before = len(state.completion_calls), len(state.embedding_calls)
    monkeypatch.setattr(flow, "can_prompt", lambda console: True)
    monkeypatch.setattr(flow, "choose_one", lambda *args, **kwargs: PickerResult(values=("back",)))

    def unexpected(*args: object, **kwargs: object) -> None:
        """Fail if authorization or client construction happens after cancellation."""
        pytest.fail("cancel reached provider construction or authorization")

    monkeypatch.setattr(flow, "RuntimeModelCatalog", unexpected)
    monkeypatch.setattr(flow, "require_spend_consent", unexpected)
    result = CliRunner().invoke(
        app, ["eval", "support", "--root", str(project.paths.root), "--resume", run.run_id]
    )
    assert result.exit_code == 0, result.output
    assert load_run(project, run.run_id).status == "prepared"
    assert before == (len(state.completion_calls), len(state.embedding_calls))


def test_unbuilt_project_requires_build_before_setup(tmp_path: Path) -> None:
    """Eval never imports traces, creates a project, or prompts for providers implicitly."""
    result = CliRunner().invoke(
        app, ["eval", "powerset", "--root", str(tmp_path), "--models", "a,b", "--non-interactive"]
    )
    assert result.exit_code != 0
    assert "exp build powerset" in result.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("interactive", [False, True])
def test_uncapped_launch_requires_yes_or_the_reviewed_start_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive: bool
) -> None:
    """A reviewed Start authorizes once; a cheap noninteractive estimate authorizes nothing."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    set_maximum_command_cost_usd(1_000, project.paths.root)
    assert run.spending_limit_usd is None
    assert run.prepared.cost.estimated_cost_usd < 500
    monkeypatch.setattr(flow, "can_prompt", lambda console: interactive)
    monkeypatch.setattr(consent, "can_prompt", lambda console: False)
    reviewed: list[str] = []
    constructed: list[ModelCatalog] = []

    def choose(*args: object, **kwargs: object) -> PickerResult:
        """Record the explicit Start action after the no-limit review is displayed."""
        del args, kwargs
        reviewed.append("start")
        return PickerResult(values=("start",))

    def runtime(catalog: ModelCatalog) -> RuntimeModelCatalog:
        """Stop at the actual provider-construction boundary without making any call."""
        constructed.append(catalog)
        raise ValueError("authorized provider boundary")

    monkeypatch.setattr(flow, "choose_one", choose)
    monkeypatch.setattr(flow, "RuntimeModelCatalog", runtime)
    before = len(state.completion_calls), len(state.embedding_calls), state.credential_resolutions
    result = CliRunner().invoke(
        app, ["eval", "support", "--root", str(project.paths.root), "--resume", run.run_id]
    )
    assert result.exit_code != 0
    if interactive:
        assert reviewed == ["start"]
        assert len(constructed) == 1
        assert "authorized provider boundary" in result.output
    else:
        assert reviewed == constructed == []
        assert "no spending limit and requires explicit confirmation" in " ".join(
            result.output.split()
        )
        assert "--yes" in result.output
    assert before == (
        len(state.completion_calls),
        len(state.embedding_calls),
        state.credential_resolutions,
    )


def test_execution_shows_progress_before_runtime_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pressing Start announces preparation before any potentially slow runtime setup."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    output = StringIO()
    monkeypatch.setattr(flow, "_console", Console(file=output, width=100))
    before = len(state.completion_calls), len(state.embedding_calls), state.credential_resolutions

    def inspect_runtime(catalog: ModelCatalog) -> RuntimeModelCatalog:
        """Stop before client construction after checking that progress is already visible."""
        del catalog
        assert "Preparing evaluation" in output.getvalue()
        raise ValueError("stop before provider initialization")

    monkeypatch.setattr(flow, "RuntimeModelCatalog", inspect_runtime)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "support",
            "--root",
            str(project.paths.root),
            "--resume",
            run.run_id,
            "--yes",
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0
    assert "stop before provider initialization" in result.output
    assert before == (
        len(state.completion_calls),
        len(state.embedding_calls),
        state.credential_resolutions,
    )


def test_cell_contention_pauses_with_resume_command_and_no_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An occupied cell leaves a resumable run instead of a traceback or failed rollout."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    monkeypatch.setattr(
        flow, "RuntimeModelCatalog", lambda catalog: _RuntimeCatalog(catalog, state)
    )
    monkeypatch.setattr(
        TextCellLeaseStore,
        "acquire",
        lambda *args, **kwargs: TextCellLeaseClaim(TextCellLeaseState.CONTENDED, None, None),
    )
    before = len(state.completion_calls), len(state.embedding_calls)
    rollouts_before = tuple(
        key
        for key in project.artifacts.list_ids()
        if project.artifacts.read(key).manifest.artifact_type == "rollout"
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "support",
            "--root",
            str(project.paths.root),
            "--resume",
            run.run_id,
            "--yes",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Paused" in result.output
    assert "Completed work saved" in result.output
    assert f"--resume {run.run_id}" in " ".join(result.output.split())
    saved = load_run(project, run.run_id)
    assert saved.status == "paused"
    assert saved.report_id is None
    assert (
        tuple(
            key
            for key in project.artifacts.list_ids()
            if project.artifacts.read(key).manifest.artifact_type == "rollout"
        )
        == rollouts_before
    )
    assert before == (len(state.completion_calls), len(state.embedding_calls))


@pytest.mark.parametrize("answer", ["n", "y"])
def test_eval_over_budget_can_decline_or_complete_the_saved_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """An over-budget eval keeps its prepared run on decline and can execute after approval."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(maximum_steps=1),
        ),
        code_revision=_REVISION,
    )
    run = run.model_copy(update={"spending_limit_usd": 100.0})
    save_run(project, run)
    set_maximum_command_cost_usd(0.0, project.paths.root)
    before = len(state.completion_calls), len(state.embedding_calls)
    monkeypatch.setattr(flow, "can_prompt", lambda _console: True)
    monkeypatch.setattr(consent, "can_prompt", lambda _console: True)
    monkeypatch.setattr(flow, "_review", lambda _project, run: run)
    monkeypatch.setattr(flow, "_results", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        flow, "RuntimeModelCatalog", lambda catalog: _RuntimeCatalog(catalog, state)
    )

    result = CliRunner().invoke(
        app,
        ["eval", "support", "--root", str(project.paths.root), "--resume", run.run_id],
        input=f"{answer}\n",
    )

    assert result.exit_code == 0, result.output
    assert "exceeds the $0.00 budget" in result.output
    assert "Proceed anyway" in result.output
    if answer == "y":
        assert load_run(project, run.run_id).status == "completed"
        assert len(state.completion_calls) > before[0]
    else:
        assert load_run(project, run.run_id).status == "prepared"
        assert before == (len(state.completion_calls), len(state.embedding_calls))


def test_cli_spending_pause_is_saved_without_report_or_invalid_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public command pauses rather than converting an allowance boundary into a failure."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(maximum_steps=1),
        ),
        code_revision=_REVISION,
    )
    run = run.model_copy(update={"spending_limit_usd": 0.2})
    save_run(project, run)
    monkeypatch.setattr(
        flow, "RuntimeModelCatalog", lambda catalog: _RuntimeCatalog(catalog, state)
    )
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "support",
            "--root",
            str(project.paths.root),
            "--resume",
            run.run_id,
            "--non-interactive",
            "--yes",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Paused at the $0.20 spending limit" in result.output
    assert "Completed calls saved" in result.output
    saved = load_run(project, run.run_id)
    assert saved.status == "paused"
    assert saved.report_id is None
    assert saved.required_spending_limit_usd is not None
    assert saved.spending_limit_usd is not None
    assert saved.required_spending_limit_usd > saved.spending_limit_usd


@pytest.mark.parametrize("choice", ["start", "back"])
def test_uncapped_warning_precedes_the_single_launch_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    """An above-threshold uncapped review warns before authorization and defaults to Back."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(models=("candidate-a", "candidate-b")),
        code_revision=_REVISION,
    )
    set_maximum_command_cost_usd(0.01, project.paths.root)
    assert run.prepared.cost.estimated_cost_usd > 0.01
    output = StringIO()
    monkeypatch.setattr(flow, "_console", Console(file=output, width=200))
    monkeypatch.setattr(flow, "can_prompt", lambda console: True)
    monkeypatch.setattr(consent, "can_prompt", lambda console: False)
    reviewed: list[str] = []
    constructed: list[ModelCatalog] = []

    def choose(*args: object, **kwargs: object) -> PickerResult:
        """Verify the warning is already visible before accepting the single user choice."""
        del args
        assert "configured $0.01 warning threshold" in output.getvalue()
        assert "Spending limit off" in output.getvalue()
        assert kwargs["default"] == "back"
        reviewed.append(choice)
        return PickerResult(values=(choice,))

    def runtime(catalog: ModelCatalog) -> RuntimeModelCatalog:
        """Record authorization without constructing a live provider."""
        constructed.append(catalog)
        raise ValueError("authorized provider boundary")

    def unexpected_confirmation(*args: object, **kwargs: object) -> bool:
        """Reject any second confirmation after the reviewed launch action."""
        del args, kwargs
        pytest.fail("reviewed Start must be the only authorization")

    monkeypatch.setattr(flow, "choose_one", choose)
    monkeypatch.setattr(flow, "RuntimeModelCatalog", runtime)
    monkeypatch.setattr(consent.Confirm, "ask", unexpected_confirmation)
    before = len(state.completion_calls), len(state.embedding_calls), state.credential_resolutions
    result = CliRunner().invoke(
        app, ["eval", "support", "--root", str(project.paths.root), "--resume", run.run_id]
    )
    assert reviewed == [choice], result.output
    assert len(constructed) == (1 if choice == "start" else 0)
    assert result.exit_code == (2 if choice == "start" else 0), result.output
    assert output.getvalue().count("configured $0.01 warning threshold") == 1
    assert before == (
        len(state.completion_calls),
        len(state.embedding_calls),
        state.credential_resolutions,
    )


@pytest.mark.parametrize("affordable", [False, True])
def test_resumed_progress_retires_cap_hint_and_ledger_admits_lower_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, affordable: bool
) -> None:
    """A saved reservation hint never vetoes review; current ledger admission owns the cap."""
    project, catalog, state = _twenty_scenarios(tmp_path)
    complete = _CompletionClient.complete

    def one_attempt(client: _CompletionClient, request: ModelRequest) -> ModelResponse:
        """Report the scripted client's one real call so its unused retry holds settle."""
        response = complete(client, request)
        return response.model_copy(
            update={"economics": response.economics.model_copy(update={"provider_attempts": 1})}
        )

    monkeypatch.setattr(_CompletionClient, "complete", one_attempt)
    run = prepare_run(
        project,
        catalog,
        EvaluationDefaults(
            models=("candidate-a", "candidate-b"),
            options=ModelEvaluationOptions(
                maximum_steps=1, maximum_concurrency=1, maximum_judge_output_tokens=16
            ),
        ),
        code_revision=_REVISION,
    ).model_copy(update={"spending_limit_usd": 1e-12})
    save_run(project, run)
    runtime = cast(RuntimeModelCatalog, _RuntimeCatalog(catalog, state))
    before = tuple(state.completion_calls), tuple(state.embedding_calls)
    with pytest.raises(SpendLimitReached):
        execute_run(project, run, runtime, provider_spend_consented=True)
    paused = load_run(project, run.run_id)
    old_hint = paused.required_spending_limit_usd
    assert old_hint is not None and old_hint > 1e-12
    assert before == (tuple(state.completion_calls), tuple(state.embedding_calls))
    choices = iter(("start", "back"))
    monkeypatch.setattr(
        flow, "choose_one", lambda *args, **kwargs: PickerResult(values=(next(choices),))
    )
    assert flow._review(project, paused) == paused
    with pytest.raises(SpendLimitReached):
        execute_run(project, paused, runtime, provider_spend_consented=True)
    assert before == (tuple(state.completion_calls), tuple(state.embedding_calls))

    resumed = paused.model_copy(update={"spending_limit_usd": run.prepared.cost.maximum_cost_usd})
    save_run(project, resumed)

    def interrupt(event: ProgressEvent) -> None:
        """Stop after the previously blocked calls settle and before cheaper judging starts."""
        if event.stage == "judging":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_run(project, resumed, runtime, provider_spend_consented=True, progress=interrupt)
    interrupted = load_run(project, run.run_id)
    assert interrupted.status == "interrupted"
    assert interrupted.required_spending_limit_usd is None
    lowered = interrupted.model_copy(
        update={"spending_limit_usd": old_hint * 0.75 if affordable else 1e-12}
    )
    choices = iter(("start", "back"))
    assert flow._review(project, lowered) == lowered
    save_run(project, lowered)
    before_judging = tuple(state.completion_calls), tuple(state.embedding_calls)
    if affordable:
        result = execute_run(project, lowered, runtime, provider_spend_consented=True)
        assert result.report.compared_cells == 20
        assert result.simulation_cost_usd is not None
        assert result.simulation_cost_usd + result.judge_cost_usd < old_hint * 0.75
        assert load_run(project, run.run_id).status == "completed"
    else:
        with pytest.raises(SpendLimitReached):
            execute_run(project, lowered, runtime, provider_spend_consented=True)
        fresh = load_run(project, run.run_id)
        assert fresh.status == "paused"
        assert fresh.required_spending_limit_usd is not None
        assert fresh.required_spending_limit_usd < old_hint
        assert before_judging == (tuple(state.completion_calls), tuple(state.embedding_calls))
    assert load_run(project, run.run_id).judging_revision == run.judging_revision
