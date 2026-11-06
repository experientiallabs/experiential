"""The ``decision-variance`` action: the default plan, consent refusal and a consented run."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.decision_variance_action import DecisionVarianceAction
from exp.common.project import ProjectStore
from exp.optimize.evaluation.spending import SpendPolicy
from exp.optimize.evaluation.variance import pipeline
from exp.optimize.evaluation.variance.dense_prompt import dense_profile
from exp.optimize.evaluation.variance.pipeline_dense import DenseSettings
from exp.optimize.evaluation.variance.pipeline_dense_test import TAG
from exp.optimize.evaluation.variance.pipeline_spec import STAGE_TYPE, TraceMLSettings
from exp.optimize.evaluation.variance.pipeline_test import FixtureBuild, Run, make_spec, wired


def _command(project: ProjectStore, request: Path, *extra: str) -> list[str]:
    """The public ``exp eval`` command for one continuation spec."""
    paths = project.paths
    return [
        "eval",
        paths.project_id,
        "--root",
        str(paths.root),
        "--continuations",
        str(request),
        *extra,
    ]


def _request(run: Run, root: Path, *, execute: bool) -> Path:
    """Write a single-tier spec without TraceML and its operation file."""
    spec = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",), crosstalk=False)
    # The CLI run has no injected System One client and the stand-in rules owner learns no
    # memory, so the rules stage keeps the research profile here.
    rules = spec.rules.model_copy(update={"jev_memory": False})
    traceml = TraceMLSettings(
        enabled=False, bank_root=run.bank, decision_alias=spec.traceml.decision_alias
    )
    # The catalog has no typed decision alias to route dense steps, so this run freezes its own
    # dense vocabulary judged in full rather than the in-flow hybrid one.
    settings = run.fixture.settings
    dense = DenseSettings(
        tags=(TAG,),
        settings=settings,
        profile=dense_profile(
            model_alias=settings.model_alias,
            model=settings.model,
            maximum_output_tokens=settings.output_tokens,
            maximum_input_bytes=settings.maximum_input_bytes,
            reasoning_effort=settings.reasoning_effort,
        ),
    )
    records = spec.records.model_copy(update={"dense": dense})
    spec = spec.model_copy(update={"traceml": traceml, "rules": rules, "records": records})
    spec_path = root / "decision-variance.json"
    spec_path.write_text(spec.model_dump_json())
    command = ContinuationCommand(
        action="decision-variance",
        decision_variance=DecisionVarianceAction(spec=spec_path, execute=execute),
        spend=SpendPolicy(unpriced_attempts=100_000),
    )
    request = root / ("execute.json" if execute else "plan.json")
    request.write_text(command.model_dump_json())
    return request


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """A wired grounded project whose build stage is the fixture stand-in for every action."""
    wired_run = wired(tmp_path, monkeypatch)
    monkeypatch.setitem(
        pipeline.DEFAULT_EXECUTORS,
        "variance-build",
        FixtureBuild(wired_run.fixture),
    )
    return wired_run


def test_the_default_prints_the_stage_plan_and_bounds_and_makes_no_call(
    run: Run, tmp_path: Path
) -> None:
    """Without execute the action prints every stage, its status and its bound, then stops."""
    before = run.calls()
    result = CliRunner().invoke(app, _command(run.project, _request(run, tmp_path, execute=False)))

    assert result.exit_code == 0, result.output
    for stage in ("variance-build", "rules", "traceml", "error-mining", "families", "records"):
        assert f". {stage} [" in result.output
    assert "Error mining: 1 tiers (actor), each G=3 J=1 C=1 K=2 M=3" in result.output
    assert "physical attempts" in result.output and "unpriced" in result.output
    assert "Dry run: stage plan and upper-bound costs only" in result.output
    assert run.calls() == before
    assert not run.project.artifacts.list_ids(artifact_type=STAGE_TYPE)


def test_refused_consent_runs_no_stage(run: Run, tmp_path: Path) -> None:
    """Unpriced work without ``--yes`` in a non-interactive session is refused before any call."""
    before = run.calls()
    request = _request(run, tmp_path, execute=True)
    result = CliRunner().invoke(app, _command(run.project, request, "--non-interactive"))

    assert result.exit_code == 2, result.output
    assert "authorization:" in result.output
    assert run.calls() == before
    assert not run.project.artifacts.list_ids(artifact_type=STAGE_TYPE)


def test_a_consented_run_saves_every_stage_and_replays_without_consent(
    run: Run, tmp_path: Path
) -> None:
    """``--yes`` authorizes the plan once; a second invocation replays with no call."""
    request = _request(run, tmp_path, execute=True)
    runner = CliRunner()
    result = runner.invoke(app, _command(run.project, request, "--yes", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert '"complete": true' in result.output and "Report: " in result.output
    assert len(run.project.artifacts.list_ids(artifact_type=STAGE_TYPE)) == 6
    after = run.calls()
    replay = runner.invoke(app, _command(run.project, request, "--non-interactive"))

    assert replay.exit_code == 0, replay.output
    assert "saved, replays with no call" in replay.output
    assert run.calls() == after
