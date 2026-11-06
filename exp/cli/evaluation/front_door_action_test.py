"""The front-door action: a dry-run plan from a trace file, and the ``exp build`` it would run."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.decision_variance_action import DecisionVarianceAction
from exp.cli.evaluation.front_door_action import ExpBuild
from exp.optimize.evaluation.spending import SpendPolicy
from exp.optimize.evaluation.variance.frontdoor import TraceSource
from exp.optimize.evaluation.variance.frontdoor_test import front_door


def test_the_dry_run_separates_completed_remaining_and_unpriced_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointed at traces, the plan prints every stage and makes no call or artifact."""
    run = front_door(tmp_path, monkeypatch)
    spec_path = tmp_path / "front-door.json"
    spec_path.write_text(run.spec.model_dump_json())
    request = tmp_path / "plan.json"
    request.write_text(
        ContinuationCommand(
            action="decision-variance",
            decision_variance=DecisionVarianceAction(front_door=spec_path),
            spend=SpendPolicy(unpriced_attempts=1_000),
        ).model_dump_json()
    )
    before = set(run.project.artifacts.list_ids())
    paths = run.project.paths
    result = CliRunner().invoke(
        app,
        ["eval", paths.project_id, "--root", str(paths.root), "--continuations", str(request)],
    )
    assert result.exit_code == 0, result.output
    output = result.output
    assert "1. project-build: up to $0.000000" in output
    assert "2. bootstrap-preparation: waiting on an earlier stage" in output
    assert "3. pipelines: waiting on an earlier stage" in output
    assert "Completed work: none." in output
    assert "Remaining calls: up to $0.000000 priced across 1 pending stages" in output
    assert "Unpriced components:" in output and "exp build's own preflight" in output
    assert "priced once the preparation registers it" in output
    assert "Dry run: front-door plan" in output
    assert set(run.project.artifacts.list_ids()) == before
    assert run.builder.calls == []


def test_the_action_takes_exactly_one_spec() -> None:
    """A pipeline spec and a front-door spec are alternatives."""
    with pytest.raises(ValidationError, match="exactly one"):
        DecisionVarianceAction()
    with pytest.raises(ValidationError, match="exactly one"):
        DecisionVarianceAction(spec=Path("a.json"), front_door=Path("b.json"))


def test_a_pending_build_runs_exp_build_non_interactively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder never answers a cost question itself and reports a stopped build."""
    commands: list[list[str]] = []

    def record(command: list[str], *, check: bool) -> subprocess.CompletedProcess[bytes]:
        """Record the argument vector and fail like a build that stopped at its estimate."""
        assert check is False
        commands.append(command)
        return subprocess.CompletedProcess(command, 2)

    run = front_door(tmp_path, monkeypatch)
    monkeypatch.setattr("exp.cli.evaluation.front_door_action.subprocess.run", record)
    builder = ExpBuild(root=run.project.paths.root, project=run.project.paths.project_id)
    source = TraceSource(path=tmp_path / "traces.jsonl", source="otlp")
    with pytest.raises(ValueError, match="exited with status 2"):
        builder(run.project, source)
    (command,) = commands
    assert command[1:4] == ["-m", "exp", "build"]
    assert "--non-interactive" in command and "--yes" not in command
    assert command[command.index("--traces") + 1] == str(source.path)
