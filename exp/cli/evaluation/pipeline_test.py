"""The public pipeline command drives real native packages with bounded, explicit spending."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.runner_test import factory_fixture


def test_pipeline_dry_run_execution_and_saved_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the maintained CLI from preflight to XLSX execution, export and offline replay."""
    project, spec, transport = factory_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("exp.cli.evaluation.pipeline.installed_execution_revision", lambda: "test")
    path = tmp_path / "pipeline.json"
    path.write_text(spec.model_dump_json())
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--pipeline",
        str(path),
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "No new calls" in dry.output
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test"))
    denied = runner.invoke(app, args)
    assert denied.exit_code == 2, denied.output
    assert not transport.requests
    run = runner.invoke(app, [*args, "--yes"])
    assert run.exit_code == 0, run.output
    assert '"packaged_candidates": 2' in run.output
    assert '"selected_components": 1' in run.output
    assert "no protected release confirmation" in run.output
    assert len(transport.requests) == 12
    load_executable_task(project, spec.families[0].proposal.base).job.runtime_file.unlink()
    project.model_catalog_path.unlink()
    replay = runner.invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert '"packaged_candidates": 2' in replay.output
    assert len(transport.requests) == 12


@pytest.mark.parametrize("extra", [["--continuations", "other.json"], ["--models", "a,b"]])
def test_pipeline_refuses_other_modes_before_loading_inputs(extra: list[str]) -> None:
    """A frozen manifest cannot be silently altered through unrelated evaluation flags."""
    result = CliRunner().invoke(app, ["eval", "example", "--pipeline", "none.json", *extra])
    assert result.exit_code == 2
    assert "cannot combine" in " ".join(result.output.split())
