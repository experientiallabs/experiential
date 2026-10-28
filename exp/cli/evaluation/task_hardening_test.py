"""Task validity and hardening run through ``exp eval --continuations`` with spend consent."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.factory.hardening_test import record_hardening
from exp.optimize.evaluation.factory.leakage import ValidityRule
from exp.optimize.evaluation.factory.validity_test import closed, judge, record_validity, script


def test_check_task_validity_runs_with_consent_and_saves_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry run and refused consent make no draw; the run prints a summary and saves the report."""
    project, spec = record_validity(
        tmp_path, monkeypatch, rule=ValidityRule(specified_values=("new",))
    )
    transport = script(monkeypatch, judge(), closed())
    path = tmp_path / "validity.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="check-task-validity", task_validity=spec
        ).model_dump_json()
    )
    args = ["eval", "records", "--root", str(tmp_path), "--continuations", str(path)]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run", "--non-interactive"])
    assert dry.exit_code == 0, dry.output
    assert not transport.requests
    assert not project.artifacts.list_ids(artifact_type="task-validity-registration")
    refused = runner.invoke(app, [*args, "--non-interactive"])
    assert refused.exit_code != 0 and not transport.requests
    run = runner.invoke(app, [*args, "--yes", "--non-interactive"])
    assert run.exit_code == 0, run.output
    assert "Task validity: valid" in run.output and "Report:" in run.output
    assert "mechanical_leak: pass" in run.output
    identity = spec.identity(installed_execution_revision())
    report = project.paths.runtime_directory / "task-validity" / identity / "report.txt"
    assert json.loads(report.read_text())["status"] == "valid"
    assert len(transport.requests) == 2


def test_harden_tasks_runs_with_consent_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saturated record world is hardened once; a rerun replays the saved report."""
    project, spec, author, actor = record_hardening(tmp_path, monkeypatch)
    path = tmp_path / "hardening.json"
    path.write_text(
        ContinuationCommand(
            spend=UNPRICED_SPEND, action="harden-tasks", hardening=spec
        ).model_dump_json()
    )
    args = ["eval", "records", "--root", str(tmp_path), "--continuations", str(path)]
    runner = CliRunner()
    seeded = len(author.requests)
    dry = runner.invoke(app, [*args, "--dry-run", "--non-interactive"])
    assert dry.exit_code == 0, dry.output
    assert "up to 2 harder levels, 2 panel actors, 1 attempts each" in dry.output
    assert len(author.requests) == seeded and not actor.requests
    assert not project.artifacts.list_ids(artifact_type="task-hardening-registration")
    run = runner.invoke(app, [*args, "--yes", "--non-interactive"])
    assert run.exit_code == 0, run.output
    assert "Task hardening: kept" in run.output
    assert "level 0 world 1: saturated, pass rate 1.00" in run.output
    assert "level 1 world 0: kept, pass rate 0.50" in run.output
    assert "level 1 world 1: unqualified" in run.output and "Report:" in run.output
    assert (len(author.requests), len(actor.requests)) == (6, 8)
    again = runner.invoke(app, [*args, "--non-interactive"])
    assert again.exit_code == 0, again.output
    assert "Task hardening: kept" in again.output
    assert (len(author.requests), len(actor.requests)) == (6, 8)
