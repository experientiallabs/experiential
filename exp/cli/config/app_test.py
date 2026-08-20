"""CLI tests for project-local EXP settings commands."""

from __future__ import annotations

from pathlib import Path

from click import unstyle
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.config.settings import load_settings

_RUNNER = CliRunner()


def test_budget_status_reports_the_default_without_writing(tmp_path: Path) -> None:
    """Reading the default is deterministic and does not create settings state."""
    root = tmp_path / ".exp"

    result = _RUNNER.invoke(app, ["config", "budget", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "maximum command cost: $10.00" in unstyle(result.output)
    assert not root.exists()


def test_budget_command_persists_an_exact_noninteractive_limit(tmp_path: Path) -> None:
    """Agents can set the shared ceiling with one flag-complete command."""
    root = tmp_path / ".exp"

    result = _RUNNER.invoke(app, ["config", "budget", "12.5", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "maximum command cost: $12.50" in unstyle(result.output)
    assert load_settings(root).commands.maximum_cost_usd == 12.5


def test_budget_command_rejects_a_negative_limit_without_writing(tmp_path: Path) -> None:
    """The CLI cannot persist a limit below zero."""
    root = tmp_path / ".exp"

    result = _RUNNER.invoke(app, ["config", "budget", "-0.01", "--root", str(root)])

    assert result.exit_code == 2
    assert not root.exists()
