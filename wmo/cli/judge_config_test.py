"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app


def test_judge_commands_render_as_nested_config_commands() -> None:
    """Both manual judge stages are discoverable without expanding the root surface."""
    runner = CliRunner()

    setup = runner.invoke(app, ["config", "judge", "setup", "--help"])
    calibrate = runner.invoke(app, ["config", "judge", "calibrate", "--help"])

    assert setup.exit_code == 0, setup.output
    assert calibrate.exit_code == 0, calibrate.output
    setup_output = unstyle(setup.output)
    calibrate_output = unstyle(calibrate.output)
    assert "--approve" in setup_output
    assert "--yes" in calibrate_output
    assert "--approve" in calibrate_output
