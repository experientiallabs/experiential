"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app
from wmo.cli.router_app import _require_approved_manual_calibration
from wmo.common.project import ProjectConfig, ProjectStore


def test_judge_commands_render_as_nested_config_commands() -> None:
    """Both manual judge stages are discoverable without expanding the root surface."""
    runner = CliRunner()

    setup = runner.invoke(app, ["config", "judge", "setup", "--help"])
    calibrate = runner.invoke(app, ["config", "judge", "calibrate", "--help"])

    assert setup.exit_code == 0, setup.output
    assert calibrate.exit_code == 0, calibrate.output
    assert "--approve" in setup.output
    assert "--yes" in calibrate.output
    assert "--approve" in calibrate.output


def test_human_calibrated_optimize_error_names_both_manual_steps(tmp_path: Path) -> None:
    """A missing approval directs users through setup and calibration instead of failing vaguely."""
    store = ProjectStore(tmp_path / ".wmo", "support")
    store.initialize(ProjectConfig(project_id="support"))

    with pytest.raises(ValueError) as error:
        _require_approved_manual_calibration(store)

    message = str(error.value)
    assert "wmo config judge setup PROJECT" in message
    assert "wmo config judge calibrate PROJECT --approve" in message
