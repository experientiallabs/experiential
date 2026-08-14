"""CLI surface tests for explicit manual judge configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app


@pytest.mark.parametrize(
    "arguments",
    [
        ["config", "judge", "setup", "support"],
        [
            "config",
            "judge",
            "calibrate",
            "support",
            "--input-usd-per-million",
            "0",
            "--output-usd-per-million",
            "0",
        ],
    ],
)
def test_malformed_release_revision_fails_before_judge_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """Reject malformed producer identity before judge project or artifact writes.

    Args:
        tmp_path: Pytest-owned state directory.
        monkeypatch: Scoped invalid release revision override.
        arguments: Exact nested judge command invocation.
    """
    root = tmp_path / ".wmo"
    monkeypatch.setenv("WMO_RELEASE_REVISION", "HEAD")

    result = CliRunner().invoke(app, [*arguments, "--root", str(root)])

    assert result.exit_code == 2
    assert "full lowercase 40-hex" in result.output
    assert not root.exists()


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
