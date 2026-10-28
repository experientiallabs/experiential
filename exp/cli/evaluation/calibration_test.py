"""Calibration CLI rejects mode mixing before opening artifacts or providers."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app


@pytest.mark.parametrize(
    "extra", [["--models", "actor"], ["--fidelity", "other.json"], ["--refine-task", "other.json"]]
)
def test_calibration_mode_cannot_silently_change_other_workloads(
    tmp_path: Path, extra: list[str]
) -> None:
    """An explicit frozen specification cannot combine with model or mode overrides."""
    result = CliRunner().invoke(
        app, ["eval", "example", "--root", str(tmp_path), "--calibrate", "pack.json", *extra]
    )
    assert result.exit_code != 0
    assert "cannot combine" in result.output
    assert not list(tmp_path.iterdir())
