"""Native materialization is an explicit mode, separate from provider evaluation."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app


@pytest.mark.parametrize(
    "extra", [["--models", "actor"], ["--calibrate", "pack.json"], ["--fidelity", "other.json"]]
)
def test_materialization_refuses_mode_mixing(tmp_path: Path, extra: list[str]) -> None:
    """No other mode can silently alter a frozen handoff or trigger provider setup."""
    result = CliRunner().invoke(
        app, ["eval", "example", "--root", str(tmp_path), "--materialize", "native.json", *extra]
    )
    assert result.exit_code != 0
    assert "cannot combine" in result.output
    assert not list(tmp_path.iterdir())
