"""Regression coverage for the locked customer CLI help surface."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from wmo.cli.app import app


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["build", "--help"],
        ["config", "telemetry", "--help"],
        ["config", "providers", "--help"],
        ["optimize", "router", "--help"],
        ["optimize", "model", "--help"],
        ["run", "--help"],
    ],
    ids=["root", "build", "telemetry", "providers", "router", "model", "run"],
)
def test_help_renders_only_user_facing_descriptions(argv: list[str]) -> None:
    """Command help exposes the approved surface without docstring scaffolding."""
    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 0, result.output
    for marker in ("Args:", "Returns:", "Inputs accepted by this callable"):
        assert marker not in result.output
