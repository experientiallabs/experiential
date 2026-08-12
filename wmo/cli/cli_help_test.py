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
        ["optimize", "router", "--help"],
        ["optimize", "model", "--help"],
        ["run", "--help"],
    ],
    ids=["root", "build", "telemetry", "router", "model", "run"],
)
def test_help_renders_only_user_facing_descriptions(argv: list[str]) -> None:
    """Command help exposes the approved surface without docstring scaffolding."""
    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 0, result.output
    for marker in ("Args:", "Returns:", "Inputs accepted by this callable"):
        assert marker not in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["providers"],
        ["list"],
        ["download"],
        ["eval"],
        ["knowledge"],
        ["serve"],
        ["optimize", "route"],
        ["optimize", "router", "fit"],
        ["optimize", "router", "report"],
        ["optimize", "distill"],
    ],
)
def test_obsolete_commands_are_not_callable(argv: list[str]) -> None:
    """Clean-break router and root aliases do not survive as hidden commands."""
    result = CliRunner().invoke(app, argv)

    assert result.exit_code == 2
