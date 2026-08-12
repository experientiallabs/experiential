"""Regression coverage for user-facing CLI help rendering."""

from __future__ import annotations

import pytest

from wmo.cli.cli_fixtures_test import app, runner


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["build", "--help"],
        ["list", "--help"],
        ["download", "--help"],
        ["eval", "--help"],
        ["knowledge", "--help"],
        ["providers", "set", "--help"],
        ["providers", "verify", "--help"],
        ["config", "telemetry", "--help"],
        ["optimize", "router", "--help"],
        ["optimize", "router", "fit", "--help"],
        ["optimize", "router", "report", "--help"],
    ],
    ids=[
        "root",
        "build",
        "list",
        "download",
        "eval",
        "knowledge",
        "provider-set",
        "provider-verify",
        "telemetry",
        "router",
        "router-fit",
        "router-report",
    ],
)
def test_help_renders_only_user_facing_descriptions(argv: list[str]) -> None:
    """Command help never exposes generated Python docstring placeholder sections."""
    result = runner.invoke(app, argv)

    assert result.exit_code == 0, result.output
    for marker in (
        "Args:",
        "Returns:",
        "Inputs accepted by this callable",
        "The value produced by this callable",
    ):
        assert marker not in result.output


def test_removed_route_owner_is_not_callable() -> None:
    """The W10 clean break leaves no parallel legacy router CLI owner."""
    result = runner.invoke(app, ["optimize", "route", "--help"])

    assert result.exit_code == 2
    assert "No such command 'route'" in result.output
