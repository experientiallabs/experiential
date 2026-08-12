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
        ["serve", "--help"],
        ["eval", "--help"],
        ["knowledge", "--help"],
        ["providers", "set", "--help"],
        ["providers", "verify", "--help"],
        ["config", "telemetry", "--help"],
        ["optimize", "model", "--help"],
        ["optimize", "route", "sweep", "--help"],
        ["optimize", "route", "fit", "--help"],
        ["optimize", "route", "tune", "--help"],
        ["optimize", "route", "report", "--help"],
        ["optimize", "route", "student", "--help"],
        ["optimize", "route", "pin", "--help"],
        ["optimize", "route", "convert-deepswe", "--help"],
        ["scenarios", "build", "--help"],
        ["scenarios", "verify", "--help"],
    ],
    ids=[
        "root",
        "build",
        "list",
        "download",
        "serve",
        "eval",
        "knowledge",
        "provider-set",
        "provider-verify",
        "telemetry",
        "nested-model",
        "route-sweep",
        "route-fit",
        "route-tune",
        "route-report",
        "route-student",
        "route-pin",
        "route-deepswe",
        "scenarios-build",
        "scenarios-verify",
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
