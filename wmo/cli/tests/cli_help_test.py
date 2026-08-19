"""Regression coverage for the locked customer CLI help surface."""

from __future__ import annotations

import pytest
from click import unstyle
from typer.testing import CliRunner

from wmo.cli.app import app


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["build", "--help"],
        ["config", "telemetry", "--help"],
        ["config", "budget", "--help"],
        ["config", "providers", "--help"],
        ["config", "judge", "setup", "--help"],
        ["config", "judge", "calibrate", "--help"],
        ["optimize", "router", "--help"],
        ["optimize", "model", "--help"],
        ["run", "--help"],
    ],
    ids=[
        "root",
        "build",
        "telemetry",
        "budget",
        "providers",
        "judge-setup",
        "judge-calibrate",
        "router",
        "model",
        "run",
    ],
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
        ["build", "--help"],
        ["config", "judge", "calibrate", "--help"],
        ["optimize", "router", "--help"],
        ["optimize", "model", "--help"],
    ],
)
def test_every_paid_command_exposes_deterministic_agent_flags(argv: list[str]) -> None:
    """Paid CLI surfaces expose explicit confirmation and noninteractive mode."""
    result = CliRunner().invoke(app, argv, color=True)
    help_text = unstyle(result.output)

    assert result.exit_code == 0, result.output
    assert "--yes" in help_text
    assert "--non-interactive" in help_text


def test_config_gateway_deferred_names_match_real_gateway_commands() -> None:
    """The deferred gateway group advertises exactly the registered subcommands."""
    from typing import cast

    import click
    from typer.main import get_command

    from wmo.cli.gateway.app import gateway_app

    root_group = cast(click.Group, get_command(app))
    config_group = cast(click.Group, root_group.commands["config"])
    gateway_group = cast(click.Group, config_group.commands["gateway"])
    deferred_names = set(gateway_group.list_commands(cast(click.Context, None)))
    real_names = set(cast(click.Group, get_command(gateway_app)).commands)

    assert deferred_names == real_names
    assert {"pool", "budget"}.issubset(deferred_names)
