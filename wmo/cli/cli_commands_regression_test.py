"""Root command inventory tests for the router customer slice."""

from __future__ import annotations

from typer.testing import CliRunner

from wmo.cli.app import app


def test_root_command_surface_is_locked() -> None:
    """Only build, config, optimize, and run are registered at the root."""
    command_names = {command.name for command in app.registered_commands}
    group_names = {group.name for group in app.registered_groups}

    assert command_names == {"build", "run"}
    assert group_names == {"config", "optimize"}


def test_config_requires_only_the_telemetry_subtree() -> None:
    """The retained config command is the thin local telemetry preference service."""
    result = CliRunner().invoke(app, ["config", "telemetry", "status"])

    assert result.exit_code == 0, result.output
    assert "telemetry enabled" in result.output
