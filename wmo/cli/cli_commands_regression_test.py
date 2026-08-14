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


def test_config_exposes_telemetry_provider_and_manual_judge_setup() -> None:
    """Config contains local telemetry, model-provider, and manual judge state."""
    result = CliRunner().invoke(app, ["config", "--help"])

    assert result.exit_code == 0, result.output
    assert "telemetry" in result.output
    assert "providers" in result.output
    assert "judge" in result.output
