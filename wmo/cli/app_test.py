"""Tests for the locked command surface exposed by the root Typer application."""

from __future__ import annotations

from typer import Context
from typer.core import TyperGroup
from typer.main import get_group

from wmo.cli.app import app

EXPECTED_SUBCOMMANDS = {
    "config": {"providers", "telemetry"},
    "optimize": {"model", "router"},
}


def test_root_cli_and_subgroups_are_exact() -> None:
    """The public CLI remains build, config, optimize, and run with two exact subgroups."""
    root = get_group(app)
    root_context = Context(root)
    assert set(root.list_commands(root_context)) == {"build", "config", "optimize", "run"}

    for name, expected in EXPECTED_SUBCOMMANDS.items():
        command = root.get_command(root_context, name)
        assert isinstance(command, TyperGroup)
        context = Context(command, parent=root_context, info_name=name)
        assert set(command.list_commands(context)) == expected
