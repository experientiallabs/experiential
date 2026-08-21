"""Tests for the locked command surface exposed by the root Typer application."""

from __future__ import annotations

from importlib import import_module

import pytest
from typer import Context
from typer.core import TyperGroup
from typer.main import get_group

from exp.cli.app import app

app_module = import_module("exp.cli.app")

EXPECTED_SUBCOMMANDS = {
    "config": {"budget", "gateway", "judge", "providers", "telemetry"},
    "optimize": {"model", "router"},
}


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], ["run"]),
        (["--help"], ["--help"]),
        (["help"], ["--help"]),
        (["--install-completion"], ["--install-completion"]),
        (["--show-completion"], ["--show-completion"]),
        (["build", "--help"], ["build", "--help"]),
        (["config", "gateway", "status"], ["config", "gateway", "status"]),
        (["run", "project-a"], ["run", "project-a"]),
        (["project-a", "--check"], ["run", "project-a", "--check"]),
        (["--check", "--json"], ["run", "--check", "--json"]),
    ],
)
def test_root_dispatch_preserves_help_and_existing_commands(
    arguments: list[str],
    expected: list[str],
) -> None:
    """Route bare gateway forms without changing root help or explicit subcommands."""
    assert app_module._dispatch_arguments(arguments) == expected


def test_main_routes_bare_invocation_through_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed entrypoint turns a bare invocation into the existing run command."""
    captured: list[list[str]] = []

    def capture_app(*, args: list[str]) -> None:
        """Capture parsed arguments passed to the root application."""
        captured.append(args)

    monkeypatch.setattr(app_module, "app", capture_app)
    monkeypatch.setattr(app_module, "load_env_file", lambda: None)
    monkeypatch.setattr(app_module.sys, "argv", ["exp"])

    app_module.main()

    assert captured == [["run"]]


def test_root_cli_and_subgroups_are_exact() -> None:
    """Prove the root command and nested subgroup surfaces remain exact.

    The test enumerates the public root commands and then verifies each configured subgroup exposes
    only its approved child commands.
    """
    root = get_group(app)
    root_context = Context(root)
    assert set(root.list_commands(root_context)) == {"build", "config", "optimize", "run"}

    for name, expected in EXPECTED_SUBCOMMANDS.items():
        command = root.get_command(root_context, name)
        assert isinstance(command, TyperGroup)
        context = Context(command, parent=root_context, info_name=name)
        assert set(command.list_commands(context)) == expected
