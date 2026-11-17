"""Tests for the supported command surface exposed by the root Typer application."""

from __future__ import annotations

import importlib

import pytest
from typer import Context
from typer.core import TyperGroup
from typer.main import get_group

from exp.cli.app import app
from exp.common.core.host_scope import HostScopeError

EXPECTED_SUBCOMMANDS = {
    "capture": set(),
    "config": {"budget", "gateway", "judge", "providers", "telemetry"},
    "optimize": {"model", "router"},
}


def test_root_cli_and_subgroups_are_exact() -> None:
    """Prove the root command and nested subgroup surfaces remain exact.

    The test enumerates the public root commands and then verifies each configured subgroup exposes
    only its approved child commands.
    """
    root = get_group(app)
    root_context = Context(root)
    assert set(root.list_commands(root_context)) == {
        "build",
        "capture",
        "code",
        "config",
        "eval",
        "login",
        "optimize",
        "run",
    }

    for name, expected in EXPECTED_SUBCOMMANDS.items():
        command = root.get_command(root_context, name)
        assert isinstance(command, TyperGroup)
        context = Context(command, parent=root_context, info_name=name)
        assert set(command.list_commands(context)) == expected


def test_a_project_owned_by_another_machine_exits_with_its_explanation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The single-host refusal is an operator message and exit status, never a traceback."""

    def refuse() -> None:
        """Stand in for a command whose first coordinated write is refused."""
        raise HostScopeError("coordinated by another machine; run on one machine")

    root = importlib.import_module("exp.cli.app")  # the package re-exports the Typer as `app`
    monkeypatch.setattr(root, "app", refuse)
    monkeypatch.setattr(root, "load_env_file", lambda: None)
    with pytest.raises(SystemExit) as exited:
        root.main()
    assert exited.value.code == 1
    assert capsys.readouterr().err == "Error: coordinated by another machine; run on one machine\n"
