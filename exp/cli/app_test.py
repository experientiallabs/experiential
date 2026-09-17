"""Tests for the supported command surface exposed by the root Typer application."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer import Context
from typer.core import TyperGroup
from typer.main import get_group

from exp.cli.app import app, main

EXPECTED_SUBCOMMANDS = {
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
    assert set(root.list_commands(root_context)) == {"build", "config", "login", "optimize", "run"}

    for name, expected in EXPECTED_SUBCOMMANDS.items():
        command = root.get_command(root_context, name)
        assert isinstance(command, TyperGroup)
        context = Context(command, parent=root_context, info_name=name)
        assert set(command.list_commands(context)) == expected


def test_main_reports_an_invalid_dotenv_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI startup should frame an invalid environment file without dispatch or traceback."""
    (tmp_path / ".env").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as captured:
        main()

    assert captured.value.code == 2
    output = capsys.readouterr()
    assert "Invalid value for .env" in output.err
    assert "regular UTF-8 text file: .env" in output.err
    assert "remove it" in output.err
    assert "Traceback" not in output.err
