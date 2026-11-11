"""Tests for the ``exp code`` declaration: arguments pass through untouched."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from typer.testing import CliRunner

from exp.cli import code as code_package
from exp.cli.app import app


def test_every_argument_reaches_the_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags exp does not know, including --help, go to exp-code, and its status is returned."""
    seen: list[list[str]] = []

    def launch_from_cli(arguments: Sequence[str]) -> int:
        """Record the arguments and exit with status 7."""
        seen.append(list(arguments))
        return 7

    monkeypatch.setattr("exp.cli.code.launcher.launch_from_cli", launch_from_cli)
    result = CliRunner().invoke(app, ["code", "--help", "-p", "hi", "--mode", "json"])
    assert result.exit_code == 7
    assert seen == [["--help", "-p", "hi", "--mode", "json"]]
    assert code_package.__doc__


def test_root_help_lists_code() -> None:
    """`exp --help` lists the command with its one-line description."""
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "code" in result.output
    assert "exp-code" in result.output
