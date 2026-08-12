"""Tests for deferred Typer sub-app loading: the root CLI must stay light until it must not.

The lazy group is what keeps `wmo --help` off the optimize/serving import chains (the whole-CLI
budget is checked by the startup section of `app_test.py`). Here the mechanism is exercised
directly, against a sub-app registered under a module name that does not exist: any accidental
import fails loudly instead of silently costing a second.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import pytest
import typer
from typer.testing import CliRunner

from wmo.cli.defer import DeferredTyperGroup, add_deferred_typer

if TYPE_CHECKING:
    from collections.abc import Iterator

_MISSING_MODULE = "wmo.cli.__no_such_deferred_module__"
_LOADED_MODULE = "wmo_deferred_test_target"

runner = CliRunner()


@pytest.fixture
def target_module() -> Iterator[list[str]]:
    """Install an importable module holding a real Typer app; yields its invocation log."""
    calls: list[str] = []
    module = types.ModuleType(_LOADED_MODULE)
    sub_app = typer.Typer(help="the real thing", no_args_is_help=True)

    @sub_app.command("fit")
    def _fit() -> None:
        """Fit the thing."""
        calls.append("fit")

    @sub_app.command("tune")
    def _tune() -> None:
        """Tune the thing."""
        calls.append("tune")

    module.sub_app = sub_app  # ty: ignore[unresolved-attribute]
    sys.modules[_LOADED_MODULE] = module
    try:
        yield calls
    finally:
        del sys.modules[_LOADED_MODULE]


def _parent(module: str, known_names: tuple[str, ...] = ("fit", "tune")) -> typer.Typer:
    parent = typer.Typer(no_args_is_help=True)

    @parent.command("local")
    def _local() -> None:
        """A command the parent owns outright."""

    add_deferred_typer(
        parent,
        name="deferred",
        module=module,
        attr="sub_app",
        help="Deferred group help.",
        known_names=known_names,
    )
    return parent


def test_parent_help_lists_the_group_without_importing_it() -> None:
    result = runner.invoke(_parent(_MISSING_MODULE), ["--help"])

    assert result.exit_code == 0, result.output
    assert "deferred" in result.output
    assert "Deferred group help." in result.output
    assert _MISSING_MODULE not in sys.modules


def test_known_names_answer_list_commands_with_no_import() -> None:
    group = DeferredTyperGroup(
        name="deferred",
        import_path=_MISSING_MODULE,
        attr="sub_app",
        known_names=("fit", "tune"),
    )

    assert group.list_commands(None) == ["fit", "tune"]  # ty: ignore[invalid-argument-type]


def test_resolving_a_child_imports_the_real_app_and_runs_it(target_module: list[str]) -> None:
    result = runner.invoke(_parent(_LOADED_MODULE), ["deferred", "fit"])

    assert result.exit_code == 0, result.output
    assert target_module == ["fit"]


def test_the_real_app_is_imported_once_and_reused(
    target_module: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    imports: list[str] = []
    real_import = importlib.import_module

    def _counting_import(name: str, package: str | None = None) -> types.ModuleType:
        imports.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _counting_import)
    group = DeferredTyperGroup(
        name="deferred", import_path=_LOADED_MODULE, attr="sub_app", known_names=("fit",)
    )

    assert group.get_command(None, "fit") is not None  # ty: ignore[invalid-argument-type]
    assert group.get_command(None, "tune") is not None  # ty: ignore[invalid-argument-type]

    assert imports.count(_LOADED_MODULE) == 1


def test_after_loading_list_commands_comes_from_the_real_app(target_module: list[str]) -> None:
    # `known_names` is a cheap stand-in, not a second source of truth: once the app is loaded the
    # real command list wins, so a stale hint cannot hide a command that exists.
    group = DeferredTyperGroup(
        name="deferred", import_path=_LOADED_MODULE, attr="sub_app", known_names=("fit",)
    )
    group.get_command(None, "fit")  # ty: ignore[invalid-argument-type]

    assert group.list_commands(None) == ["fit", "tune"]  # ty: ignore[invalid-argument-type]


def test_without_known_names_list_commands_falls_back_to_the_real_app(
    target_module: list[str],
) -> None:
    group = DeferredTyperGroup(
        name="deferred", import_path=_LOADED_MODULE, attr="sub_app", known_names=()
    )

    assert group.list_commands(None) == ["fit", "tune"]  # ty: ignore[invalid-argument-type]


def test_an_unknown_child_is_a_usage_error_not_a_crash(target_module: list[str]) -> None:
    result = runner.invoke(_parent(_LOADED_MODULE), ["deferred", "nope"])

    assert result.exit_code == 2
    assert "No such command" in result.output


def test_the_group_with_no_args_prints_its_help(target_module: list[str]) -> None:
    result = runner.invoke(_parent(_LOADED_MODULE), ["deferred"])

    assert result.exit_code == 2  # no_args_is_help: help plus a usage-error exit
    assert "fit" in result.output
    assert "tune" in result.output


def test_a_sibling_command_still_runs_when_the_deferred_module_is_broken() -> None:
    # The whole point of deferring: a group nobody asked for cannot break the rest of the CLI.
    result = runner.invoke(_parent(_MISSING_MODULE), ["local"])

    assert result.exit_code == 0, result.output
