"""Tests for `wmo optimize`: the three product optimizers behind one switch, still deferred."""

from __future__ import annotations

from typer.main import get_group
from typer.testing import CliRunner

from wmo.cli.defer import DeferredTyperGroup
from wmo.cli.optimize_app import optimize_app

runner = CliRunner()


def test_the_switch_offers_exactly_route_model_and_distill() -> None:
    result = runner.invoke(optimize_app, ["--help"])

    assert result.exit_code == 0, result.output
    for name in ("route", "model", "distill"):
        assert name in result.output
    # Harness search moved to the private agent-optimization repo; it must not reappear as a
    # command that resolves to nothing.
    assert "harness" not in result.output.split("Commands")[-1]


def test_route_and_distill_are_deferred_groups_that_still_name_their_children() -> None:
    # `wmo optimize --help` must not pay for the route and distill import chains, so each group
    # answers with its declared child names rather than loading the real app; the mechanism is
    # covered in defer_test.py.
    group = get_group(optimize_app)

    for name, child_name in (("route", "sweep"), ("distill", "run")):
        child = group.get_command(None, name)  # ty: ignore[invalid-argument-type]
        assert isinstance(child, DeferredTyperGroup), f"{name} is not a deferred group"
        assert child_name in child.list_commands(None)  # ty: ignore[invalid-argument-type]


def test_route_and_distill_resolve_to_their_real_apps() -> None:
    # The deferred registration is only correct if the module/attr pair actually loads; a typo
    # would otherwise surface the first time a user ran the command.
    route_help = runner.invoke(optimize_app, ["route", "--help"])
    distill_help = runner.invoke(optimize_app, ["distill", "--help"])

    assert route_help.exit_code == 0, route_help.output
    assert distill_help.exit_code == 0, distill_help.output
    for name in ("sweep", "fit", "tune", "report", "pin", "student"):
        assert name in route_help.output
    for name in ("run", "probe", "report"):
        assert name in distill_help.output


def test_model_is_registered_directly_and_keeps_its_full_signature() -> None:
    # `optimize model` is not deferred (its module is light at import time), which is what lets
    # its help show real options instead of a forwarding stub's.
    result = runner.invoke(optimize_app, ["model", "--help"])

    assert result.exit_code == 0, result.output
    assert "--force-from" in result.output
    assert "--max-usd" in result.output


def test_bare_optimize_prints_help() -> None:
    result = runner.invoke(optimize_app, [])

    assert result.exit_code == 2  # no_args_is_help: help plus a usage-error exit
    assert "route" in result.output
