"""Tests for the public package API surface."""

from __future__ import annotations

import subprocess
import sys

import wmo
from wmo.optimize.router.workflow import fit_router, optimize_router, report_router
from wmo.runtime.router.runtime import RouterRuntime
from wmo.simulation.build import build_project
from wmo.workflow.router import compose_router


def test_public_api_matches_quickstart() -> None:
    """The customer workflow uses only deliberate package-root service exports."""
    assert wmo.build_project is build_project
    assert wmo.optimize_router is optimize_router
    assert wmo.fit_router is fit_router
    assert wmo.report_router is report_router
    assert wmo.RouterRuntime is RouterRuntime
    assert wmo.compose_router is compose_router
    assert "ActionKind" not in wmo.__all__


def test_runtime_router_import_isolated_from_simulation_and_offline_optimizer() -> None:
    """A fresh runtime import must not initialize simulation or optimizer owners."""
    code = """
import sys
import wmo.runtime.router

offline = sorted(name for name in sys.modules if name.startswith("wmo.optimize"))
gepa = sorted(name for name in sys.modules if name == "gepa" or name.startswith("gepa."))
simulation_model = sorted(
    name for name in sys.modules if name.startswith("wmo.simulation.model")
)
assert not offline, offline
assert not gepa, gepa
assert not simulation_model, simulation_model
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
