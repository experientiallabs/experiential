"""Tests for the public package API surface."""

from __future__ import annotations

import subprocess
import sys
from inspect import signature

import wmo
import wmo.cli.router_app as router_cli
from wmo.optimize.router import automatic_router
from wmo.optimize.router.activation import load_project_router, load_router
from wmo.optimize.router.composition import compose_router
from wmo.optimize.router.workflow import fit_router, optimize_router, report_router
from wmo.runtime.router.application import (
    create_project_router_app,
)
from wmo.runtime.router.runtime import RouterRuntime
from wmo.simulation.build import build_project
from wmo.simulation.world_model.application import (
    WorldModel,
    WorldModelLoadError,
    WorldModelObservation,
    WorldModelSession,
    WorldModelSessionError,
    WorldModelSessionLimits,
    load_world_model,
)


def test_public_api_matches_quickstart() -> None:
    """The quickstart uses only deliberate package-root services.

    Every supported build, optimization, router, and world-model entrypoint resolves to its owning
    implementation while unsupported conveniences remain absent.
    """
    assert wmo.build_project is build_project
    assert wmo.optimize_router is optimize_router
    assert wmo.fit_router is fit_router
    assert wmo.report_router is report_router
    assert wmo.RouterRuntime is RouterRuntime
    assert wmo.compose_router is compose_router
    assert automatic_router.compose_router is compose_router
    assert router_cli.optimize_project_router is automatic_router.optimize_project_router
    assert wmo.load_project_router is load_project_router
    assert wmo.load_router is load_router
    assert "ghost" in signature(wmo.load_router).parameters
    assert wmo.create_project_router_app is create_project_router_app
    assert wmo.load_world_model is load_world_model
    assert wmo.WorldModel is WorldModel
    assert wmo.WorldModelLoadError is WorldModelLoadError
    assert wmo.WorldModelSession is WorldModelSession
    assert wmo.WorldModelSessionError is WorldModelSessionError
    assert wmo.WorldModelSessionLimits is WorldModelSessionLimits
    assert wmo.WorldModelObservation is WorldModelObservation
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
