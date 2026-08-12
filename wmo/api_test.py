"""Tests for the public package API surface."""

from __future__ import annotations

import subprocess
import sys

import wmo
from wmo.common.core.types import ActionKind


def test_public_api_matches_quickstart() -> None:
    # README/docstring quickstart imports ActionKind from the package root.
    assert "ActionKind" in wmo.__all__
    assert wmo.ActionKind is ActionKind


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
