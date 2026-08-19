"""Startup / import-isolation tests for the light CLI shell.

Each probe runs in a fresh subprocess so pytest collection of other CLI tests cannot pollute
``sys.modules``. Light paths must stay free of FastAPI, sklearn, scipy, uvicorn, and the heavy
serving / SFT / route bodies.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> None:
    """Run one isolated Python startup probe."""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_cli_import_stays_free_of_heavy_third_parties() -> None:
    """Importing the CLI shell does not load heavy serving dependencies."""
    _run(
        """
import sys
import wmo.cli
banned = ("fastapi", "sklearn", "scipy", "uvicorn")
roots = {k.split(".")[0] for k in sys.modules}
bad = sorted(b for b in banned if b in roots)
assert not bad, f"heavy packages loaded on import wmo.cli: {bad}"
"""
    )


def test_cli_help_stays_free_of_heavy_product_modules() -> None:
    """Rendering root help does not initialize heavy product modules."""
    _run(
        """
import sys
from typer.testing import CliRunner
from wmo.cli.app import app

result = CliRunner().invoke(app, ["--help"])
assert result.exit_code == 0, result.output
banned = (
    "fastapi",
    "sklearn",
    "scipy",
    "uvicorn",
    "wmo.simulation.serving.server",
    "wmo.optimize.model",
    "wmo.cli.route_app",
)
roots = {k.split(".")[0] for k in sys.modules}
bad = []
for name in banned:
    if name in sys.modules:
        bad.append(name)
    elif "." not in name and name in roots:
        bad.append(name)
assert not bad, f"heavy modules loaded for wmo --help: {bad}"
"""
    )


def test_config_does_not_load_router_runtime_or_optimizer() -> None:
    """Telemetry configuration stays independent of router product owners."""
    _run(
        """
import sys
from typer.testing import CliRunner
from wmo.cli.app import app

runner = CliRunner()
assert runner.invoke(app, ["config", "telemetry", "status"]).exit_code == 0
banned = (
    "wmo.runtime.router",
    "wmo.optimize.router",
    "fastapi",
    "sklearn",
    "uvicorn",
)
roots = {k.split(".")[0] for k in sys.modules}
bad = []
for name in banned:
    if name in sys.modules:
        bad.append(name)
    elif "." not in name and name in roots:
        bad.append(name)
assert not bad, f"light commands pulled: {bad}"
"""
    )


def test_world_model_public_import_resolves_after_lazy_package_import() -> None:
    """Resolve the public world-model loader through the lazy package export."""
    _run(
        """
from wmo.simulation.world_model import load_world_model
assert callable(load_world_model)
assert load_world_model.__module__ == "wmo.simulation.world_model.application"
"""
    )


def test_router_selection_and_optimizer_activation_imports_resolve_lazily() -> None:
    """Expose selection activation without restoring a router HTTP server."""
    _run(
        """
import wmo.runtime.router as router
import wmo.optimize.router as optimizer
assert not {"create_router_endpoint", "create_project_router_app"}.intersection(dir(router))
assert {"load_project_router", "load_router"}.issubset(dir(optimizer))
try:
    getattr(router, "p17_unknown_router_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown router export resolved")
from wmo.optimize.router import load_router
from wmo.optimize.router.activation import load_router as nested_load_router
assert load_router.__module__ == "wmo.optimize.router.activation"
assert load_router is nested_load_router
"""
    )
