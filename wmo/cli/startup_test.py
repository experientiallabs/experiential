"""Startup / import-isolation tests for the light CLI shell (#344).

Each probe runs in a fresh subprocess so pytest collection of other CLI tests cannot pollute
``sys.modules``. Light paths must stay free of FastAPI, sklearn, scipy, uvicorn, and the heavy
serving / SFT / route bodies.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_cli_import_stays_free_of_heavy_third_parties() -> None:
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
    "wmo.simulation.evaluation.open_loop",
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
