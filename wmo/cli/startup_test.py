"""Startup / import-isolation tests for the light CLI shell (#344).

Each probe runs in a fresh subprocess so pytest collection of other CLI tests cannot pollute
``sys.modules``. Light paths must stay free of FastAPI, sklearn, scipy, uvicorn, and the heavy
serving / distill / route bodies.
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
    "wmo.serving.server",
    "wmo.distill",
    "wmo.cli.route_app",
    "wmo.evals.open_loop",
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


def test_list_and_config_do_not_load_serve_or_distill() -> None:
    _run(
        """
import sys
from typer.testing import CliRunner
from wmo.cli.app import app

runner = CliRunner()
assert runner.invoke(app, ["list"]).exit_code == 0
assert runner.invoke(app, ["config", "telemetry", "status"]).exit_code == 0
banned = (
    "wmo.serving.server",
    "wmo.distill",
    "wmo.cli.route_app",
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


def test_serve_help_loads_uvicorn_path_when_handler_imports() -> None:
    """Invoking the serve command body must be able to import the serve stack.

    ``serve --help`` only needs the signature (still light). Importing the serve handler's
    dependencies on demand is checked by importing create_app the same way ``serve`` does.
    """
    _run(
        """
import sys
from wmo.serving.server import create_app
assert callable(create_app)
assert "fastapi" in {m.split(".")[0] for m in sys.modules}
"""
    )
