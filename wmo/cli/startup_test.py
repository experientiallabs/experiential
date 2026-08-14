"""Startup / import-isolation tests for the light CLI shell.

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


def test_runtime_rag_refresh_public_imports_resolve_without_eager_router_import() -> None:
    """Resolve runtime refresh services without loading router application code eagerly."""
    _run(
        """
import sys
import wmo.simulation.retrieval as retrieval
names = {"RuntimeRAGRefresh", "RuntimeTraceStitchingError", "refresh_runtime_trace_rag"}
assert names.issubset(dir(retrieval))
assert "wmo.runtime.router" not in sys.modules
try:
    getattr(retrieval, "unknown_retrieval_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown retrieval export resolved")
from wmo.simulation.retrieval import RuntimeRAGRefresh, refresh_runtime_trace_rag
from wmo.simulation.retrieval.refresh import RuntimeRAGRefresh as nested_refresh_type
from wmo.simulation.retrieval.refresh import refresh_runtime_trace_rag as nested_refresh
assert RuntimeRAGRefresh is nested_refresh_type
assert refresh_runtime_trace_rag is nested_refresh
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


def test_router_server_public_imports_resolve_after_lazy_package_import() -> None:
    """Resolve public router application and endpoint services through lazy exports."""
    _run(
        """
import wmo.runtime.router as router
assert {"create_router_endpoint", "load_router"}.issubset(dir(router))
try:
    getattr(router, "p17_unknown_router_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown router export resolved")
from wmo.runtime.router import create_router_endpoint, load_router
from wmo.runtime.router.application import load_router as nested_load_router
from wmo.runtime.router.endpoint import create_router_endpoint as nested_endpoint
assert create_router_endpoint.__module__ == "wmo.runtime.router.endpoint"
assert load_router.__module__ == "wmo.runtime.router.application"
assert create_router_endpoint is nested_endpoint
assert load_router is nested_load_router
"""
    )


def test_runtime_rag_refresh_public_imports_resolve_after_lazy_package_import() -> None:
    """Resolve public runtime refresh services through lazy retrieval exports."""
    _run(
        """
import sys
import wmo.simulation.retrieval as retrieval
names = {"RuntimeRAGRefresh", "RuntimeTraceStitchingError", "refresh_runtime_trace_rag"}
assert names.issubset(dir(retrieval))
assert "wmo.runtime.router" not in sys.modules
try:
    getattr(retrieval, "p17_unknown_retrieval_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown retrieval export resolved")
from wmo.simulation.retrieval import RuntimeRAGRefresh, refresh_runtime_trace_rag
from wmo.simulation.retrieval.refresh import RuntimeRAGRefresh as nested_refresh_type
from wmo.simulation.retrieval.refresh import refresh_runtime_trace_rag as nested_refresh
assert RuntimeRAGRefresh is nested_refresh_type
assert refresh_runtime_trace_rag is nested_refresh
"""
    )
