"""Local FastAPI backend — the live environment agents call over HTTP."""

from wmo.simulation.serving.server import create_app

__all__ = ["create_app"]
