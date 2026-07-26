"""Local FastAPI backend — the live environment agents call over HTTP."""

from wmo.serving.server import create_app

__all__ = ["create_app"]
