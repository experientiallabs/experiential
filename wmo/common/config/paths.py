"""Shared user-level paths that do not belong to a product domain."""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "WMO_HOME"


def wmo_home() -> Path:
    """Return the user-level WMO directory from ``$WMO_HOME`` or ``~/.wmo``."""
    override = os.environ.get(ENV_HOME)
    return Path(override) if override else Path.home() / ".wmo"
