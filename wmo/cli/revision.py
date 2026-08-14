"""The local code revision recorded in immutable artifacts written by CLI commands."""

from __future__ import annotations

import subprocess
from pathlib import Path

_UNVERSIONED_REVISION = "local-unversioned"


def current_revision() -> str:
    """Return the local Git revision without changing repository or provider state.

    Returns:
        The exact ``HEAD`` commit of the checkout that owns this package, or
        ``local-unversioned`` when the revision cannot be read.
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else _UNVERSIONED_REVISION
