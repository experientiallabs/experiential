"""Minimal read-only `.env` loading for CLI startup."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = ".env"


def load_env_file(path: str | Path = ENV_FILE) -> None:
    """Load simple environment assignments without overriding process values.

    Args:
        path: File containing ``KEY=VALUE`` lines.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Strip only a MATCHED surrounding quote pair; a secret legitimately ending in a
        # quote character must survive the round-trip.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and value and key not in os.environ:
            os.environ[key] = value
