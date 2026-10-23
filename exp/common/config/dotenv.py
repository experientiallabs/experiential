"""Minimal read-only `.env` loading for CLI startup."""

from __future__ import annotations

import os
import stat
from pathlib import Path

ENV_FILE = ".env"
_INVALID_ENV_FILE = (
    "environment file must be a readable regular UTF-8 text file: {path}; "
    "repair or replace it, or remove it to continue without one"
)


def load_env_file(path: str | Path = ENV_FILE) -> None:
    """Load simple environment assignments without overriding process values.

    Args:
        path: File containing ``KEY=VALUE`` lines.

    Raises:
        ValueError: The existing path is not a readable regular UTF-8 text file.
    """
    env_path = Path(path)
    try:
        descriptor = os.open(env_path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(_INVALID_ENV_FILE.format(path=env_path)) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(_INVALID_ENV_FILE.format(path=env_path))
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            contents = handle.read()
    except (OSError, UnicodeError) as exc:
        raise ValueError(_INVALID_ENV_FILE.format(path=env_path)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    for raw_line in contents.splitlines():
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
