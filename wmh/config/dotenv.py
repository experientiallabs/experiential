"""Minimal `.env` support: loaded on CLI startup, written by the wizard's credential prompts.

No third-party dotenv dependency — the harness only needs KEY=VALUE lines. Values entered in
the build wizard are persisted here so the next `wmh` invocation has them without re-prompting.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

ENV_FILE = ".env"


def load_env_file(path: str | Path = ENV_FILE) -> None:
    """Read KEY=VALUE lines from `path` into os.environ without overriding already-set vars."""
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


def upsert_env_var(var: str, value: str, path: str | Path = ENV_FILE) -> None:
    """Set `var` in os.environ and persist it to `path`, replacing any existing line for it."""
    os.environ[var] = value
    env_path = Path(path)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    rendered = f"{var}={value}"
    for i, line in enumerate(lines):
        if line.partition("=")[0].strip() == var:
            lines[i] = rendered
            break
    else:
        lines.append(rendered)
    # Credentials file: owner-only from birth (no umask window) and tightened before the
    # secret is written if the file pre-existed with a looser mode.
    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        fh.write("\n".join(lines) + "\n")
