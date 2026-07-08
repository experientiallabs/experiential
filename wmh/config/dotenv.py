"""Minimal `.env` support: loaded on CLI startup, written by the wizard's credential prompts.

No third-party dotenv dependency — the harness only needs KEY=VALUE lines. Values entered in
the build wizard are persisted here so the next `wmh` invocation has them without re-prompting.
"""

from __future__ import annotations

import os
import tempfile
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
    """Set `var` in os.environ and persist it to `path`, replacing any existing line for it.

    Raises ValueError if `path` is a symlink: a credential rewrite must never end up in
    whatever file the link happens to point at.
    """
    env_path = Path(path)
    if env_path.is_symlink():
        raise ValueError(
            f"refusing to write credentials through the symlink {env_path}; "
            f"set {var} in the link target or your shell instead"
        )
    os.environ[var] = value
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    rendered = f"{var}={value}"
    for i, line in enumerate(lines):
        if line.partition("=")[0].strip() == var:
            lines[i] = rendered
            break
    else:
        lines.append(rendered)
    # Write-then-rename: mkstemp creates the temp file 0600 (owner-only, no umask window) and
    # cannot hit a planted symlink; os.replace swaps the path atomically WITHOUT following a
    # link that appeared after the check above, so the secret can never land in a linked-to
    # file. This needs no platform-dependent open flags.
    fd, tmp_name = tempfile.mkstemp(dir=env_path.parent, prefix=f"{env_path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp_name, env_path)
    except BaseException:
        os.unlink(tmp_name)
        raise
