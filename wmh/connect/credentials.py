"""Connector credentials, stored once per user.

Like the platform credential (`wmh/platform/credentials.py`), connector tokens are user-global:
`~/.wmh/connectors.toml` (directory overridable via `$WMH_HOME`, exact file overridable via
`$WMH_CONNECTORS_PATH`), one TOML table per connector name. A `WMH_<NAME>_TOKEN` environment
variable injects a token-kind credential without a file, so CI and headless runs never write one
to disk; a few connectors additionally honor a service-native variable (`ENV_TOKEN_ALIASES`)
that deployments already carry.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

import tomli_w
from pydantic import ValidationError

from wmh.connect.types import ConnectError, ConnectorAuth
from wmh.core.types import JsonObject
from wmh.platform.credentials import wmh_home

ENV_CONNECTORS_PATH = "WMH_CONNECTORS_PATH"

CONNECTORS_FILENAME = "connectors.toml"


def connectors_path() -> Path:
    """Where the connector credential file lives (`$WMH_CONNECTORS_PATH` wins over `$WMH_HOME`)."""
    override = os.environ.get(ENV_CONNECTORS_PATH)
    return Path(override) if override else wmh_home() / CONNECTORS_FILENAME


def token_env_var(name: str) -> str:
    """The env var that injects a plain token for connector `name`: WMH_<NAME>_TOKEN."""
    return f"WMH_{name.upper().replace('-', '_')}_TOKEN"


# Service-native env vars accepted as token sources per connector, consulted after the generic
# WMH_<NAME>_TOKEN override. BRAVE_SEARCH_API_KEY is the key the grounding engine already uses
# (wmh/engine/grounding.py) and deployments already carry, so the brave connector honors it too.
ENV_TOKEN_ALIASES: dict[str, tuple[str, ...]] = {"brave": ("BRAVE_SEARCH_API_KEY",)}


def token_env_vars(name: str) -> list[str]:
    """Every env var that can inject a token for connector `name`, in precedence order."""
    return [token_env_var(name), *ENV_TOKEN_ALIASES.get(name, ())]


def resolve_env_token(name: str) -> tuple[str, str] | None:
    """The first set env token for connector `name` as (var, token), or None when none is set.

    Set-but-empty vars are treated as unset.
    """
    for var in token_env_vars(name):
        token = os.environ.get(var)
        if token:
            return var, token
    return None


def load_connector_auth(name: str) -> ConnectorAuth | None:
    """Load the stored credential for one connector, or None when not connected.

    A non-empty env token (`WMH_<NAME>_TOKEN`, then any `ENV_TOKEN_ALIASES` entry) takes
    precedence over the file and yields a token-kind auth (a set-but-empty var is treated as
    unset).
    """
    resolved = resolve_env_token(name)
    if resolved is not None:
        return ConnectorAuth(kind="token", access_token=resolved[1])
    section = _read_sections().get(name)
    if section is None:
        return None
    return _validate_section(name, section)


def save_connector_auth(name: str, auth: ConnectorAuth) -> Path:
    """Persist one connector's credential, keeping every other connector's table intact.

    Mirrors the platform credential writer: refuses symlinks, writes through a 0600 mkstemp,
    and swaps into place atomically.

    Raises:
        ValueError: If the target path is a symlink.
    """
    sections = _read_sections()
    sections[name] = {
        key: value for key, value in auth.model_dump(mode="json").items() if value is not None
    }
    return _write_sections(sections)


def delete_connector_auth(name: str) -> bool:
    """Remove one connector's credential; returns whether one existed.

    Deleting the last entry removes the file entirely.
    """
    sections = _read_sections()
    if name not in sections:
        return False
    del sections[name]
    if sections:
        _write_sections(sections)
    else:
        connectors_path().unlink()
    return True


def list_connected() -> dict[str, ConnectorAuth]:
    """Every connector with a stored credential, by name (sorted).

    Only reads the file: connections injected purely via `WMH_<NAME>_TOKEN` env vars are not
    enumerable and do not appear here (per-name lookups still see them via
    `load_connector_auth`).
    """
    return {
        name: _validate_section(name, section) for name, section in sorted(_read_sections().items())
    }


def _read_sections() -> dict[str, JsonObject]:
    """All connector tables in the credential file (empty when the file is absent)."""
    path = connectors_path()
    if not path.exists():
        return {}
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return {name: section for name, section in document.items() if isinstance(section, dict)}


def _validate_section(name: str, section: JsonObject) -> ConnectorAuth:
    """Parse one connector table, turning pydantic errors into an actionable ConnectError."""
    try:
        return ConnectorAuth.model_validate(section)
    except ValidationError as exc:
        raise ConnectError(
            f"invalid [{name}] entry in {connectors_path()}: {exc}; "
            f"the stored credential for {name} is malformed; provide a valid token or "
            "reauthorize the connection"
        ) from exc


def _write_sections(sections: dict[str, JsonObject]) -> Path:
    """Atomically rewrite the credential file with owner-only permissions."""
    path = connectors_path()
    if path.is_symlink():
        msg = (
            f"refusing to write connector credentials through the symlink {path}; "
            f"remove the link or point ${ENV_CONNECTORS_PATH} elsewhere"
        )
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates the file 0600: that IS the owner-only mechanism.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tomli_w.dumps(sections))
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise
    return path
