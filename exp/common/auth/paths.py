"""Platform user-data paths for the persistent provider credential file.

The file lives outside repositories. Linux and other XDG hosts use
``$XDG_DATA_HOME/exp/auth.json`` or ``~/.local/share/exp/auth.json``, matching the
OpenCode data-directory rule. macOS and Windows use their native application-data
directories when ``XDG_DATA_HOME`` is unset.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

AUTH_FILE_NAME = "auth.json"
_APP_DIR_NAME = "exp"


def provider_data_dir(
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> Path:
    """Return the platform user-data directory that owns ``auth.json``.

    Args:
        environment: Optional mapping used instead of the process environment.
        home: Optional home directory used instead of ``Path.home()``.
        platform: Optional ``sys.platform`` value used by deterministic tests.

    Returns:
        Directory that should contain ``auth.json``.
    """
    values = os.environ if environment is None else environment
    xdg = values.get("XDG_DATA_HOME", "").strip()
    if xdg:
        return Path(xdg) / _APP_DIR_NAME
    host = sys.platform if platform is None else platform
    base = Path.home() if home is None else home
    if host == "darwin":
        return base / "Library" / "Application Support" / _APP_DIR_NAME
    if host == "win32":
        roaming = values.get("APPDATA", "").strip()
        local = values.get("LOCALAPPDATA", "").strip()
        root = Path(local or roaming) if (local or roaming) else base / "AppData" / "Roaming"
        return root / _APP_DIR_NAME
    return base / ".local" / "share" / _APP_DIR_NAME


def default_auth_path(
    *,
    environment: Mapping[str, str] | None = None,
    home: Path | None = None,
    platform: str | None = None,
) -> Path:
    """Return the default user-data path of the provider credential file.

    Args:
        environment: Optional mapping used instead of the process environment.
        home: Optional home directory used instead of ``Path.home()``.
        platform: Optional ``sys.platform`` value used by deterministic tests.

    Returns:
        Absolute or home-relative path of ``auth.json``.
    """
    return provider_data_dir(environment=environment, home=home, platform=platform) / AUTH_FILE_NAME
