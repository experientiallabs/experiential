"""Platform login credentials, stored once per user.

Unlike everything else in wmh (project-local under `./.wmh/`), the platform credential is
user-global: `~/.wmh/credentials.toml`, directory overridable via `$WMH_HOME`. Environment
variables override the file so CI and headless runs never need one written to disk.
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel

ENV_HOME = "WMH_HOME"
ENV_WEB_URL = "WMH_PLATFORM_URL"
ENV_API_URL = "WMH_PLATFORM_API_URL"
ENV_TOKEN = "WMH_PLATFORM_TOKEN"
ENV_ORG = "WMH_PLATFORM_ORG"

CREDENTIALS_FILENAME = "credentials.toml"

# The hosted platform a bare `wmh login` connects to; `--url` (previews,
# self-hosted, the local stack) and saved credentials both take precedence.
DEFAULT_WEB_URL = "https://platform.experientiallabs.ai"


class PlatformCredentials(BaseModel):
    """The saved connection: where the platform lives and which key acts for us."""

    web_url: str | None = None  # the browser-facing app (login page, keys page)
    api_url: str | None = None  # the backend host requests go to
    token: str | None = None  # org API key (xpl_…)
    default_org: str | None = None  # organization id used when --org is omitted

    def is_complete(self) -> bool:
        """Whether requests can be made without further configuration."""
        return bool(self.api_url and self.token)


def wmh_home() -> Path:
    """The user-global wmh directory (`$WMH_HOME` or `~/.wmh`)."""
    override = os.environ.get(ENV_HOME)
    return Path(override) if override else Path.home() / ".wmh"


def credentials_path() -> Path:
    """Where the credential file lives."""
    return wmh_home() / CREDENTIALS_FILENAME


def load_credentials() -> PlatformCredentials:
    """Read the credential file, then apply environment overrides.

    Env alone is sufficient (no file needed); a set-but-empty env var is
    treated as unset rather than clearing a file value.
    """
    data: dict[str, str] = {}
    path = credentials_path()
    if path.exists():
        section = tomllib.loads(path.read_text(encoding="utf-8")).get("platform", {})
        data = {key: value for key, value in section.items() if isinstance(value, str)}
        # Files written before the platform's org-only change carry the old
        # default_project key. A project id is NOT an org id, so carrying the
        # value over would send a guaranteed-miss id to /api/orgs/{org_id}/...;
        # discard it instead, so the user gets the clear "pass --org" prompt
        # (or the sole-org auto-pick at the next login) rather than a
        # confusing org-not-found. The stale key drops on the next save.
        data.pop("default_project", None)
    credentials = PlatformCredentials.model_validate(data)
    overrides = {
        "web_url": os.environ.get(ENV_WEB_URL),
        "api_url": os.environ.get(ENV_API_URL),
        "token": os.environ.get(ENV_TOKEN),
        "default_org": os.environ.get(ENV_ORG),
    }
    updates = {key: value for key, value in overrides.items() if value}
    return credentials.model_copy(update=updates) if updates else credentials


def save_credentials(credentials: PlatformCredentials) -> Path:
    """Persist the credential file with owner-only permissions.

    Mirrors the dotenv writer: refuses symlinks (a credential rewrite must never
    land in whatever a link points at), writes through a 0600 mkstemp, and swaps
    into place atomically.

    Raises:
        ValueError: If the target path is a symlink.
    """
    path = credentials_path()
    if path.is_symlink():
        msg = (
            f"refusing to write credentials through the symlink {path}; "
            "remove the link or point $WMH_HOME elsewhere"
        )
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "platform": {
            key: value
            for key, value in credentials.model_dump(mode="json").items()
            if value is not None
        }
    }
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tomli_w.dumps(payload))
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise
    return path


def clear_credentials() -> bool:
    """Delete the credential file; returns whether one existed."""
    path = credentials_path()
    if not path.exists():
        return False
    path.unlink()
    return True
