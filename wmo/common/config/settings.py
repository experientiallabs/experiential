"""Project-local settings stored under the selected harness root."""

from __future__ import annotations

import tomllib
import uuid
from pathlib import Path

import tomli_w
from pydantic import BaseModel, Field, ValidationError

from wmo.common.config.paths import ARTIFACT_DIR
from wmo.common.core.files import write_text_atomic
from wmo.common.core.locks import file_write_lock

SETTINGS_FILENAME = "settings.toml"


class TelemetrySettings(BaseModel):
    """Usage telemetry preferences for this harness project."""

    enabled: bool = True
    anonymous_id: str | None = Field(
        default=None,
        min_length=32,
        max_length=32,
        pattern=r"^[0-9a-f]{32}$",
    )


class ProjectSettings(BaseModel):
    """Anonymous product-telemetry settings local to one project root."""

    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)


def settings_path(root: str | Path = ARTIFACT_DIR) -> Path:
    """Return the project-local telemetry settings path."""
    return Path(root) / SETTINGS_FILENAME


_SETTINGS_REPAIR = (
    "fix the file, or delete it and rerun `wmo config telemetry status` to use defaults"
)
"""How to recover a settings file this loader refuses. Every raise below names it.

Telemetry commands read the file before changing it, so a malformed file must be repaired or
removed before the local default can be used.
"""


def load_settings(root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """Read `<root>/settings.toml`, defaulting when it does not exist.

    Raises:
        ValueError: The file exists but cannot be read, is not valid TOML, or does not match the
            current settings schema. The message names the path and the repair.
    """
    path = settings_path(root)
    if not path.exists():
        return ProjectSettings()
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        # A TOML document is UTF-8 by definition, so bytes that do not decode are the same
        # "not valid TOML" answer; `tomllib.load` decodes before it parses, and the raw
        # UnicodeDecodeError names neither the file nor the way out.
        raise ValueError(f"{path} is not valid TOML ({exc}); {_SETTINGS_REPAIR}") from exc
    except OSError as exc:
        raise ValueError(f"{path} could not be read ({exc}); {_SETTINGS_REPAIR}") from exc
    try:
        return ProjectSettings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"{path} does not match the current settings schema ({exc}); {_SETTINGS_REPAIR}"
        ) from exc


def save_settings(settings: ProjectSettings, root: str | Path = ARTIFACT_DIR) -> None:
    """Atomically persist the project-local telemetry settings.

    Args:
        settings: Validated settings to persist.
        root: Directory that owns the settings file.
    """
    path = settings_path(root)
    data = settings.model_dump(mode="json", exclude_none=True)
    write_text_atomic(path, tomli_w.dumps(data))


def set_telemetry_enabled(enabled: bool, root: str | Path = ARTIFACT_DIR) -> ProjectSettings:
    """Persist an explicit product-telemetry preference.

    Args:
        enabled: Whether product telemetry is allowed.
        root: Directory that owns the settings file.

    Returns:
        The updated settings.
    """
    path = settings_path(root)
    with file_write_lock(path, what="product telemetry settings"):
        settings = load_settings(root)
        settings.telemetry.enabled = enabled
        save_settings(settings, root)
    return settings


def ensure_telemetry_anonymous_id(root: str | Path = ARTIFACT_DIR) -> str:
    """Return the stable local anonymous ID, creating it only when needed.

    Args:
        root: Directory that owns the settings file.

    Returns:
        The stable anonymous identifier.
    """
    path = settings_path(root)
    with file_write_lock(path, what="product telemetry identity"):
        settings = load_settings(root)
        if settings.telemetry.anonymous_id is None:
            settings.telemetry.anonymous_id = uuid.uuid4().hex
            save_settings(settings, root)
    assert settings.telemetry.anonymous_id is not None
    return settings.telemetry.anonymous_id
