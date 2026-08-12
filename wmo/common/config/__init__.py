"""Minimal environment, path, and product-telemetry configuration exports."""

from wmo.common.config.dotenv import load_env_file
from wmo.common.config.paths import ARTIFACT_DIR, ENV_HOME, wmo_home
from wmo.common.config.settings import (
    ProjectSettings,
    TelemetrySettings,
    ensure_telemetry_anonymous_id,
    load_settings,
    save_settings,
    set_telemetry_enabled,
    settings_path,
)

__all__ = [
    "ARTIFACT_DIR",
    "ENV_HOME",
    "ProjectSettings",
    "TelemetrySettings",
    "ensure_telemetry_anonymous_id",
    "load_env_file",
    "load_settings",
    "save_settings",
    "set_telemetry_enabled",
    "settings_path",
    "wmo_home",
]
