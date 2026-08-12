"""Minimal environment loading, artifact path, and product-telemetry settings exports."""

from wmo.common.config.dotenv import load_env_file
from wmo.common.config.paths import ARTIFACT_DIR
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
    "ProjectSettings",
    "TelemetrySettings",
    "ensure_telemetry_anonymous_id",
    "load_env_file",
    "load_settings",
    "save_settings",
    "set_telemetry_enabled",
    "settings_path",
]
