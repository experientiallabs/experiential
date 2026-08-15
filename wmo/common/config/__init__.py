"""Minimal environment loading, artifact path, and product-telemetry settings exports."""

from wmo.common.config.dotenv import load_env_file
from wmo.common.config.paths import ARTIFACT_DIR
from wmo.common.config.settings import (
    load_settings,
    set_telemetry_enabled,
    settings_path,
)

__all__ = [
    "ARTIFACT_DIR",
    "load_env_file",
    "load_settings",
    "set_telemetry_enabled",
    "settings_path",
]
