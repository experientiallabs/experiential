"""Metadata-only anonymous product telemetry."""

from wmo.common.observability.telemetry import (
    BuildTelemetryStats,
    capture_build_completed,
    capture_completion_once,
)

__all__ = [
    "BuildTelemetryStats",
    "capture_build_completed",
    "capture_completion_once",
]
