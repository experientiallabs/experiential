"""Metadata-only anonymous product telemetry."""

from wmo.common.observability.telemetry import BuildTelemetryStats, capture, capture_build_completed

__all__ = [
    "BuildTelemetryStats",
    "capture",
    "capture_build_completed",
]
