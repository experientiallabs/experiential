"""Shields.io endpoint payload for the README gateway-latency badge."""

from __future__ import annotations

import json
from pathlib import Path

from exp.common.core.artifacts import JsonObject

BADGE_BRANCH = "badges"
BADGE_FILENAME = "gateway-latency.json"
SHIELDS_LABEL = "gateway latency"
SHIELDS_COLOR = "0070f3"
SHIELDS_CACHE_SECONDS = 300
RAW_ENDPOINT_URL = (
    "https://raw.githubusercontent.com/experientiallabs/experiential/"
    f"{BADGE_BRANCH}/{BADGE_FILENAME}"
)
SHIELDS_IMAGE_URL = (
    "https://img.shields.io/endpoint?url="
    "https%3A%2F%2Fraw.githubusercontent.com%2Fexperientiallabs%2Fexperiential%2F"
    f"{BADGE_BRANCH}%2F{BADGE_FILENAME}"
)


def format_latency_ms(p50_ms: float) -> str:
    """Format representative gateway p50 request latency for the badge message.

    Args:
        p50_ms: Client-observed gateway p50, in milliseconds.

    Returns:
        One-decimal millisecond message such as ``22.2 ms``.
    """
    return f"{p50_ms:.1f} ms"


def shields_endpoint(*, p50_ms: float) -> JsonObject:
    """Build a Shields endpoint document from one measured gateway p50.

    Args:
        p50_ms: Representative non-stream gateway p50, in milliseconds.

    Returns:
        Shields schemaVersion 1 object. ``message`` is the formatted latency.
    """
    return {
        "schemaVersion": 1,
        "label": SHIELDS_LABEL,
        "message": format_latency_ms(p50_ms),
        "color": SHIELDS_COLOR,
        "cacheSeconds": SHIELDS_CACHE_SECONDS,
    }


def write_shields_endpoint(*, p50_ms: float, path: Path) -> JsonObject:
    """Write the Shields endpoint JSON for ``p50_ms`` and return the payload.

    Args:
        p50_ms: Representative non-stream gateway p50, in milliseconds.
        path: Destination file. Parent directories are created when missing.

    Returns:
        The JSON object written to ``path``.
    """
    payload = shields_endpoint(p50_ms=p50_ms)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
