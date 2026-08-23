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


def gateway_p50_ms_from_report_json(path: Path) -> float:
    """Read representative non-stream gateway p50 from a report file.

    Args:
        path: Versioned ``exp.gateway.latency_report`` JSON artifact.

    Returns:
        ``representative_run.gateway.p50_ms``.

    Raises:
        ValueError: The file is not a report object with a numeric p50.
        OSError: The file cannot be read.
        json.JSONDecodeError: The file is not JSON.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("latency report must be a JSON object")
    run = raw.get("representative_run")
    if not isinstance(run, dict):
        raise ValueError("latency report is missing representative_run")
    gateway = run.get("gateway")
    if not isinstance(gateway, dict):
        raise ValueError("latency report is missing representative_run.gateway")
    p50 = gateway.get("p50_ms")
    if isinstance(p50, bool) or not isinstance(p50, int | float):
        raise ValueError("latency report representative_run.gateway.p50_ms must be a number")
    return float(p50)
