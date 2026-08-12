"""Best-effort anonymous usage telemetry."""

from __future__ import annotations

import math
import os
import sys
from atexit import register
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from posthog import Posthog

from wmo.common.config import ARTIFACT_DIR
from wmo.common.config.settings import ensure_telemetry_anonymous_id, load_settings

POSTHOG_PROJECT_API_KEY = "phc_rPFfCufWpxyctR7duEZTTXovP4k5kbHqSqzd4Z4MQJdL"
POSTHOG_HOST = "https://us.i.posthog.com"

TelemetryValue = str | int | float | bool | None
TelemetryProperties = Mapping[str, TelemetryValue]

_FALSE_VALUES = {"0", "false", "off", "no"}
_TRUE_VALUES = {"1", "true", "on", "yes"}
_CLIENTS: dict[tuple[str, str], Posthog] = {}
_ALLOWED_EVENT_PROPERTIES: dict[str, frozenset[str]] = {
    "wmo build completed": frozenset(
        {
            "success",
            "input_trace_count",
            "input_step_count",
            "train_trace_count",
            "val_trace_count",
            "heldout_trace_count",
            "indexed_step_count",
            "rollouts_used",
            "frontier_size",
            "duration_seconds",
            "llm_call_count",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        }
    ),
    "wmo eval completed": frozenset(
        {
            "success",
            "eval_mode",
            "file_count",
            "scored_step_count",
            "rag_enabled",
            "train_split",
            "top_k",
        }
    ),
    "wmo generated trace started": frozenset({"generated_trace_count"}),
    "wmo generated step failed": frozenset({"success", "duration_seconds"}),
    "wmo generated step completed": frozenset(
        {
            "success",
            "generated_step_count",
            "session_step_count",
            "duration_seconds",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        }
    ),
}
_BOOLEAN_PROPERTIES = frozenset({"success", "rag_enabled"})
_COUNT_PROPERTIES = frozenset(
    {
        "input_trace_count",
        "input_step_count",
        "train_trace_count",
        "val_trace_count",
        "heldout_trace_count",
        "indexed_step_count",
        "rollouts_used",
        "frontier_size",
        "llm_call_count",
        "input_tokens",
        "output_tokens",
        "file_count",
        "scored_step_count",
        "top_k",
        "generated_trace_count",
        "generated_step_count",
        "session_step_count",
    }
)
_NONNEGATIVE_MEASUREMENTS = frozenset({"duration_seconds", "cost_usd"})
_EVAL_MODES = frozenset({"ad_hoc", "suite"})


@dataclass
class BuildTelemetryStats:
    """Corpus and split sizes a build reported, accumulated for one telemetry event."""

    input_trace_count: int = 0
    input_step_count: int = 0
    train_trace_count: int = 0
    val_trace_count: int = 0
    heldout_trace_count: int = 0
    indexed_step_count: int = 0
    rollouts_used: int = 0
    frontier_size: int = 0
    duration_seconds: float = 0.0
    llm_call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def capture(
    event: str,
    properties: TelemetryProperties | None = None,
    *,
    root: str | Path = ARTIFACT_DIR,
) -> bool:
    """Send one anonymous metadata-only event. Returns False when skipped or failed."""
    safe_properties = _sanitize_properties(event, properties)
    if safe_properties is None:
        return False
    if not _enabled(root):
        return False
    api_key = os.getenv("WMO_POSTHOG_PROJECT_API_KEY", POSTHOG_PROJECT_API_KEY).strip()
    if not api_key:
        return False
    host = os.getenv("WMO_POSTHOG_HOST", POSTHOG_HOST).rstrip("/")
    try:
        distinct_id = ensure_telemetry_anonymous_id(root)
        event_properties = {
            "$process_person_profile": False,
            "wmo_version": _wmo_version(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        event_properties.update(safe_properties)
        message_id = _posthog_client(api_key, host).capture(
            event,
            distinct_id=distinct_id,
            properties=event_properties,
        )
        return message_id is not None
    except Exception:  # noqa: BLE001
        return False


def capture_build_completed(
    *,
    stats: BuildTelemetryStats,
    root: str | Path,
) -> None:
    """Capture one metadata-only aggregate for a completed local build.

    Args:
        stats: Aggregate build measurements without prompt or response content.
        root: Project root that owns telemetry preferences and identity.
    """
    capture(
        "wmo build completed",
        {
            "success": True,
            "input_trace_count": stats.input_trace_count,
            "input_step_count": stats.input_step_count,
            "train_trace_count": stats.train_trace_count,
            "val_trace_count": stats.val_trace_count,
            "heldout_trace_count": stats.heldout_trace_count,
            "indexed_step_count": stats.indexed_step_count,
            "rollouts_used": stats.rollouts_used,
            "frontier_size": stats.frontier_size,
            "duration_seconds": round(stats.duration_seconds, 3),
            "llm_call_count": stats.llm_call_count,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "cost_usd": round(stats.cost_usd, 6),
        },
        root=root,
    )


def _sanitize_properties(
    event: str,
    properties: TelemetryProperties | None,
) -> dict[str, TelemetryValue] | None:
    """Return validated aggregate metadata, rejecting arbitrary telemetry egress."""
    allowed_properties = _ALLOWED_EVENT_PROPERTIES.get(event)
    if allowed_properties is None:
        return None
    supplied_properties: TelemetryProperties = {} if properties is None else properties
    if not set(supplied_properties).issubset(allowed_properties):
        return None
    safe_properties: dict[str, TelemetryValue] = {}
    for name, value in supplied_properties.items():
        if value is None:
            continue
        if not _is_safe_property_value(name, value):
            return None
        safe_properties[name] = value
    return safe_properties


def _is_safe_property_value(name: str, value: TelemetryValue) -> bool:
    """Validate one explicitly permitted aggregate property value."""
    if name in _BOOLEAN_PROPERTIES:
        return isinstance(value, bool)
    if name in _COUNT_PROPERTIES:
        return type(value) is int and value >= 0
    if name in _NONNEGATIVE_MEASUREMENTS:
        return _is_nonnegative_finite_number(value)
    if name == "train_split":
        return _is_unit_interval_number(value)
    if name == "eval_mode":
        return isinstance(value, str) and value in _EVAL_MODES
    return False


def _is_nonnegative_finite_number(value: TelemetryValue) -> bool:
    """Return whether a telemetry measurement is finite and nonnegative."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
    )


def _is_unit_interval_number(value: TelemetryValue) -> bool:
    """Return whether a telemetry fraction is finite and falls from zero through one."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def capture_eval_completed(
    *,
    mode: str,
    file_count: int,
    scored_step_count: int,
    rag_enabled: bool,
    sample_turns: str,
    train_split: float,
    top_k: int,
    root: str | Path,
) -> None:
    capture(
        "wmo eval completed",
        {
            "success": True,
            "eval_mode": mode,
            "file_count": file_count,
            "scored_step_count": scored_step_count,
            "rag_enabled": rag_enabled,
            "train_split": train_split,
            "top_k": top_k,
        },
        root=root,
    )


def settings_root_from_results_root(results_root: str) -> Path:
    path = Path(results_root)
    return path.parent if path.name == "evals" else Path(ARTIFACT_DIR)


def _enabled(root: str | Path) -> bool:
    if _env_truthy("DO_NOT_TRACK"):
        return False
    env = os.getenv("WMO_TELEMETRY")
    if env is not None:
        return env.strip().lower() in _TRUE_VALUES
    if os.getenv("PYTEST_CURRENT_TEST"):
        return False
    try:
        return load_settings(root).telemetry.enabled
    except (OSError, ValueError):
        return False


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _FALSE_VALUES:
        return False
    return bool(normalized)


def _wmo_version() -> str:
    try:
        return version("world-model-optimizer")
    except PackageNotFoundError:
        return "unknown"


def _posthog_client(api_key: str, host: str) -> Posthog:
    key = (api_key, host)
    client = _CLIENTS.get(key)
    if client is None:
        client = Posthog(
            api_key,
            host=host,
            flush_interval=1.0,
            max_retries=1,
            timeout=0.5,
        )
        _CLIENTS[key] = client
        register(client.shutdown)
    return client
