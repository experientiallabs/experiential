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
from typing import Literal
from uuid import NAMESPACE_URL, UUID, uuid5

from posthog import Posthog
from pydantic import BaseModel, ConfigDict, ValidationError

from wmo.common.config import ARTIFACT_DIR
from wmo.common.config.settings import ensure_telemetry_anonymous_id, load_settings
from wmo.common.core.artifacts import canonical_json_bytes, sha256_json, validate_artifact_id
from wmo.common.core.files import write_bytes_atomic
from wmo.common.core.locks import file_write_lock

POSTHOG_PROJECT_API_KEY = "phc_rPFfCufWpxyctR7duEZTTXovP4k5kbHqSqzd4Z4MQJdL"
POSTHOG_HOST = "https://us.i.posthog.com"

TelemetryValue = str | int | float | bool | None
TelemetryProperties = Mapping[str, TelemetryValue]

_FALSE_VALUES = {"0", "false", "off", "no"}
_TRUE_VALUES = {"1", "true", "on", "yes"}
_CLIENTS: dict[tuple[str, str, bool], Posthog] = {}
_COMPLETION_RECEIPT_DIRECTORY = "telemetry-receipts"
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
    "wmo router completed": frozenset(
        {
            "success",
            "candidate_count",
            "duration_seconds",
        }
    ),
    "wmo simulation completed": frozenset(
        {
            "success",
            "rollout_count",
            "duration_seconds",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        }
    ),
    "wmo sft completed": frozenset(
        {
            "success",
            "train_example_count",
            "heldout_example_count",
            "training_step_count",
            "duration_seconds",
            "cost_usd",
        }
    ),
}
_BOOLEAN_PROPERTIES = frozenset({"success"})
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
        "candidate_count",
        "rollout_count",
        "train_example_count",
        "heldout_example_count",
        "training_step_count",
    }
)
_NONNEGATIVE_MEASUREMENTS = frozenset({"duration_seconds", "cost_usd"})


class _CompletionTelemetryReceipt(BaseModel):
    """Durable local delivery state for one content-addressed completion event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    event: str
    completion_id: str
    properties: dict[str, TelemetryValue]
    delivery_status: Literal["pending", "delivered"]


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


def capture_completion_once(
    event: str,
    completion_id: str,
    properties: TelemetryProperties,
    *,
    root: str | Path = ARTIFACT_DIR,
) -> bool:
    """Retry one metadata-only completion until its deterministic event is delivered.

    Pending state is written before egress and delivered state afterwards. A crash in between is
    retried with the same PostHog UUID, so ingestion deduplicates the ambiguous delivery attempt.
    Callers must invoke this on every successful replay of the named durable completion.

    Args:
        event: One allowlisted completion event.
        completion_id: Canonical immutable artifact ID for the durable completion.
        properties: Allowlisted aggregate metadata without user content.
        root: Project root that owns telemetry settings and receipts.

    Returns:
        True only when this call advanced the receipt to delivered.
    """
    safe_properties = _sanitize_properties(event, properties)
    if safe_properties is None or not _enabled(root):
        return False
    try:
        validated_completion_id = validate_artifact_id(completion_id)
        if len(validated_completion_id) > 128:
            return False
        pending = _CompletionTelemetryReceipt(
            event=event,
            completion_id=validated_completion_id,
            properties=dict(safe_properties),
            delivery_status="pending",
        )
        receipt_path = _completion_receipt_path(root, event, validated_completion_id)
        with file_write_lock(receipt_path, what="anonymous completion telemetry receipt"):
            if receipt_path.exists():
                pending = _read_completion_receipt(
                    receipt_path,
                    event,
                    validated_completion_id,
                )
                if pending.delivery_status == "delivered":
                    return False
                if _stable_completion_properties(pending.properties) != (
                    _stable_completion_properties(safe_properties)
                ):
                    return False
            else:
                write_bytes_atomic(receipt_path, canonical_json_bytes(pending) + b"\n")
            delivered = _capture_sanitized(
                event,
                pending.properties,
                root=root,
                event_uuid=_completion_event_uuid(event, validated_completion_id),
            )
            if not delivered:
                return False
            write_bytes_atomic(
                receipt_path,
                canonical_json_bytes(pending.model_copy(update={"delivery_status": "delivered"}))
                + b"\n",
            )
            return True
    except (OSError, RuntimeError, ValueError):
        return False


def _capture_sanitized(
    event: str,
    safe_properties: Mapping[str, TelemetryValue],
    *,
    root: str | Path,
    event_uuid: UUID | None = None,
) -> bool:
    """Deliver already validated aggregate metadata with failure isolation."""
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
        client = _posthog_client(api_key, host, synchronous=event_uuid is not None)
        if event_uuid is None:
            message_id = client.capture(
                event,
                distinct_id=distinct_id,
                properties=event_properties,
            )
        else:
            message_id = client.capture(
                event,
                distinct_id=distinct_id,
                properties=event_properties,
                uuid=event_uuid,
            )
        return message_id is not None
    except Exception:  # noqa: BLE001
        return False


def capture_build_completed(
    *,
    completion_id: str,
    stats: BuildTelemetryStats,
    root: str | Path,
) -> None:
    """Capture one metadata-only aggregate for a completed local build.

    Args:
        completion_id: Immutable task-set artifact ID produced by the build.
        stats: Aggregate build measurements without prompt or response content.
        root: Project root that owns telemetry preferences and identity.
    """
    capture_completion_once(
        "wmo build completed",
        completion_id,
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


def _completion_receipt_path(root: str | Path, event: str, completion_id: str) -> Path:
    """Return the content-addressed local receipt path for one completion event."""
    receipt_id = sha256_json({"event": event, "completion_id": completion_id})
    return Path(root) / _COMPLETION_RECEIPT_DIRECTORY / f"{receipt_id}.json"


def _completion_event_uuid(event: str, completion_id: str) -> UUID:
    """Return the deterministic PostHog ingestion UUID for one immutable completion."""
    return uuid5(NAMESPACE_URL, f"world-model-optimizer:{event}:{completion_id}")


def _read_completion_receipt(
    path: Path,
    event: str,
    completion_id: str,
) -> _CompletionTelemetryReceipt:
    """Load canonical delivery state matching its content-addressed completion."""
    try:
        payload = path.read_bytes()
        receipt = _CompletionTelemetryReceipt.model_validate_json(payload)
    except (OSError, ValidationError, ValueError) as exc:
        raise ValueError("completion telemetry receipt is invalid") from exc
    if (
        receipt.event != event
        or receipt.completion_id != completion_id
        or canonical_json_bytes(receipt) + b"\n" != payload
        or _sanitize_properties(receipt.event, receipt.properties) != receipt.properties
    ):
        raise ValueError("completion telemetry receipt does not match its durable completion")
    return receipt


def _stable_completion_properties(
    properties: Mapping[str, TelemetryValue],
) -> dict[str, TelemetryValue]:
    """Exclude invocation-local duration while comparing immutable replay aggregates."""
    return {name: value for name, value in properties.items() if name != "duration_seconds"}


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
    return False


def _is_nonnegative_finite_number(value: TelemetryValue) -> bool:
    """Return whether a telemetry measurement is finite and nonnegative."""
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
    )


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


def _posthog_client(api_key: str, host: str, *, synchronous: bool = False) -> Posthog:
    """Return one bounded official PostHog client for async or confirmed delivery."""
    key = (api_key, host, synchronous)
    client = _CLIENTS.get(key)
    if client is None:
        client = Posthog(
            api_key,
            host=host,
            flush_interval=1.0,
            max_retries=1,
            sync_mode=synchronous,
            timeout=0.5,
        )
        _CLIENTS[key] = client
        register(client.shutdown)
    return client
