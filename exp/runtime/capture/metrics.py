"""Adapt passive provider observations to the gateway's shared capture measurements."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.common.traces.capture import CaptureMetrics, CaptureUsage


def observed_metrics(
    protocol: str,
    response: JsonObject,
    *,
    started_ns: int,
    ended_ns: int,
    terminal: bool,
) -> CaptureMetrics:
    """Keep observed counts separate from proof of a complete exchange.

    Passive capture cannot recover first-token timing from a completed wire copy.
    Its timestamps describe the observed connection, not a gateway deployment or
    any earlier requests on the same application session.
    """
    usage = _usage(protocol, response.get("usage"))
    complete = (
        terminal
        and usage is not None
        and usage.input_tokens is not None
        and usage.output_tokens is not None
        and (usage.input_tokens > 0 or usage.output_tokens > 0)
    )
    return CaptureMetrics(
        started_at=started_ns / 1_000_000_000,
        first_token_at=None,
        terminal_at=max(started_ns, ended_ns) / 1_000_000_000 if terminal else None,
        duration_ms=max(0, ended_ns - started_ns) / 1_000_000 if terminal else None,
        usage=usage,
        usage_complete=complete,
    )


def _count(value: JsonValue) -> int | None:
    """Retain nonnegative reported integers, never bools, estimates, or defaults."""
    return value if type(value) is int and value >= 0 else None


def _detail(value: JsonValue, key: str) -> int | None:
    """Read one optional provider detail without inventing an absent measurement."""
    return _count(value.get(key)) if isinstance(value, dict) else None


def _usage(protocol: str, raw: JsonValue) -> CaptureUsage | None:
    """Normalize provider totals and retain cache and reasoning subsets separately."""
    if not isinstance(raw, dict):
        return None
    inputs = _count(raw.get("input_tokens", raw.get("prompt_tokens")))
    outputs = _count(raw.get("output_tokens", raw.get("completion_tokens")))
    creation: int | None = None
    creation_1h: int | None = None
    if protocol == "messages":
        cached = _count(raw.get("cache_read_input_tokens"))
        creation = _count(raw.get("cache_creation_input_tokens"))
        creation_1h = _detail(raw.get("cache_creation"), "ephemeral_1h_input_tokens")
        for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
            if key in raw and _count(raw[key]) is None:
                inputs = None
        if inputs is not None:
            inputs += (cached or 0) + (creation or 0)
        reasoning = None
    else:
        cached = _detail(
            raw.get("input_tokens_details", raw.get("prompt_tokens_details")), "cached_tokens"
        )
        reasoning = _detail(
            raw.get("output_tokens_details", raw.get("completion_tokens_details")),
            "reasoning_tokens",
        )
    return CaptureUsage(
        input_tokens=inputs,
        output_tokens=outputs,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
        cache_creation_input_tokens=creation,
        cache_creation_1h_input_tokens=creation_1h,
    )
