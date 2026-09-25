"""Capture response and measurement contracts shared by passive and gateway observers."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.models import Usage


class CaptureJsonResponse(ContractModel):
    """A complete JSON response body, before host-specific presentation decoration.

    Attributes:
        kind: JSON response discriminator, always json.
        status: Observed HTTP response status.
        body: Queryable response projection.
        source_json: Exact escaped source when normalization was needed, otherwise None.
    """

    kind: Literal["json"] = "json"
    status: int = Field(ge=100, le=599)
    body: JsonValue
    source_json: str | None = None


class CaptureSseResponse(ContractModel):
    """Ordered SSE data payloads with explicit loss and disconnect indicators.

    Attributes:
        kind: Streaming response discriminator, always sse.
        status: Observed HTTP response status.
        frames: Ordered complete SSE data payloads.
        truncated: Whether capture limits excluded part of the response.
        client_disconnected: Whether the consumer stopped before body completion.
        source_json: Exact escaped frames when normalization was needed, otherwise None.
    """

    kind: Literal["sse"] = "sse"
    status: int = Field(ge=100, le=599)
    frames: tuple[JsonValue, ...]
    truncated: bool
    client_disconnected: bool
    source_json: str | None = None


class CaptureUsage(ContractModel):
    """Provider reported counts, with every absent counter remaining unknown.

    Attributes:
        input_tokens: Total input tokens, including cache subsets when reported.
        output_tokens: Total output tokens, including the reasoning subset.
        cached_input_tokens: Observed cache-read subset, otherwise None.
        reasoning_tokens: Observed reasoning subset, otherwise None.
        cache_creation_input_tokens: Cache-write subset, otherwise None.
        cache_creation_1h_input_tokens: One-hour cache-write subset, otherwise None.
    """

    input_tokens: int | None = Field(ge=0)
    output_tokens: int | None = Field(ge=0)
    cached_input_tokens: int | None = Field(ge=0)
    reasoning_tokens: int | None = Field(ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_1h_input_tokens: int | None = Field(default=None, ge=0)


class CaptureMetrics(ContractModel):
    """Measurements for one observed exchange, never aggregate fallback billing.

    Attributes:
        started_at: Observed exchange start as Unix seconds.
        first_token_at: Observed first output-token time, otherwise unknown.
        terminal_at: Provider terminal time, absent for an interrupted stream.
        duration_ms: Observed exchange duration, absent before a provider terminal.
        usage: Observed normalized provider counts; unknown fields remain None.
        usage_complete: Whether a terminal and credible complete token totals were observed.
    """

    started_at: float = Field(ge=0, allow_inf_nan=False)
    first_token_at: float | None = Field(ge=0, allow_inf_nan=False)
    terminal_at: float | None = Field(ge=0, allow_inf_nan=False)
    duration_ms: float | None = Field(ge=0, allow_inf_nan=False)
    usage: CaptureUsage | None
    usage_complete: bool


def capture_usage(metrics: CaptureMetrics) -> Usage | None:
    """Expose complete measured totals, keeping partial counts only as source evidence."""
    meter = metrics.usage
    if (
        not metrics.usage_complete
        or meter is None
        or meter.input_tokens is None
        or meter.output_tokens is None
    ):
        return None
    return Usage(
        input_tokens=meter.input_tokens,
        output_tokens=meter.output_tokens,
        cached_input_tokens=meter.cached_input_tokens,
        cache_write_input_tokens=meter.cache_creation_input_tokens,
    )


def capture_metric_attributes(metrics: CaptureMetrics, *, owns_usage: bool = True) -> JsonObject:
    """Retain measurements and emit token counters once per observed exchange.

    Args:
        metrics: Observed timing and provider counts, with missing facts left unknown.
        owns_usage: Whether this span owns the exchange's totals. Sibling tool spans
            retain the measurements without multiplying their token counts.

    Returns:
        Shared capture metadata and complete GenAI token attributes for the owning span.
    """
    attributes: JsonObject = {
        "exp.capture.metrics": metrics.model_dump_json(),
        "exp.capture.usage.complete": metrics.usage_complete,
    }
    if not owns_usage:
        return attributes
    meter = metrics.usage
    if meter is not None and meter.reasoning_tokens is not None:
        attributes["gen_ai.usage.reasoning_tokens"] = meter.reasoning_tokens
    if metrics.usage_complete:
        attributes.update(capture_usage_attributes(meter))
    return attributes


def capture_usage_attributes(usage: CaptureUsage | None) -> JsonObject:
    """Map reported counts without claiming that the provider exchange completed.

    Desktop Capture includes reported partial counts in its live totals. Gateway
    evidence uses this projection only when its observer certifies complete usage.
    Both preserve the same cache subsets and never substitute zero for absence.
    """
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return {}
    attributes: JsonObject = {
        "gen_ai.usage.input_tokens": usage.input_tokens,
        "gen_ai.usage.output_tokens": usage.output_tokens,
    }
    for name, value in (
        ("cached_input_tokens", usage.cached_input_tokens),
        ("cache_creation_input_tokens", usage.cache_creation_input_tokens),
        ("reasoning_tokens", usage.reasoning_tokens),
    ):
        if value is not None:
            attributes[f"gen_ai.usage.{name}"] = value
    return attributes
