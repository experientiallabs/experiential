"""Project capture measurements into the same trace fields for every observer."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models import Usage
from exp.common.traces.capture import CaptureMetrics, CaptureUsage


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
