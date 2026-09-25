"""Attach measured gateway turn economics without attributing them to echoed history."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from exp.common.traces import Trace, TraceSpan
from exp.common.traces.capture import CaptureMetrics
from exp.common.traces.ingest.capture import capture_metric_attributes, capture_usage
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.common.traces.ingest.vendor_trace import SOURCE_SPAN_ATTRIBUTE


def apply_gateway_metrics(result: TraceNormalizationResult) -> TraceNormalizationResult:
    """Map native facts onto this exchange's output spans, leaving unknown history explicit."""
    return replace(result, traces=tuple(measured_trace(trace) for trace in result.traces))


def measured_trace(trace: Trace) -> Trace:
    """Charge a measured turn once even when parallel tools emit several model spans."""
    charged = False
    spans: list[TraceSpan] = []
    owner: str | None = None
    for span in trace.spans:
        context = span.attributes.get("exp.request.context")
        if not isinstance(context, dict) or span.name != "agent.model_call":
            spans.append(span)
            continue
        output_start = context.get("output_message_start")
        source_id = span.attributes.get(SOURCE_SPAN_ATTRIBUTE)
        if (
            not isinstance(output_start, int)
            or not isinstance(source_id, str)
            or not source_id.startswith("message-")
            or not source_id.removeprefix("message-").isdigit()
            or int(source_id.removeprefix("message-")) < output_start
        ):
            spans.append(span)
            continue
        capture = context.get("capture_output")
        raw_metrics = capture.get("metrics") if isinstance(capture, dict) else None
        if raw_metrics is None:
            spans.append(span)
            continue
        metrics = CaptureMetrics.model_validate(raw_metrics)
        attributes = dict(span.attributes)
        attributes.update(capture_metric_attributes(metrics, owns_usage=not charged))
        attributes["exp.gateway.metrics.scope"] = "selected_attempt"
        attributes["exp.gateway.metrics"] = metrics.model_dump(mode="json")
        usage = None
        meter = metrics.usage
        if not charged:
            owner = span.span_id
            if meter is not None:
                attributes["exp.gateway.usage.complete"] = metrics.usage_complete
                usage = capture_usage(metrics)
            charged = True
        else:
            attributes["exp.gateway.usage.shared_with"] = owner
            attributes["exp.capture.usage.shared_with"] = owner
        updates = {"attributes": attributes, "usage": usage}
        if metrics.terminal_at is not None and metrics.duration_ms is not None:
            started = datetime.fromtimestamp(metrics.started_at, UTC)
            # Monotonic elapsed time keeps wall-clock corrections from inverting a span.
            updates.update(
                started_at=started, ended_at=started + timedelta(milliseconds=metrics.duration_ms)
            )
            attributes["exp.source.time.synthetic"] = False
        spans.append(span.model_copy(update=updates))
    return trace.model_copy(update={"spans": tuple(spans)})
