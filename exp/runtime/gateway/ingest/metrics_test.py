"""Measured economics belong to one captured turn, not every echoed message or tool."""

from datetime import UTC, datetime

from exp.common.core.artifacts import SourceIdentity
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.common.traces.capture import CaptureMetrics, CaptureUsage
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.runtime.gateway.ingest.metrics import apply_gateway_metrics


def _result(metrics: CaptureMetrics | None) -> TraceNormalizationResult:
    """Build historical and parallel-output spans for one captured exchange."""
    context = {
        "output_message_start": 3,
        "capture_output": {"metrics": None if metrics is None else metrics.model_dump(mode="json")},
    }
    trace = Trace(
        trace_id="trace",
        task="Inspect the fixture",
        source=TraceSource(
            identity=SourceIdentity(kind="production", source_id="capture", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
        spans=tuple(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=datetime(1970, 1, 1, tzinfo=UTC),
                ended_at=datetime(1970, 1, 1, tzinfo=UTC),
                attributes={
                    "exp.request.context": context,
                    "exp.source.span.id": f"message-{source_index}",
                    "exp.source.time.synthetic": True,
                },
            )
            for index, source_index in enumerate([1, 3, 3])
        ),
    )
    return TraceNormalizationResult(traces=(trace,), issues=())


def _metrics() -> CaptureMetrics:
    """Declare a known native meter including reasoning and cache-write subsets."""
    return CaptureMetrics(
        started_at=1_800_000_000,
        first_token_at=1_800_000_001,
        terminal_at=1_800_000_002,
        duration_ms=2000,
        usage=CaptureUsage(
            input_tokens=100,
            output_tokens=20,
            cached_input_tokens=None,
            reasoning_tokens=12,
            cache_creation_input_tokens=10,
            cache_creation_1h_input_tokens=4,
        ),
        usage_complete=True,
    )


def test_current_turn_gets_measured_time_and_usage_once() -> None:
    """Multiple tool spans share timing but neither history nor sibling spans double bill."""
    result = apply_gateway_metrics(_result(_metrics()))
    history, first, parallel = result.traces[0].spans
    assert history.started_at.year == 1970
    assert history.usage is None
    assert history.attributes["exp.source.time.synthetic"] is True
    assert first.started_at.timestamp() == 1_800_000_000
    assert (first.ended_at - first.started_at).total_seconds() == 2
    assert first.attributes["exp.source.time.synthetic"] is False
    assert first.usage is not None and first.usage.output_tokens == 20
    assert first.usage.cached_input_tokens is None
    assert first.usage.cache_write_input_tokens == 10
    assert first.attributes["gen_ai.usage.reasoning_tokens"] == 12
    assert parallel.usage is None
    assert parallel.attributes["exp.gateway.usage.shared_with"] == first.span_id
    assert parallel.started_at == first.started_at


def test_missing_and_partial_meters_never_become_zero_or_final() -> None:
    """Incomplete observations remain available as evidence, not trustworthy totals."""
    missing = _result(None)
    assert apply_gateway_metrics(missing) == missing
    meter = _metrics().usage
    assert meter is not None
    metrics = _metrics().model_copy(
        update={
            "terminal_at": None,
            "duration_ms": None,
            "usage_complete": False,
            "usage": meter.model_copy(update={"output_tokens": None}),
        }
    )
    first = apply_gateway_metrics(_result(metrics)).traces[0].spans[1]
    assert first.usage is None
    assert first.attributes["exp.gateway.usage.complete"] is False
    assert first.attributes["exp.source.time.synthetic"] is True


def test_noncredible_zero_meter_remains_raw_without_zero_cost_trace_usage() -> None:
    """Native settlement's all-zero credibility decision survives trace export."""
    meter = _metrics().usage
    assert meter is not None
    metrics = _metrics().model_copy(
        update={
            "usage_complete": False,
            "usage": meter.model_copy(update={"input_tokens": 0, "output_tokens": 0}),
        }
    )
    first = apply_gateway_metrics(_result(metrics)).traces[0].spans[1]
    assert first.usage is None
    assert first.attributes["exp.gateway.usage.complete"] is False
    assert first.attributes["exp.gateway.metrics"] == metrics.model_dump(mode="json")
    assert first.attributes["exp.source.time.synthetic"] is False
