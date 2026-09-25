"""Source-independent capture projection preserves counts, subsets, and ownership."""

from exp.common.traces.capture import CaptureMetrics, CaptureUsage
from exp.common.traces.ingest.capture import (
    capture_metric_attributes,
    capture_usage,
    capture_usage_attributes,
)


def test_measured_usage_is_charged_once_and_preserves_subsets() -> None:
    """Parallel tools retain evidence without each receiving the exchange's token totals."""
    metrics = CaptureMetrics(
        started_at=1,
        first_token_at=None,
        terminal_at=3,
        duration_ms=2000,
        usage=CaptureUsage(
            input_tokens=113,
            output_tokens=8,
            cached_input_tokens=100,
            reasoning_tokens=4,
            cache_creation_input_tokens=10,
            cache_creation_1h_input_tokens=6,
        ),
        usage_complete=True,
    )
    owner = capture_metric_attributes(metrics)
    sibling = capture_metric_attributes(metrics, owns_usage=False)
    assert owner["gen_ai.usage.input_tokens"] == 113
    assert owner["gen_ai.usage.output_tokens"] == 8
    assert owner["gen_ai.usage.cached_input_tokens"] == 100
    assert owner["gen_ai.usage.cache_creation_input_tokens"] == 10
    assert owner["gen_ai.usage.reasoning_tokens"] == 4
    assert sibling == {
        "exp.capture.metrics": metrics.model_dump_json(),
        "exp.capture.usage.complete": True,
    }
    usage = capture_usage(metrics)
    assert usage is not None and usage.cache_write_input_tokens == 10
    partial = metrics.model_copy(update={"terminal_at": None, "usage_complete": False})
    assert capture_usage(partial) is None
    assert "gen_ai.usage.input_tokens" not in capture_metric_attributes(partial)
    assert capture_usage_attributes(partial.usage)["gen_ai.usage.input_tokens"] == 113
