"""Capture evidence retains explicit absence and loss across its JSON boundary."""

from exp.common.traces.capture import CaptureMetrics, CaptureSseResponse, CaptureUsage


def test_shared_contract_keeps_partial_usage_and_response_loss() -> None:
    """An interrupted stream can retain observations without certifying final totals."""
    metrics = CaptureMetrics(
        started_at=1,
        first_token_at=None,
        terminal_at=None,
        duration_ms=None,
        usage=CaptureUsage(
            input_tokens=12,
            output_tokens=None,
            cached_input_tokens=0,
            reasoning_tokens=None,
        ),
        usage_complete=False,
    )
    assert CaptureMetrics.model_validate_json(metrics.model_dump_json()) == metrics
    response = CaptureSseResponse(
        status=200,
        frames=({"type": "response.created", "unknown": "preserved"},),
        truncated=False,
        client_disconnected=True,
    )
    assert CaptureSseResponse.model_validate_json(response.model_dump_json()) == response
