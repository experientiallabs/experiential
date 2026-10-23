"""Tests for the python mirror of the estimated reasoning count."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind, GatewayUsage
from exp.runtime.openai_protocol.reasoning_estimate import ReasoningEstimate


def _usage(output: int, reasoning: int | None = None) -> GatewayUsage:
    return GatewayUsage(input_tokens=48, output_tokens=output, reasoning_tokens=reasoning)


def _events(*items: tuple[GatewayEventKind, dict[str, object]]) -> tuple[GatewayEvent, ...]:
    return tuple(
        GatewayEvent.model_validate({"kind": kind, "sequence_number": index, **fields})
        for index, (kind, fields) in enumerate(items)
    )


def _thinking_then(answer: str) -> tuple[GatewayEvent, ...]:
    return _events(
        (
            GatewayEventKind.THINKING_DELTA,
            {"reasoning_block_index": 0, "text_delta": "1001 = 7 x 143, so composite."},
        ),
        (GatewayEventKind.TEXT_DELTA, {"text_delta": answer}),
    )


def test_a_thinking_turn_without_a_count_estimates_output_less_the_visible_answer() -> None:
    """20 visible bytes (the multiplication sign is two) are 5 tokens of 259."""
    estimate = ReasoningEstimate.from_events(_thinking_then("No.\n\n1001 = 7 × 143"))
    assert estimate.estimate(_usage(259)) == 254


def test_tool_arguments_count_and_redacted_thinking_counts_as_thinking() -> None:
    """Tool arguments are visible output; redacted thinking alone is evidence of thinking."""
    events = _events(
        (
            GatewayEventKind.REDACTED_THINKING,
            {"reasoning_block_index": 0, "redacted_thinking_data": "opaque"},
        ),
        (
            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            {"tool_call_index": 0, "raw_arguments_delta": '{"path":"a"}'},
        ),
    )
    assert ReasoningEstimate.from_events(events).estimate(_usage(10)) == 7


def test_no_estimate_without_thinking_or_with_a_reported_count() -> None:
    """A provider count is never replaced, and a turn that did not think has none."""
    plain = ReasoningEstimate.from_events(
        _events((GatewayEventKind.TEXT_DELTA, {"text_delta": "ready"}))
    )
    assert plain.estimate(_usage(5)) is None
    assert ReasoningEstimate.from_events(_thinking_then("ok")).estimate(_usage(40, 12)) is None
    long = ReasoningEstimate.from_events(_thinking_then("x" * 400))
    assert long.estimate(_usage(20)) == 0


def test_annotate_moves_reasoning_from_unreported_to_estimated() -> None:
    """The wire marks the count as an estimate and drops it from the unreported list."""
    estimate = ReasoningEstimate.from_events(_thinking_then("No."))
    wire: JsonObject = {
        "completion_tokens": 30,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "unreported_token_details": ["cached_tokens", "reasoning_tokens"],
    }
    annotated = estimate.annotate(wire, _usage(30), "completion_tokens_details")
    assert annotated is not None
    assert annotated["completion_tokens_details"] == {"reasoning_tokens": 29}
    assert annotated["unreported_token_details"] == ["cached_tokens"]
    assert annotated["estimated_token_details"] == ["reasoning_tokens"]
    assert estimate.annotate(None, _usage(30), "completion_tokens_details") is None
    known: JsonObject = {"completion_tokens_details": {"reasoning_tokens": 7}}
    assert estimate.annotate(dict(known), _usage(30, 7), "completion_tokens_details") == known
