"""Tests for completion projection and text-only replacement."""

from __future__ import annotations

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import GatewayEvent, GatewayEventKind
from exp.runtime.gateway.guardrails.completion import apply_text_replacement, completion_from_events


def test_completion_from_events_projects_text_and_tool_calls() -> None:
    """The output-check subject is the winning text plus completed tool calls."""
    events = (
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="hello"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=1,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="lookup",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=2,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"q": "x"},
                raw_arguments='{"q":"x"}',
            ),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=3),
    )

    completion = completion_from_events(events)

    assert completion.text == "hello"
    assert completion.tool_calls[0].arguments == '{"q":"x"}'


def test_text_replacement_does_not_edit_tool_call_events() -> None:
    """Replacement collapses text deltas and leaves tool-call events intact."""
    events = (
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="hel"),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="lo"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=2,
            tool_call_index=0,
            tool_call=ToolCall(call_id="call-1", name="lookup", arguments={"q": "x"}),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=3),
    )

    rewritten = apply_text_replacement(events, "safe")
    texts = tuple(
        event.text_delta for event in rewritten if event.kind is GatewayEventKind.TEXT_DELTA
    )

    assert texts == ("safe",)
    assert any(event.kind is GatewayEventKind.TOOL_CALL_COMPLETED for event in rewritten)
