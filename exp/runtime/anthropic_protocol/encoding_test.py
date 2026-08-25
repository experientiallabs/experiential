"""Golden-fixture tests for the Anthropic Messages encoders.

The committed frames below are the wire contract for the Messages surface.
The Rust data plane must match them byte for byte (see the parity tests in
``exp/runtime/gateway/native_bridge_test.py``), so a change here is a public
protocol change, not a refactor.
"""

from __future__ import annotations

import json

import pytest

from exp.common.models.model import ToolCall
from exp.runtime.anthropic_protocol.encoding import (
    MessagesSseEncoder,
    completed_messages_body,
    messages_usage,
)
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_MESSAGE_START = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_ff596a02add1b410b6dc664a47e25b3e",'
    '"type":"message","role":"assistant","model":"coding","content":[],"stop_reason":null,'
    '"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
)
_PING = 'event: ping\ndata: {"type":"ping"}\n\n'


def tool_stream_events() -> tuple[GatewayEvent, ...]:
    """Return one canonical text-then-tool stream ending completed."""
    return (
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="Hel"),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=1, text_delta="lo é"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta='{"q": ',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=4,
            tool_call_index=0,
            raw_arguments_delta='"x"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=5,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="search",
                arguments={"q": "x"},
                raw_arguments='{"q": "x"}',
            ),
        ),
        GatewayEvent(
            kind=GatewayEventKind.USAGE,
            sequence_number=6,
            usage=GatewayUsage(input_tokens=10, output_tokens=4, cached_input_tokens=3),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=7),
    )


# The committed golden lifecycle for `tool_stream_events`.
TOOL_STREAM_GOLDEN_FRAMES = (
    _MESSAGE_START,
    _PING,
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"lo é"}}\n\n',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use",'
    '"id":"call-1","name":"search","input":{}}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"input_json_delta","partial_json":"{\\"q\\": "}}\n\n',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"input_json_delta","partial_json":"\\"x\\"}"}}\n\n',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},'
    '"usage":{"input_tokens":7,"output_tokens":4,"cache_read_input_tokens":3}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)

# The committed golden non-streaming body for `tool_stream_events`.
TOOL_STREAM_GOLDEN_BODY = (
    '{"id":"msg_ff596a02add1b410b6dc664a47e25b3e","type":"message","role":"assistant",'
    '"model":"coding","content":[{"type":"text","text":"Hello é"},'
    '{"type":"tool_use","id":"call-1","name":"search","input":{"q":"x"}}],'
    '"stop_reason":"tool_use","stop_sequence":null,'
    '"usage":{"input_tokens":7,"output_tokens":4,"cache_read_input_tokens":3}}'
)


def encode(events: tuple[GatewayEvent, ...]) -> tuple[str, ...]:
    """Encode one complete stream with a fresh Messages encoder."""
    encoder = MessagesSseEncoder(request_id="request-abc", model="coding")
    frames = list(encoder.start())
    for event in events:
        frames.extend(encoder.feed(event))
    return tuple(frames)


def test_tool_stream_matches_the_committed_golden_frames() -> None:
    """The full text-and-tool lifecycle is byte-stable against the fixture."""
    assert encode(tool_stream_events()) == TOOL_STREAM_GOLDEN_FRAMES


def test_completed_body_matches_the_committed_golden_body() -> None:
    """The non-streaming message object is byte-stable against the fixture."""
    body = completed_messages_body(
        request_id="request-abc", model="coding", events=tool_stream_events()
    )
    assert json.dumps(body, separators=(",", ":"), ensure_ascii=False) == TOOL_STREAM_GOLDEN_BODY


def test_incomplete_terminal_reports_max_tokens_and_zero_unknown_usage() -> None:
    """An incomplete stream stops with max_tokens and zeroed unknown usage."""
    frames = encode(
        (
            GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="part"),
            GatewayEvent(kind=GatewayEventKind.INCOMPLETE, sequence_number=1),
        )
    )
    assert frames[-2] == (
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},'
        '"usage":{"input_tokens":0,"output_tokens":0}}\n\n'
    )
    assert frames[-1] == 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


def test_failed_terminal_emits_one_anthropic_error_event() -> None:
    """A failed terminal ends the stream with one Anthropic error event."""
    frames = encode(
        (
            GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="oops"),
            GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=1,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider stream failed",
                ),
            ),
        )
    )
    assert frames[-1] == (
        "event: error\n"
        'data: {"type":"error","error":{"type":"api_error",'
        '"message":"provider stream failed"}}\n\n'
    )


def test_refusal_deltas_are_withheld_and_terminate_as_an_error() -> None:
    """Refusal content never renders as text; the terminal is an error event."""
    encoder = MessagesSseEncoder(request_id="request-abc", model="coding")
    encoder.start()
    assert (
        encoder.feed(
            GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=0, text_delta="no")
        )
        == ()
    )
    frames = encoder.feed(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1))
    assert frames == (
        "event: error\n"
        'data: {"type":"error","error":{"type":"api_error",'
        '"message":"provider refused the request"}}\n\n',
    )


def test_completed_body_rejects_refusal_content() -> None:
    """The non-streaming aggregation fails closed on refusal content."""
    events = (
        GatewayEvent(kind=GatewayEventKind.REFUSAL_DELTA, sequence_number=0, text_delta="no"),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
    )
    with pytest.raises(OpenAIProtocolError) as excinfo:
        completed_messages_body(request_id="request-abc", model="coding", events=events)
    assert excinfo.value.status_code == 502


def test_encoder_enforces_start_order_terminal_and_tool_identity() -> None:
    """State violations raise the sanitized invalid provider-stream error."""
    encoder = MessagesSseEncoder(request_id="request-abc", model="coding")
    with pytest.raises(OpenAIProtocolError, match="must be started"):
        encoder.feed(
            GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="x")
        )
    encoder.start()
    with pytest.raises(OpenAIProtocolError, match="started more than once"):
        encoder.start()
    with pytest.raises(OpenAIProtocolError, match="before tool-call start"):
        encoder.feed(
            GatewayEvent(
                kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                sequence_number=1,
                tool_call_index=0,
                raw_arguments_delta="{}",
            )
        )
    encoder.feed(
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        )
    )
    encoder.feed(
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=0,
            raw_arguments_delta='{"q":1}',
        )
    )
    with pytest.raises(OpenAIProtocolError, match="changed streamed identity or bytes"):
        encoder.feed(
            GatewayEvent(
                kind=GatewayEventKind.TOOL_CALL_COMPLETED,
                sequence_number=4,
                tool_call_index=0,
                tool_call=ToolCall(
                    call_id="call-1",
                    name="search",
                    arguments={"q": 2},
                    raw_arguments='{"q":2}',
                ),
            )
        )
    encoder.feed(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=5))
    with pytest.raises(OpenAIProtocolError, match="after its terminal"):
        encoder.feed(GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=6))


def test_usage_mapping_reports_cached_reads_out_of_the_input_total() -> None:
    """Cached reads come back out of the folded normalized input count."""
    usage = GatewayUsage(input_tokens=10, output_tokens=4, cached_input_tokens=3)
    assert messages_usage(usage) == {
        "input_tokens": 7,
        "output_tokens": 4,
        "cache_read_input_tokens": 3,
    }
    assert messages_usage(None) == {"input_tokens": 0, "output_tokens": 0}
    plain = GatewayUsage(input_tokens=5, output_tokens=2)
    assert messages_usage(plain) == {"input_tokens": 5, "output_tokens": 2}


def test_completed_body_preserves_interleaved_block_order() -> None:
    """Content blocks keep provider order and merge only adjacent text."""
    events = (
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=0,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=1,
            tool_call_index=0,
            raw_arguments_delta="{}",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=2,
            tool_call_index=0,
            tool_call=ToolCall(call_id="call-1", name="search", arguments={}, raw_arguments="{}"),
        ),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=3, text_delta="after "),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=4, text_delta="the tool"),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=5),
    )
    body = completed_messages_body(request_id="request-abc", model="coding", events=events)
    assert body["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "search", "input": {}},
        {"type": "text", "text": "after the tool"},
    ]
    assert body["stop_reason"] == "tool_use"


def test_deferred_tool_completion_streams_and_aggregates_consistently() -> None:
    """OpenAI-compatible streams complete tools only at their terminal.

    Text may therefore arrive between a tool's arguments and its completion.
    The streaming encoder keeps the tool block open across the text block and
    stops it at the deferred completion, and the aggregation must anchor the
    tool block at its start position so both renderings order blocks
    identically.
    """
    events = (
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=0,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=1,
            tool_call_index=0,
            raw_arguments_delta="{}",
        ),
        GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=2, text_delta="after"),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=3,
            tool_call_index=0,
            tool_call=ToolCall(call_id="call-1", name="search", arguments={}, raw_arguments="{}"),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=4),
    )
    frames = encode(events)
    names = [frame.split("\n", 1)[0].removeprefix("event: ") for frame in frames]
    assert names == [
        "message_start",
        "ping",
        "content_block_start",
        "content_block_delta",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    body = completed_messages_body(request_id="request-abc", model="coding", events=events)
    assert body["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "search", "input": {}},
        {"type": "text", "text": "after"},
    ]


def test_parallel_tool_deltas_interleave_across_open_blocks() -> None:
    """Interleaved parallel tool fragments target each tool's own block.

    Providers stream parallel tool calls concurrently, so a fragment for an
    earlier tool may arrive after a later tool started. Each fragment must
    land on the block its tool opened and each completion must stop that
    same block.
    """
    events = (
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=0,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=1,
            tool_call_index=0,
            raw_arguments_delta='{"q": ',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=2,
            tool_call_index=1,
            tool_call_id="call-2",
            tool_name="fetch",
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=3,
            tool_call_index=1,
            raw_arguments_delta='{"u": "y"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=4,
            tool_call_index=0,
            raw_arguments_delta='"x"}',
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=5,
            tool_call_index=0,
            tool_call=ToolCall(
                call_id="call-1",
                name="search",
                arguments={"q": "x"},
                raw_arguments='{"q": "x"}',
            ),
        ),
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_COMPLETED,
            sequence_number=6,
            tool_call_index=1,
            tool_call=ToolCall(
                call_id="call-2",
                name="fetch",
                arguments={"u": "y"},
                raw_arguments='{"u": "y"}',
            ),
        ),
        GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=7),
    )
    frames = encode(events)
    payloads = [
        json.loads(frame.split("data: ", 1)[1]) for frame in frames if "content_block" in frame
    ]
    deltas = [
        (payload["index"], payload["delta"]["partial_json"])
        for payload in payloads
        if payload["type"] == "content_block_delta"
    ]
    assert deltas == [(0, '{"q": '), (1, '{"u": "y"}'), (0, '"x"}')]
    stops = [payload["index"] for payload in payloads if payload["type"] == "content_block_stop"]
    assert stops == [0, 1]
    body = completed_messages_body(request_id="request-abc", model="coding", events=events)
    assert body["content"] == [
        {"type": "tool_use", "id": "call-1", "name": "search", "input": {"q": "x"}},
        {"type": "tool_use", "id": "call-2", "name": "fetch", "input": {"u": "y"}},
    ]


def test_repeated_tool_completion_is_still_rejected() -> None:
    """A second completion for the same tool stays an invalid provider stream."""
    encoder = MessagesSseEncoder(request_id="request-abc", model="coding")
    encoder.start()
    encoder.feed(
        GatewayEvent(
            kind=GatewayEventKind.TOOL_CALL_STARTED,
            sequence_number=0,
            tool_call_index=0,
            tool_call_id="call-1",
            tool_name="search",
        )
    )
    encoder.feed(
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=1,
            tool_call_index=0,
            raw_arguments_delta="{}",
        )
    )
    completion = GatewayEvent(
        kind=GatewayEventKind.TOOL_CALL_COMPLETED,
        sequence_number=2,
        tool_call_index=0,
        tool_call=ToolCall(call_id="call-1", name="search", arguments={}, raw_arguments="{}"),
    )
    encoder.feed(completion)
    with pytest.raises(OpenAIProtocolError, match="omitted its started tool call"):
        encoder.feed(completion.model_copy(update={"sequence_number": 3}))
