"""Protocol normalization preserves evidence while removing transport credentials."""

import json
import time
from dataclasses import replace

import brotli
import pytest
import zstandard

from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.runtime.capture.normalization import CapturedExchange, capture_protocol, normalize_exchange
from exp.simulation.ingest.otlp import normalize_otlp_payload


def _exchange(**changes: str | bytes | int | bool) -> CapturedExchange:
    """Return a bounded synthetic request and response for this test."""
    exchange = CapturedExchange(
        protocol="responses",
        host="api.openai.com",
        path="/v1/responses",
        started_ns=time.time_ns(),
        ended_ns=time.time_ns(),
        request=b'{"model":"gpt-test","input":"hello"}',
        response=b'{"model":"gpt-test","output":[],"usage":{"input_tokens":3,"output_tokens":7}}',
        status=200,
    )
    return replace(exchange, **changes)


def _attributes(exchange: CapturedExchange) -> JsonObject:
    """Normalize a synthetic exchange through the canonical OTLP contract."""
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096))
    result = normalize_otlp_payload(
        payload, source=SourceIdentity(kind="otlp", source_id="synthetic-capture")
    )
    assert result.issues == ()
    assert len(result.traces) == 1
    assert result.traces[0].conversation_id is None
    return result.traces[0].spans[0].attributes


def test_known_usage_and_redacted_copies_normalize_through_existing_cloud_contract() -> None:
    """Preserve known usage while stripping credential fields from copied payloads."""
    request = json.dumps(
        {
            "model": "gpt-test",
            "input": "hello Bearer TOP-SECRET",
            "api_key": "TOP-SECRET",
            "metadata": {"Cookie": "private-cookie", "token": "sk-proj-abcdefghijklmnop"},
        }
    ).encode()
    payload = normalize_exchange(_exchange(request=request), max_body_bytes=4096)
    assert b"TOP-SECRET" not in payload
    assert b"private-cookie" not in payload
    assert b"sk-proj-" not in payload
    attributes = _attributes(_exchange(request=request))
    assert attributes["gen_ai.usage.input_tokens"] == 3
    assert attributes["gen_ai.usage.output_tokens"] == 7


def test_cancelled_sse_keeps_request_and_complete_events_without_inventing_usage() -> None:
    """Retain interrupted stream evidence without manufacturing token counts."""
    body = b'data: {"type":"response.output_text.delta","delta":"partial"}\n\ndata: {"type":'
    attributes = _attributes(
        _exchange(response=body, response_content_type="text/event-stream", failed=True)
    )
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "partial" in str(attributes["exp.capture.response"])
    assert attributes["exp.capture.interrupted"] is True


def test_anthropic_stream_merges_tool_arguments_and_usage() -> None:
    """Reconstruct Anthropic tool input and token counts from stream events."""
    events = [
        {
            "type": "message_start",
            "message": {"model": "claude-test", "usage": {"input_tokens": 9}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "name": "f", "id": "t", "input": {}},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"partial_json": '{"x":1}'}},
        {"type": "message_delta", "usage": {"output_tokens": 4}},
    ]
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    attributes = _attributes(
        _exchange(
            protocol="messages",
            request=b'{"model":"claude-test","messages":[{"role":"user","content":"go"}]}',
            response=body,
            response_content_type="text/event-stream",
        )
    )
    raw_response = attributes["exp.capture.response"]
    assert isinstance(raw_response, str)
    response = json.loads(raw_response)
    assert response["content"][0]["input"] == {"x": 1}
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.output_tokens"] == 4


@pytest.mark.parametrize("encoding", ["br", "zstd"])
def test_compressed_copies_are_bounded(encoding: str) -> None:
    """Decode valid compressed bodies and reject decompression beyond the cap."""
    raw = _exchange().request
    compressed = (
        brotli.compress(raw) if encoding == "br" else zstandard.ZstdCompressor().compress(raw)
    )
    assert (
        _attributes(_exchange(request=compressed, request_encoding=encoding))[
            "gen_ai.request.model"
        ]
        == "gpt-test"
    )
    expanded = b"x" * 100000
    bomb = (
        brotli.compress(expanded)
        if encoding == "br"
        else zstandard.ZstdCompressor().compress(expanded)
    )
    with pytest.raises(ValueError, match="limit"):
        normalize_exchange(_exchange(request=bomb, request_encoding=encoding), max_body_bytes=4096)


def test_paths_exclude_login_billing_and_unrelated_traffic() -> None:
    """Select inference endpoints while excluding authentication and other traffic."""
    assert capture_protocol("POST", "/backend-api/codex/responses?key=secret") == "responses"
    assert capture_protocol("GET", "/v1/responses") == "responses"
    for path in ("/oauth/token", "/v1/messages/batches", "/backend-api/accounts", "/login"):
        assert capture_protocol("POST", path) is None


def test_chat_stream_reassembles_tool_arguments_and_final_usage() -> None:
    """Join streamed chat tool arguments and retain final provider token counts."""
    events = [
        {
            "model": "chat-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {"name": "f", "arguments": '{"x":'},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        },
    ]
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    attributes = _attributes(
        _exchange(
            protocol="chat",
            request=b'{"model":"chat-test","messages":[{"role":"user","content":"go"}]}',
            response=body,
            response_content_type="text/event-stream",
        )
    )
    raw = attributes["exp.capture.response"]
    assert isinstance(raw, str)
    response = json.loads(raw)
    assert response["choices"][0]["message"]["tool_calls"][0]["function"] == {
        "name": "f",
        "arguments": '{"x":1}',
    }
    assert attributes["gen_ai.usage.input_tokens"] == 2
    assert attributes["gen_ai.usage.output_tokens"] == 3


def test_provider_stream_errors_are_failed_spans_even_with_http_200() -> None:
    """Mark provider stream failures independently of the HTTP status code."""
    body = b'data: {"type":"error","error":{"message":"provider unavailable"}}\n\n'
    payload = json.loads(
        normalize_exchange(
            _exchange(
                protocol="messages",
                response=body,
                response_content_type="text/event-stream",
            ),
            max_body_bytes=4096,
        )
    )
    span = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["status"]["code"] == 2


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "content_filter"])
def test_chat_refusals_preserve_text_and_finish_reason_as_failed_spans(
    streamed: bool, finish_reason: str
) -> None:
    """Keep explicit refusal evidence for streamed and ordinary Chat Completions."""
    message = {"role": "assistant", "content": None, "refusal": "Cannot provide that."}
    if streamed:
        events = [
            {"choices": [{"index": 0, "delta": {"refusal": "Cannot provide "}}]},
            {"choices": [{"index": 0, "delta": {"refusal": "that."}}]},
            {"choices": [{"index": 0, "finish_reason": finish_reason}]},
        ]
        body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    else:
        body = json.dumps(
            {"choices": [{"message": message, "finish_reason": finish_reason}]}
        ).encode()
    exchange = _exchange(
        protocol="chat",
        response=body,
        response_content_type="text/event-stream" if streamed else "application/json",
    )
    attributes = _attributes(exchange)
    response = json.loads(str(attributes["exp.capture.response"]))
    assert response["choices"][0]["message"]["refusal"] == message["refusal"]
    assert response["choices"][0]["finish_reason"] == finish_reason
    assert attributes["exp.capture.refused"] is True
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096))
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"] == 2


def test_chat_content_filter_is_a_refusal_without_text_but_ordinary_text_is_not() -> None:
    """Use provider refusal fields rather than guessing from model-authored prose."""
    filtered = _exchange(
        protocol="chat",
        response=b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"content_filter"}]}\n\n',
        response_content_type="text/event-stream",
    )
    assert _attributes(filtered)["exp.capture.refused"] is True
    ordinary = _exchange(
        protocol="chat",
        response=(
            b'{"choices":[{"message":{"role":"assistant","content":"I refuse"},'
            b'"finish_reason":"stop"}]}'
        ),
    )
    assert _attributes(ordinary)["exp.capture.refused"] is False


@pytest.mark.parametrize("protocol", ["responses", "messages"])
@pytest.mark.parametrize("streamed", [False, True])
def test_responses_and_messages_refusals_preserve_provider_evidence(
    protocol: str, streamed: bool
) -> None:
    """Treat explicit Responses refusal blocks and Anthropic refusal stops consistently."""
    if protocol == "responses":
        response = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "Cannot."}]}],
        }
        events = [{"type": "response.completed", "response": response}]
    else:
        response = {"content": [{"type": "text", "text": "Cannot."}], "stop_reason": "refusal"}
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "Cannot."},
            },
            {"type": "message_delta", "delta": {"stop_reason": "refusal"}},
        ]
    body = (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        if streamed
        else json.dumps(response).encode()
    )
    exchange = _exchange(
        protocol=protocol,
        response=body,
        response_content_type="text/event-stream" if streamed else "application/json",
    )
    attributes = _attributes(exchange)
    assert "Cannot." in str(attributes["gen_ai.output.messages"])
    assert attributes["exp.capture.refused"] is True
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096))
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"] == 2
