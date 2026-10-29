"""Protocol normalization preserves evidence while removing transport credentials."""

import json
import time
import zlib
from dataclasses import replace

import brotli
import pytest
import zstandard

from exp.common.core.artifacts import JsonObject, JsonValue, SourceIdentity
from exp.common.traces.ingest.otlp import normalize_otlp_payload
from exp.runtime.capture.normalization import CapturedExchange, capture_protocol, normalize_exchange


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
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096)[0])
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
    payload = normalize_exchange(_exchange(request=request), max_body_bytes=4096)[0]
    assert b"TOP-SECRET" not in payload
    assert b"private-cookie" not in payload
    assert b"sk-proj-" not in payload
    attributes = _attributes(_exchange(request=request))
    assert attributes["gen_ai.usage.input_tokens"] == 3
    assert attributes["gen_ai.usage.output_tokens"] == 7


@pytest.mark.parametrize("protocol", ["responses", "chat"])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("array_arguments", [False, True])
def test_json_tool_arguments_redact_known_credentials_in_every_uploaded_copy(
    protocol: str, streamed: bool, array_arguments: bool
) -> None:
    """Request history and returned tool arguments redact known keys inside JSON strings."""
    canary = "SYNTHETIC_CAPTURE_PASSWORD_CANARY"
    arguments: JsonValue = {
        "password": canary,
        "nested": [{"api_key": canary}],
        "query": "retain useful tool content",
        "events": [
            "login",
            "logout",
            {"arguments": '{"city":"SF"}', "partial_json": "literal value"},
        ],
    }
    if array_arguments:
        arguments = [arguments]
    encoded = json.dumps(arguments)
    if protocol == "responses":
        tool: JsonObject = {"type": "function_call", "name": "lookup", "arguments": encoded}
        request: JsonObject = {"model": "gpt-test", "input": [tool]}
        response: JsonObject = {"model": "gpt-test", "output": [tool]}
        events = [{"type": "response.completed", "response": response}]
    else:
        tool = {"type": "function", "function": {"name": "lookup", "arguments": encoded}}
        message: JsonObject = {"role": "assistant", "tool_calls": [tool]}
        request = {"model": "gpt-test", "messages": [message]}
        response = {"model": "gpt-test", "choices": [{"message": message}]}
        midpoint = len(encoded) // 2
        events = [
            {
                "model": "gpt-test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "lookup", "arguments": encoded[:midpoint]},
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
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": encoded[midpoint:]}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
    body = (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        if streamed
        else json.dumps(response).encode()
    )
    exchange = _exchange(
        protocol=protocol,
        request=json.dumps(request).encode(),
        response=body,
        response_content_type="text/event-stream" if streamed else "application/json",
    )
    payload = normalize_exchange(exchange, max_body_bytes=4096)[0]
    assert canary.encode() not in payload
    attributes = _attributes(exchange)
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        messages = json.loads(str(attributes[key]))
        captured_tool = messages[0] if protocol == "responses" else messages[0]["tool_calls"][0]
        captured_arguments = (
            captured_tool["arguments"]
            if protocol == "responses"
            else captured_tool["function"]["arguments"]
        )
        assert isinstance(captured_arguments, str)
        decoded = json.loads(captured_arguments)
        if array_arguments:
            decoded = decoded[0]
        assert decoded == {
            "password": "[REDACTED]",
            "nested": [{"api_key": "[REDACTED]"}],
            "query": "retain useful tool content",
            "events": [
                "login",
                "logout",
                {"arguments": '{"city":"SF"}', "partial_json": "literal value"},
            ],
        }


@pytest.mark.parametrize(
    "arguments",
    [' { "query" : ["keep formatting", 2] }\n', '"a scalar"'],
)
def test_benign_json_tool_arguments_keep_original_string(arguments: str) -> None:
    """Argument redaction never reformats benign valid JSON."""
    request = {"model": "gpt-test", "input": [{"type": "function_call", "arguments": arguments}]}
    attributes = _attributes(_exchange(request=json.dumps(request).encode()))
    assert json.loads(str(attributes["exp.capture.request"]))["input"][0]["arguments"] == arguments


@pytest.mark.parametrize(
    "arguments", ['{"password":"ordinary-secret', "Bearer ordinary-secret", ""]
)
def test_malformed_tool_arguments_are_redacted(arguments: str) -> None:
    """Unreadable arguments cannot skip field-aware credential redaction."""
    request = {
        "model": "gpt-test",
        "input": [{"type": "function_call", "arguments": arguments}],
    }
    exchange = _exchange(request=json.dumps(request).encode())
    payload = normalize_exchange(exchange, max_body_bytes=4096)[0]
    assert b"ordinary-secret" not in payload
    attributes = _attributes(exchange)
    captured = json.loads(str(attributes["exp.capture.request"]))
    assert captured["input"][0]["arguments"] == "[REDACTED_INVALID_TOOL_ARGUMENTS]"


@pytest.mark.parametrize("protocol", ["responses", "chat", "messages"])
def test_interrupted_tool_streams_redact_partial_credentials_everywhere(protocol: str) -> None:
    """Incomplete tool JSON and isolated argument fragments cannot survive in raw events."""
    canary = "SYNTHETIC_PARTIAL_PASSWORD_CANARY"
    fragments = ['{"password":"', canary]
    events: list[JsonObject]
    if protocol == "responses":
        events = [
            {"type": "response.output_text.delta", "delta": "retain ordinary output"},
        ]
        for fragment in fragments:
            events.append({"type": "response.function_call_arguments.delta", "delta": fragment})
    elif protocol == "chat":
        events = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": "retain ordinary output Bearer "
                            if index == 0
                            else "SYNTHETIC_SPLIT_SECRET",
                            "tool_calls": [{"index": 0, "function": {"arguments": fragment}}],
                        },
                    }
                ]
            }
            for index, fragment in enumerate(fragments)
        ]
    else:
        events = [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "retain ordinary output Bearer "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "SYNTHETIC_SPLIT_SECRET"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tool", "name": "lookup", "input": {}},
            },
        ]
        for fragment in fragments:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                }
            )
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    body += b'data: {"password":"' + canary.encode() + b'",\n\n'
    payload = normalize_exchange(
        _exchange(
            protocol=protocol,
            response=body,
            response_content_type="text/event-stream",
            failed=True,
        ),
        max_body_bytes=4096,
    )[0]
    assert canary.encode() not in payload
    assert b"SYNTHETIC_SPLIT_SECRET" not in payload
    assert b"[REDACTED_INVALID_TOOL_ARGUMENTS]" in payload
    assert b"retain ordinary output" in payload


def test_custom_tool_freeform_input_remains_captured() -> None:
    """Custom tool input uses its own freeform field and is unaffected by JSON argument rules."""
    request = {
        "model": "gpt-test",
        "input": [{"type": "custom_tool_call", "name": "shell", "input": "echo hello"}],
    }
    attributes = _attributes(_exchange(request=json.dumps(request).encode()))
    assert json.loads(str(attributes["exp.capture.request"]))["input"][0]["input"] == "echo hello"


@pytest.mark.parametrize("nesting", [60, 1500])
def test_deep_json_tool_arguments_are_bounded_and_redacted(nesting: int) -> None:
    """Parsed deep values and parser recursion failures cannot retain known credentials."""
    canary = "SYNTHETIC_DEEP_CREDENTIAL_CANARY"
    arguments = "[" * nesting + json.dumps({"password": canary}) + "]" * nesting
    request = {"model": "gpt-test", "input": [{"type": "function_call", "arguments": arguments}]}
    payload = normalize_exchange(
        _exchange(request=json.dumps(request).encode()), max_body_bytes=4096
    )[0]
    assert canary.encode() not in payload
    assert b"[REDACTED_DEEP_VALUE]" in payload


def test_cancelled_sse_keeps_request_and_complete_events_without_inventing_usage() -> None:
    """Retain interrupted stream evidence without manufacturing token counts."""
    body = b'data: {"type":"response.output_text.delta","delta":"partial"}\n\ndata: {"type":'
    attributes = _attributes(
        _exchange(response=body, response_content_type="text/event-stream", failed=True)
    )
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "partial" in str(attributes["exp.capture.events"])
    assert attributes["exp.capture.interrupted"] is True


@pytest.mark.parametrize("streamed", [True, False])
@pytest.mark.parametrize("completed", [True, False])
def test_terminal_response_survives_later_transport_failure(
    streamed: bool, completed: bool
) -> None:
    """A disconnect after provider completion cannot turn a finished model call into a failure."""
    response = {
        "id": "resp_test_completion",
        "status": "completed" if completed else "incomplete",
        "model": "test",
        "output": [],
        "metadata": {"record_id": 18446744073709551617},
        "usage": {"input_tokens": 23, "output_tokens": 17},
    }
    body = json.dumps(response).encode()
    if streamed:
        event = {"type": f"response.{response['status']}", "response": response}
        body = b"data: " + json.dumps(event).encode() + b"\n\n"
    exchange = _exchange(
        response=body,
        response_content_type="text/event-stream" if streamed else "application/json",
        failed=True,
    )
    attributes = _attributes(exchange)
    assert attributes["exp.capture.transport_error"] is True
    assert attributes["exp.capture.interrupted"] is not completed
    assert attributes["gen_ai.usage.input_tokens"] == 23
    assert attributes["gen_ai.usage.output_tokens"] == 17
    assert json.loads(str(attributes["exp.capture.response"]))["metadata"] == response["metadata"]
    assert "exp.capture.events" not in attributes
    span = json.loads(normalize_exchange(exchange, max_body_bytes=4096)[0])["resourceSpans"][0][
        "scopeSpans"
    ][0]["spans"][0]
    assert span["status"]["code"] == (1 if completed else 2)


@pytest.mark.parametrize("protocol", ["chat", "messages"])
@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("provider_error", [False, True])
def test_stream_completion_and_provider_errors_remain_separate_from_transport_errors(
    protocol: str, completed: bool, provider_error: bool
) -> None:
    """Terminal SSE events survive late disconnects without masking actual provider failures."""
    events: list[JsonObject]
    if protocol == "messages":
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
            {"type": "message_delta", "usage": {"output_tokens": 7}},
        ]
        if completed:
            events.append({"type": "message_stop"})
    else:
        events = [{"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 7}}]
    if provider_error:
        events.append({"type": "error", "error": {"type": "overloaded_error"}})
    body = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    if protocol == "chat" and completed:
        body += b"data: [DONE]\n\n"
    exchange = _exchange(
        protocol=protocol, response=body, response_content_type="text/event-stream", failed=True
    )
    attributes = _attributes(exchange)
    assert attributes["exp.capture.completed"] is completed
    assert attributes["exp.capture.interrupted"] is not completed
    assert attributes["gen_ai.usage.input_tokens"] == 3
    assert attributes["gen_ai.usage.output_tokens"] == 7
    span = json.loads(normalize_exchange(exchange, max_body_bytes=4096)[0])["resourceSpans"][0][
        "scopeSpans"
    ][0]["spans"][0]
    assert span["status"]["code"] == (2 if provider_error or not completed else 1)


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "zstd"])
@pytest.mark.parametrize("completed", [False, True])
def test_interrupted_compressed_sse_retains_complete_events_without_compression_footer(
    encoding: str, completed: bool
) -> None:
    """A missing compression footer must not discard already received model output or usage."""
    body = b'data: {"type":"response.output_text.delta","delta":"retained text"}\n\n'
    if completed:
        body += b'data: {"type":"response.completed","response":{"output":[],'
        body += b'"usage":{"input_tokens":3,"output_tokens":7}}}\n\n'
    if encoding == "br":
        brotli_compressor = brotli.Compressor()
        compressed = brotli_compressor.process(body) + brotli_compressor.flush()
    elif encoding == "zstd":
        zstd_compressor = zstandard.ZstdCompressor().compressobj()
        compressed = zstd_compressor.compress(body) + zstd_compressor.flush(
            zstandard.COMPRESSOBJ_FLUSH_BLOCK
        )
    else:
        compressor = zlib.compressobj(wbits=31 if encoding == "gzip" else 15)
        compressed = compressor.compress(body) + compressor.flush(zlib.Z_SYNC_FLUSH)
    exchange = _exchange(
        response=compressed,
        response_encoding=encoding,
        response_content_type="text/event-stream",
        failed=True,
    )
    attributes = _attributes(exchange)
    assert attributes["exp.capture.interrupted"] is not completed
    assert attributes["exp.capture.completed"] is completed
    if completed:
        assert attributes["gen_ai.usage.input_tokens"] == 3
        assert attributes["gen_ai.usage.output_tokens"] == 7
    else:
        assert "retained text" in str(attributes["exp.capture.events"])
        assert "gen_ai.usage.input_tokens" not in attributes


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br"])
@pytest.mark.parametrize("oversized", [False, True])
def test_partial_compression_never_invents_a_terminal_event_or_bypasses_limits(
    encoding: str, oversized: bool
) -> None:
    """Incomplete SSE records and excessive decompression cannot produce reported usage."""
    body = b'data: {"type":"response.completed","response":{"usage":'
    body += b'{"input_tokens":3,"output_tokens":7}}}'
    if oversized:
        body += b"\n\n:" + b"x" * 5000 + b"\n\n"
    if encoding == "br":
        brotli_compressor = brotli.Compressor()
        compressed = brotli_compressor.process(body) + brotli_compressor.flush()
    else:
        compressor = zlib.compressobj(wbits=31 if encoding == "gzip" else 15)
        compressed = compressor.compress(body) + compressor.flush(zlib.Z_SYNC_FLUSH)
    attributes = _attributes(
        _exchange(
            response=compressed,
            response_encoding=encoding,
            response_content_type="text/event-stream",
            failed=True,
        )
    )
    assert attributes["exp.capture.completed"] is False
    assert attributes["exp.capture.interrupted"] is True
    assert "gen_ai.usage.input_tokens" not in attributes


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br"])
def test_complete_http_response_still_requires_a_complete_compression_frame(encoding: str) -> None:
    """Partial decompression is restricted to transport-interrupted SSE captures."""
    body = b'data: {"type":"response.completed","response":{"output":[]}}\n\n'
    if encoding == "br":
        brotli_compressor = brotli.Compressor()
        compressed = brotli_compressor.process(body) + brotli_compressor.flush()
    else:
        compressor = zlib.compressobj(wbits=31 if encoding == "gzip" else 15)
        compressed = compressor.compress(body) + compressor.flush(zlib.Z_SYNC_FLUSH)
    with pytest.raises(ValueError, match="incomplete"):
        _attributes(
            _exchange(
                response=compressed,
                response_encoding=encoding,
                response_content_type="text/event-stream",
                failed=False,
            )
        )


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
        {"type": "content_block_stop", "index": 99},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"partial_json": '{"x":18446744073709551617,"password":"TOOL_SECRET"}'},
        },
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
    assert response["content"][0]["input"] == {
        "x": 18446744073709551617,
        "password": "[REDACTED]",
    }
    assert "TOOL_SECRET" not in str(attributes)
    assert "capture_input_source_json" not in str(attributes)
    assert attributes["gen_ai.usage.input_tokens"] == 9
    assert attributes["gen_ai.usage.output_tokens"] == 4


def _anthropic_usage_attributes(usage: JsonObject, *, streamed: bool) -> JsonObject:
    """Normalize the same synthetic Anthropic usage through JSON or separate SSE events."""
    if streamed:
        initial = {key: value for key, value in usage.items() if key != "output_tokens"}
        events = [
            {"type": "message_start", "message": {"usage": initial}},
            {"type": "message_delta", "usage": {"output_tokens": usage["output_tokens"]}},
        ]
        response = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    else:
        response = json.dumps({"content": [], "usage": usage}).encode()
    return _attributes(
        _exchange(
            protocol="messages",
            response=response,
            response_content_type="text/event-stream" if streamed else "application/json",
        )
    )


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    ("cache_usage", "expected_input"),
    [
        ({}, 3),
        ({"cache_creation_input_tokens": 100}, 103),
        ({"cache_read_input_tokens": 1_000}, 1_003),
        ({"cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}, 3),
        (
            {
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 1_000,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 40,
                    "ephemeral_1h_input_tokens": 60,
                },
            },
            1_103,
        ),
    ],
)
def test_anthropic_input_total_includes_cache_once_and_preserves_raw_usage(
    streamed: bool, cache_usage: JsonObject, expected_input: int
) -> None:
    """Count each Anthropic input category once without changing the copied provider breakdown."""
    usage: JsonObject = {"input_tokens": 3, "output_tokens": 7, **cache_usage}
    attributes = _anthropic_usage_attributes(usage, streamed=streamed)
    assert attributes["gen_ai.usage.input_tokens"] == expected_input
    assert attributes["gen_ai.usage.output_tokens"] == 7
    assert json.loads(str(attributes["exp.capture.response"]))["usage"] == usage


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("cache_key", ["cache_creation_input_tokens", "cache_read_input_tokens"])
@pytest.mark.parametrize("invalid", [True, -1, None, "100", 100.0])
def test_anthropic_invalid_cache_count_keeps_total_usage_unknown(
    streamed: bool, cache_key: str, invalid: JsonValue
) -> None:
    """An invalid reported cache category cannot produce a misleading complete token total."""
    usage: JsonObject = {"input_tokens": 3, "output_tokens": 7, cache_key: invalid}
    attributes = _anthropic_usage_attributes(usage, streamed=streamed)
    assert "gen_ai.usage.input_tokens" not in attributes
    assert "gen_ai.usage.output_tokens" not in attributes
    assert json.loads(str(attributes["exp.capture.response"]))["usage"] == usage


@pytest.mark.parametrize("protocol", ["responses", "chat"])
@pytest.mark.parametrize("streamed", [False, True])
def test_openai_cached_input_is_already_in_the_provider_total(
    protocol: str, streamed: bool
) -> None:
    """Preserve OpenAI's inclusive total without adding its nested cached-token breakdown."""
    if protocol == "responses":
        usage = {
            "input_tokens": 1_003,
            "output_tokens": 7,
            "input_tokens_details": {"cached_tokens": 1_000},
        }
        response = {"output": [], "usage": usage}
        events = [{"type": "response.completed", "response": response}]
    else:
        usage = {
            "prompt_tokens": 1_003,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 1_000},
        }
        response = {"choices": [], "usage": usage}
        events = [response]
    body = (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        if streamed
        else json.dumps(response).encode()
    )
    attributes = _attributes(
        _exchange(
            protocol=protocol,
            response=body,
            response_content_type="text/event-stream" if streamed else "application/json",
        )
    )
    assert attributes["gen_ai.usage.input_tokens"] == 1_003
    assert attributes["gen_ai.usage.output_tokens"] == 7
    assert json.loads(str(attributes["exp.capture.response"]))["usage"] == usage


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
        normalize_exchange(_exchange(request=bomb, request_encoding=encoding), max_body_bytes=4096)[
            0
        ]


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
    events.append({"choices": [{"delta": {"content": "unindexed"}}]})
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
        )[0]
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
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096)[0])
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
    payload = json.loads(normalize_exchange(exchange, max_body_bytes=4096)[0])
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["status"]["code"] == 2
