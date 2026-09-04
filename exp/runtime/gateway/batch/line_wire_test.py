"""Tests for per-line wire shaping: Anthropic params, result rendering, usage."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.batch.contracts import BatchLine, BatchSubmitError, BatchSurface
from exp.runtime.gateway.batch.line_wire import (
    anthropic_line_params,
    anthropic_result_body,
    line_usage,
)
from exp.runtime.models.providers.errors import ProviderResponseError


def _object(value: JsonValue) -> JsonObject:
    """Narrow one JSON value to an object for assertions."""
    assert isinstance(value, dict)
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    """Narrow one JSON value to an array for assertions."""
    assert isinstance(value, list)
    return value


def _choice(body: JsonObject) -> JsonObject:
    """Return choices[0] of one chat completion body."""
    return _object(_array(body["choices"])[0])


def _line(surface: BatchSurface, body: JsonObject, custom_id: str = "line-0") -> BatchLine:
    """Build one accepted line on the given surface."""
    return BatchLine(
        custom_id=custom_id,
        surface=surface,
        model="claude-haiku-4.5-batch",
        provider_model="claude-haiku-4-5",
        body=body,
        estimated_input_tokens=8,
        maximum_output_tokens=64,
    )


_CHAT_BODY: JsonObject = {
    "messages": [
        {"role": "system", "content": "Answer tersely."},
        {"role": "user", "content": "Weather in Zürich?"},
    ],
    "max_tokens": 64,
    "temperature": 0.2,
    "stop": ["END"],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ],
    "tool_choice": "required",
}

_ANTHROPIC_MESSAGE: JsonObject = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Bern"}},
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 15,
        "cache_creation_input_tokens": 4501,
        "cache_read_input_tokens": 0,
        "output_tokens": 4,
    },
}


def test_chat_line_becomes_messages_params_through_the_sync_builder() -> None:
    """System rides top-level, tools become input_schema, required is any, stop maps,
    the model is the provider id, and the streaming flag is gone."""
    params = anthropic_line_params(_line("/v1/chat/completions", _CHAT_BODY))
    assert params["model"] == "claude-haiku-4-5"
    assert params["system"] == "Answer tersely."
    assert params["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Zürich?"}]}
    ]
    assert params["max_tokens"] == 64
    assert params["temperature"] == 0.2
    assert params["stop_sequences"] == ["END"]
    assert params["tools"] == [
        {
            "name": "lookup",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "description": "Look up weather",
        }
    ]
    assert params["tool_choice"] == {"type": "any"}
    assert "stream" not in params


def test_responses_line_becomes_messages_params() -> None:
    """A Responses body decodes through the shared decoder onto the same wire."""
    params = anthropic_line_params(
        _line("/v1/responses", {"input": "Say hi", "max_output_tokens": 32})
    )
    assert params["model"] == "claude-haiku-4-5"
    assert params["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}]
    assert params["max_tokens"] == 32
    assert "stream" not in params


def test_messages_line_travels_verbatim_under_the_provider_model() -> None:
    """A Messages line is already on this wire; only the model id is rebound."""
    body: JsonObject = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    assert anthropic_line_params(_line("/v1/messages", body)) == {
        **body,
        "model": "claude-haiku-4-5",
    }


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"messages": [{"role": "user", "content": "hi"}], "bogus": 1}, "not a valid"),
        ({"messages": [{"role": "user", "content": "hi"}], "stream": True}, "sets stream"),
        ({"messages": [], "max_tokens": 4}, "not a valid"),
    ],
)
def test_untranslatable_chat_lines_are_submit_rejections(body: JsonObject, match: str) -> None:
    """Protocol failures and stream requests are per-line submit errors with the cause."""
    with pytest.raises(BatchSubmitError, match=match):
        anthropic_line_params(_line("/v1/chat/completions", body))


@pytest.mark.parametrize(
    ("surface", "body", "match"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "logprobs": True},
            "sets logprobs",
        ),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "logprobs": True, "top_logprobs": 3},
            "top_logprobs",
        ),
        # The shared decoder already refuses this one; it must still surface as
        # the same per-line submit rejection.
        ("/v1/responses", {"input": "hi", "top_logprobs": 2}, "top_logprobs"),
        (
            "/v1/responses",
            {"input": "hi", "tools": [{"type": "web_search"}]},
            "native Responses tool declarations",
        ),
    ],
)
def test_controls_the_messages_wire_cannot_carry_are_submit_rejections(
    surface: BatchSurface, body: JsonObject, match: str
) -> None:
    """logprobs and native Responses tools are refused per line at submit, as the
    synchronous lane refuses them before dispatch, never silently dropped."""
    with pytest.raises(BatchSubmitError, match=match):
        anthropic_line_params(_line(surface, body))


def test_responses_continuation_is_refused_in_a_batch() -> None:
    """A batch line carries the whole conversation; previous_response_id has no meaning."""
    with pytest.raises(BatchSubmitError, match="previous_response_id"):
        anthropic_line_params(
            _line("/v1/responses", {"input": "more", "previous_response_id": "resp_1"})
        )


def test_message_result_renders_as_a_chat_completion_with_cached_usage() -> None:
    """Text and tool_use become choices[0].message, finish_reason is tool_calls, and the
    usage is chat-shaped with the cache legs folded into prompt_tokens."""
    body = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY),
        _ANTHROPIC_MESSAGE,
        request_id="batch_x:line-0",
        created_at=1_700_000_000.0,
    )
    assert body["object"] == "chat.completion"
    assert body["model"] == "claude-haiku-4.5-batch"
    assert body["created"] == 1_700_000_000
    choice = _choice(body)
    assert choice["finish_reason"] == "tool_calls"
    message = _object(choice["message"])
    assert message["content"] == "Checking."
    assert message["tool_calls"] == [
        {
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city": "Bern"}'},
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 4516,
        "completion_tokens": 4,
        "total_tokens": 4520,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": None,
    }


def test_message_result_maps_max_tokens_and_refusal_finish_reasons() -> None:
    """max_tokens ends the chat turn `length`; a refusal ends it `content_filter`."""
    truncated = {**_ANTHROPIC_MESSAGE, "content": [{"type": "text", "text": "x"}]}
    truncated["stop_reason"] = "max_tokens"
    body = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), truncated, request_id="r", created_at=0.0
    )
    assert _choice(body)["finish_reason"] == "length"
    refused = {**_ANTHROPIC_MESSAGE, "content": [], "stop_reason": "refusal"}
    body = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), refused, request_id="r", created_at=0.0
    )
    assert _choice(body)["finish_reason"] == "content_filter"
    assert _object(_choice(body)["message"])["content"] is None


def test_server_tool_blocks_render_as_the_cited_answer_not_a_failure() -> None:
    """Provider-run web search blocks are not client tool calls: the chat completion
    carries the answer text, no tool_calls, and finish_reason stop."""
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "weather"},
            },
            {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
            {"type": "text", "text": "Sunny in Bern."},
        ],
        "stop_reason": "end_turn",
    }
    body = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), message, request_id="r", created_at=0.0
    )
    choice = _choice(body)
    assert choice["finish_reason"] == "stop"
    rendered = _object(choice["message"])
    assert rendered["content"] == "Sunny in Bern."
    assert rendered["tool_calls"] is None


def test_message_result_renders_as_a_response_object_on_the_responses_surface() -> None:
    """A Responses line receives the synchronous lane's response object."""
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
    }
    body = anthropic_result_body(
        _line("/v1/responses", {"input": "Say hi"}), message, request_id="r", created_at=0.0
    )
    assert body["object"] == "response"
    assert body["status"] == "completed"
    texts = [
        _object(part)["text"]
        for item in map(_object, _array(body["output"]))
        if item.get("type") == "message"
        for part in _array(item["content"])
        if _object(part).get("type") == "output_text"
    ]
    assert texts == ["hi"]
    assert _object(body["usage"])["input_tokens"] == 4516


def test_thinking_blocks_render_as_reasoning_on_responses_and_vanish_on_chat() -> None:
    """A thinking block's text reaches a Responses caller as a reasoning summary (what
    they paid for); the chat shape has no slot for it and carries only the answer."""
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [
            {"type": "thinking", "thinking": "Consider the forecast.", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "Sunny."},
        ],
        "stop_reason": "end_turn",
    }
    responses = anthropic_result_body(
        _line("/v1/responses", {"input": "Weather?"}), message, request_id="r", created_at=0.0
    )
    items = [_object(item) for item in _array(responses["output"])]
    reasoning = [item for item in items if item.get("type") == "reasoning"]
    assert len(reasoning) == 1
    summaries = [_object(part)["text"] for part in _array(reasoning[0]["summary"])]
    assert summaries == ["Consider the forecast."]
    texts = [
        _object(part)["text"]
        for item in items
        if item.get("type") == "message"
        for part in _array(item["content"])
    ]
    assert texts == ["Sunny."]
    chat = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), message, request_id="r", created_at=0.0
    )
    assert _object(_choice(chat)["message"])["content"] == "Sunny."
    bad = {**_ANTHROPIC_MESSAGE, "content": [{"type": "thinking", "thinking": 4}]}
    with pytest.raises(ProviderResponseError, match="thinking must be text"):
        anthropic_result_body(
            _line("/v1/chat/completions", _CHAT_BODY), bad, request_id="r", created_at=0.0
        )


def test_anthropic_usage_without_cache_legs_reports_a_zero_cached_leg() -> None:
    """The Anthropic wire always has a cache-read leg; when nothing was read the
    rendered chat usage says cached_tokens: 0, as the synchronous normalizer does."""
    plain = line_usage({"usage": {"input_tokens": 9, "output_tokens": 2}})
    assert (plain.input_tokens, plain.output_tokens) == (9, 2)
    assert plain.cached_input_tokens == 0 and plain.cache_creation_input_tokens is None
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 9, "output_tokens": 2},
    }
    chat = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), message, request_id="r", created_at=0.0
    )
    assert _object(chat["usage"])["prompt_tokens_details"] == {"cached_tokens": 0}
    # OpenAI-shaped usage without a details object keeps the subset unknown.
    assert (
        line_usage({"usage": {"prompt_tokens": 3, "completion_tokens": 1}}).cached_input_tokens
        is None
    )
    responses_shaped = line_usage(
        {"usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}
    )
    assert responses_shaped.cached_input_tokens is None


def test_messages_surface_result_is_the_message_verbatim() -> None:
    """No translation on the native surface."""
    line = _line("/v1/messages", {"messages": [{"role": "user", "content": "hi"}]})
    assert (
        anthropic_result_body(line, _ANTHROPIC_MESSAGE, request_id="r", created_at=0.0)
        is _ANTHROPIC_MESSAGE
    )


def test_line_usage_reads_every_wire_shape() -> None:
    """Anthropic legs fold into input; chat and responses subsets pass through by name."""
    anthropic = line_usage(_ANTHROPIC_MESSAGE)
    assert (anthropic.input_tokens, anthropic.output_tokens) == (4516, 4)
    assert anthropic.cached_input_tokens == 0 and anthropic.cache_creation_input_tokens == 4501
    chat = line_usage(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 148,
                "prompt_tokens_details": {"cached_tokens": 60},
                "completion_tokens_details": {"reasoning_tokens": 8},
            }
        }
    )
    assert (chat.input_tokens, chat.output_tokens) == (100, 48)
    assert (chat.cached_input_tokens, chat.reasoning_tokens) == (60, 8)
    assert line_usage({"usage": "nope"}).input_tokens == 0
    assert line_usage(None).output_tokens == 0
    # A present but malformed count is a malformed provider response (the
    # synchronous require_integer contract), never a silent zero.
    for malformed in (
        {"prompt_tokens": -1, "completion_tokens": 2},
        {"prompt_tokens": 1, "completion_tokens": True},
        {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": "3"},
        {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 1.5},
        },
    ):
        with pytest.raises(ProviderResponseError, match="non-negative integer"):
            line_usage({"usage": malformed})


def test_line_without_an_output_ceiling_carries_the_reserved_ceiling() -> None:
    """A body naming no max_tokens sends the line's reserved ceiling, never the
    payload builder's generic default, so the wire cannot exceed the reservation."""
    body: JsonObject = {"messages": [{"role": "user", "content": "hi"}]}
    line = _line("/v1/chat/completions", body)
    assert anthropic_line_params(line)["max_tokens"] == line.maximum_output_tokens == 64
    explicit = _line("/v1/chat/completions", {**body, "max_tokens": 7})
    assert anthropic_line_params(explicit)["max_tokens"] == 7


def test_unsupported_reasoning_effort_is_a_submit_rejection() -> None:
    """An effort no Anthropic model carries is reported per line at submit, never
    raised past the engine as an unclassified error."""
    body: JsonObject = {
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "ultra",
    }
    # An effort-generation model (adaptive thinking) is where the effort ladder
    # is checked; a budget-only model realizes any effort as a token budget.
    line = _line("/v1/chat/completions", body).model_copy(
        update={"provider_model": "claude-sonnet-4-5"}
    )
    with pytest.raises(BatchSubmitError, match="cannot be served on the Anthropic wire"):
        anthropic_line_params(line)


def test_refusal_with_partial_text_renders_content_free_on_both_surfaces() -> None:
    """A refusal stop after some text is the whole visible turn: the chat message
    carries the provider's refusal text and no content, and the Responses object
    carries one refusal part (text and refusal never mix on that wire)."""
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [
            {"type": "text", "text": "I was going to say"},
            {"type": "refusal", "refusal": "I can't help with that."},
        ],
        "stop_reason": "refusal",
    }
    chat = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), message, request_id="r", created_at=0.0
    )
    choice = _choice(chat)
    assert choice["finish_reason"] == "content_filter"
    rendered = _object(choice["message"])
    assert rendered["content"] is None
    assert rendered["refusal"] == "I can't help with that."
    assert rendered["tool_calls"] is None
    responses = anthropic_result_body(
        _line("/v1/responses", {"input": "Say hi"}), message, request_id="r", created_at=0.0
    )
    parts = [
        _object(part)
        for item in map(_object, _array(responses["output"]))
        if item.get("type") == "message"
        for part in _array(item["content"])
    ]
    assert [part["type"] for part in parts] == ["refusal"]
    assert parts[0]["refusal"] == "I can't help with that."


def test_unknown_block_kinds_are_skipped_not_rejected() -> None:
    """Block kinds with no output on these surfaces (MCP tool traffic, future
    kinds) are skipped exactly as the streaming normalizer skips them."""
    message = {
        **_ANTHROPIC_MESSAGE,
        "content": [
            {"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "search", "input": {}},
            {"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "content": []},
            {"type": "compaction", "content": "..."},
            {"type": "text", "text": "Done."},
        ],
        "stop_reason": "end_turn",
    }
    body = anthropic_result_body(
        _line("/v1/chat/completions", _CHAT_BODY), message, request_id="r", created_at=0.0
    )
    assert _object(_choice(body)["message"])["content"] == "Done."


def test_malformed_blocks_are_a_typed_rendering_failure() -> None:
    """A tool_use with an empty id (or a non-text refusal) is reported as a
    provider response error, never as a bare validation error the results
    parser cannot classify."""
    empty_id = {
        **_ANTHROPIC_MESSAGE,
        "content": [{"type": "tool_use", "id": "", "name": "lookup", "input": {}}],
    }
    with pytest.raises(ProviderResponseError, match="tool_use is malformed"):
        anthropic_result_body(
            _line("/v1/chat/completions", _CHAT_BODY), empty_id, request_id="r", created_at=0.0
        )
    bad_refusal = {**_ANTHROPIC_MESSAGE, "content": [{"type": "refusal", "refusal": 3}]}
    with pytest.raises(ProviderResponseError, match="refusal must be text"):
        anthropic_result_body(
            _line("/v1/chat/completions", _CHAT_BODY), bad_refusal, request_id="r", created_at=0.0
        )
    # A block with no discriminator is malformed output, not a hidden kind:
    # it must never render as a successful (possibly empty) completion.
    for untyped in ({"text": "lost"}, {"type": 7, "text": "lost"}, {"type": "", "text": "lost"}):
        unnamed = {**_ANTHROPIC_MESSAGE, "content": [untyped]}
        with pytest.raises(ProviderResponseError, match="type must be text"):
            anthropic_result_body(
                _line("/v1/chat/completions", _CHAT_BODY), unnamed, request_id="r", created_at=0.0
            )
