"""Tests for shared native request-body decoding."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.native_decode import (
    NativeDecodeError,
    decode_native_body,
    decode_native_count_tokens_body,
    decode_native_embeddings_body,
)


def test_decode_native_body_accepts_chat_and_responses() -> None:
    """Both public surfaces use the shared protocol decoders."""
    chat = decode_native_body(
        '{"model":"public-model","messages":[{"role":"user","content":"hi"}]}'
    )
    responses = decode_native_body(
        '{"model":"public-model","input":"hi"}',
        surface="responses",
    )

    assert chat.alias == "public-model"
    assert responses.request.messages[0].content == "hi"


def test_invalid_json_is_a_native_decode_error() -> None:
    """Malformed bodies keep the shared public invalid-JSON shape."""
    with pytest.raises(NativeDecodeError) as raised:
        decode_native_body("{not json")

    assert raised.value.error.status_code == 400
    assert raised.value.error.detail.code == "invalid_json"


def test_messages_surface_threads_the_caller_beta_header() -> None:
    """The Messages decode receives the caller anthropic-beta header so
    allowlisted tokens (the 1M context window) survive to dispatch."""
    decoded = decode_native_body(
        '{"model":"public-model","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}',
        surface="messages",
        anthropic_beta="claude-code-20250219,context-1m-2025-08-07",
    )
    assert decoded.request.provider_beta_tokens == ("context-1m-2025-08-07",)
    assert decoded.request.ignored_parameters == ("anthropic-beta.claude-code-20250219",)


def test_decode_native_embeddings_body_accepts_the_embeddings_surface() -> None:
    """The embeddings surface decodes through its own message-less entrypoint."""
    decoded = decode_native_embeddings_body('{"model":"text-embedding-3-small","input":"hi"}')

    assert decoded.alias == "text-embedding-3-small"
    assert decoded.request.surface == GatewayApiSurface.EMBEDDINGS
    assert decoded.request.inputs == ("hi",)


def test_decode_native_embeddings_body_rejects_malformed_bodies() -> None:
    """Invalid JSON and non-object bodies keep the shared public error shapes."""
    with pytest.raises(NativeDecodeError) as invalid_json:
        decode_native_embeddings_body("{not json")
    assert invalid_json.value.error.detail.code == "invalid_json"

    with pytest.raises(NativeDecodeError) as not_object:
        decode_native_embeddings_body("[]")
    assert not_object.value.error.status_code == 400


_LONE = "\\ud83d"
"""A lone high surrogate exactly as JSON text."""


def test_lone_surrogate_content_is_repaired_once_for_every_text_surface() -> None:
    """Chat, Responses, and Messages share one repair and one disclosure shape."""
    chat = decode_native_body(
        '{"model":"m","messages":[{"role":"user","content":"hi ' + _LONE + '"}]}'
    )
    responses = decode_native_body('{"model":"m","input":"hi ' + _LONE + '"}', surface="responses")
    messages = decode_native_body(
        '{"model":"m","max_tokens":8,"system":"be ' + _LONE + '",'
        '"messages":[{"role":"user","content":"hi"}]}',
        surface="messages",
    )
    assert chat.request.messages[0].content == "hi \ufffd"
    assert chat.request.ignored_parameters == ("messages.content->replaced(lone_surrogate)",)
    assert responses.request.messages[0].content == "hi \ufffd"
    assert responses.request.ignored_parameters == ("input->replaced(lone_surrogate)",)
    assert messages.request.ignored_parameters == ("system->replaced(lone_surrogate)",)


def test_repair_disclosures_follow_the_decoder_s_own() -> None:
    """A decoder disclosure keeps its place; the repair is appended after it."""
    decoded = decode_native_body(
        '{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi ' + _LONE + '"}]}',
        surface="messages",
        anthropic_beta="claude-code-20250219",
    )
    assert decoded.request.ignored_parameters == (
        "anthropic-beta.claude-code-20250219",
        "messages.content->replaced(lone_surrogate)",
    )


def test_a_lone_surrogate_name_is_a_field_specific_400() -> None:
    """The model names a route; it is refused by field, never rewritten."""
    with pytest.raises(NativeDecodeError) as refused:
        decode_native_body('{"model":"m' + _LONE + '","input":"hi"}', surface="responses")
    assert refused.value.error.status_code == 400
    assert refused.value.error.detail.code == "invalid_parameter"
    assert refused.value.error.detail.param == "model"


def test_a_paired_escape_is_neither_repaired_nor_disclosed() -> None:
    """A complete pair decodes to one scalar, so there is nothing to repair."""
    decoded = decode_native_body('{"model":"m","input":"\\ud83d\\ude00"}', surface="responses")
    assert decoded.request.messages[0].content == "\U0001f600"
    assert decoded.request.ignored_parameters == ()


def test_count_tokens_discloses_the_same_repair() -> None:
    """The Messages count decodes through the same repaired load."""
    decoded = decode_native_count_tokens_body(
        '{"model":"m","messages":[{"role":"user","content":"hi ' + _LONE + '"}]}'
    )
    assert decoded.request.ignored_parameters == ("messages.content->replaced(lone_surrogate)",)


def test_surfaces_without_a_disclosure_channel_refuse_lone_surrogates() -> None:
    """Embeddings cannot announce a rewrite, so the input is a typed 400, not a 500."""
    with pytest.raises(NativeDecodeError) as refused:
        decode_native_embeddings_body('{"model":"e","input":["ok","cut ' + _LONE + '"]}')
    assert refused.value.error.status_code == 400
    assert refused.value.error.detail.param == "input.1"


def test_an_escaped_backslash_spelling_a_nested_escape_is_still_repaired() -> None:
    """``\\u005c`` spells the backslash of a nested escape the raw body never shows."""
    arguments = '"{\\"q\\":\\"\\u005cud83d\\"}"'
    body = (
        '{"model":"m","messages":[{"role":"user","content":"hi"},'
        '{"role":"assistant","tool_calls":[{"id":"c","type":"function",'
        '"function":{"name":"f","arguments":' + arguments + "}}]},"
        '{"role":"tool","tool_call_id":"c","content":"r"}]}'
    )
    assert "\\ud83d" not in body
    decoded = decode_native_body(body, surface="chat")
    assert decoded.request.ignored_parameters == (
        "messages.tool_calls.function.arguments->replaced(lone_surrogate)",
    )
