"""Tests for the disclosed removal of unknown keys on conversation objects."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.capture_context import capture_context_document
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)
from exp.runtime.openai_protocol.unknown_message_fields import (
    MAXIMUM_KEY_CHARACTERS,
    MAXIMUM_NAMED_DISCLOSURES,
    DroppedKeys,
    drop_unknown_chat_message_fields,
    key_token,
)

_WORKBUDDY_HISTORY: list[JsonValue] = [
    {"role": "system", "content": "You are helpful.", "conversationRequestId": "req-0"},
    {"role": "user", "content": "hi", "conversationRequestId": "req-1"},
    {"role": "assistant", "content": "hello", "conversationRequestId": "req-1"},
    {"role": "user", "content": "and now?", "conversationRequestId": "req-2"},
]
"""A WorkBuddy-shaped history: every kept message carries its request id."""


def _without(messages: list[JsonValue], *keys: str) -> list[JsonValue]:
    """Return ``messages`` with ``keys`` removed from every message object."""
    return [
        {key: value for key, value in message.items() if key not in keys}
        if isinstance(message, dict)
        else message
        for message in messages
    ]


def _rejection(payload: JsonObject) -> OpenAIProtocolError:
    """Decode a Chat body that must be refused and return the refusal."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_chat(payload)
    return captured.value


def test_a_body_without_unknown_keys_is_returned_unchanged() -> None:
    """No unknown key means no copy and no disclosure."""
    payload: JsonObject = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    repaired, disclosures = drop_unknown_chat_message_fields(payload)
    assert repaired is payload
    assert disclosures == ()


def test_workbuddy_history_decodes_exactly_like_the_plain_history() -> None:
    """``conversationRequestId`` on every message is dropped and named once."""
    tagged = decode_chat({"model": "coding", "messages": _WORKBUDDY_HISTORY})
    plain = decode_chat(
        {"model": "coding", "messages": _without(_WORKBUDDY_HISTORY, "conversationRequestId")}
    )
    assert tagged.request.messages == plain.request.messages
    assert tagged.request.ignored_parameters == (
        "messages.conversationRequestId->dropped(unknown_message_field)",
    )
    # Neither replay identity nor the captured effective request sees the key.
    assert canonical_request_sha256(tagged.request) == canonical_request_sha256(plain.request)
    tagged_capture = capture_context_document(tagged.request)
    plain_capture = capture_context_document(plain.request)
    assert tagged_capture["request"] == plain_capture["request"]
    assert "conversationRequestId" not in str(tagged_capture["request"])


def test_the_caller_body_is_not_mutated() -> None:
    """The repair copies; the caller's own objects keep their keys."""
    payload: JsonObject = {"model": "m", "messages": [{"role": "user", "content": "hi", "x": 1}]}
    repaired, _ = drop_unknown_chat_message_fields(payload)
    assert repaired["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["messages"] == [{"role": "user", "content": "hi", "x": 1}]


def test_internal_tool_call_flag_and_echoed_model_fields_are_dropped() -> None:
    """``isInternalToolCall``, ``model_id`` and a per-message ``max_tokens`` go, in order."""
    messages: list[JsonValue] = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": None,
            "isInternalToolCall": True,
            "model_id": "glm-5.3",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok", "isInternalToolCall": True},
        {"role": "user", "content": "thanks", "max_tokens": 512},
    ]
    tagged = decode_chat({"model": "coding", "messages": messages})
    plain = decode_chat(
        {
            "model": "coding",
            "messages": _without(messages, "isInternalToolCall", "model_id", "max_tokens"),
        }
    )
    assert tagged.request.messages == plain.request.messages
    assert tagged.request.ignored_parameters == (
        "messages.isInternalToolCall->dropped(unknown_message_field)",
        "messages.model_id->dropped(unknown_message_field)",
        "messages.max_tokens->dropped(unknown_message_field)",
    )


def test_unknown_tool_call_and_content_part_keys_are_dropped() -> None:
    """Assistant tool calls (function and custom) and content parts tolerate extras."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look", "providerOptions": {}},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.test/a.png"},
                            "uiState": "shown",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                            "status": "done",
                        },
                        {
                            "id": "call_2",
                            "type": "custom",
                            "custom": {"name": "patch", "input": "x"},
                            "status": "done",
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
                {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
            ],
        }
    )
    assert decoded.request.ignored_parameters == (
        "messages.content.providerOptions->dropped(unknown_message_field)",
        "messages.content.uiState->dropped(unknown_message_field)",
        "messages.tool_calls.status->dropped(unknown_message_field)",
    )
    assert decoded.request.messages[0].content == "look"


def test_ai_sdk_image_part_provider_options_are_dropped_inside_image_url() -> None:
    """The Vercel AI SDK's ``image_url.providerOptions`` is dropped; the image is unchanged."""
    image: JsonObject = {"url": "https://example.test/a.png", "detail": "high"}
    tagged = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {**image, "providerOptions": {"openai": {}}},
                        },
                    ],
                }
            ],
        }
    )
    plain = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": image},
                    ],
                }
            ],
        }
    )
    assert tagged.request.messages == plain.request.messages
    assert tagged.request.ignored_parameters == (
        "messages.content.image_url.providerOptions->dropped(unknown_message_field)",
    )
    assert canonical_request_sha256(tagged.request) == canonical_request_sha256(plain.request)


def test_audio_and_file_media_objects_tolerate_unknown_keys() -> None:
    """``input_audio`` and ``file`` objects drop extras like ``image_url`` does."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "UklGRg==", "format": "wav", "uiId": 1},
                        },
                        {
                            "type": "file",
                            "file": {"file_id": "file-abc", "providerOptions": {}},
                        },
                    ],
                }
            ],
        }
    )
    assert decoded.request.ignored_parameters == (
        "messages.content.input_audio.uiId->dropped(unknown_message_field)",
        "messages.content.file.providerOptions->dropped(unknown_message_field)",
    )


@pytest.mark.parametrize(
    ("part", "param"),
    (
        (
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.test/a.png",
                    "detail": "ultra",
                    "providerOptions": {},
                },
            },
            "messages.0.content.0.image_url.detail",
        ),
        (
            {"type": "image_url", "image_url": {"url": "", "providerOptions": {}}},
            "messages.0.content.0.image_url.url",
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "", "format": "wav", "x": 1}},
            "messages.0.content.0.input_audio.data",
        ),
        (
            {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": 7, "x": 1}},
            "messages.0.content.0.input_audio.format",
        ),
    ),
)
def test_a_media_object_field_with_a_bad_value_is_still_refused(
    part: JsonObject, param: str
) -> None:
    """Dropping unknown media keys never loosens ``url``, ``detail``, ``data`` or ``format``."""
    refusal = _rejection({"model": "coding", "messages": [{"role": "user", "content": [part]}]})
    assert refusal.status_code == 400
    assert refusal.detail.code == "invalid_parameter"
    assert refusal.detail.param == param


def test_researched_client_annotations_keep_their_reason() -> None:
    """Keys traced to one client (Copilot, WorkBuddy CLI) stay ``client_annotation``."""
    decoded = decode_chat(
        {
            "model": "coding",
            "messages": [
                {"role": "user", "content": "hi", "cot_id": "a", "agent": "build"},
                {"role": "user", "content": "again", "conversationRequestId": "r"},
            ],
        }
    )
    assert decoded.request.ignored_parameters == (
        "messages.cot_id->dropped(client_annotation)",
        "messages.agent->dropped(client_annotation)",
        "messages.conversationRequestId->dropped(unknown_message_field)",
    )


@pytest.mark.parametrize(
    ("message", "param"),
    (
        ({"role": "user", "content": 5}, "messages.0.content"),
        (
            {"role": "assistant", "content": "x", "reasoning_content": 5},
            "messages.0.reasoning_content",
        ),
        ({"role": "user", "content": [{"type": "text", "text": 5}]}, "messages.0.content.0.text"),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                        "index": -1,
                    }
                ],
            },
            "messages.0.tool_calls.0.index",
        ),
        (
            {"role": "user", "content": "hi", "cache_control": {"type": "persistent"}},
            "messages.0.cache_control",
        ),
    ),
)
def test_a_defined_field_with_a_bad_value_is_still_refused(message: JsonObject, param: str) -> None:
    """Tolerance covers unknown keys only; every defined field keeps its typed 400."""
    refusal = _rejection(
        {"model": "coding", "messages": [message, {"role": "user", "content": "x"}]}
    )
    assert refusal.status_code == 400
    assert refusal.detail.code == "invalid_parameter"
    assert refusal.detail.param == param


def test_unknown_keys_below_a_tool_calls_function_object_are_still_refused() -> None:
    """The ``function`` object is the call itself, so it stays closed."""
    refusal = _rejection(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}", "extra": 1},
                        }
                    ],
                }
            ],
        }
    )
    assert refusal.detail.param == "messages.0.tool_calls.0.function.extra"


@pytest.mark.parametrize(
    "decode",
    (
        lambda: decode_chat(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "conversationRequestId": "r",
            }
        ),
        lambda: decode_responses({"model": "m", "input": "hi", "conversationRequestId": "r"}),
        lambda: decode_messages(
            {
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
                "conversationRequestId": "r",
            }
        ),
    ),
)
def test_an_unknown_top_level_parameter_is_still_refused(
    decode: Callable[[], DecodedGatewayRequest],
) -> None:
    """Top-level arguments keep the per-profile ``unsupported_parameter`` refusal."""
    with pytest.raises(OpenAIProtocolError) as captured:
        decode()
    assert captured.value.detail.code == "unsupported_parameter"
    assert captured.value.detail.param == "conversationRequestId"


def test_responses_message_items_tolerate_unknown_keys_and_other_items_do_not() -> None:
    """A Responses ``message`` item drops extras; a ``function_call_output`` stays closed."""
    decoded = decode_responses(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "hi", "conversationRequestId": "r"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "more", "uiState": 1}],
                },
            ],
        }
    )
    assert decoded.request.ignored_parameters == (
        "input.conversationRequestId->dropped(unknown_message_field)",
        "input.content.uiState->dropped(unknown_message_field)",
    )
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_responses(
            {
                "model": "m",
                "input": [{"type": "function_call_output", "call_id": "c", "output": "x", "z": 1}],
            }
        )
    assert captured.value.detail.param == "input.0.z"


def test_anthropic_turns_tolerate_unknown_keys_and_content_blocks_do_not() -> None:
    """A Messages turn drops extras; its content blocks keep their own validation."""
    decoded = decode_messages(
        {
            "model": "m",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "output_config": {"effort": "high"},
                    "startsNewUserRequest": True,
                }
            ],
        }
    )
    assert decoded.request.ignored_parameters[:2] == (
        "messages.output_config->dropped(unknown_message_field)",
        "messages.startsNewUserRequest->dropped(unknown_message_field)",
    )
    with pytest.raises(OpenAIProtocolError) as captured:
        decode_messages(
            {
                "model": "m",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "z": 1}]}],
            }
        )
    assert captured.value.detail.param == "messages.0.content.0.z"


@pytest.mark.parametrize(
    ("key", "token"),
    (
        ("conversationRequestId", "conversationRequestId"),
        ("x-trace.id", "x_trace_id"),
        ("<script>", "_script_"),
        ("", "_"),
        ("ключ", "____"),
        ("k" * 500, "k" * MAXIMUM_KEY_CHARACTERS),
    ),
)
def test_key_tokens_are_identifier_shaped_and_bounded(key: str, token: str) -> None:
    """Every disclosure token matches ``[A-Za-z0-9_]{1,64}`` whatever the caller sent."""
    assert key_token(key) == token


def test_a_hostile_history_cannot_grow_the_disclosure_list() -> None:
    """Hundreds of distinct keys name at most the cap, then one counting entry."""
    messages: list[JsonValue] = [
        {"role": "user", "content": "hi", **{f"k{index}": index for index in range(300)}},
        {"role": "user", "content": "again", **{f"k{index}": index for index in range(300)}},
    ]
    decoded = decode_chat({"model": "coding", "messages": messages})
    disclosures = decoded.request.ignored_parameters
    assert len(disclosures) == MAXIMUM_NAMED_DISCLOSURES + 1
    assert disclosures[0] == "messages.k0->dropped(unknown_message_field)"
    hidden = 300 - MAXIMUM_NAMED_DISCLOSURES
    assert disclosures[-1] == f"messages.*->dropped({hidden}_more_unknown_message_fields)"
    assert sum(len(entry) for entry in disclosures) < 2_048


def test_keys_that_sanitize_alike_are_named_once() -> None:
    """Two caller spellings of one token produce one disclosure entry."""
    dropped = DroppedKeys("messages")
    dropped.strip({"a.b": 1, "a-b": 2, "role": "user"}, frozenset({"role"}))
    assert dropped.disclosures() == ("messages.a_b->dropped(unknown_message_field)",)
