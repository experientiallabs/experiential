"""Tests for the disclosed translation of OpenAI's deprecated Chat functions API."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.models.providers.streaming_requests import openai_compatible_stream_payload
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.legacy_functions import (
    LEGACY_FUNCTION_CALL_TRANSLATED,
    LEGACY_FUNCTION_CHOICE_TRANSLATED,
    LEGACY_FUNCTION_NULL_RESULT_TRANSLATED,
    LEGACY_FUNCTION_ROLE_TRANSLATED,
    LEGACY_FUNCTIONS_TRANSLATED,
    translate_legacy_functions,
)
from exp.runtime.openai_protocol.requests import decode_chat

_WEATHER = {
    "name": "get_weather",
    "description": "Weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _legacy_body() -> JsonObject:
    """Return one conversation in the deprecated ``functions`` shapes (AsyncOpenAI replay)."""
    return {
        "model": "coding",
        "functions": [_WEATHER],
        "function_call": "auto",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            },
            {"role": "function", "name": "get_weather", "content": "sunny"},
            {"role": "user", "content": "And now?"},
        ],
    }


def _modern_body() -> JsonObject:
    """Return the same conversation written with ``tools`` under the synthetic id."""
    return {
        "model": "coding",
        "tools": [{"type": "function", "function": _WEATHER}],
        "tool_choice": "auto",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_legacy_function_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_legacy_function_1",
                "name": "get_weather",
                "content": "sunny",
            },
            {"role": "user", "content": "And now?"},
        ],
    }


def test_legacy_conversation_decodes_exactly_like_its_tools_spelling() -> None:
    """Every legacy shape becomes its modern twin, disclosed once per rewrite kind."""
    legacy = decode_chat(_legacy_body())
    modern = decode_chat(_modern_body())
    assert legacy.request.ignored_parameters == (
        LEGACY_FUNCTION_CALL_TRANSLATED,
        LEGACY_FUNCTION_ROLE_TRANSLATED,
        LEGACY_FUNCTIONS_TRANSLATED,
        LEGACY_FUNCTION_CHOICE_TRANSLATED,
    )
    assert LEGACY_FUNCTION_CALL_TRANSLATED == "messages.function_call->translated(tool_calls)"
    assert LEGACY_FUNCTION_ROLE_TRANSLATED == "messages.role->translated(function_as_tool)"
    assert LEGACY_FUNCTIONS_TRANSLATED == "functions->translated(tools)"
    assert LEGACY_FUNCTION_CHOICE_TRANSLATED == "function_call->translated(tool_choice)"
    assert modern.request.ignored_parameters == ()
    assert legacy.request.messages == modern.request.messages
    assert legacy.request.tools == modern.request.tools
    assert legacy.request.tool_choice == modern.request.tool_choice
    # The disclosure is excluded from replay identity; the rewrite is not a new request.
    assert canonical_request_sha256(legacy.request) == canonical_request_sha256(modern.request)
    assert legacy.request.messages[2].provider_tool_name == "get_weather"


def test_legacy_history_reaches_a_chat_wire_only_in_the_tools_spelling() -> None:
    """The upstream Chat body carries ``tool_calls`` and a ``tool`` result, not the legacy keys."""
    payload = openai_compatible_stream_payload("chat-fixture", decode_chat(_legacy_body()).request)
    messages = cast("list[JsonObject]", payload["messages"])
    assert "functions" not in payload and "function_call" not in payload
    assert all("function_call" not in message for message in messages)
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    assert messages[1]["tool_calls"] == [
        {
            "id": "call_legacy_function_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]
    assert messages[2] == {
        "role": "tool",
        "content": "sunny",
        "tool_call_id": "call_legacy_function_1",
        "name": "get_weather",
    }


def test_production_shape_late_in_a_long_history_admits() -> None:
    """The 2026-10-24 repro: ``messages.44.function_call`` replayed by AsyncOpenAI 1.109.1."""
    filler: list[JsonObject] = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(44)
    ]
    body: JsonObject = {
        "model": "coding",
        "messages": [
            *filler,
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "search", "arguments": '{"q":"x"}'},
            },
            {"role": "function", "name": "search", "content": "found"},
        ],
    }
    decoded = decode_chat(body)
    call = decoded.request.messages[44].tool_calls[0]
    assert call.call_id == "call_legacy_function_44"
    assert decoded.request.messages[45].tool_call_id == "call_legacy_function_44"


def test_results_pair_with_the_nearest_unanswered_call_of_their_name() -> None:
    """Repeated calls of one function pair in order; ids follow the call's message index."""
    body: JsonObject = {
        "model": "coding",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "function_call": {"name": "f", "arguments": "{}"}},
            {"role": "function", "name": "f", "content": "one"},
            {"role": "assistant", "function_call": {"name": "f", "arguments": '{"n":2}'}},
            {"role": "function", "name": "f", "content": None},
        ],
    }
    decoded = decode_chat(body)
    messages = decoded.request.messages
    # The documented ``content: null`` result is carried as the empty string, disclosed.
    assert messages[4].content == ""
    assert decoded.request.ignored_parameters == (
        LEGACY_FUNCTION_CALL_TRANSLATED,
        LEGACY_FUNCTION_ROLE_TRANSLATED,
        LEGACY_FUNCTION_NULL_RESULT_TRANSLATED,
    )
    assert LEGACY_FUNCTION_NULL_RESULT_TRANSLATED == (
        "messages.content->translated(null_function_result_as_empty)"
    )
    assert [message.tool_call_id for message in messages] == [
        None,
        None,
        "call_legacy_function_1",
        None,
        "call_legacy_function_3",
    ]
    assert messages[3].tool_calls[0].call_id == "call_legacy_function_3"


def test_appending_turns_never_changes_an_earlier_synthetic_id() -> None:
    """Ids depend on the call's position only, so the replayed prefix stays byte-stable."""
    short = cast("list[JsonObject]", _legacy_body()["messages"])[:3]
    longer = [
        *short,
        {"role": "user", "content": "again"},
        {"role": "assistant", "function_call": {"name": "get_weather", "arguments": "{}"}},
        {"role": "function", "name": "get_weather", "content": "rain"},
    ]
    first, _ = translate_legacy_functions({"messages": short})
    second, _ = translate_legacy_functions({"messages": longer})
    first_messages = cast("list[JsonObject]", first["messages"])
    second_messages = cast("list[JsonObject]", second["messages"])
    assert second_messages[:3] == first_messages
    assert second_messages[5]["tool_call_id"] == "call_legacy_function_4"


def test_a_synthetic_id_steps_around_an_id_the_caller_already_uses() -> None:
    """A caller's own id equal to the synthetic one is never shadowed."""
    payload, _ = translate_legacy_functions(
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_legacy_function_2",
                            "type": "function",
                            "function": {"name": "g", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_legacy_function_2", "content": "x"},
                {"role": "assistant", "function_call": {"name": "f", "arguments": "{}"}},
                {"role": "function", "name": "f", "content": "y"},
            ]
        }
    )
    messages = cast("list[JsonObject]", payload["messages"])
    assert messages[3]["tool_call_id"] == "call_legacy_function_2_"


@pytest.mark.parametrize(
    ("choice", "expected"),
    (
        ("none", "none"),
        ("auto", "auto"),
        ({"name": "get_weather"}, {"type": "function", "function": {"name": "get_weather"}}),
    ),
)
def test_function_call_parameter_becomes_tool_choice(
    choice: JsonValue, expected: JsonValue
) -> None:
    """Each documented ``function_call`` value has one ``tool_choice`` twin."""
    payload, disclosures = translate_legacy_functions(
        {"model": "coding", "functions": [_WEATHER], "function_call": choice, "messages": []}
    )
    assert payload["tool_choice"] == expected
    assert payload["tools"] == [{"type": "function", "function": _WEATHER}]
    assert "functions" not in payload and "function_call" not in payload
    assert disclosures == (LEGACY_FUNCTIONS_TRANSLATED, LEGACY_FUNCTION_CHOICE_TRANSLATED)


def test_forced_function_choice_decodes_like_the_named_tool_choice() -> None:
    """``function_call: {"name"}`` forces the same tool a named ``tool_choice`` does."""
    legacy = decode_chat(
        {
            "model": "coding",
            "functions": [_WEATHER],
            "function_call": {"name": "get_weather"},
            "messages": [{"role": "user", "content": "go"}],
        }
    )
    modern = decode_chat(
        {
            "model": "coding",
            "tools": [{"type": "function", "function": _WEATHER}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "messages": [{"role": "user", "content": "go"}],
        }
    )
    assert legacy.request.tool_choice == modern.request.tool_choice
    assert legacy.request.tools == modern.request.tools


def test_null_legacy_fields_are_removed_without_disclosure() -> None:
    """Null legacy fields ask for nothing, so they leave no disclosure."""
    body: JsonObject = {
        "model": "coding",
        "functions": None,
        "function_call": None,
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "function_call": None},
            {"role": "user", "content": "again"},
        ],
    }
    payload, disclosures = translate_legacy_functions(body)
    assert disclosures == ()
    assert "functions" not in payload and "function_call" not in payload
    assert decode_chat(body).request.ignored_parameters == ()


def test_a_body_without_legacy_shapes_is_returned_unchanged() -> None:
    """The pass is the identity on modern bodies."""
    body = _modern_body()
    payload, disclosures = translate_legacy_functions(body)
    assert payload is body
    assert disclosures == ()


def _call(name: str = "f", arguments: object = "{}") -> JsonObject:
    """Return one assistant message with a legacy call."""
    return {
        "role": "assistant",
        "content": None,
        "function_call": {"name": name, "arguments": arguments},
    }


_USER: JsonObject = {"role": "user", "content": "go"}


@pytest.mark.parametrize(
    ("body", "param"),
    (
        pytest.param(
            {"messages": [_USER, {"role": "function", "name": "f", "content": "r"}]},
            "messages.1.name",
            id="result-without-a-call",
        ),
        pytest.param(
            {"messages": [_USER, _call("f"), {"role": "function", "name": "g", "content": "r"}]},
            "messages.2.name",
            id="result-of-another-function",
        ),
        pytest.param(
            {
                "messages": [
                    _USER,
                    _call("f"),
                    {"role": "function", "name": "f", "content": "r"},
                    {"role": "function", "name": "f", "content": "again"},
                ]
            },
            "messages.3.name",
            id="second-result-for-one-call",
        ),
        pytest.param(
            {"messages": [_USER, _call("f"), {"role": "function", "content": "r"}]},
            "messages.2.name",
            id="result-without-name",
        ),
        pytest.param(
            {
                "messages": [
                    _USER,
                    _call("f"),
                    {"role": "function", "name": "f", "tool_call_id": "x", "content": "r"},
                ]
            },
            "messages.2.tool_call_id",
            id="result-with-tool-call-id",
        ),
        pytest.param(
            {"messages": [_USER, _call("f")]},
            "messages.1.function_call",
            id="call-without-a-result",
        ),
        pytest.param(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "go",
                        "function_call": {"name": "f", "arguments": "{}"},
                    }
                ]
            },
            "messages.0.function_call",
            id="call-on-a-user-message",
        ),
        pytest.param(
            {
                "messages": [
                    _USER,
                    {
                        **_call("f"),
                        "tool_calls": [
                            {
                                "id": "c",
                                "type": "function",
                                "function": {"name": "g", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "function", "name": "f", "content": "r"},
                ]
            },
            "messages.1.function_call",
            id="call-beside-tool-calls",
        ),
        pytest.param(
            {"messages": [_USER, {"role": "assistant", "function_call": "f"}]},
            "messages.1.function_call",
            id="call-not-an-object",
        ),
        pytest.param(
            {"messages": [_USER, _call(""), {"role": "function", "name": "f", "content": "r"}]},
            "messages.1.function_call.name",
            id="call-empty-name",
        ),
        pytest.param(
            {"messages": [_USER, _call("f", {"a": 1})]},
            "messages.1.function_call.arguments",
            id="call-arguments-not-a-string",
        ),
        pytest.param(
            {
                "messages": [
                    _USER,
                    {
                        "role": "assistant",
                        "function_call": {"name": "f", "arguments": "{}", "id": "x"},
                    },
                ]
            },
            "messages.1.function_call.id",
            id="call-unknown-key",
        ),
        pytest.param(
            {
                "functions": [_WEATHER],
                "tools": [{"type": "function", "function": _WEATHER}],
                "messages": [_USER],
            },
            "functions",
            id="functions-beside-tools",
        ),
        pytest.param(
            {"function_call": "auto", "tool_choice": "auto", "messages": [_USER]},
            "function_call",
            id="function-call-beside-tool-choice",
        ),
        pytest.param(
            {"functions": {"name": "f"}, "messages": [_USER]}, "functions", id="functions-object"
        ),
        pytest.param(
            {"functions": ["f"], "messages": [_USER]}, "functions.0", id="functions-entry-string"
        ),
        pytest.param(
            {"functions": [{"description": "x"}], "messages": [_USER]},
            "functions.0.name",
            id="functions-entry-without-name",
        ),
        pytest.param(
            {"functions": [_WEATHER], "function_call": "required", "messages": [_USER]},
            "function_call",
            id="function-call-unknown-mode",
        ),
        pytest.param(
            {
                "functions": [_WEATHER],
                "function_call": {"name": "get_weather", "type": "function"},
                "messages": [_USER],
            },
            "function_call",
            id="function-call-extra-key",
        ),
    ),
)
def test_unpairable_or_malformed_legacy_shapes_are_refused_naming_the_field(
    body: JsonObject, param: str
) -> None:
    """A shape the rewrite cannot carry exactly is a typed 400 naming the field."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat({"model": "coding", **body})
    assert error.value.status_code == 400
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == param
