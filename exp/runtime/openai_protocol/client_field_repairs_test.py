"""Tests for the disclosed client-field repairs on the OpenAI surfaces."""

from __future__ import annotations

import json
from typing import cast

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.openai_protocol.client_field_repairs import (
    INCLUDE_REASONING_TRANSLATED,
    OUTPUT_CONFIG_DROPPED,
    OUTPUT_CONFIG_EFFORT_IGNORED,
    OUTPUT_CONFIG_EFFORT_TRANSLATED,
    OUTPUT_PHASE_DROPPED,
    OUTPUT_STATUS_DROPPED,
    PROMPT_CACHE_RETENTION_IGNORED,
    RESPONSES_ARGUMENTS_DECODED,
    RESPONSES_ARGUMENTS_SERIALIZED,
    SEED_IGNORED,
    STREAM_OBFUSCATION_IGNORED,
    STREAM_OPTIONS_NOT_STREAMING,
    THINKING_DISPLAY_DROPPED,
    accept_responses_stream_options,
    drop_unanchored_output_markers,
    drop_unforwarded_hints,
    translate_chat_include_reasoning,
    translate_chat_output_config,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


def test_markers_with_an_item_id_are_kept() -> None:
    """A provider-issued output item replays exactly; nothing is repaired."""
    payload: JsonObject = {
        "input": [
            {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed"},
            {"type": "function_call", "status": "completed", "call_id": "c", "name": "f"},
        ]
    }
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired is payload
    assert disclosures == ()


def test_input_status_on_non_assistant_messages_is_left_for_the_wire_model() -> None:
    """Official input-message status needs no repair (the wire model accepts it)."""
    payload: JsonObject = {"input": [{"role": "user", "content": "x", "status": "completed"}]}
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired is payload
    assert disclosures == ()


def test_unanchored_status_and_phase_are_dropped_and_each_kind_disclosed_once() -> None:
    """Status (assistant only) and phase (any role) go when the item has no id."""
    payload: JsonObject = {
        "input": [
            {"role": "assistant", "content": "a", "status": "completed"},
            {"role": "assistant", "content": "b", "status": "completed", "phase": "commentary"},
            {"role": "user", "content": "c", "phase": "final_answer"},
        ]
    }
    repaired, disclosures = drop_unanchored_output_markers(payload)
    assert repaired["input"] == [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert disclosures == (OUTPUT_STATUS_DROPPED, OUTPUT_PHASE_DROPPED)


@pytest.mark.parametrize(
    ("payload", "disclosures"),
    (
        ({"stream": True, "stream_options": {"include_obfuscation": False}}, ()),
        ({"stream": True, "stream_options": None}, ()),
        (
            {"stream": True, "stream_options": {"include_obfuscation": True}},
            (STREAM_OBFUSCATION_IGNORED,),
        ),
        ({"stream_options": {}}, (STREAM_OPTIONS_NOT_STREAMING,)),
    ),
)
def test_stream_options_are_consumed(payload: JsonObject, disclosures: tuple[str, ...]) -> None:
    """The object never reaches the wire models; only ignored settings are disclosed."""
    repaired, named = accept_responses_stream_options(payload)
    assert "stream_options" not in repaired
    assert named == disclosures


@pytest.mark.parametrize(
    ("options", "param"),
    (
        ("yes", "stream_options"),
        ({"include_usage": True}, "stream_options.include_usage"),
        ({"include_obfuscation": "no"}, "stream_options.include_obfuscation"),
    ),
)
def test_malformed_stream_options_keep_a_named_rejection(options: JsonValue, param: str) -> None:
    """A malformed object or unknown key is the caller's named 400."""
    with pytest.raises(OpenAIProtocolError) as error:
        accept_responses_stream_options({"stream": True, "stream_options": options})
    assert error.value.detail.param == param


_CHAT_BASE: JsonObject = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_chat_output_config_effort_becomes_reasoning_effort() -> None:
    """OpenCode's Anthropic ``output_config.effort`` reads as ``reasoning_effort``."""
    decoded = decode_chat({**_CHAT_BASE, "output_config": {"effort": "high"}})
    assert decoded.request.reasoning_effort == "high"
    assert decoded.request.ignored_parameters == (OUTPUT_CONFIG_EFFORT_TRANSLATED,)


def test_chat_output_config_yields_to_explicit_reasoning_effort() -> None:
    """An explicit ``reasoning_effort`` wins; the rest of the object is dropped."""
    repaired, disclosures = translate_chat_output_config(
        {
            **_CHAT_BASE,
            "reasoning_effort": "low",
            "output_config": {"effort": "high", "format": {"type": "json_schema"}},
        }
    )
    assert "output_config" not in repaired
    assert repaired["reasoning_effort"] == "low"
    assert disclosures == (OUTPUT_CONFIG_EFFORT_IGNORED, OUTPUT_CONFIG_DROPPED)


def test_chat_output_config_with_unknown_effort_is_dropped() -> None:
    """An effort no route knows is not guessed at."""
    repaired, disclosures = translate_chat_output_config(
        {**_CHAT_BASE, "output_config": {"effort": "turbo"}}
    )
    assert "reasoning_effort" not in repaired
    assert disclosures == (OUTPUT_CONFIG_DROPPED,)


_EXCLUDE_DROPPED = "reasoning.exclude->dropped(not_carried)"


def test_chat_include_reasoning_false_decodes_as_reasoning_exclude() -> None:
    """OpenRouter's ``include_reasoning: false`` is ``reasoning: {exclude: true}``."""
    decoded = decode_chat({**_CHAT_BASE, "include_reasoning": False})
    reference = decode_chat({**_CHAT_BASE, "reasoning": {"exclude": True}})
    assert decoded.request.reasoning_effort == reference.request.reasoning_effort is None
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == (INCLUDE_REASONING_TRANSLATED, _EXCLUDE_DROPPED)


@pytest.mark.parametrize("value", (True, None))
def test_chat_include_reasoning_default_values_are_consumed_silently(value: bool | None) -> None:
    """``true`` asks for the default and ``null`` for nothing: neither is disclosed."""
    decoded = decode_chat({**_CHAT_BASE, "include_reasoning": value})
    assert decoded.request.reasoning_effort is None
    assert decoded.request.thinking_default_enable is False
    assert decoded.request.ignored_parameters == ()


def test_chat_include_reasoning_false_merges_into_the_reasoning_object() -> None:
    """The other reasoning controls the caller sent keep their meaning."""
    repaired, disclosures = translate_chat_include_reasoning(
        {**_CHAT_BASE, "include_reasoning": False, "reasoning": {"effort": "high"}}
    )
    assert "include_reasoning" not in repaired
    assert repaired["reasoning"] == {"effort": "high", "exclude": True}
    assert disclosures == (INCLUDE_REASONING_TRANSLATED,)
    decoded = decode_chat(
        {**_CHAT_BASE, "include_reasoning": False, "reasoning": {"effort": "high"}}
    )
    assert decoded.request.reasoning_effort == "high"
    assert decoded.request.ignored_parameters == (
        INCLUDE_REASONING_TRANSLATED,
        _EXCLUDE_DROPPED,
        "reasoning->translated(reasoning_effort)",
    )


def test_chat_include_reasoning_false_fills_a_null_reasoning_object() -> None:
    repaired, _ = translate_chat_include_reasoning(
        {**_CHAT_BASE, "include_reasoning": False, "reasoning": None}
    )
    assert repaired["reasoning"] == {"exclude": True}


@pytest.mark.parametrize(("include", "exclude"), ((True, True), (False, False)))
def test_chat_include_reasoning_contradicting_reasoning_exclude_is_refused(
    include: bool, exclude: bool
) -> None:
    """A conflict is a typed 400 naming both fields, never a silent winner."""
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat({**_CHAT_BASE, "include_reasoning": include, "reasoning": {"exclude": exclude}})
    assert error.value.status_code == 400
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == "include_reasoning"
    assert "include_reasoning" in error.value.detail.message
    assert "reasoning.exclude" in error.value.detail.message


@pytest.mark.parametrize("include", (True, False))
def test_chat_include_reasoning_agreeing_with_reasoning_exclude_is_served(include: bool) -> None:
    decoded = decode_chat(
        {**_CHAT_BASE, "include_reasoning": include, "reasoning": {"exclude": not include}}
    )
    assert decoded.request.reasoning_effort is None


@pytest.mark.parametrize("value", ("false", 0, {"exclude": True}))
def test_chat_include_reasoning_non_boolean_is_a_named_400(value: JsonValue) -> None:
    with pytest.raises(OpenAIProtocolError) as error:
        decode_chat({**_CHAT_BASE, "include_reasoning": value})
    assert error.value.detail.code == "invalid_parameter"
    assert error.value.detail.param == "include_reasoning"


def test_chat_include_reasoning_beside_explicit_reasoning_effort_is_served() -> None:
    decoded = decode_chat({**_CHAT_BASE, "include_reasoning": False, "reasoning_effort": "low"})
    assert decoded.request.reasoning_effort == "low"
    assert INCLUDE_REASONING_TRANSLATED in decoded.request.ignored_parameters


@pytest.mark.parametrize("modalities", (["image", "text"], ["text"], ["image"], None))
def test_chat_text_and_image_modalities_serve(modalities: list[str] | None) -> None:
    """OpenRouter-style image clients' ``modalities`` never 400; the route decides output."""
    decoded = decode_chat({**_CHAT_BASE, "modalities": cast(JsonValue, modalities)})
    assert decoded.request.ignored_parameters == ()


@pytest.mark.parametrize(
    "modalities", (["text", "audio"], ["image", "image"], [], "image", [{"type": "image"}])
)
def test_chat_audio_or_malformed_modalities_keep_their_refusal(modalities: JsonValue) -> None:
    """Audio output, repeats, and malformed lists keep the named unsupported-field refusal."""
    with pytest.raises(OpenAIProtocolError) as raised:
        decode_chat({**_CHAT_BASE, "modalities": modalities})
    assert raised.value.detail.param == "modalities"


@pytest.mark.parametrize("stream", (False, None))
def test_chat_stream_options_without_streaming_are_ignored(stream: bool | None) -> None:
    """``stream_options`` on a non-streaming Chat call serves, disclosed."""
    payload: JsonObject = {**_CHAT_BASE, "stream_options": {"include_usage": True}}
    if stream is not None:
        payload["stream"] = stream
    decoded = decode_chat(payload)
    assert decoded.request.include_usage is False
    assert decoded.request.ignored_parameters == (STREAM_OPTIONS_NOT_STREAMING,)


def test_chat_stream_options_on_a_stream_still_configure_usage() -> None:
    """A streaming call keeps its ``include_usage`` request."""
    decoded = decode_chat({**_CHAT_BASE, "stream": True, "stream_options": {"include_usage": True}})
    assert decoded.request.include_usage is True
    assert decoded.request.ignored_parameters == ()


def test_chat_image_dimensions_and_message_agent_are_stripped() -> None:
    """WorkBuddy's ``agent`` tag and an image part's ``dimensions`` hint never 400."""
    decoded = decode_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "agent": "build",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.test/a.png",
                                "dimensions": {"width": 10, "height": 20},
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert decoded.request.ignored_parameters == (
        "messages.agent->dropped(client_annotation)",
        "messages.content.image_url.dimensions->dropped(client_annotation)",
    )


def test_chat_thinking_display_is_dropped_and_thinking_kept() -> None:
    """``thinking.display`` is removed; the budget still reaches the request."""
    decoded = decode_chat(
        {
            **_CHAT_BASE,
            "thinking": {"type": "enabled", "budget_tokens": 2048, "display": "summarized"},
        }
    )
    assert THINKING_DISPLAY_DROPPED in decoded.request.ignored_parameters
    assert decoded.request.provider_thinking_config == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


@pytest.mark.parametrize(
    ("field", "value", "disclosure"),
    (
        ("seed", 42, SEED_IGNORED),
        ("prompt_cache_retention", "24h", PROMPT_CACHE_RETENTION_IGNORED),
    ),
)
def test_unforwarded_hints_are_ignored_on_both_surfaces(
    field: str, value: JsonValue, disclosure: str
) -> None:
    """``seed`` and ``prompt_cache_retention`` serve, disclosed, instead of a named 400."""
    chat = decode_chat({**_CHAT_BASE, field: value})
    responses = decode_responses({"model": "m", "input": "hi", field: value})
    assert chat.request.ignored_parameters == (disclosure,)
    assert responses.request.ignored_parameters == (disclosure,)


def test_null_hints_are_removed_silently() -> None:
    """A ``null`` hint asked for nothing, so nothing is disclosed."""
    repaired, disclosures = drop_unforwarded_hints({**_CHAT_BASE, "seed": None})
    assert "seed" not in repaired
    assert disclosures == ()


def _function_call_input(arguments: JsonValue) -> JsonObject:
    """Build a Responses body replaying one function call and its output."""
    return {
        "model": "m",
        "input": [
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": arguments},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ],
    }


@pytest.mark.parametrize(
    ("arguments", "disclosure"),
    (
        ({"city": "Paris"}, RESPONSES_ARGUMENTS_SERIALIZED),
        (json.dumps(json.dumps({"city": "Paris"})), RESPONSES_ARGUMENTS_DECODED),
    ),
)
def test_responses_call_arguments_object_or_double_encoded_are_normalized(
    arguments: JsonValue, disclosure: str
) -> None:
    """An object or a double-encoded string names one argument object unambiguously."""
    decoded = decode_responses(_function_call_input(arguments))
    assert decoded.request.ignored_parameters == (disclosure,)
    calls = [call for message in decoded.request.messages for call in message.tool_calls]
    assert [call.arguments for call in calls] == [{"city": "Paris"}]


@pytest.mark.parametrize(
    ("arguments", "treatment"),
    (
        ("not json", "invalid_json"),
        ("[1, 2]", "non_object_json"),
        (json.dumps("plain text"), "non_object_json"),
    ),
)
def test_responses_call_arguments_that_name_no_object_are_served_with_a_stand_in(
    arguments: str, treatment: str
) -> None:
    """A replayed call whose text names no object keeps its text and gains a wrapper.

    Refusing it bricked the session (every later turn replays the same call);
    the rung decides which form it sends and discloses a rewrite there.
    """
    decoded = decode_responses(_function_call_input(arguments))
    assert decoded.request.ignored_parameters == ()
    (call,) = [call for message in decoded.request.messages for call in message.tool_calls]
    assert call.raw_arguments == arguments
    assert call.arguments == {"_unparsed_arguments": arguments}
    assert call.arguments_treatment == treatment
