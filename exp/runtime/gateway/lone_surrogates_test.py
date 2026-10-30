"""Lone surrogate repair and refusal over decoded request JSON."""

from __future__ import annotations

import json
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.lone_surrogates import (
    SURROGATE_ESCAPE,
    LoneSurrogateRefusal,
    has_lone_surrogate,
    refuse_lone_surrogates,
    repaired_escapes,
    repaired_text,
    replace_lone_surrogates,
)


def _all_strings(value: JsonValue) -> list[str]:
    """Every string and key in ``value``, plus those of any JSON text in an argument."""
    found: list[str] = []
    pending: list[JsonValue] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            found.append(item)
            try:
                pending.append(json.loads(item))
            except json.JSONDecodeError:
                pass
        elif isinstance(item, dict):
            found.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


@pytest.mark.parametrize(
    ("text", "expected", "changed"),
    [
        ("cut \ud83d", "cut \ufffd", True),
        ("\ude00 cut", "\ufffd cut", True),
        ("\ude00\ud83d", "\ufffd\ufffd", True),
        ("\U0001f600 whole", "\U0001f600 whole", False),
        ("ascii", "ascii", False),
    ],
)
def test_repaired_text_replaces_each_lone_surrogate_once(
    text: str, expected: str, changed: bool
) -> None:
    """One U+FFFD per lone surrogate; a scalar and ASCII are untouched."""
    assert repaired_text(text) == (expected, changed)
    assert has_lone_surrogate(text) is changed


@pytest.mark.parametrize(
    ("json_text", "expected"),
    [
        ('{"q": "cut \\ud83d"}', '{"q": "cut \\ufffd"}'),
        ('{"q": "\\ude00 cut"}', '{"q": "\\ufffd cut"}'),
        ('{"q": "\\ud83d\\ude00"}', '{"q": "\\ud83d\\ude00"}'),
        ('{"q": "\\ude00\\ud83d"}', '{"q": "\\ufffd\\ufffd"}'),
        ('{"q": "\\ud83d"}', '{"q": "\\ufffd"}'),
        ('{"q": "\\\\ud83d"}', '{"q": "\\\\ud83d"}'),
        ('{"q": "\\\\\\ud83d"}', '{"q": "\\\\\\ufffd"}'),
        ('{ "q" :  "\\ud83dx" ,"r":1}', '{ "q" :  "\\ufffdx" ,"r":1}'),
    ],
)
def test_repaired_escapes_rewrites_only_unpaired_escapes_and_keeps_every_other_byte(
    json_text: str, expected: str
) -> None:
    """An escaped backslash is not an escape; a pair is kept; spacing survives."""
    repaired, changed = repaired_escapes(json_text)
    assert repaired == expected
    assert changed is (repaired != json_text)
    assert not any(has_lone_surrogate(text) for text in _all_strings(json.loads(repaired)))


def test_the_gate_matches_both_decoded_and_nested_escapes() -> None:
    """One raw-body search finds a lone escape at either JSON level."""
    assert SURROGATE_ESCAPE.search('{"a":"\\uD83D"}')
    assert SURROGATE_ESCAPE.search(json.dumps({"arguments": json.dumps({"q": "\ud83d"})}))
    assert not SURROGATE_ESCAPE.search('{"a":"\\u00e9 \\uE000"}')


def _payload() -> JsonObject:
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys \ud83d"},
            {"role": "user", "name": "Jo \ud83d", "content": [{"type": "text", "text": "a\ude00"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": json.dumps({"q": "x\ud83d"})},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"id": {"title": "\ud83d"}}},
                },
            }
        ],
        "metadata": {"name": "label \ud83d"},
    }


def test_content_is_repaired_in_place_with_one_disclosure_per_field_path() -> None:
    """List positions collapse, caller-keyed objects stop the path, and nothing is left."""
    payload = _payload()
    assert replace_lone_surrogates(payload) == (
        "messages.content->replaced(lone_surrogate)",
        "messages.content.text->replaced(lone_surrogate)",
        "messages.name->replaced(lone_surrogate)",
        "messages.tool_calls.function.arguments->replaced(lone_surrogate)",
        "metadata->replaced(lone_surrogate)",
        "tools.function.parameters->replaced(lone_surrogate)",
    )
    assert not any(has_lone_surrogate(text) for text in _all_strings(payload))
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0] == {"role": "system", "content": "sys \ufffd"}
    calls = cast(list[JsonObject], messages[2]["tool_calls"])
    function = cast(JsonObject, calls[0]["function"])
    assert json.loads(cast(str, function["arguments"])) == {"q": "x\ufffd"}


def test_a_body_without_lone_surrogates_is_untouched_and_undisclosed() -> None:
    """Nothing to replace: the payload is byte-identical and nothing is disclosed."""
    payload: JsonObject = {"model": "m", "input": "\U0001f600", "metadata": {"k": "v"}}
    snapshot = json.dumps(payload)
    assert replace_lone_surrogates(payload) == ()
    assert json.dumps(payload) == snapshot


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"model": "m\ud83d"}, "model"),
        ({"model": "m", "tools": [{"function": {"name": "f\ud83d"}}]}, "tools.0.function.name"),
        (
            {"model": "m", "messages": [{"role": "tool", "tool_call_id": "c\ud83d"}]},
            "messages.0.tool_call_id",
        ),
        (
            {"model": "m", "messages": [{"role": "tool", "name": "f\ud83d", "content": "r"}]},
            "messages.0.name",
        ),
        (
            {"model": "m", "input": [{"type": "function_call", "call_id": "c\udc00"}]},
            "input.0.call_id",
        ),
        (
            {
                "model": "m",
                "messages": [{"content": [{"type": "thinking", "signature": "\ud83d"}]}],
            },
            "messages.0.content.0.signature",
        ),
    ],
)
def test_identifiers_are_refused_by_field(payload: JsonObject, param: str) -> None:
    """A name is a typed refusal that names its field and never echoes its text."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        replace_lone_surrogates(payload)
    assert refused.value.param == param
    assert param in refused.value.message
    assert not has_lone_surrogate(refused.value.message)


def test_names_inside_caller_keyed_objects_are_content() -> None:
    """A tool call's ``input`` holds the caller's arguments, so an ``id`` there is text."""
    payload: JsonObject = {
        "messages": [{"content": [{"type": "tool_use", "id": "t", "input": {"id": "\ud83d"}}]}]
    }
    assert replace_lone_surrogates(payload) == ("messages.content.input->replaced(lone_surrogate)",)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"\ud83d": 1}, "body"),
        ({"messages": [{"role": "user", "\ud83d": 1}]}, "messages.0"),
        ({"tools": [{"parameters": {"properties": {"q\ud83d": {}}}}]}, "tools.0.parameters"),
    ],
)
def test_object_keys_are_refused_without_echoing_caller_keys(
    payload: JsonObject, param: str
) -> None:
    """Keys are names; the refusal stops at the caller-keyed object that holds them."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        replace_lone_surrogates(payload)
    assert refused.value.param == param
    assert "properties" not in refused.value.message


def test_deep_nesting_is_walked_without_recursion() -> None:
    """A body nested far past the recursion limit is repaired, not a RecursionError."""
    deep: JsonValue = "cut \ud83d"
    for _ in range(5_000):
        deep = [deep]
    payload: JsonObject = {"input": deep}
    assert replace_lone_surrogates(payload) == ("input->replaced(lone_surrogate)",)


def test_refusing_surfaces_name_the_first_lone_surrogate() -> None:
    """Surfaces with no disclosure channel refuse content and keys alike."""
    refuse_lone_surrogates({"model": "e", "input": ["ok"]})
    with pytest.raises(LoneSurrogateRefusal) as value:
        refuse_lone_surrogates({"model": "e", "input": ["ok", "cut \ud83d"]})
    assert value.value.param == "input.1"
    with pytest.raises(LoneSurrogateRefusal) as key:
        refuse_lone_surrogates({"model": "e", "metadata": {"\ud83d": "v"}})
    assert key.value.param == "metadata"
