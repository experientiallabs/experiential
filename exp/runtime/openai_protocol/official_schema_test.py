"""Tests for official OpenAI SDK schema validation."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.official_schema import (
    CHAT_OFFICIAL,
    RESPONSES_OFFICIAL,
    OfficialBranches,
    stream_requested,
    streaming_branch,
    validate_official,
)

_MESSAGES: list[JsonValue] = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("branches", [CHAT_OFFICIAL, RESPONSES_OFFICIAL])
def test_only_a_true_stream_selects_the_streaming_branch(branches: OfficialBranches) -> None:
    """Only values the lax ``Literal[True]`` admits stream; the rest validate as non-streaming."""
    non_streaming, streaming = branches
    for value in (True, 1, 1.0):
        assert streaming_branch(branches, {"stream": value}) is streaming
    payloads: list[JsonObject] = [
        {},
        {"stream": False},
        {"stream": 0},
        {"stream": None},
        {"stream": "true"},
        {"stream": 2},
    ]
    for payload in payloads:
        assert streaming_branch(branches, payload) is non_streaming


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"model": {"value": "coding"}, "messages": _MESSAGES}, "model"),
        ({"model": None, "messages": _MESSAGES}, "model"),
        ({"model": "coding", "messages": _MESSAGES, "tool_choice": None}, "tool_choice"),
        ({"model": "coding", "messages": _MESSAGES, "response_format": "json"}, "response_format"),
        ({"model": "coding", "messages": _MESSAGES, "stop": 5}, "stop"),
        ({"model": "coding", "messages": _MESSAGES, "stream": True, "stop": 5}, "stop"),
    ],
)
def test_a_fault_outside_stream_is_named_not_the_union_branch_tag(
    payload: JsonObject, param: str
) -> None:
    """The faulty field is named, never the other union branch's missing ``stream``."""
    with pytest.raises(OpenAIProtocolError) as raised:
        validate_official(streaming_branch(CHAT_OFFICIAL, payload), payload)
    assert raised.value.detail.param == param
    assert "expected a value" not in str(raised.value)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [(True, True), (1, True), (1.0, True), (False, False), (0, False), (2, False), ("true", False)],
)
def test_stream_requested_reads_the_lax_literal_true(stream: JsonValue, expected: bool) -> None:
    """Exactly the values the official ``Literal[True]`` admits ask for a stream."""
    assert stream_requested({"stream": stream}) is expected
    assert stream_requested({}) is False
