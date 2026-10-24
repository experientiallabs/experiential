"""Tests for official OpenAI SDK schema validation."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.official_schema import (
    CHAT_OFFICIAL,
    RESPONSES_OFFICIAL,
    OfficialBranches,
    streaming_branch,
    validate_official,
)

_MESSAGES: list[JsonValue] = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("branches", [CHAT_OFFICIAL, RESPONSES_OFFICIAL])
def test_only_stream_true_selects_the_streaming_branch(branches: OfficialBranches) -> None:
    """Absent, false and otherwise-valued ``stream`` all validate as non-streaming."""
    non_streaming, streaming = branches
    assert streaming_branch(branches, {"stream": True}) is streaming
    payloads: list[JsonObject] = [{}, {"stream": False}, {"stream": None}, {"stream": "true"}]
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
    """The union used to report the streaming branch's missing ``stream`` for these bodies."""
    with pytest.raises(OpenAIProtocolError) as raised:
        validate_official(streaming_branch(CHAT_OFFICIAL, payload), payload)
    assert raised.value.detail.param == param
    assert "expected a value" not in str(raised.value)
