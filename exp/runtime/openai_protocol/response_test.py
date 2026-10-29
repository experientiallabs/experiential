"""Tests for response assembly from normalized serving events."""

import pytest

from exp.runtime.gateway.contracts import (
    ChoiceLogprobs,
    ChoiceLogprobsDelta,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayRequest,
    TokenLogprob,
)
from exp.runtime.openai_protocol.response import _chat_logprobs, completed_body


@pytest.mark.parametrize("terminal", [GatewayEventKind.COMPLETED, GatewayEventKind.INCOMPLETE])
def test_responses_body_truncates_fractional_epoch_seconds(terminal: GatewayEventKind) -> None:
    """A fractional wall clock yields integer timestamps for strict Responses clients.

    Args:
        terminal: The terminal status whose timestamp contract is checked.
    """
    body = completed_body(
        request=GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        request_id="request-one",
        model="coding",
        created_at=1_700_000_000.75,
        events=(GatewayEvent(kind=terminal, sequence_number=0),),
    )
    assert type(body["created_at"]) is int
    assert body["created_at"] == 1_700_000_000
    if terminal == GatewayEventKind.COMPLETED:
        assert type(body["completed_at"]) is int
        assert body["completed_at"] == body["created_at"]
    else:
        assert body["completed_at"] is None


def test_chat_message_without_tool_calls_omits_the_key() -> None:
    """A Chat message that made no tool calls carries no ``tool_calls`` key.

    OpenAI documents the field as an optional array and omits it when empty.
    Strict OpenAI-schema consumers (the OpenRouter SDK's parser) reject
    ``"tool_calls": null`` while accepting an absent key, which broke a
    customer's nightly gate on a Gemini completion.
    """
    body = completed_body(
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="hello"),),
        ),
        request_id="request-one",
        model="coding",
        created_at=1_700_000_000.0,
        events=(
            GatewayEvent(kind=GatewayEventKind.TEXT_DELTA, sequence_number=0, text_delta="Hi."),
            GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
        ),
    )
    choices = body["choices"]
    assert isinstance(choices, list)
    choice = choices[0]
    assert isinstance(choice, dict)
    message = choice["message"]
    assert isinstance(message, dict)
    assert "tool_calls" not in message
    assert message["content"] == "Hi."
    assert message["refusal"] is None
    assert choice["finish_reason"] == "stop"


def _update(value: ChoiceLogprobs | None) -> GatewayEvent:
    """Build one typed choice observation."""
    return GatewayEvent(
        kind=GatewayEventKind.CHOICE_LOGPROBS_DELTA,
        sequence_number=0,
        choice_logprobs_delta=ChoiceLogprobsDelta(choice_index=0, logprobs=value),
    )


def test_null_updates_never_erase_content_or_refusal_records() -> None:
    """Append both channels once in provider order, retaining empty alternatives and bytes."""
    token = TokenLogprob(token="é", logprob=0.0, bytes=(195,), top_logprobs=())
    result = _chat_logprobs(
        (
            _update(None),
            _update(ChoiceLogprobs(content=(token,))),
            _update(None),
            _update(ChoiceLogprobs(content=(token,), refusal=())),
            _update(ChoiceLogprobs(refusal=(token,))),
            _update(None),
        )
    )
    expected = {"token": "é", "logprob": 0.0, "bytes": [195], "top_logprobs": []}
    assert result == {"content": [expected, expected], "refusal": [expected]}


def test_empty_and_absent_probability_channels_remain_distinct() -> None:
    """An empty observed array stays an array; absent or null metadata stays null."""
    assert _chat_logprobs(()) is None
    assert _chat_logprobs((_update(None),)) is None
    assert _chat_logprobs((_update(ChoiceLogprobs(content=())),)) == {
        "content": [],
        "refusal": None,
    }
