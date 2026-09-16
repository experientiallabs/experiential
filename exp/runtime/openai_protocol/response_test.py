"""Tests for aggregate Chat probability projection."""

from exp.runtime.gateway.contracts import (
    ChoiceLogprobs,
    ChoiceLogprobsDelta,
    GatewayEvent,
    GatewayEventKind,
    TokenLogprob,
)
from exp.runtime.openai_protocol.response import _chat_logprobs


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
