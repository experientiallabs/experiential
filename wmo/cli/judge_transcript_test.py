"""Tests for readable calibration transcript rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from wmo.cli.judge_transcript import span_evidence_text, user_input
from wmo.common.traces import TraceSpan

_TIME = datetime(2026, 8, 19, tzinfo=UTC)


def test_user_input_reads_current_prompt_attribute_when_messages_are_absent() -> None:
    """Normalized GenAI spans still record the request on gen_ai.prompt."""
    assert user_input({"gen_ai.prompt": "delete the file"}) == "delete the file"


def test_user_input_prefers_captured_messages_over_prompt() -> None:
    """Structured input messages are the primary request text when both exist."""
    assert (
        user_input(
            {
                "gen_ai.prompt": "older prompt text",
                "gen_ai.input.messages": [
                    {"role": "user", "content": "latest user message"},
                ],
            }
        )
        == "latest user message"
    )


def test_span_evidence_includes_prompt_only_user_text() -> None:
    """Cited span evidence keeps the recorded request when messages were never captured."""
    span = TraceSpan(
        span_id="span-1",
        name="chat",
        started_at=_TIME,
        ended_at=_TIME,
        attributes={"gen_ai.prompt": "Run two commands"},
    )

    assert span_evidence_text(span) == "User message: Run two commands"
