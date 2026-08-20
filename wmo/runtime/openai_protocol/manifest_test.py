"""Tests for the closed accepted-field sets of both OpenAI surfaces."""

from __future__ import annotations

from wmo.runtime.openai_protocol.manifest import CHAT_ACCEPTED_FIELDS, RESPONSES_ACCEPTED_FIELDS


def test_chat_accepted_fields_pin_the_closed_surface() -> None:
    """The Chat Completions gateway surface accepts exactly these top-level fields."""
    assert CHAT_ACCEPTED_FIELDS == frozenset(
        {
            "model",
            "messages",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stop",
            "temperature",
            "stream",
            "stream_options",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "response_format",
            "metadata",
        }
    )


def test_responses_accepted_fields_pin_the_closed_surface() -> None:
    """The Responses gateway surface accepts exactly these top-level fields."""
    assert RESPONSES_ACCEPTED_FIELDS == frozenset(
        {
            "model",
            "input",
            "instructions",
            "previous_response_id",
            "max_output_tokens",
            "temperature",
            "stream",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "text",
            "metadata",
        }
    )


def test_explicit_exclusions_stay_outside_both_surfaces() -> None:
    """Multimodal, hosted, background, logprob, and multi-choice features stay excluded."""
    assert {"audio", "n", "logprobs"}.isdisjoint(CHAT_ACCEPTED_FIELDS)
    assert {"background", "conversation", "include"}.isdisjoint(RESPONSES_ACCEPTED_FIELDS)
