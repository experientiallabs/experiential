"""Accepted top-level request fields for both shared OpenAI surfaces."""

from __future__ import annotations

CHAT_ACCEPTED_FIELDS: frozenset[str] = frozenset(
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

RESPONSES_ACCEPTED_FIELDS: frozenset[str] = frozenset(
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
