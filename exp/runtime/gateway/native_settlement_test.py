"""Tests for native settlement payload normalization."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.runtime.gateway.contracts import GatewayEventKind, GatewayFailureClass
from exp.runtime.gateway.native_settlement import (
    _usage_from_payload,  # noqa: PLC2701 - direct unit coverage for normalization.
    first_token_at_from_settlement,
    terminal_from_settlement,
)


def test_first_token_at_parses_the_native_plane_rfc3339_wire_format() -> None:
    # The exact string the Rust data plane emits (settlement.rs
    # `system_time_to_rfc3339`): explicit +00:00 offset, millisecond fraction.
    # This pins the producer/consumer contract so a format drift on either side
    # fails loudly instead of silently dropping time-to-first-token.
    parsed = first_token_at_from_settlement({"first_token_at": "2023-11-14T22:13:20.500+00:00"})
    assert parsed == datetime(2023, 11, 14, 22, 13, 20, 500_000, tzinfo=UTC)


def test_first_token_at_is_none_when_absent_or_malformed() -> None:
    # A non-streaming attempt observes no first token, so the field is absent;
    # a malformed value never crashes accounting.
    assert first_token_at_from_settlement({}) is None
    assert first_token_at_from_settlement({"first_token_at": None}) is None
    assert first_token_at_from_settlement({"first_token_at": 1_700_000_000}) is None
    assert first_token_at_from_settlement({"first_token_at": "not-a-timestamp"}) is None


def test_usage_from_payload_handles_tokens_and_tool_names() -> None:
    """Settlement usage covers token totals, tool-only, and absent cases."""
    assert _usage_from_payload(None, []) is None
    tools_only = _usage_from_payload(None, ["search"])
    assert tools_only is not None and tools_only.tool_names == ("search",)
    complete = _usage_from_payload(
        {"input_tokens": 10, "output_tokens": 3, "cached_input_tokens": 2},
        [],
    )
    assert complete is not None
    assert complete.input_tokens == 10
    assert complete.output_tokens == 3
    assert complete.cached_input_tokens == 2


def test_usage_from_payload_preserves_cache_write_leg() -> None:
    """Cache-write tokens survive settlement even when cache-read is absent."""
    usage = _usage_from_payload(
        {
            "input_tokens": 1_000,
            "output_tokens": 10,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 1_000,
        },
        [],
    )
    assert usage is not None
    assert usage.input_tokens == 1_000
    assert usage.cached_input_tokens == 0
    assert usage.cache_creation_input_tokens == 1_000

    # Fresh and cache-write streams must not collapse to the same usage.
    fresh = _usage_from_payload(
        {"input_tokens": 1_000, "output_tokens": 10},
        [],
    )
    assert fresh is not None
    assert fresh.cache_creation_input_tokens is None
    assert usage != fresh
    assert usage.model_dump(exclude_none=True) != fresh.model_dump(exclude_none=True)


def test_terminal_from_settlement_preserves_cache_write_usage() -> None:
    """Terminal events carry the cache-write leg end-to-end."""
    terminal, _failure = terminal_from_settlement(
        {
            "outcome": "completed",
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 10,
                "cache_creation_input_tokens": 1_000,
            },
            "tool_names": [],
            "failure": None,
        }
    )
    assert terminal.usage is not None
    assert terminal.usage.cache_creation_input_tokens == 1_000


def test_terminal_from_settlement_normalizes_usage_and_tools() -> None:
    """Completed payloads retain token counts and ordered tool names."""
    terminal, failure = terminal_from_settlement(
        {
            "outcome": "completed",
            "usage": {
                "input_tokens": 8,
                "output_tokens": 3,
                "cached_input_tokens": 2,
                "reasoning_tokens": 1,
            },
            "tool_names": ["search", "fetch"],
            "failure": None,
        }
    )

    assert failure is None
    assert terminal.kind == GatewayEventKind.COMPLETED
    assert terminal.usage is not None
    assert terminal.usage.input_tokens == 8
    assert terminal.usage.tool_names == ("search", "fetch")


def test_terminal_from_settlement_normalizes_failure() -> None:
    """Failed payloads attach the sanitized failure to the terminal."""
    terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": None,
            "tool_names": [],
            "failure": {
                "failure_class": "transport",
                "safe_message": "provider transport failed",
            },
        }
    )

    assert failure is not None
    assert failure.failure_class == GatewayFailureClass.TRANSPORT
    assert terminal.failure == failure


def test_terminal_from_settlement_carries_the_provider_detail() -> None:
    """A client-error settlement threads the sanitized provider sentence through."""
    _terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": None,
            "tool_names": [],
            "failure": {
                "failure_class": "invalid_request",
                "safe_message": "provider rejected the request",
                "provider_detail": "max_tokens must be greater than thinking budget.",
            },
        }
    )

    assert failure is not None
    assert failure.provider_detail == "max_tokens must be greater than thinking budget."

    # An empty or absent detail resolves to None rather than an empty string.
    _t, blank = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": None,
            "tool_names": [],
            "failure": {
                "failure_class": "invalid_request",
                "safe_message": "provider rejected the request",
                "provider_detail": "",
            },
        }
    )
    assert blank is not None
    assert blank.provider_detail is None
