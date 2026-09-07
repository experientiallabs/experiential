"""Tests for native settlement payload normalization."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.runtime.gateway.contracts import (
    GatewayEventKind,
    GatewayFailureClass,
    GatewayRefusalReason,
)
from exp.runtime.gateway.native_settlement import (
    _usage_from_payload,  # noqa: PLC2701 - direct unit coverage for normalization.
    first_token_at_from_settlement,
    settlement_rate_limit,
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


def test_terminal_from_settlement_carries_the_refusal_reason() -> None:
    """A refusal settlement threads the bounded category through, and an
    unknown token fails closed to None instead of raising."""
    _terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": None,
            "tool_names": [],
            "failure": {
                "failure_class": "refusal",
                "safe_message": "provider refused the request: cybersecurity policy",
                "refusal_reason": "cyber_policy",
            },
        }
    )
    assert failure is not None
    assert failure.refusal_reason is GatewayRefusalReason.CYBER_POLICY

    _t, unknown = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": None,
            "tool_names": [],
            "failure": {
                "failure_class": "refusal",
                "safe_message": "provider refused the request",
                "refusal_reason": "reason_a_stale_worker_does_not_know",
            },
        }
    )
    assert unknown is not None
    assert unknown.refusal_reason is None


def test_customer_owned_failures_settle_as_the_callers_invalid_request() -> None:
    """A BYOK credential failure keeps its ladder class in the data plane but files client-side."""
    terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "provider_authentication",
                "safe_message": "your connected openai credential was rejected by the provider",
                "customer_owned": True,
            },
        }
    )
    assert failure is not None
    assert failure.failure_class == GatewayFailureClass.INVALID_REQUEST
    assert failure.safe_message.startswith("your connected openai credential")
    assert terminal.failure is failure

    _terminal, house = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "provider_authentication",
                "safe_message": "provider authentication failed",
            },
        }
    )
    assert house is not None
    assert house.failure_class == GatewayFailureClass.PROVIDER_AUTHENTICATION


def test_throttled_settlement_takes_retry_after_from_harvested_headers() -> None:
    """A throttled failure without its own wait borrows the header's wait."""
    _terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "throttled",
                "safe_message": "provider throttled the request",
            },
            "rate_limit_headers": {"retry-after": "3600"},
        }
    )
    assert failure is not None
    assert failure.retry_after_seconds == 3_600


def test_failure_payloads_own_retry_after_wins_over_the_headers() -> None:
    """A wait the failure payload names is kept verbatim."""
    _terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "throttled",
                "safe_message": "provider throttled the request",
                "retry_after_seconds": 42,
            },
            "rate_limit_headers": {"retry-after": "3600"},
        }
    )
    assert failure is not None
    assert failure.retry_after_seconds == 42


def test_non_throttled_failures_never_borrow_a_retry_after() -> None:
    """Only the throttled class reads the harvested wait; garbage stays None."""
    _terminal, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "provider_internal",
                "safe_message": "provider service failed",
            },
            "rate_limit_headers": {"retry-after": "3600"},
        }
    )
    assert failure is not None
    assert failure.retry_after_seconds is None
    _terminal, garbled = terminal_from_settlement(
        {
            "outcome": "failed",
            "failure": {
                "failure_class": "throttled",
                "safe_message": "provider throttled the request",
                "retry_after_seconds": "soon",
            },
            "rate_limit_headers": {"retry-after": "eventually"},
        }
    )
    assert garbled is not None
    assert garbled.retry_after_seconds is None


def test_settlement_rate_limit_reads_the_optional_header_map() -> None:
    """The typed observation parses when present and stays empty when absent."""
    observation = settlement_rate_limit(
        {
            "outcome": "completed",
            "rate_limit_headers": {
                "anthropic-ratelimit-requests-limit": "10000",
                "anthropic-ratelimit-requests-remaining": "9500",
                "retry-after": "7",
            },
        }
    )
    assert observation.limit_requests == 10_000
    assert observation.remaining_requests == 9_500
    assert observation.retry_after_seconds == 7
    assert settlement_rate_limit({"outcome": "completed"}).is_empty
