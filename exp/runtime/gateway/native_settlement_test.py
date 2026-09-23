"""Tests for native settlement payload normalization."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.contracts import (
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRefusalReason,
    GatewayUsage,
)
from exp.runtime.gateway.native_settlement import (
    NativeSettlementPayload,
    StreamedOutput,
    _usage_from_payload,  # noqa: PLC2701 - direct unit coverage for normalization.
    accepts_keyword,
    exhausted_attempt_payload,
    first_token_at_from_settlement,
    settlement_metadata,
    settlement_rate_limit,
    streamed_output_from_settlement,
    terminal_from_settlement,
    tool_search_requests_from_settlement,
    tool_search_requests_from_terminal,
    tool_search_requests_kwarg,
    upstream_provider_from_settlement,
    upstream_provider_kwarg,
    web_search_requests_from_settlement,
    web_search_requests_from_terminal,
    web_search_requests_kwarg,
)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"input_tokens": 19},
        {"output_tokens": 7},
        {"input_tokens": 19, "output_tokens": 0},
        {"input_tokens": 19, "output_tokens": 7},
    ],
)
def test_cancelled_disconnect_marker_retains_unknown_final_meter(usage: JsonObject | None) -> None:
    """No observed-to-date token count certifies a cancelled attempt's final meter."""
    data: JsonObject = {
        "outcome": "failed",
        "failure": {"failure_class": "cancelled", "safe_message": "cancelled"},
        "usage": usage,
        "dispatched": True,
        "usage_incomplete_due_to_disconnect": True,
    }
    terminal, _ = terminal_from_settlement(data)
    assert terminal.usage_incomplete_due_to_disconnect is True
    if usage is not None:
        assert terminal.usage is not None
        assert terminal.usage.input_tokens == usage.get("input_tokens")
        assert terminal.usage.output_tokens == usage.get("output_tokens")
    assert "usage_incomplete_due_to_disconnect" not in terminal.model_dump()
    replayed, _ = terminal_from_settlement(data)
    assert replayed.usage_incomplete_due_to_disconnect is True


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, None),
        ("Hello", None),
        ({"text": 7}, None),
        ({"text": "Hi", "text_overflow_chars": -1}, None),
        ({}, StreamedOutput()),
        (
            {
                "text": "Hi",
                "reasoning": "hmm",
                "text_overflow_chars": 0,
                "reasoning_overflow_chars": 12,
            },
            StreamedOutput(text="Hi", reasoning="hmm", reasoning_overflow_chars=12),
        ),
        ({"images": 2}, StreamedOutput(images=2)),
        ({"images": "2"}, None),
        ({"unexpected": "text"}, None),
    ],
)
def test_streamed_output_parses_only_the_typed_shape(
    payload: JsonValue | None, expected: StreamedOutput | None
) -> None:
    """Generated-text evidence is typed; a malformed object is dropped, never guessed at."""
    data: JsonObject = {"outcome": "failed"}
    if payload is not None:
        data["streamed_output"] = payload
    assert streamed_output_from_settlement(data) == expected
    assert streamed_output_from_settlement(None) is None


@pytest.mark.parametrize("marker", ["true", 1, None, [], {}])
def test_disconnect_marker_rejects_nonboolean_values(marker: object) -> None:
    """Caller-shaped values never coerce into financial provenance."""
    data: JsonObject = {
        "outcome": "failed",
        "failure": {"failure_class": "cancelled", "safe_message": "cancelled"},
        "dispatched": True,
    }
    # model_validate receives the untrusted wire object before typed routing.
    with pytest.raises(ValueError):
        NativeSettlementPayload.model_validate(
            {**data, "usage_incomplete_due_to_disconnect": marker}
        )


@pytest.mark.parametrize(
    "outcome,failure,dispatched",
    [
        ("completed", None, True),
        ("failed", "provider_internal", True),
        ("failed", "cancelled", False),
    ],
)
def test_disconnect_marker_rejects_inconsistent_work_state(
    outcome: str, failure: str | None, dispatched: bool
) -> None:
    """Only dispatched cancellation can carry the unresolved-reservation signal."""
    data: JsonObject = {
        "outcome": outcome,
        "dispatched": dispatched,
        "usage_incomplete_due_to_disconnect": True,
        "failure": None
        if failure is None
        else {"failure_class": failure, "safe_message": "failed"},
    }
    with pytest.raises(ValueError, match="dispatched cancelled"):
        terminal_from_settlement(data)


@pytest.mark.parametrize("finalize", [False, "true", 1, None])
def test_disconnect_hold_requires_strict_request_finalization(finalize: object) -> None:
    """A held cancelled attempt must close its request, never continue the ladder."""
    payload = {
        "outcome": "failed",
        "failure": {"failure_class": "cancelled", "safe_message": "cancelled"},
        "dispatched": True,
        "usage_incomplete_due_to_disconnect": True,
        "finalize": finalize,
    }
    with pytest.raises(ValueError):
        provenance = NativeSettlementPayload.model_validate(payload)
        provenance.validate_disconnect(
            GatewayFailure(failure_class=GatewayFailureClass.CANCELLED, safe_message="cancelled"),
            GatewayEventKind.FAILED,
        )


def test_ordinary_terminal_without_meter_does_not_infer_disconnect_hold() -> None:
    """Absent evidence leaves the separate terminal-without-meter policy unchanged."""
    terminal, _ = terminal_from_settlement({"outcome": "completed", "usage": None})
    assert terminal.usage_incomplete_due_to_disconnect is False


@pytest.mark.parametrize("full", [False, True])
def test_exhausted_attempt_payload_preserves_exact_optional_boundary_fields(full: bool) -> None:
    """Exhausted selection preserves native wire fields without adding absent values."""
    fields: JsonObject = {"failure_class": "refusal", "safe_message": "request refused"}
    if full:
        fields.update(
            {
                "customer_owned": True,
                "rejected_parameter": "tools",
                "provider_detail": "provider rejected tools",
                "refusal_reason": "cyber_policy",
                "retry_after_seconds": 5,
            }
        )
    failure = GatewayFailure.model_validate(fields)
    assert exhausted_attempt_payload(failure) == json.dumps(
        {"exhausted": True, "failure": fields}, separators=(",", ":")
    )


def test_settlement_metadata_is_exact_content_free_and_host_capability_bound() -> None:
    """Shared projection retains observed time without forwarding unrelated payload contents."""

    def current(*, upstream_provider: str | None = None) -> None:
        """A host accepting the new optional observation."""
        del upstream_provider

    def legacy() -> None:
        """A host whose signature predates the optional observation."""

    fields = settlement_metadata(
        {
            "first_token_at": "2026-09-18T01:02:03+00:00",
            "upstream_provider": "Azure",
            "rate_limit_headers": {"retry-after": "12"},
            "content": "private prompt",
            "authorization": "secret",
        },
        current,
    )
    assert fields == {
        "first_token_at": datetime(2026, 9, 18, 1, 2, 3, tzinfo=UTC),
        "retry_after_seconds": 12,
        "ratelimit_limit_requests": None,
        "ratelimit_remaining_requests": None,
        "ratelimit_limit_tokens": None,
        "ratelimit_remaining_tokens": None,
        "upstream_provider": "Azure",
    }
    assert "upstream_provider" not in settlement_metadata({"upstream_provider": "Azure"}, legacy)
    assert settlement_metadata(None, current)["first_token_at"] is None
    assert settlement_metadata({}, current)["first_token_at"] is None


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


def test_terminal_from_settlement_preserves_cache_write_usage_and_upstream_provider() -> None:
    """Settlement retains the TTL breakdown alongside selected upstream evidence."""
    payload: JsonObject = {
        "outcome": "completed",
        "usage": {
            "input_tokens": 1_000,
            "output_tokens": 10,
            "cache_creation_input_tokens": 600,
            "cache_creation_1h_input_tokens": 200,
        },
        "tool_names": [],
        "failure": None,
        "upstream_provider": "Azure",
    }
    terminal, _failure = terminal_from_settlement(payload)
    assert terminal.usage is not None
    assert terminal.usage.cache_creation_input_tokens == 600
    assert terminal.usage.cache_creation_1h_input_tokens == 200
    assert upstream_provider_from_settlement(payload) == "Azure"


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


def test_all_zero_token_report_on_a_finished_attempt_settles_as_unknown() -> None:
    """A finished attempt whose provider report is zero everywhere carries no usage.

    The OpenAI lane's truncations intermittently report zero input and zero
    output tokens for a prompt the provider processed; the ledger must file
    that as unknown usage, never as an observed free attempt.
    """
    zero = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
    }
    for outcome in ("completed", "incomplete"):
        terminal, failure = terminal_from_settlement(
            {"outcome": outcome, "usage": dict(zero), "tool_names": [], "failure": None}
        )
        assert failure is None
        assert terminal.kind == GatewayEventKind(outcome)
        assert terminal.usage is None

    # Tool names ride the same usage object and the control plane files any
    # non-null usage as observed, so an all-zero report drops them too: a
    # tool call is output the meter should have counted.
    with_tools, _ = terminal_from_settlement(
        {"outcome": "incomplete", "usage": dict(zero), "tool_names": ["search"], "failure": None}
    )
    assert with_tools.usage is None


def test_partial_zero_reports_and_failed_zero_reports_stay_observed() -> None:
    """Only the all-zero FINISHED report is demoted; every other shape is kept verbatim."""
    # A truncation that processed the prompt but produced nothing (OpenRouter
    # codex lanes at a 16-token budget) is a real observation.
    input_only, _ = terminal_from_settlement(
        {
            "outcome": "incomplete",
            "usage": {"input_tokens": 13, "output_tokens": 0, "reasoning_tokens": 0},
            "tool_names": [],
            "failure": None,
        }
    )
    assert input_only.usage is not None
    assert input_only.usage.input_tokens == 13
    assert input_only.usage.output_tokens == 0

    # A failed terminal's zeros settle at nothing either way and a billed
    # refusal keys on positive counts, so the report is kept as sent.
    failed, failure = terminal_from_settlement(
        {
            "outcome": "failed",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "tool_names": [],
            "failure": {"failure_class": "provider_internal", "safe_message": "boom"},
        }
    )
    assert failure is not None
    assert failed.usage is not None
    assert failed.usage.input_tokens == 0


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


def test_upstream_provider_parses_the_aggregators_label_and_nothing_else() -> None:
    """A named upstream threads through; absent, blank, typed-wrong or over-long yields None."""
    assert upstream_provider_from_settlement({"upstream_provider": "Azure"}) == "Azure"
    assert upstream_provider_from_settlement({"upstream_provider": "  Amazon Bedrock "}) == (
        "Amazon Bedrock"
    )
    assert upstream_provider_from_settlement({}) is None
    assert upstream_provider_from_settlement({"upstream_provider": None}) is None
    assert upstream_provider_from_settlement({"upstream_provider": ""}) is None
    assert upstream_provider_from_settlement({"upstream_provider": 7}) is None
    assert upstream_provider_from_settlement({"upstream_provider": "x" * 129}) is None


def test_accepts_keyword_reads_named_and_variadic_signatures() -> None:
    """Named, keyword-only, ``**kwargs`` accept; absent and unreadable do not."""

    def named(*, upstream_provider: str | None = None) -> None:
        del upstream_provider

    def positional(upstream_provider: str | None = None) -> None:
        del upstream_provider

    def variadic(**kwargs: object) -> None:
        del kwargs

    def absent(*, other: int = 0) -> None:
        del other

    assert accepts_keyword(named, "upstream_provider")
    assert accepts_keyword(positional, "upstream_provider")
    assert accepts_keyword(variadic, "upstream_provider")
    assert not accepts_keyword(absent, "upstream_provider")
    assert upstream_provider_kwarg(named, "Azure") == {"upstream_provider": "Azure"}
    assert upstream_provider_kwarg(variadic, None) == {"upstream_provider": None}
    assert upstream_provider_kwarg(absent, "Azure") == {}


@pytest.mark.parametrize("writes", [None, 0, 6108])
def test_cache_write_count_crosses_the_settlement_boundary(writes: int | None) -> None:
    """A reported write count, including zero, reaches hosted settlement unchanged."""
    usage = _usage_from_payload(
        {
            "input_tokens": 6119,
            "output_tokens": 5,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": writes,
        },
        [],
    )
    assert usage is not None
    assert usage.cache_creation_input_tokens == writes


def test_web_search_requests_parses_the_top_level_count_and_nothing_else() -> None:
    """The count sits beside ``usage`` in the settle argument; anything unusable reads as zero."""
    assert web_search_requests_from_settlement({"web_search_requests": 2}) == 2
    assert web_search_requests_from_settlement({}) == 0
    assert web_search_requests_from_settlement(None) == 0
    assert web_search_requests_from_settlement({"web_search_requests": None}) == 0
    assert web_search_requests_from_settlement({"web_search_requests": "2"}) == 0
    assert web_search_requests_from_settlement({"web_search_requests": True}) == 0
    assert web_search_requests_from_settlement({"web_search_requests": -1}) == 0
    # Inside "usage" is the wrong place: the engine never puts it there.
    assert web_search_requests_from_settlement({"usage": {"web_search_requests": 2}}) == 0


def test_web_search_requests_ride_on_the_settled_usage() -> None:
    """Token-bearing and tool-only usage both carry the count; no usage means no carrier."""
    tokens: JsonObject = {"input_tokens": 8, "output_tokens": 3}
    terminal, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": tokens, "tool_names": [], "web_search_requests": 2}
    )
    assert terminal.usage is not None and terminal.usage.web_search_requests == 2
    assert web_search_requests_from_terminal(terminal) == 2

    tools_only, _failure = terminal_from_settlement(
        {
            "outcome": "completed",
            "usage": None,
            "tool_names": ["web_search"],
            "web_search_requests": 1,
        }
    )
    assert tools_only.usage is not None and tools_only.usage.web_search_requests == 1
    assert not tools_only.usage.has_token_counts

    # A count with neither tokens nor tool names has nothing to ride on: the
    # contract's validator keeps a bare count from being usage, so it is dropped.
    bare, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": None, "tool_names": [], "web_search_requests": 3}
    )
    assert bare.usage is None
    assert web_search_requests_from_terminal(bare) == 0
    assert web_search_requests_from_terminal(None) == 0

    # Absent stays byte-identical to the pre-field engine: zero on the usage.
    absent, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": tokens, "tool_names": []}
    )
    assert absent.usage is not None and absent.usage.web_search_requests == 0
    assert _usage_from_payload(tokens, [], web_search_requests=4) == GatewayUsage(
        input_tokens=8, output_tokens=3, web_search_requests=4
    )


def test_web_search_requests_kwarg_is_withheld_at_zero_and_from_a_legacy_ledger() -> None:
    """Same seam as ``upstream_provider_kwarg``, plus: a zero count sends nothing at all."""

    def named(*, web_search_requests: int = 0) -> None:
        del web_search_requests

    def variadic(**kwargs: object) -> None:
        del kwargs

    def absent(*, upstream_provider: str | None = None) -> None:
        del upstream_provider

    assert web_search_requests_kwarg(named, 2) == {"web_search_requests": 2}
    assert web_search_requests_kwarg(variadic, 1) == {"web_search_requests": 1}
    assert web_search_requests_kwarg(absent, 2) == {}
    assert web_search_requests_kwarg(named, 0) == {}
    assert web_search_requests_kwarg(variadic, 0) == {}


def test_tool_search_requests_parses_the_top_level_count_and_nothing_else() -> None:
    """The count sits beside ``usage`` in the settle argument; anything unusable reads as zero."""
    assert tool_search_requests_from_settlement({"tool_search_requests": 2}) == 2
    assert tool_search_requests_from_settlement({}) == 0
    assert tool_search_requests_from_settlement(None) == 0
    assert tool_search_requests_from_settlement({"tool_search_requests": None}) == 0
    assert tool_search_requests_from_settlement({"tool_search_requests": "2"}) == 0
    assert tool_search_requests_from_settlement({"tool_search_requests": True}) == 0
    assert tool_search_requests_from_settlement({"tool_search_requests": -1}) == 0
    # Inside "usage" is the wrong place: the engine never puts it there.
    assert tool_search_requests_from_settlement({"usage": {"tool_search_requests": 2}}) == 0
    # The two meters are independent keys: one never reads as the other.
    assert tool_search_requests_from_settlement({"web_search_requests": 2}) == 0
    assert web_search_requests_from_settlement({"tool_search_requests": 2}) == 0


def test_tool_search_requests_ride_on_the_settled_usage() -> None:
    """Token-bearing and tool-only usage both carry the count; no usage means no carrier."""
    tokens: JsonObject = {"input_tokens": 8, "output_tokens": 3}
    terminal, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": tokens, "tool_names": [], "tool_search_requests": 2}
    )
    assert terminal.usage is not None and terminal.usage.tool_search_requests == 2
    assert terminal.usage.web_search_requests == 0
    assert tool_search_requests_from_terminal(terminal) == 2

    tools_only, _failure = terminal_from_settlement(
        {
            "outcome": "completed",
            "usage": None,
            "tool_names": ["tool_search"],
            "tool_search_requests": 1,
        }
    )
    assert tools_only.usage is not None and tools_only.usage.tool_search_requests == 1
    assert not tools_only.usage.has_token_counts

    # A count with neither tokens nor tool names has nothing to ride on: the
    # contract's validator keeps a bare count from being usage, so it is dropped.
    bare, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": None, "tool_names": [], "tool_search_requests": 3}
    )
    assert bare.usage is None
    assert tool_search_requests_from_terminal(bare) == 0
    assert tool_search_requests_from_terminal(None) == 0

    # Absent stays byte-identical to the pre-field engine: zero on the usage.
    absent, _failure = terminal_from_settlement(
        {"outcome": "completed", "usage": tokens, "tool_names": []}
    )
    assert absent.usage is not None and absent.usage.tool_search_requests == 0
    assert _usage_from_payload(tokens, [], tool_search_requests=4) == GatewayUsage(
        input_tokens=8, output_tokens=3, tool_search_requests=4
    )
    # Both meters settle side by side on one usage.
    both, _failure = terminal_from_settlement(
        {
            "outcome": "completed",
            "usage": tokens,
            "tool_names": [],
            "web_search_requests": 1,
            "tool_search_requests": 2,
        }
    )
    assert both.usage is not None
    assert (both.usage.web_search_requests, both.usage.tool_search_requests) == (1, 2)


def test_tool_search_requests_kwarg_is_withheld_at_zero_and_from_a_legacy_ledger() -> None:
    """Same seam as ``web_search_requests_kwarg``: zero or a pre-meter ledger gets nothing."""

    def named(*, tool_search_requests: int = 0) -> None:
        del tool_search_requests

    def variadic(**kwargs: object) -> None:
        del kwargs

    def pre_tool_search(*, web_search_requests: int = 0) -> None:
        del web_search_requests

    assert tool_search_requests_kwarg(named, 2) == {"tool_search_requests": 2}
    assert tool_search_requests_kwarg(variadic, 1) == {"tool_search_requests": 1}
    assert tool_search_requests_kwarg(pre_tool_search, 2) == {}
    assert tool_search_requests_kwarg(named, 0) == {}
    assert tool_search_requests_kwarg(variadic, 0) == {}
