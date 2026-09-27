"""Retry-After parsing and provider header-map normalization tests."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.runtime.gateway.rate_limit_headers import (
    MAXIMUM_RETRY_AFTER_SECONDS,
    RateLimitObservation,
    parse_retry_after_seconds,
    rate_limit_observation,
    rate_limit_observation_from_payload,
)

_NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)


class TestParseRetryAfter:
    """Both RFC 9110 forms parse; garbage degrades to None."""

    def test_integer_seconds_floor_at_one(self) -> None:
        """Plain seconds parse; zero still expresses a minimal backoff."""
        assert parse_retry_after_seconds("30") == 30
        assert parse_retry_after_seconds(" 7 ") == 7
        assert parse_retry_after_seconds("0") == 1

    def test_http_date_measures_from_now(self) -> None:
        """An HTTP-date wait is the ceiling of the remaining seconds."""
        assert parse_retry_after_seconds("Mon, 07 Sep 2026 12:02:00 GMT", now=_NOW) == 120
        # An already-passed date still floors at one second of backoff.
        assert parse_retry_after_seconds("Mon, 07 Sep 2026 11:00:00 GMT", now=_NOW) == 1

    def test_garbage_and_negatives_yield_none(self) -> None:
        """Unparseable values never raise and never invent a wait."""
        for value in ("", "soon", "-5", "Mon, 99 Foo 2026", "12.5"):
            assert parse_retry_after_seconds(value, now=_NOW) is None

    def test_absurd_waits_clamp_to_the_ceiling(self) -> None:
        """A wait past the week ceiling clamps for both RFC forms.

        Python parses arbitrary-precision integers, and an unbounded value
        would overflow the ledger's signed 64-bit column and wedge the
        settlement in a retry loop, so one hostile BYOK server header must
        never cross the boundary unbounded.
        """
        assert parse_retry_after_seconds("9" * 40, now=_NOW) == MAXIMUM_RETRY_AFTER_SECONDS
        assert (
            parse_retry_after_seconds("Fri, 01 Jan 2100 00:00:00 GMT", now=_NOW)
            == MAXIMUM_RETRY_AFTER_SECONDS
        )
        # A day-scale quota reset stays exact: the ceiling only cuts absurdity.
        assert parse_retry_after_seconds("86400", now=_NOW) == 86_400


class TestHeaderMap:
    """The small OpenAI and Anthropic header families normalize identically."""

    def test_openai_family(self) -> None:
        """x-ratelimit-* headers map onto the four count fields."""
        observation = rate_limit_observation(
            {
                "X-RateLimit-Limit-Requests": "10000",
                "x-ratelimit-remaining-requests": "9998",
                "x-ratelimit-limit-tokens": "180000000",
                "x-ratelimit-remaining-tokens": "179000000",
                "retry-after": "12",
            }
        )
        assert observation == RateLimitObservation(
            retry_after_seconds=12,
            limit_requests=10_000,
            remaining_requests=9_998,
            limit_tokens=180_000_000,
            remaining_tokens=179_000_000,
        )

    def test_anthropic_family(self) -> None:
        """anthropic-ratelimit-* headers map onto the same fields."""
        observation = rate_limit_observation(
            {
                "anthropic-ratelimit-requests-limit": "10000",
                "anthropic-ratelimit-requests-remaining": "9500",
                "anthropic-ratelimit-tokens-limit": "12000000",
                "anthropic-ratelimit-tokens-remaining": "11000000",
            }
        )
        assert observation.limit_requests == 10_000
        assert observation.remaining_requests == 9_500
        assert observation.limit_tokens == 12_000_000
        assert observation.remaining_tokens == 11_000_000
        assert observation.retry_after_seconds is None

    def test_garbled_counts_stay_none_and_unknown_headers_are_ignored(self) -> None:
        """A garbled or negative count yields None; unrelated headers do nothing."""
        observation = rate_limit_observation(
            {
                "x-ratelimit-limit-requests": "many",
                "x-ratelimit-remaining-requests": "-2",
                "content-type": "application/json",
            }
        )
        assert observation.limit_requests is None
        assert observation.remaining_requests is None
        assert observation.is_empty

    def test_counts_past_the_sanity_ceiling_read_as_absent(self) -> None:
        """An arbitrary-precision count never reaches the 64-bit ledger columns."""
        observation = rate_limit_observation(
            {
                "x-ratelimit-limit-tokens": "9" * 40,
                "x-ratelimit-remaining-tokens": "15000000000",
            }
        )
        assert observation.limit_tokens is None
        # A large-but-real published quota (OpenAI scale-tier TPM) stays exact.
        assert observation.remaining_tokens == 15_000_000_000


class TestPayloadTolerance:
    """The boundary reader degrades to the empty observation, never an error."""

    def test_absent_or_malformed_payload_is_empty(self) -> None:
        """None, wrong shapes, and non-string values all degrade quietly."""
        assert rate_limit_observation_from_payload(None).is_empty
        assert rate_limit_observation_from_payload("headers").is_empty
        assert rate_limit_observation_from_payload(["retry-after"]).is_empty
        assert rate_limit_observation_from_payload({"retry-after": 30}).is_empty

    def test_mapping_payload_parses(self) -> None:
        """A well-shaped payload map parses exactly like raw headers."""
        observation = rate_limit_observation_from_payload({"retry-after": "45"})
        assert observation.retry_after_seconds == 45


class TestPlanWindows:
    """The ChatGPT plan backend's ``x-codex-*`` usage windows become typed observations."""

    def test_both_windows_parse_primary_first(self) -> None:
        """Percent, reset, and length ride each window; the observation is not empty."""
        observation = rate_limit_observation(
            {
                "X-Codex-Primary-Used-Percent": "22",
                "x-codex-primary-reset-after-seconds": "11511",
                "x-codex-primary-window-minutes": "300",
                "x-codex-secondary-used-percent": "4",
                "x-codex-secondary-reset-after-seconds": "598311",
                "x-codex-secondary-window-minutes": "10080",
            }
        )

        assert [window.window for window in observation.subscription_windows] == [
            "primary",
            "secondary",
        ]
        primary = observation.subscription_window("primary")
        assert primary is not None
        assert (primary.used_percent, primary.reset_after_seconds, primary.window_minutes) == (
            22,
            11_511,
            300,
        )
        assert not observation.is_empty
        assert observation.exhausted_reset_after_seconds is None

    def test_exhaustion_takes_the_longest_reset_among_spent_windows(self) -> None:
        """A spent long window outlasts a spent short one, so its reset is the wait."""
        observation = rate_limit_observation(
            {
                "x-codex-primary-used-percent": "100",
                "x-codex-primary-reset-after-seconds": "600",
                "x-codex-secondary-used-percent": "100",
                "x-codex-secondary-reset-after-seconds": "80000",
            }
        )

        assert observation.exhausted_reset_after_seconds == 80_000

    def test_a_window_without_a_percent_is_absent_and_garbage_fields_stay_none(self) -> None:
        """Only a parseable used-percent reports a window; other garbled fields degrade."""
        observation = rate_limit_observation(
            {
                "x-codex-primary-used-percent": "137",
                "x-codex-primary-reset-after-seconds": "soon",
                "x-codex-secondary-reset-after-seconds": "600",
            }
        )

        primary = observation.subscription_window("primary")
        assert primary is not None
        assert primary.used_percent == 100
        assert primary.exhausted
        assert primary.reset_after_seconds is None
        assert observation.subscription_window("secondary") is None
        assert observation.exhausted_reset_after_seconds is None

    def test_payload_without_plan_headers_has_no_windows(self) -> None:
        """API-key rungs keep an empty window tuple."""
        assert rate_limit_observation_from_payload({"retry-after": "5"}).subscription_windows == ()


class TestClaudePlanWindows:
    """A Claude plan's ``anthropic-ratelimit-unified-*`` windows map onto the plan window shape."""

    _NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)

    def test_fraction_utilization_and_second_resets_parse_short_window_first(self) -> None:
        """A 0..1 utilization becomes a percent; the reset is measured from now."""
        reset = int(self._NOW.timestamp()) + 3_600
        observation = rate_limit_observation(
            {
                "anthropic-ratelimit-unified-5h-utilization": "0.42",
                "anthropic-ratelimit-unified-5h-reset": str(reset),
                "anthropic-ratelimit-unified-7d-utilization": "7",
                "anthropic-ratelimit-unified-7d-reset": str(reset * 1_000),
            },
            now=self._NOW,
        )

        primary = observation.subscription_window("primary")
        secondary = observation.subscription_window("secondary")
        assert primary is not None
        assert secondary is not None
        assert (primary.used_percent, primary.reset_after_seconds, primary.window_minutes) == (
            42,
            3_600,
            300,
        )
        assert (secondary.used_percent, secondary.reset_after_seconds) == (7, 3_600)
        assert observation.exhausted_reset_after_seconds is None

    def test_a_rejected_status_is_a_used_up_window_whatever_the_utilization(self) -> None:
        """The provider's own refusal verdict rotates the plan for the stated reset."""
        reset = int(self._NOW.timestamp()) + 90_000
        observation = rate_limit_observation(
            {
                "anthropic-ratelimit-unified-7d-status": "rejected",
                "anthropic-ratelimit-unified-7d-reset": str(reset),
            },
            now=self._NOW,
        )

        assert observation.exhausted_reset_after_seconds == 90_000

    def test_garbage_utilization_without_a_verdict_reports_no_window(self) -> None:
        """An unparseable utilization and no rejected status is simply absent."""
        observation = rate_limit_observation(
            {"anthropic-ratelimit-unified-5h-utilization": "lots"}, now=self._NOW
        )

        assert observation.subscription_windows == ()

    def test_the_written_form_decides_fraction_or_percent_and_floors_the_reading(self) -> None:
        """``1.05`` is past the allowance, ``1`` is one percent, and 99.6 percent stays open."""
        observation = rate_limit_observation(
            {
                "anthropic-ratelimit-unified-5h-utilization": "1.05",
                "anthropic-ratelimit-unified-5h-reset": str(int(self._NOW.timestamp()) + 60),
                "anthropic-ratelimit-unified-7d-utilization": "0.996",
            },
            now=self._NOW,
        )

        primary = observation.subscription_window("primary")
        secondary = observation.subscription_window("secondary")
        assert primary is not None and primary.exhausted
        assert secondary is not None and secondary.used_percent == 99
        assert not secondary.exhausted
        one_percent = rate_limit_observation(
            {"anthropic-ratelimit-unified-5h-utilization": "1"}, now=self._NOW
        ).subscription_window("primary")
        assert one_percent is not None and one_percent.used_percent == 1

    def test_a_reset_already_past_still_waits_at_least_one_second(self) -> None:
        """An exhausted window whose reset has passed yields the Retry-After floor, never 0."""
        observation = rate_limit_observation(
            {
                "anthropic-ratelimit-unified-5h-status": "rejected",
                "anthropic-ratelimit-unified-5h-reset": str(int(self._NOW.timestamp()) - 30),
            },
            now=self._NOW,
        )

        assert observation.exhausted_reset_after_seconds == 1


class TestChatGptDecimalWindows:
    """The ChatGPT backend states used-percent as a decimal and may state the reset as a time."""

    _NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)

    def test_decimal_percent_and_reset_timestamp_exhaust_the_window(self) -> None:
        """``100.0`` is a spent window and ``reset-at`` supplies its wait."""
        observation = rate_limit_observation(
            {
                "x-codex-primary-used-percent": "12.5",
                "x-codex-secondary-used-percent": "100.0",
                "x-codex-secondary-reset-at": str(int(self._NOW.timestamp()) + 86_400),
            },
            now=self._NOW,
        )

        primary = observation.subscription_window("primary")
        assert primary is not None and primary.used_percent == 12
        assert observation.exhausted_reset_after_seconds == 86_400

    def test_an_unbounded_reset_is_clamped_to_the_retry_after_ceiling(self) -> None:
        """A garbage-large stated reset never outruns the Retry-After ceiling."""
        observation = rate_limit_observation(
            {
                "x-codex-primary-used-percent": "100",
                "x-codex-primary-reset-after-seconds": str(10**12),
            }
        )

        assert observation.exhausted_reset_after_seconds == MAXIMUM_RETRY_AFTER_SECONDS
