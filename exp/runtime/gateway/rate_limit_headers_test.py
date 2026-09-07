"""Retry-After parsing and provider header-map normalization tests."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.runtime.gateway.rate_limit_headers import (
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
