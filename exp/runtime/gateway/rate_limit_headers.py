"""Normalize provider rate-limit response headers into one typed observation.

The native data plane forwards the small allowlisted subset of provider
response headers that describe rate limiting (``retry-after`` plus the OpenAI
``x-ratelimit-*`` and Anthropic ``anthropic-ratelimit-*`` families, and the
ChatGPT plan backend's ``x-codex-*`` usage windows) on the
settlement payload, for successes and failures alike. This module owns the
one place those raw header strings become typed integers: the observation
rides the attempt ledger for calibration analytics, and a throttled
settlement's ``retry-after`` sizes the deployment's throttle window instead
of the fixed default. Parsing is deliberately tolerant: a missing, garbled,
or unknown header yields ``None`` for its field, never an error, because a
provider's header hygiene must not be able to fail a settlement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal

from exp.common.core.artifacts import ContractModel

SubscriptionWindowName = Literal["primary", "secondary"]
"""A plan's rolling usage windows: the short one (hours) and the long one (a week)."""

EXHAUSTED_USED_PERCENT = 100
"""The used-percent reading at which a plan window admits no more requests."""


class SubscriptionWindowObservation(ContractModel):
    """One plan usage window as the ChatGPT backend reports it on every response."""

    window: SubscriptionWindowName
    used_percent: int
    """Share of the window's allowance already spent, 0 to 100."""
    reset_after_seconds: int | None = None
    """Seconds until the window rolls over, when the provider states it."""
    window_minutes: int | None = None
    """The window's length, when the provider states it."""

    @property
    def exhausted(self) -> bool:
        """Whether the window admits no more requests until it resets."""
        return self.used_percent >= EXHAUSTED_USED_PERCENT


class RateLimitObservation(ContractModel):
    """Typed rate-limit facts harvested from one provider response's headers."""

    retry_after_seconds: int | None = None
    """Provider-requested wait before the next attempt, floored at one second."""
    limit_requests: int | None = None
    """The account's request-rate ceiling as the provider states it."""
    remaining_requests: int | None = None
    """Requests left in the provider's current window."""
    limit_tokens: int | None = None
    """The account's token-rate ceiling as the provider states it."""
    remaining_tokens: int | None = None
    """Tokens left in the provider's current window."""
    subscription_windows: tuple[SubscriptionWindowObservation, ...] = ()
    """Plan usage windows, in primary-then-secondary order, when the rung is a plan."""

    @property
    def is_empty(self) -> bool:
        """Whether no header produced a value."""
        return (
            self.retry_after_seconds is None
            and self.limit_requests is None
            and self.remaining_requests is None
            and self.limit_tokens is None
            and self.remaining_tokens is None
            and not self.subscription_windows
        )

    def subscription_window(
        self, window: SubscriptionWindowName
    ) -> SubscriptionWindowObservation | None:
        """Return one named plan window, or ``None`` when the response carried none."""
        return next((item for item in self.subscription_windows if item.window == window), None)

    @property
    def exhausted_reset_after_seconds(self) -> int | None:
        """The longest stated reset among exhausted plan windows, or ``None`` when none is.

        A plan whose short window is spent stays unusable until that window rolls over,
        and a spent long window outlasts a spent short one, so the longest reset is the
        wait that actually reopens the rung.
        """
        waits = [
            item.reset_after_seconds
            for item in self.subscription_windows
            if item.exhausted and item.reset_after_seconds is not None
        ]
        return max(waits) if waits else None


_EMPTY_OBSERVATION = RateLimitObservation()

# Bounds on what a provider header may claim. The raw strings cross the
# boundary unbounded (Python integers are arbitrary precision), and a value
# past SQLite's signed 64-bit column would fail every settlement write for the
# attempt — a wedge one hostile BYOK server header must never be able to
# cause. A wait is clamped to a week (a longer ask is still "come back much
# later" for the ledger; the health window clamps far tighter anyway); a
# limit/remaining count past the sanity ceiling is garbage and reads as absent.
MAXIMUM_RETRY_AFTER_SECONDS = 7 * 24 * 3_600
_MAXIMUM_OBSERVED_COUNT = 10**15

# The small provider header map: one canonical field per provider spelling.
# OpenAI-compatible wires send x-ratelimit-*; Anthropic sends
# anthropic-ratelimit-*. Nothing else is special-cased per provider.
_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("x-ratelimit-limit-requests", "limit_requests"),
    ("x-ratelimit-remaining-requests", "remaining_requests"),
    ("x-ratelimit-limit-tokens", "limit_tokens"),
    ("x-ratelimit-remaining-tokens", "remaining_tokens"),
    ("anthropic-ratelimit-requests-limit", "limit_requests"),
    ("anthropic-ratelimit-requests-remaining", "remaining_requests"),
    ("anthropic-ratelimit-tokens-limit", "limit_tokens"),
    ("anthropic-ratelimit-tokens-remaining", "remaining_tokens"),
)


def parse_retry_after_seconds(value: str, *, now: datetime | None = None) -> int | None:
    """Parse one raw ``Retry-After`` header value into whole seconds.

    Both RFC 9110 forms are accepted: a nonnegative integer second count and
    an HTTP-date, whose wait is measured from ``now``. A parseable wait is
    floored at one second so a zero or already-passed date still expresses
    "back off briefly" rather than vanishing, and capped at
    ``MAXIMUM_RETRY_AFTER_SECONDS`` so one absurd header can neither distort
    the ledger nor overflow an integer column; anything unparseable yields
    ``None`` so garbage degrades to the caller's default window.

    Args:
        value: Raw header value.
        now: Reference wall-clock time for the HTTP-date form; defaults to
            the current UTC time.

    Returns:
        Whole seconds to wait, or ``None`` when the value is unparseable.
    """
    text = value.strip()
    if not text:
        return None
    try:
        seconds = int(text)
    except ValueError:
        pass
    else:
        if seconds < 0:
            return None
        return min(max(1, seconds), MAXIMUM_RETRY_AFTER_SECONDS)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    reference = now if now is not None else datetime.now(UTC)
    return min(max(1, math.ceil((when - reference).total_seconds())), MAXIMUM_RETRY_AFTER_SECONDS)


def rate_limit_observation(headers: Mapping[str, str]) -> RateLimitObservation:
    """Build one typed observation from raw provider response headers.

    Args:
        headers: Header names (any case) mapped to raw string values; only
            the allowlisted rate-limit subset is read.

    Returns:
        The parsed observation; fields whose headers are absent or garbled
        stay ``None``.
    """
    lowered = {str(name).lower(): str(value) for name, value in headers.items()}
    values: dict[str, int | None] = {}
    for header, field_name in _HEADER_FIELDS:
        if values.get(field_name) is not None:
            continue
        raw = lowered.get(header)
        if raw is not None:
            values[field_name] = _parse_count(raw)
    retry_after = lowered.get("retry-after")
    return RateLimitObservation(
        retry_after_seconds=(
            None if retry_after is None else parse_retry_after_seconds(retry_after)
        ),
        limit_requests=values.get("limit_requests"),
        remaining_requests=values.get("remaining_requests"),
        limit_tokens=values.get("limit_tokens"),
        remaining_tokens=values.get("remaining_tokens"),
        subscription_windows=_subscription_windows(lowered),
    )


def _subscription_windows(lowered: Mapping[str, str]) -> tuple[SubscriptionWindowObservation, ...]:
    """Read the ChatGPT plan backend's ``x-codex-*`` usage windows.

    A window is reported only when its used-percent header parses; the reset and
    length stay ``None`` when theirs do not, so one garbled header never hides the
    window's exhaustion.

    Args:
        lowered: Lowercased header names mapped to raw values.

    Returns:
        The windows present, primary first.
    """
    windows: list[SubscriptionWindowObservation] = []
    for name in ("primary", "secondary"):
        used_raw = lowered.get(f"x-codex-{name}-used-percent")
        used = None if used_raw is None else _parse_count(used_raw)
        if used is None:
            continue
        reset_raw = lowered.get(f"x-codex-{name}-reset-after-seconds")
        length_raw = lowered.get(f"x-codex-{name}-window-minutes")
        windows.append(
            SubscriptionWindowObservation(
                window=name,
                used_percent=min(used, EXHAUSTED_USED_PERCENT),
                reset_after_seconds=None if reset_raw is None else _parse_count(reset_raw),
                window_minutes=None if length_raw is None else _parse_count(length_raw),
            )
        )
    return tuple(windows)


def rate_limit_observation_from_payload(payload: object) -> RateLimitObservation:
    """Read the optional raw-header map off one boundary settlement payload.

    Args:
        payload: The settlement's ``rate_limit_headers`` value: a mapping of
            raw header names to string values when the data plane harvested
            any, else anything (absent, null, wrong shape).

    Returns:
        The parsed observation; a missing or malformed payload yields the
        empty observation rather than an error.
    """
    if not isinstance(payload, Mapping):
        return _EMPTY_OBSERVATION
    return rate_limit_observation(
        {str(name): str(value) for name, value in payload.items() if isinstance(value, str)}
    )


def _parse_count(value: str) -> int | None:
    """Parse one nonnegative integer header value, tolerating garbage.

    A count past the sanity ceiling is treated as garbage rather than clamped:
    no provider states a real quota there, and an unbounded integer would
    overflow the ledger's 64-bit columns and fail the settlement write.
    """
    try:
        count = int(value.strip())
    except ValueError:
        return None
    return count if 0 <= count <= _MAXIMUM_OBSERVED_COUNT else None
