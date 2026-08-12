"""Bounded same-endpoint retry classification for provider requests."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from wmo.runtime.models.providers.transport import ProviderTransportError

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryClassification:
    """Whether an error merits one or more same-endpoint retry attempts."""

    retryable: bool
    reason: str


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry policy without provider failover semantics."""

    maximum_attempts: int = 3
    initial_delay_seconds: float = 0.25
    maximum_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least one")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError("maximum_delay_seconds cannot be smaller than initial_delay_seconds")


def classify_retry(exception: Exception) -> RetryClassification:
    """Classify one error without consulting provider-specific failover policy.

    Args:
        exception: Error raised by one request attempt.

    Returns:
        A stable retry decision and concise reason.
    """
    if isinstance(exception, ProviderTransportError):
        if exception.status_code is None:
            return RetryClassification(retryable=True, reason="transport")
        if exception.status_code in _RETRYABLE_STATUS_CODES:
            return RetryClassification(retryable=True, reason=f"http_{exception.status_code}")
        return RetryClassification(retryable=False, reason=f"http_{exception.status_code}")
    if isinstance(exception, TimeoutError):
        return RetryClassification(retryable=True, reason="timeout")
    if isinstance(exception, OSError):
        return RetryClassification(retryable=True, reason="os_error")
    return RetryClassification(retryable=False, reason="non_transport_error")


def run_with_retry[ResultT](
    operation: Callable[[], ResultT],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> ResultT:
    """Run one idempotent request operation with bounded same-endpoint retries.

    Args:
        operation: A single idempotent provider request attempt.
        policy: Attempt and delay limits.
        sleep: Delay function, injectable for deterministic tests.

    Returns:
        The operation's first successful result.

    Raises:
        Exception: The first non-retryable error or last retryable error.
    """
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.maximum_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            classification = classify_retry(exc)
            if not classification.retryable or attempt == policy.maximum_attempts:
                raise
            if delay > 0:
                sleep(delay)
            delay = min(delay * 2, policy.maximum_delay_seconds)
    raise RuntimeError("retry loop exhausted without running an attempt")
