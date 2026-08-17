"""Tests for bounded provider retry classification."""

from __future__ import annotations

import pytest

from wmo.runtime.models.providers.errors import ProviderError
from wmo.runtime.models.providers.retry import RetryPolicy, classify_retry, run_with_retry


@pytest.mark.parametrize(
    ("exception", "retryable"),
    [
        (ProviderError("unavailable", status_code=503), True),
        (ProviderError("bad request", status_code=400), False),
        (ProviderError("network"), True),
        (
            ProviderError(
                "unsupported",
                status_code=400,
                error_code="unsupported_parameter",
                rejected_parameter="temperature",
            ),
            False,
        ),
        (TimeoutError("slow"), True),
        (ValueError("invalid request"), False),
    ],
)
def test_retry_classification_is_transport_specific(exception: Exception, retryable: bool) -> None:
    """Only transport-shaped errors retry, with no semantic failover branch."""
    assert classify_retry(exception).retryable is retryable


def test_retry_runs_a_bounded_same_operation() -> None:
    """A retry returns the later success and records deterministic delay behavior."""
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderError("busy", status_code=429)
        return "ok"

    assert (
        run_with_retry(
            operation,
            policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=0.5),
            sleep=delays.append,
        )
        == "ok"
    )
    assert attempts == 2
    assert delays == [0.5]
