"""Tests for the shared transport request helpers and bounded retry classification."""

from __future__ import annotations

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
    classify_retry,
    get_json,
    run_with_retry,
)

_IMMEDIATE_RETRY = RetryPolicy(maximum_attempts=2, initial_delay_seconds=0, maximum_delay_seconds=0)


def test_get_json_returns_the_first_success_body_for_one_attempt() -> None:
    """A success status is decoded once without any additional attempt."""
    transport = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={"data": []})])

    body: JsonObject = get_json(
        transport,
        "https://provider.test/v1/models",
        headers={"Authorization": "Bearer secret"},
        timeout_seconds=1.0,
        retry_policy=_IMMEDIATE_RETRY,
    )

    assert body == {"data": []}
    assert len(transport.requests) == 1


def test_get_json_retries_one_retryable_status_before_succeeding() -> None:
    """A retryable status is retried against the same endpoint within the attempt bound."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=503, body={}),
            JsonHttpResponse(status_code=200, body={"data": [{"id": "model"}]}),
        ]
    )

    body = get_json(
        transport,
        "https://provider.test/v1/models",
        headers={"Authorization": "Bearer secret"},
        timeout_seconds=1.0,
        retry_policy=_IMMEDIATE_RETRY,
    )

    assert body == {"data": [{"id": "model"}]}
    assert [request.url for request in transport.requests] == [
        "https://provider.test/v1/models",
        "https://provider.test/v1/models",
    ]


def test_get_json_reports_a_rejected_credential_status_without_retrying() -> None:
    """A non-retryable status fails immediately and carries its status code."""
    transport = ScriptedJsonTransport([JsonHttpResponse(status_code=401, body={})])

    with pytest.raises(ProviderTransportError) as error:
        get_json(
            transport,
            "https://provider.test/v1/models",
            headers={"Authorization": "Bearer secret"},
            timeout_seconds=1.0,
            retry_policy=_IMMEDIATE_RETRY,
        )

    assert error.value.status_code == 401
    assert "secret" not in str(error.value)
    assert len(transport.requests) == 1


@pytest.mark.parametrize(
    ("exception", "retryable"),
    [
        (ProviderTransportError("unavailable", status_code=503), True),
        (ProviderTransportError("bad request", status_code=400), False),
        (ProviderTransportError("network"), True),
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
        """Fail once with a retryable status, then succeed."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderTransportError("busy", status_code=429)
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
