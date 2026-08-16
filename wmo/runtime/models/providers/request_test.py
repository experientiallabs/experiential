"""Focused tests for the shared bounded provider JSON request helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.request import get_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
)

_IMMEDIATE_RETRY = RetryPolicy(maximum_attempts=2, initial_delay_seconds=0, maximum_delay_seconds=0)


class _ScriptedTransport(JsonHttpTransport):
    """One transport that replays scripted GET responses and records every request."""

    def __init__(self, responses: Sequence[JsonHttpResponse]) -> None:
        """Store the responses returned in order by successive GET attempts."""
        self._responses = list(responses)
        self.requests: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one attempt and return the next scripted response."""
        self.requests.append((url, dict(headers)))
        return self._responses.pop(0)


def test_get_json_returns_the_first_success_body_for_one_attempt() -> None:
    """A success status is decoded once without any additional attempt."""
    transport = _ScriptedTransport([JsonHttpResponse(status_code=200, body={"data": []})])

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
    transport = _ScriptedTransport(
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
    assert [url for url, _ in transport.requests] == [
        "https://provider.test/v1/models",
        "https://provider.test/v1/models",
    ]


def test_get_json_reports_a_rejected_credential_status_without_retrying() -> None:
    """A non-retryable status fails immediately and carries its status code."""
    transport = _ScriptedTransport([JsonHttpResponse(status_code=401, body={})])

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
