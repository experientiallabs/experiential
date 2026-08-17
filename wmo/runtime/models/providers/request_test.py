"""Focused tests for the shared bounded provider JSON request helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.errors import ProviderError
from wmo.runtime.models.providers.request import get_json, post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
)

_IMMEDIATE_RETRY = RetryPolicy(maximum_attempts=2, initial_delay_seconds=0, maximum_delay_seconds=0)
_SECRET = "sk-secret-live-key-1234567890"
_PROMPT = "Score this hidden trace content."


class _ScriptedTransport(JsonHttpTransport):
    """One transport that replays scripted GET or POST responses and records every request."""

    def __init__(self, responses: Sequence[JsonHttpResponse | ProviderError]) -> None:
        """Store the responses or failures returned in order by successive attempts."""
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
        return self._next()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one POST attempt and return the next scripted response."""
        self.requests.append((url, dict(headers)))
        return self._next()

    def _next(self) -> JsonHttpResponse:
        """Return the next scripted success or raise the next scripted failure."""
        item = self._responses.pop(0)
        if isinstance(item, ProviderError):
            raise item
        return item


def test_get_json_returns_the_first_success_body_for_one_attempt() -> None:
    """A success status is decoded once without any additional attempt."""
    transport = _ScriptedTransport([JsonHttpResponse(status_code=200, body={"data": []})])

    body: JsonObject = get_json(
        transport,
        "https://provider.test/v1/models",
        headers={"Authorization": "Bearer secret"},
        timeout_seconds=1.0,
        retry_policy=_IMMEDIATE_RETRY,
        provider="openai",
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
        provider="openai",
    )

    assert body == {"data": [{"id": "model"}]}
    assert [url for url, _ in transport.requests] == [
        "https://provider.test/v1/models",
        "https://provider.test/v1/models",
    ]


def test_get_json_reports_a_rejected_credential_status_without_retrying() -> None:
    """A non-retryable status fails immediately and carries its status code."""
    transport = _ScriptedTransport([JsonHttpResponse(status_code=401, body={})])

    with pytest.raises(ProviderError) as error:
        get_json(
            transport,
            "https://provider.test/v1/models",
            headers={"Authorization": "Bearer secret"},
            timeout_seconds=1.0,
            retry_policy=_IMMEDIATE_RETRY,
            provider="openai",
        )

    assert error.value.status_code == 401
    assert error.value.provider == "openai"
    assert error.value.endpoint_class == "models"
    assert error.value.retryable is False
    assert "secret" not in str(error.value)
    assert len(transport.requests) == 1


def test_post_json_parses_openai_unsupported_temperature_without_retry_or_secrets() -> None:
    """A documented Responses rejection is typed, non-retryable, and secret-free."""
    transport = _ScriptedTransport(
        [
            JsonHttpResponse(
                status_code=400,
                body={
                    "error": {
                        "message": (
                            "Unsupported parameter: 'temperature' is not supported with this "
                            f"model. Authorization: Bearer {_SECRET} prompt={_PROMPT}"
                        ),
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_parameter",
                    }
                },
                request_id="req_luna_1",
            )
        ]
    )

    with pytest.raises(ProviderError) as error:
        post_json(
            transport,
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {_SECRET}"},
            payload={"model": "gpt-5.6-luna", "temperature": 0.0, "input": [_PROMPT]},
            timeout_seconds=1.0,
            retry_policy=_IMMEDIATE_RETRY,
            provider="openai",
            endpoint_class="responses",
        )

    failure = error.value
    rendered = str(failure)
    assert failure.status_code == 400
    assert failure.error_code == "unsupported_parameter"
    assert failure.rejected_parameter == "temperature"
    assert failure.request_id == "req_luna_1"
    assert failure.retryable is False
    assert _SECRET not in rendered
    assert _PROMPT not in rendered
    assert "Authorization" not in rendered
    assert len(transport.requests) == 1
