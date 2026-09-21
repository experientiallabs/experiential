"""Tests for authorized PostHog HogQL transport failures."""

from __future__ import annotations

from collections.abc import Mapping
from json import JSONDecodeError

import httpx
import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.simulation.ingest.posthog import (
    PostHogPullError,
    PostHogPullRequest,
    pull_posthog_traces,
)


class _Response:
    """Return invalid JSON or raise one deterministic HTTP status failure."""

    def __init__(
        self,
        *,
        status_code: int | None = None,
        decode_error: JSONDecodeError | UnicodeDecodeError | None = None,
    ) -> None:
        self._status_code = status_code
        self._decode_error = decode_error

    def raise_for_status(self) -> None:
        """Raise the configured HTTP status failure, if any."""
        if self._status_code is None:
            return
        request = httpx.Request("POST", "https://posthog.example/api/query")
        response = httpx.Response(self._status_code, request=request)
        raise httpx.HTTPStatusError("unsafe response body", request=request, response=response)

    def json(self) -> JsonValue:
        """Raise the configured decoder failure or return an empty result set."""
        if self._decode_error is not None:
            raise self._decode_error
        return {"results": []}


class _Client:
    """Return one response or raise one deterministic transport failure."""

    def __init__(
        self,
        *,
        response: _Response | None = None,
        error: httpx.RequestError | None = None,
    ) -> None:
        self._response = response or _Response()
        self._error = error

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JsonObject,
        timeout: float,
    ) -> _Response:
        """Return or raise the configured result without inspecting credentials."""
        del url, headers, json, timeout
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.parametrize(
    ("status_code", "expected_message"),
    [
        (401, "rejected the API key"),
        (403, "rejected the API key"),
        (429, "rate limit"),
        (500, "failed with HTTP status 500"),
    ],
)
def test_pull_normalizes_http_status_errors(status_code: int, expected_message: str) -> None:
    """HTTP failures use one content-safe ingestion exception contract."""
    request = PostHogPullRequest(project_id="42", api_key="fixture-secret")

    with pytest.raises(PostHogPullError, match=expected_message) as captured:
        pull_posthog_traces(request, client=_Client(response=_Response(status_code=status_code)))

    assert "fixture-secret" not in str(captured.value)
    assert "unsafe response body" not in str(captured.value)
    assert isinstance(captured.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_pull_normalizes_transport_errors(error_type: type[httpx.RequestError]) -> None:
    """Network failures provide remediation without exposing dependency details."""
    network_request = httpx.Request("POST", "https://posthog.example/api/query")
    error = error_type("unsafe transport detail", request=network_request)
    request = PostHogPullRequest(project_id="42", api_key="fixture-secret")

    with pytest.raises(PostHogPullError, match="could not reach PostHog") as captured:
        pull_posthog_traces(request, client=_Client(error=error))

    assert "fixture-secret" not in str(captured.value)
    assert "unsafe transport detail" not in str(captured.value)
    assert captured.value.__cause__ is error


@pytest.mark.parametrize(
    "decode_error",
    [
        JSONDecodeError("unsafe response body", "secret response", 0),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "unsafe response encoding"),
    ],
)
def test_pull_normalizes_invalid_json(
    decode_error: JSONDecodeError | UnicodeDecodeError,
) -> None:
    """Malformed or invalidly encoded JSON uses the stable ingestion error contract."""
    request = PostHogPullRequest(project_id="42", api_key="fixture-secret")

    with pytest.raises(PostHogPullError, match="returned invalid JSON") as captured:
        pull_posthog_traces(request, client=_Client(response=_Response(decode_error=decode_error)))

    assert "fixture-secret" not in str(captured.value)
    assert "secret response" not in str(captured.value)
    assert captured.value.__cause__ is decode_error
