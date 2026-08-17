"""JSON HTTP transport seam, bounded retries, request helpers, and a deterministic fake."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple
from uuid import uuid4

import httpx

from wmo.common.core.artifacts import JsonObject

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class JsonHttpResponse:
    """One decoded HTTP response returned by a provider endpoint."""

    status_code: int
    body: JsonObject


class ProviderTransportError(RuntimeError):
    """A non-success HTTP or transport result that contains no secret-bearing payload."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JsonHttpTransport:
    """Sends one JSON request without imposing a provider SDK on callers."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one JSON object from a provider metadata endpoint.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers, including provider authentication.
            timeout_seconds: Bounded per-attempt wall-clock timeout.

        Returns:
            The HTTP status and decoded object response.

        Raises:
            ProviderTransportError: The request failed or the endpoint returned non-object JSON.
        """
        raise NotImplementedError

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send JSON and decode a JSON object response.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers, including provider authentication.
            payload: JSON request object.
            timeout_seconds: Bounded per-attempt wall-clock timeout.

        Returns:
            The HTTP status and decoded object response.

        Raises:
            ProviderTransportError: The request failed or the endpoint returned non-object JSON.
        """
        raise NotImplementedError


class RecordedRequest(NamedTuple):
    """One request a scripted transport served, kept for wire assertions in tests.

    The payload is the JSON body a POST sent; GET reads record an empty object.
    """

    url: str
    headers: Mapping[str, str]
    payload: JsonObject


class ScriptedJsonTransport(JsonHttpTransport):
    """Deterministic transport that replays scripted answers and records every request.

    An empty script doubles as an unused-transport guard: any request raises AssertionError.
    """

    def __init__(self, responses: Sequence[JsonHttpResponse | Exception] = ()) -> None:
        """Store the answers served in order, one per expected request.

        Args:
            responses: Responses to return or exceptions to raise, consumed in order.
        """
        self._responses = list(responses)
        self.requests: list[RecordedRequest] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one GET and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers sent by the caller.
            timeout_seconds: Bounded per-attempt timeout, ignored by the fake.

        Returns:
            The next scripted response.

        Raises:
            Exception: The next scripted error, or AssertionError once the script is exhausted.
        """
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), {}))
        return self._answer()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one POST and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers sent by the caller.
            payload: JSON request object sent by the caller.
            timeout_seconds: Bounded per-attempt timeout, ignored by the fake.

        Returns:
            The next scripted response.

        Raises:
            Exception: The next scripted error, or AssertionError once the script is exhausted.
        """
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        return self._answer()

    def _answer(self) -> JsonHttpResponse:
        """Consume and serve the next scripted answer, failing closed when exhausted."""
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class HttpxJsonTransport(JsonHttpTransport):
    """Production JSON transport backed by a caller-owned-or-default httpx client."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one bounded provider metadata endpoint without logging credentials.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers, including the resolved credential.
            timeout_seconds: Per-attempt request timeout.

        Returns:
            The HTTP status and decoded JSON response object.

        Raises:
            ProviderTransportError: The request fails or the response is not a JSON object.
        """
        try:
            response = self._client.get(url, headers=dict(headers), timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        return _decoded_response(response)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send one bounded JSON request without logging content or credentials.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers, including the resolved credential.
            payload: Complete JSON request body.
            timeout_seconds: Per-attempt request timeout.

        Returns:
            The HTTP status and decoded JSON response object.

        Raises:
            ProviderTransportError: The request fails or the response is not a JSON object.
        """
        try:
            response = self._client.post(
                url,
                headers=dict(headers),
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        return _decoded_response(response)


def _decoded_response(response: httpx.Response) -> JsonHttpResponse:
    """Decode one provider response body as a JSON object without revealing content.

    Args:
        response: Completed provider HTTP response.

    Returns:
        The status code paired with the decoded JSON object body.

    Raises:
        ProviderTransportError: The body is not decodable JSON or is not a JSON object.
    """
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderTransportError(
            f"provider returned non-JSON HTTP {response.status_code}",
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, dict):
        raise ProviderTransportError(
            f"provider returned non-object JSON HTTP {response.status_code}",
            status_code=response.status_code,
        )
    return JsonHttpResponse(status_code=response.status_code, body=body)


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
        """Reject attempt and delay bounds that cannot describe a finite retry schedule."""
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


def get_json(
    transport: JsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
    retry_policy: RetryPolicy,
) -> JsonObject:
    """Read one provider metadata endpoint with bounded same-endpoint retries.

    Args:
        transport: Explicit transport used for this request.
        url: Absolute provider endpoint URL.
        headers: Provider headers, including an already-resolved credential.
        timeout_seconds: Timeout for each attempt.
        retry_policy: Retry policy that never changes provider or endpoint.

    Returns:
        A successful response JSON object.

    Raises:
        ProviderTransportError: The endpoint failed or returned a non-success status.
    """

    def send() -> JsonObject:
        """Run one GET attempt and return its successful body."""
        return _successful_body(
            transport.get(url, headers=headers, timeout_seconds=timeout_seconds)
        )

    return run_with_retry(send, policy=retry_policy)


def post_json(
    transport: JsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: JsonObject,
    timeout_seconds: float,
    retry_policy: RetryPolicy,
) -> JsonObject:
    """Send one non-streaming JSON request with bounded same-endpoint retries.

    Args:
        transport: Explicit transport used for this request.
        url: Absolute provider endpoint URL.
        headers: Provider headers, including an already-resolved credential.
        payload: Complete JSON request body.
        timeout_seconds: Timeout for each attempt.
        retry_policy: Retry policy that never changes provider or model.

    Returns:
        A successful response JSON object.

    Raises:
        ProviderTransportError: The endpoint failed or returned a non-success status.
    """
    request_headers = {
        name: value for name, value in headers.items() if name.lower() != "idempotency-key"
    }
    request_headers["Idempotency-Key"] = f"wmo-{uuid4().hex}"

    def send() -> JsonObject:
        """Run one POST attempt, reusing the idempotency key, and return its successful body."""
        return _successful_body(
            transport.post(
                url,
                headers=request_headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        )

    return run_with_retry(send, policy=retry_policy)


def _successful_body(response: JsonHttpResponse) -> JsonObject:
    """Return a 2xx response body or raise a status-bearing ProviderTransportError."""
    if 200 <= response.status_code < 300:
        return response.body
    raise ProviderTransportError(
        f"provider returned HTTP {response.status_code}",
        status_code=response.status_code,
    )
