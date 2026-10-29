"""JSON HTTP transport seam, bounded retries, request helpers, and a deterministic fake."""

from __future__ import annotations

import math
import os
import random
import ssl
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import NamedTuple

import httpx
import truststore

from exp.common.core.artifacts import JsonObject

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class JsonHttpResponse:
    """One decoded HTTP response and explicitly trusted transport facts.

    Attributes:
        status_code: HTTP response status.
        body: Decoded provider object.
        retry_after_seconds: Sanitized server minimum wait, or None when absent or invalid.
        known_unbilled: A trusted adapter certifies this 429 was rejected before any billable
            backend dispatch. Generic HTTP transports never infer this from response headers.
    """

    status_code: int
    body: JsonObject
    retry_after_seconds: float | None = None
    known_unbilled: bool = False

    def __post_init__(self) -> None:
        """Keep safe numeric delay metadata and restrict admission proof to HTTP 429."""
        object.__setattr__(self, "retry_after_seconds", _safe_retry_delay(self.retry_after_seconds))
        object.__setattr__(
            self, "known_unbilled", self.known_unbilled is True and self.status_code == 429
        )


class ProviderTransportError(RuntimeError):
    """A sanitized transport failure with safe retry metadata.

    Attributes:
        status_code: Optional HTTP status, absent for an unknown transport outcome.
        retry_after_seconds: Sanitized minimum server wait.
        known_unbilled: Trusted proof this attempt was rejected before billable dispatch.
            It says nothing about earlier attempts in the same logical request.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        known_unbilled: bool = False,
    ) -> None:
        """Retain only explicit safe metadata, never raw headers or provider error bodies."""
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = _safe_retry_delay(retry_after_seconds)
        self.known_unbilled = known_unbilled is True and status_code == 429


@dataclass(frozen=True)
class _RequestAttemptEvidence:
    """Aggregate proof attached by an owning retry loop to its terminal exception.

    Attributes:
        attempts: Total attempts actually started, including ambiguous in-flight requests.
        unbilled_attempts: Attempts positively rejected before any billable backend dispatch.
    """

    attempts: int
    unbilled_attempts: int


def retain_request_attempt_evidence(
    error: BaseException, *, attempts: int, unbilled_attempts: int
) -> None:
    """Retain aggregate dispatch proof without replacing exception or cancellation identity."""
    if not 0 <= unbilled_attempts <= attempts:
        raise ValueError("unbilled attempts must be a subset of all attempts")
    error.__dict__["_exp_request_attempt_evidence"] = _RequestAttemptEvidence(
        attempts, unbilled_attempts
    )


def is_known_unbilled_failure(error: BaseException) -> bool:
    """Return true only when the owning retry loop certifies every started attempt was free.

    A single attempt's ``known_unbilled`` flag is insufficient: any earlier unknown outcome
    keeps the entire failed request's reservation conservative. Cancellation during active
    I/O likewise has a started attempt without a matching non-dispatch receipt.
    """
    return known_unbilled_attempts(error) > 0


def known_unbilled_attempts(error: BaseException) -> int:
    """Return the exact certified count only if every started request attempt was unpaid.

    Durable failure receipts retain this count so a crash before caller-level exclusion
    persistence cannot discard the owning retry loop's aggregate accounting proof.
    """
    evidence = getattr(error, "_exp_request_attempt_evidence", None)
    return (
        evidence.attempts
        if (
            isinstance(evidence, _RequestAttemptEvidence)
            and evidence.attempts > 0
            and evidence.attempts == evidence.unbilled_attempts
        )
        else 0
    )


def propagate_request_attempt_evidence(source: BaseException, target: BaseException) -> None:
    """Preserve exact owning-loop evidence when a controlled boundary translates an error.

    This copies only one explicitly supplied exception's evidence. Following arbitrary cause
    chains could accidentally treat an earlier unpaid refusal as proof about a later dispatch.
    """
    evidence = getattr(source, "_exp_request_attempt_evidence", None)
    if isinstance(evidence, _RequestAttemptEvidence):
        target.__dict__["_exp_request_attempt_evidence"] = evidence


def _safe_retry_delay(value: float | None) -> float | None:
    """Accept only finite nonnegative numeric transport hints."""
    return value if value is not None and math.isfinite(value) and value >= 0 else None


def parse_retry_after(value: str | None, *, now_unix_seconds: float | None = None) -> float | None:
    """Parse an RFC Retry-After delay or HTTP date without retaining header content.

    Args:
        value: Raw Retry-After field, or None when absent.
        now_unix_seconds: Optional wall-clock reading for deterministic HTTP-date tests.

    Returns:
        A nonnegative minimum wait, or None for malformed input. Valid enormous delay values
        saturate at the largest finite float so policy rejects them instead of retrying early.
    """
    if value is None:
        return None
    stripped = value.strip()
    if stripped and stripped.isascii() and stripped.isdecimal():
        try:
            seconds = float(stripped)
        except ValueError:
            return None
        return min(seconds, sys.float_info.max)
    try:
        date = parsedate_to_datetime(stripped)
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        now = time.time() if now_unix_seconds is None else now_unix_seconds
        return _safe_retry_delay(max(0.0, date.timestamp() - now))
    except (ValueError, TypeError, OverflowError):
        return None


def validated_admission_origin(origin: str | None) -> httpx.URL | None:
    """Validate an explicit HTTPS authority whose admission receipts the caller trusts."""
    if origin is None:
        return None
    url = httpx.URL(origin)
    if (
        url.scheme != "https"
        or not url.host
        or url.userinfo
        or url.path != "/"
        or url.query
        or url.fragment
    ):
        raise ValueError("trusted admission origin must be an HTTPS origin without credentials")
    return url


def certified_admission_refusal(response: httpx.Response, trusted_origin: httpx.URL | None) -> bool:
    """Read a reserved non-dispatch receipt only from the exact authenticated trusted origin.

    The trusted server must strip downstream copies and emit this marker only for its own
    pre-dispatch refusals. Redirects, alternate ports, and unauthenticated responses cannot
    establish this accounting fact.
    """
    if (
        trusted_origin is None
        or response.status_code != 429
        or response.headers.get("x-gateway-admission-refused") != "true"
        or response.history
    ):
        return False
    request = response.request
    return (
        request.url.scheme == trusted_origin.scheme
        and request.url.host == trusted_origin.host
        and request.url.port == trusted_origin.port
        and not request.url.userinfo
        and bool(request.headers.get("Authorization", "").strip())
    )


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

    def __init__(
        self, client: httpx.Client | None = None, *, trusted_admission_origin: str | None = None
    ) -> None:
        """Bind an HTTP client and an optional explicit admission-receipt authority."""
        self._trusted_admission_origin = validated_admission_origin(trusted_admission_origin)
        self._client = client if client is not None else httpx.Client(verify=provider_ssl_context())

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
            raise ProviderTransportError(transport_error_message(exc)) from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError(transport_error_message(exc)) from exc
        return _decoded_response(response, trusted_origin=self._trusted_admission_origin)

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
            raise ProviderTransportError(transport_error_message(exc)) from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError(transport_error_message(exc)) from exc
        return _decoded_response(response, trusted_origin=self._trusted_admission_origin)


def provider_ssl_context() -> ssl.SSLContext:
    """Verify provider TLS with native system trust, preserving explicit CA overrides.

    Native trust can resolve intermediate certificates and system-managed roots absent from
    static bundles. Explicit SSL_CERT_FILE or SSL_CERT_DIR settings retain HTTPX's selected
    trust boundary. Caller-owned clients remain untouched; SSL is never patched globally.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return httpx.create_ssl_context()
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def transport_error_message(error: httpx.TransportError) -> str:
    """Describe network failure classes without exposing exception text, URLs, or headers."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return (
                "provider TLS certificate verification failed; check system certificate trust "
                "and SSL_CERT_FILE/SSL_CERT_DIR settings"
            )
        current = current.__cause__ or current.__context__
    for error_type, message in (
        (httpx.TimeoutException, "provider request timed out"),
        (httpx.ConnectError, "provider connection failed (ConnectError)"),
        (
            httpx.RemoteProtocolError,
            "provider connection closed unexpectedly (RemoteProtocolError)",
        ),
        (httpx.ReadError, "provider response could not be read (ReadError)"),
        (httpx.WriteError, "provider request could not be sent (WriteError)"),
        (httpx.ProxyError, "provider proxy connection failed (ProxyError)"),
    ):
        if isinstance(error, error_type):
            return message
    return "provider transport request failed"


def _decoded_response(
    response: httpx.Response, *, trusted_origin: httpx.URL | None = None
) -> JsonHttpResponse:
    """Decode one provider response body as a JSON object without revealing content.

    Args:
        response: Completed provider HTTP response.
        trusted_origin: Explicit authority for certified non-dispatch receipts, if configured.

    Returns:
        The status code paired with the decoded JSON object body.

    Raises:
        ProviderTransportError: The body is not decodable JSON or is not a JSON object.
    """
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    known_unbilled = certified_admission_refusal(response, trusted_origin)
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderTransportError(
            f"provider returned non-JSON HTTP {response.status_code}",
            status_code=response.status_code,
            retry_after_seconds=retry_after,
            known_unbilled=known_unbilled,
        ) from exc
    if not isinstance(body, dict):
        raise ProviderTransportError(
            f"provider returned non-object JSON HTTP {response.status_code}",
            status_code=response.status_code,
            retry_after_seconds=retry_after,
            known_unbilled=known_unbilled,
        )
    return JsonHttpResponse(
        status_code=response.status_code,
        body=body,
        retry_after_seconds=retry_after,
        known_unbilled=known_unbilled,
    )


@dataclass(frozen=True)
class RetryClassification:
    """Whether an error merits one or more same-endpoint retry attempts."""

    retryable: bool
    reason: str


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry policy without provider failover semantics.

    Attributes:
        maximum_attempts: Maximum potentially billable attempts. The async loop may separately
            wait for certified non-dispatch responses until its fixed request deadline.
        initial_delay_seconds: Initial backoff for ordinary transport failures.
        maximum_delay_seconds: Exponential backoff and jitter ceiling. Async server-directed
            minimum waits may exceed it within the existing absolute request deadline. The
            sync helper has no request deadline, so this remains its total per-retry wait bound.
    """

    maximum_attempts: int = 3
    initial_delay_seconds: float = 0.25
    maximum_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Reject attempt and delay bounds that cannot describe a finite retry schedule."""
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be at least one")
        if not math.isfinite(self.initial_delay_seconds) or self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be finite and nonnegative")
        if not math.isfinite(self.maximum_delay_seconds):
            raise ValueError("maximum_delay_seconds must be finite")
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
    classify: Callable[[Exception], RetryClassification] = classify_retry,
    random_sample: Callable[[], float] = random.random,
) -> ResultT:
    """Run one idempotent request operation with bounded same-endpoint retries.

    Args:
        operation: A single idempotent provider request attempt.
        policy: Attempt and delay limits.
        sleep: Delay function, injectable for deterministic tests.
        classify: Retry classifier applied to each attempt's error.

    Returns:
        The operation's first successful result.

    Raises:
        Exception: The first non-retryable error or last retryable error.
    """
    delay = policy.initial_delay_seconds
    attempts = 0
    unbilled_attempts = 0
    try:
        for attempt in range(1, policy.maximum_attempts + 1):
            attempts += 1
            try:
                return operation()
            except Exception as exc:
                unbilled_attempts += int(is_unbilled_attempt(exc))
                classification = classify(exc)
                if not classification.retryable or attempt == policy.maximum_attempts:
                    raise
                wait = retry_delay_seconds(
                    exc, delay=delay, policy=policy, random_sample=random_sample
                )
                if wait is None:
                    raise
                if wait > 0:
                    sleep(wait)
                delay = min(delay * 2, policy.maximum_delay_seconds)
    except BaseException as error:
        retain_request_attempt_evidence(
            error, attempts=attempts, unbilled_attempts=unbilled_attempts
        )
        raise
    raise RuntimeError("retry loop exhausted without running an attempt")


def is_unbilled_attempt(error: BaseException) -> bool:
    """Recognize only an explicit trusted adapter's pre-dispatch HTTP 429 assertion."""
    return (
        isinstance(error, ProviderTransportError)
        and error.status_code == 429
        and error.known_unbilled is True
    )


def retry_delay_seconds(
    error: Exception,
    *,
    delay: float,
    policy: RetryPolicy,
    random_sample: Callable[[], float],
    server_wait_ceiling_seconds: float | None = None,
) -> float | None:
    """Respect a server minimum and add bounded jitter only to directed throttling.

    Args:
        error: The preceding attempt's safe failure metadata.
        delay: Current exponential backoff, already bounded by policy.
        policy: Exponential backoff and jitter bound.
        random_sample: Injected uniform sample in [0, 1].
        server_wait_ceiling_seconds: Remaining absolute request time for async callers. A
            server minimum may exceed the backoff ceiling only within this bound. Without
            it, the sync helper keeps the policy's explicit total delay ceiling.

    Returns:
        The next wait, or None if the server minimum cannot fit its applicable bound.
        Jitter never advances the server's retry time or extends the request deadline.
    """
    hint = error.retry_after_seconds if isinstance(error, ProviderTransportError) else None
    ceiling = policy.maximum_delay_seconds
    if hint is not None and hint > ceiling and server_wait_ceiling_seconds is not None:
        ceiling = server_wait_ceiling_seconds
    floor = max(delay, hint or 0.0)
    if floor > ceiling:
        return None
    if hint is None and not is_unbilled_attempt(error):
        return floor
    sample = random_sample()
    if not math.isfinite(sample) or not 0 <= sample <= 1:
        raise ValueError("retry random sample must be finite and between zero and one")
    headroom = min(ceiling - floor, floor, policy.maximum_delay_seconds)
    return floor + headroom * sample


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


def _successful_body(response: JsonHttpResponse) -> JsonObject:
    """Return a 2xx response body or raise a status-bearing ProviderTransportError."""
    if 200 <= response.status_code < 300:
        return response.body
    raise ProviderTransportError(
        f"provider returned HTTP {response.status_code}",
        status_code=response.status_code,
        retry_after_seconds=response.retry_after_seconds,
        known_unbilled=response.known_unbilled,
    )
