"""Tests for the shared transport request helpers and bounded retry classification."""

from __future__ import annotations

import ssl
from pathlib import Path

import httpx
import pytest
import truststore

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
    classify_retry,
    get_json,
    is_known_unbilled_failure,
    parse_retry_after,
    provider_ssl_context,
    run_with_retry,
)

_IMMEDIATE_RETRY = RetryPolicy(maximum_attempts=2, initial_delay_seconds=0, maximum_delay_seconds=0)


def test_provider_tls_uses_system_trust_with_verification_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native trust never disables certificate or hostname verification."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    context = provider_ssl_context()
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


def test_provider_tls_does_not_ignore_an_explicit_ca_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken operator-selected trust boundary fails instead of silently using system roots."""
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    with pytest.raises(FileNotFoundError):
        provider_ssl_context()


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("certificate_error", [False, True])
def test_transport_names_network_failures_without_exposing_secrets(
    method: str, certificate_error: bool
) -> None:
    """Both JSON request paths identify certificate failures without logging exception text."""
    canary = "private-url-and-key-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        """Raise a realistic HTTPX error chain with deliberately private exception text."""
        error = httpx.ConnectError(canary, request=request)
        if certificate_error:
            raise error from ssl.SSLCertVerificationError(1, canary)
        raise error

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = HttpxJsonTransport(client)
        with pytest.raises(ProviderTransportError) as caught:
            if method == "get":
                transport.get(
                    "https://provider.test/v1/models",
                    headers={"Authorization": "Bearer " + canary},
                    timeout_seconds=1,
                )
            else:
                transport.post(
                    "https://provider.test/v1/embeddings",
                    headers={"Authorization": "Bearer " + canary},
                    payload={"input": canary},
                    timeout_seconds=1,
                )
    message = str(caught.value)
    assert canary not in message
    assert (
        "TLS certificate verification failed" in message
        if certificate_error
        else ("ConnectError" in message)
    )


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (" 1 ", 1),
        ("0", 0),
        ("-1", None),
        ("1.5", None),
        ("NaN", None),
        ("inf", None),
        ("garbage", None),
        ("Thu, 01 Jan 1970 00:00:12 GMT", 2),
        ("Thu, 01 Jan 1970 00:00:09 GMT", 0),
    ],
)
def test_retry_after_accepts_only_http_dates_or_nonnegative_integers(
    value: str | None, expected: float | None
) -> None:
    """Server timing survives safe parsing without inventing a wait from malformed metadata."""
    assert parse_retry_after(value, now_unix_seconds=10) == expected


@pytest.mark.parametrize("hint", [3.0, 10_000.0, parse_retry_after("9" * 1_000)])
def test_server_minimum_above_policy_ceiling_never_retries_early(hint: float) -> None:
    """An unserviceable Retry-After propagates the original failure without sleeping or retrying."""
    failure = ProviderTransportError("busy", status_code=429, retry_after_seconds=hint)
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        """Count attempts and retain original exception identity."""
        nonlocal attempts
        attempts += 1
        raise failure

    with pytest.raises(ProviderTransportError) as caught:
        run_with_retry(operation, policy=RetryPolicy(), sleep=delays.append)
    assert caught.value is failure
    assert attempts == 1
    assert delays == []
    assert not is_known_unbilled_failure(caught.value)


@pytest.mark.parametrize("hint", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_numeric_hints_cannot_create_unbounded_waits(hint: float) -> None:
    """Injected malformed numeric hints are discarded at both explicit transport seams."""
    response = JsonHttpResponse(429, {}, retry_after_seconds=hint)
    error = ProviderTransportError("busy", status_code=429, retry_after_seconds=hint)
    assert response.retry_after_seconds is None
    assert error.retry_after_seconds is None


def test_retry_after_jitter_never_precedes_server_minimum() -> None:
    """Jitter spreads retries only inside remaining explicit policy headroom."""
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        """Simulate an ordinary, potentially billed provider throttle."""
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ProviderTransportError("busy", status_code=429, retry_after_seconds=1)
        return "ok"

    assert (
        run_with_retry(
            operation, policy=RetryPolicy(), sleep=delays.append, random_sample=lambda: 0.5
        )
        == "ok"
    )
    assert attempts == 3
    assert delays == [1.5, 1.5]


@pytest.mark.parametrize("status", [200, 400, 500])
def test_trusted_unbilled_flag_is_restricted_to_admission_429(status: int) -> None:
    """A transport cannot accidentally release ordinary successes or unrelated errors."""
    assert not JsonHttpResponse(status, {}, known_unbilled=True).known_unbilled
    assert not ProviderTransportError(
        "response", status_code=status, known_unbilled=True
    ).known_unbilled


@pytest.mark.parametrize("body", [b"{}", b"not-json"])
def test_httpx_preserves_only_retry_delay_on_untrusted_response(body: bytes) -> None:
    """An arbitrary upstream marker never turns generic HTTPX into a trusted billing oracle."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a safe timing hint with an untrusted billing marker."""
        del request
        return httpx.Response(
            429, content=body, headers={"Retry-After": "1", "x-gateway-admission-refused": "true"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        transport = HttpxJsonTransport(client)
        try:
            response = transport.post(
                "https://provider.test/v1", headers={}, payload={}, timeout_seconds=1
            )
        except ProviderTransportError as error:
            assert error.retry_after_seconds == 1
            assert not error.known_unbilled
        else:
            assert response.retry_after_seconds == 1
            assert not response.known_unbilled
