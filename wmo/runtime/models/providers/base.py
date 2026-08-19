"""Shared construction, headers, and completion template for HTTP provider clients."""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping
from typing import ClassVar

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import ModelRequest, ModelResponse, ModelSnapshot
from wmo.runtime.models.providers.errors import ProviderRetryableResponseError
from wmo.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpTransport,
    RetryClassification,
    RetryPolicy,
    post_json,
    run_with_retry,
)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRY_POLICY = RetryPolicy()
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 4096

COMPLETION_SECONDS_PER_OUTPUT_TOKEN = 0.03
"""Per-token completion time allowance, a conservative ~33 output tokens per second."""

MAXIMUM_COMPLETION_TIMEOUT_SECONDS = 600.0
"""Hard ceiling on any derived completion attempt timeout, matching the Bedrock read bound."""


def completion_timeout_seconds(
    configured_timeout_seconds: float,
    maximum_output_tokens: int | None,
) -> float:
    """Derive one bounded per-attempt completion timeout from the requested output budget.

    A fixed timeout cannot cover a long generation: at a conservative decode rate of about
    33 output tokens per second, a 16000-token request needs roughly 480 seconds. The
    derived value scales linearly with the requested maximum output tokens under
    ``COMPLETION_SECONDS_PER_OUTPUT_TOKEN``, never drops below the client's configured
    timeout, and caps the scaled value at ``MAXIMUM_COMPLETION_TIMEOUT_SECONDS`` so no
    attempt waits unbounded.

    Args:
        configured_timeout_seconds: Positive per-attempt floor configured on the client.
        maximum_output_tokens: Requested output token ceiling, or ``None`` when the request
            leaves the provider default in place.

    Returns:
        A finite timeout between the configured floor and the scaled bounded ceiling.
    """
    if maximum_output_tokens is None:
        return configured_timeout_seconds
    scaled = maximum_output_tokens * COMPLETION_SECONDS_PER_OUTPUT_TOKEN
    return max(configured_timeout_seconds, min(scaled, MAXIMUM_COMPLETION_TIMEOUT_SECONDS))


class ProviderHttpClient(abc.ABC):
    """One explicit provider connection sharing validation, headers, and the completion flow."""

    default_headers: ClassVar[Mapping[str, str]] = {}

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Create a client with a single explicit endpoint and credential.

        Args:
            model: Resolved configured model identity.
            api_key: Credential already read from the named environment variable.
            base_url: Endpoint root that exposes the provider's HTTP routes.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Per-attempt timeout floor. Completion attempts scale above it
                with the requested maximum output tokens through
                ``completion_timeout_seconds``; every other route uses it exactly.
        """
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through the provider's completion route.

        A completed response that decodes to no usable output, such as a reasoning model
        spending its whole output budget without visible text, is dispatched again under the
        client's bounded retry policy before the last error surfaces to the caller. Each
        attempt's timeout scales with the requested maximum output tokens through
        ``completion_timeout_seconds`` so long generations are not cut off by the fixed
        floor, while every wait stays finite.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        payload = self._build_request(request)
        timeout_seconds = completion_timeout_seconds(
            self._timeout_seconds, request.maximum_output_tokens
        )

        def attempt() -> ModelResponse:
            """Post and parse one complete request attempt."""
            started_at = time.monotonic()
            response = self._post(self._completion_path(), payload, timeout_seconds=timeout_seconds)
            return self._parse_response(response, latency_seconds=time.monotonic() - started_at)

        return run_with_retry(
            attempt,
            policy=self._retry_policy,
            classify=_classify_empty_output_retry,
        )

    def _headers(self) -> dict[str, str]:
        """Build the authenticated JSON headers sent with every request."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }

    def _post(
        self,
        path: str,
        payload: JsonObject,
        *,
        timeout_seconds: float | None = None,
    ) -> JsonObject:
        """Post one JSON payload to a provider route below the configured base URL.

        Args:
            path: Provider route below the configured base URL.
            payload: JSON request body.
            timeout_seconds: Optional per-attempt timeout override; the configured client
                timeout applies when omitted.

        Returns:
            The decoded JSON response body.
        """
        return post_json(
            self._transport,
            f"{self._base_url}/{path}",
            headers=self._headers(),
            payload=payload,
            timeout_seconds=(self._timeout_seconds if timeout_seconds is None else timeout_seconds),
            retry_policy=self._retry_policy,
        )

    @abc.abstractmethod
    def _completion_path(self) -> str:
        """Return the completion route below the configured base URL."""

    @abc.abstractmethod
    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into the provider's wire payload."""

    @abc.abstractmethod
    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one decoded provider payload into the shared response contract."""


def _classify_empty_output_retry(exception: Exception) -> RetryClassification:
    """Retry only completed provider responses that decoded to no usable output.

    Transport failures already retry inside each posted attempt, so this outer classifier
    stays closed to everything except the explicit retryable empty-output signal.

    Args:
        exception: Error raised by one post-and-parse attempt.

    Returns:
        A stable retry decision and concise reason.
    """
    if isinstance(exception, ProviderRetryableResponseError):
        return RetryClassification(retryable=True, reason="empty_completed_output")
    return RetryClassification(retryable=False, reason="non_retryable_response")
