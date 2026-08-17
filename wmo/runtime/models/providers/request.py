"""One bounded JSON request path shared by the focused provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.errors import (
    ProviderEndpointClass,
    ProviderError,
    provider_error_from_http,
)
from wmo.runtime.models.providers.retry import RetryPolicy, run_with_retry
from wmo.runtime.models.providers.transport import JsonHttpTransport


def get_json(
    transport: JsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    timeout_seconds: float,
    retry_policy: RetryPolicy,
    provider: str,
    endpoint_class: ProviderEndpointClass = "models",
) -> JsonObject:
    """Read one provider metadata endpoint with bounded same-endpoint retries.

    Args:
        transport: Explicit transport used for this request.
        url: Absolute provider endpoint URL.
        headers: Provider headers, including an already-resolved credential.
        timeout_seconds: Timeout for each attempt.
        retry_policy: Retry policy that never changes provider or endpoint.
        provider: Catalog provider kind that issued the call.
        endpoint_class: Documented endpoint family for the call.

    Returns:
        A successful response JSON object.

    Raises:
        ProviderError: The endpoint failed or returned a non-success status.
    """

    def send() -> JsonObject:
        try:
            response = transport.get(url, headers=headers, timeout_seconds=timeout_seconds)
        except ProviderError as exc:
            raise exc.with_call_context(provider=provider, endpoint_class=endpoint_class) from exc
        if 200 <= response.status_code < 300:
            return response.body
        raise provider_error_from_http(
            provider=provider,
            endpoint_class=endpoint_class,
            status_code=response.status_code,
            body=response.body,
            request_id=response.request_id,
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
    provider: str,
    endpoint_class: ProviderEndpointClass,
) -> JsonObject:
    """Send one non-streaming JSON request with bounded same-endpoint retries.

    Args:
        transport: Explicit transport used for this request.
        url: Absolute provider endpoint URL.
        headers: Provider headers, including an already-resolved credential.
        payload: Complete JSON request body.
        timeout_seconds: Timeout for each attempt.
        retry_policy: Retry policy that never changes provider or model.
        provider: Catalog provider kind that issued the call.
        endpoint_class: Documented endpoint family for the call.

    Returns:
        A successful response JSON object.

    Raises:
        ProviderError: The endpoint failed or returned a non-success status.
    """
    request_headers = {
        name: value for name, value in headers.items() if name.lower() != "idempotency-key"
    }
    request_headers["Idempotency-Key"] = f"wmo-{uuid4().hex}"

    def send() -> JsonObject:
        try:
            response = transport.post(
                url,
                headers=request_headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        except ProviderError as exc:
            raise exc.with_call_context(provider=provider, endpoint_class=endpoint_class) from exc
        if 200 <= response.status_code < 300:
            return response.body
        raise provider_error_from_http(
            provider=provider,
            endpoint_class=endpoint_class,
            status_code=response.status_code,
            body=response.body,
            request_id=response.request_id,
        )

    return run_with_retry(send, policy=retry_policy)
