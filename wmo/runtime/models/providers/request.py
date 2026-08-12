"""One bounded JSON request path shared by the focused provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import uuid4

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.retry import RetryPolicy, run_with_retry
from wmo.runtime.models.providers.transport import JsonHttpTransport, ProviderTransportError


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
        response = transport.post(
            url,
            headers=request_headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        if 200 <= response.status_code < 300:
            return response.body
        raise ProviderTransportError(
            f"provider returned HTTP {response.status_code}",
            status_code=response.status_code,
        )

    return run_with_retry(send, policy=retry_policy)
