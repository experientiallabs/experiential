"""Production async HTTP JSON classifier adapter for dedicated endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from typing import Final
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

import httpx
from pydantic import ValidationError

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
)

_logger = logging.getLogger(__name__)

DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES: Final = 65_536
_HTTP_TIMEOUT_SECONDS: Final = 60.0
_MAX_CONNECTIONS: Final = 32
_MAX_KEEPALIVE_CONNECTIONS: Final = 16
_BEARER_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_FORBIDDEN_GATEWAY_PATHS: Final[tuple[str, ...]] = (
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/models",
)

_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()
_clients_lock = threading.Lock()


class ClassifierProtocolError(RuntimeError):
    """The classifier HTTP contract failed and must be treated as uncertainty."""


class _CookieFreeTransport(httpx.AsyncBaseTransport):
    """Strip ``Set-Cookie`` so a shared client cannot replay classifier cookies."""

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        """Wrap one connection-pooling transport.

        Args:
            inner: Transport that owns the actual connection pool.
        """
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward one request and drop cookie-setting headers.

        Args:
            request: Outbound classifier request.

        Returns:
            The classifier response without ``Set-Cookie`` headers.
        """
        response = await self._inner.handle_async_request(request)
        if "set-cookie" in response.headers:
            del response.headers["set-cookie"]
        return response

    async def aclose(self) -> None:
        """Close the wrapped connection pool."""
        await self._inner.aclose()


def validate_classifier_url(url: str) -> str:
    """Accept one dedicated classifier URL and reject public gateway paths.

    Args:
        url: Absolute ``http`` or ``https`` classifier endpoint.

    Returns:
        The validated URL.

    Raises:
        ValueError: The URL is relative, uses another scheme, embeds
            credentials, or targets a public gateway path.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("classifier url must be an absolute http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("classifier url cannot include credentials")
    path = parsed.path or "/"
    normalized = path.rstrip("/") or "/"
    for forbidden in _FORBIDDEN_GATEWAY_PATHS:
        if normalized == forbidden or normalized.startswith(f"{forbidden}/"):
            raise ValueError("classifier url must not be a public gateway path")
    return url


def validate_bearer_env_name(name: str) -> str:
    """Accept an environment-variable name and reject credential literals.

    Args:
        name: Environment variable that will supply a bearer credential.

    Returns:
        The validated name.

    Raises:
        ValueError: The name is not a safe environment identifier.
    """
    if not _BEARER_ENV_NAME.fullmatch(name):
        raise ValueError(
            "bearer_env must be an environment variable name such as CLASSIFIER_BEARER"
        )
    return name


def shared_http_json_client() -> httpx.AsyncClient:
    """Return the keep-alive client bound to the running event loop.

    Classifier calls reuse one pooled ``httpx.AsyncClient`` per loop. This
    pool is not the public gateway pool and is not the provider transport
    pool. The engine still owns per-check deadlines and cancellation.

    Returns:
        The shared client for the current event loop.
    """
    loop = asyncio.get_running_loop()
    with _clients_lock:
        client = _clients.get(loop)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                transport=_CookieFreeTransport(
                    httpx.AsyncHTTPTransport(
                        limits=httpx.Limits(
                            max_connections=_MAX_CONNECTIONS,
                            max_keepalive_connections=_MAX_KEEPALIVE_CONNECTIONS,
                        )
                    )
                ),
                timeout=httpx.Timeout(_HTTP_TIMEOUT_SECONDS),
            )
            _clients[loop] = client
        return client


class HttpJsonClassifier:
    """POST one inspect request to a dedicated classifier HTTP endpoint.

    The adapter never targets public gateway routes. It does not log request
    text, completions, replacements, or authorization material. Timeouts and
    cancellation stay with ``GuardrailEngine``.
    """

    def __init__(
        self,
        *,
        adapter_id: str,
        url: str,
        bearer_env: str | None = None,
        max_response_bytes: int = DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Bind one dedicated endpoint and optional environment-sourced auth.

        Args:
            adapter_id: Policy adapter identity used in content-free logs.
            url: Dedicated classifier URL. Public gateway paths are rejected.
            bearer_env: Optional environment-variable name for bearer auth.
            max_response_bytes: Maximum accepted response body size.
            client: Optional injected client. ``None`` uses the shared pool.

        Raises:
            ValueError: The URL, auth name, or byte bound is invalid.
        """
        if max_response_bytes < 1 or max_response_bytes > 1_048_576:
            raise ValueError("http_json max_response_bytes must be between 1 and 1048576")
        self._adapter_id = adapter_id
        self._url = validate_classifier_url(url)
        self._bearer_env = None if bearer_env is None else validate_bearer_env_name(bearer_env)
        self._max_response_bytes = max_response_bytes
        self._client = client

    async def inspect_input(
        self,
        *,
        request: GatewayRequest,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one canonical request and return a validated verdict."""
        return await self._inspect(
            check=check,
            payload=_inspect_payload(check=check, request=request, completion=None),
        )

    async def inspect_output(
        self,
        *,
        completion: GuardrailCompletion,
        check: GuardrailCheck,
    ) -> ClassifierVerdict:
        """Inspect one winning completion and return a validated verdict."""
        return await self._inspect(
            check=check,
            payload=_inspect_payload(check=check, request=None, completion=completion),
        )

    async def _inspect(self, *, check: GuardrailCheck, payload: dict[str, object]) -> ClassifierVerdict:
        """POST the inspect envelope and validate the classifier contract."""
        headers = {"accept": "application/json"}
        authorization = _authorization_header(self._bearer_env)
        if authorization is not None:
            headers["authorization"] = authorization
        client = self._client or shared_http_json_client()
        try:
            async with client.stream("POST", self._url, json=payload, headers=headers) as response:
                status = response.status_code
                _logger.info(
                    "http_json classifier adapter_id=%s capability=%s stage=%s status=%s",
                    self._adapter_id,
                    check.capability.value,
                    check.stage.value,
                    status,
                )
                if status < 200 or status >= 300:
                    raise ClassifierProtocolError(f"classifier returned HTTP {status}")
                body = await _read_bounded(response, self._max_response_bytes)
        except ClassifierProtocolError:
            raise
        except asyncio.CancelledError:
            raise
        except httpx.HTTPError as exc:
            raise ClassifierProtocolError("classifier transport failed") from exc
        return _verdict_from_body(body, stage=check.stage)


def _authorization_header(bearer_env: str | None) -> str | None:
    """Resolve optional bearer auth from the process environment.

    Args:
        bearer_env: Environment-variable name, or ``None`` when unused.

    Returns:
        A bearer header value, or ``None`` when auth is not configured.

    Raises:
        ClassifierProtocolError: The named variable is missing or empty.
    """
    if bearer_env is None:
        return None
    value = os.environ.get(bearer_env, "")
    if not value:
        raise ClassifierProtocolError("classifier bearer environment variable is unset")
    return f"Bearer {value}"


def _inspect_payload(
    *,
    check: GuardrailCheck,
    request: GatewayRequest | None,
    completion: GuardrailCompletion | None,
) -> dict[str, object]:
    """Build the outbound inspect envelope with exactly one subject."""
    if (request is None) == (completion is None):
        raise ClassifierProtocolError("classifier inspect requires exactly one subject")
    payload: dict[str, object] = {
        "capability": check.capability.value,
        "stage": check.stage.value,
        "action": check.action.value,
        "check_id": check.check_id,
    }
    if request is not None:
        payload["request"] = request.model_dump(mode="json")
        return payload
    if completion is None:
        raise ClassifierProtocolError("classifier inspect requires exactly one subject")
    payload["completion"] = completion.model_dump(mode="json")
    return payload


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    """Read a response body and reject oversized payloads.

    Args:
        response: Open streaming response.
        limit: Inclusive maximum accepted byte count.

    Returns:
        The complete response body.

    Raises:
        ClassifierProtocolError: The body exceeds ``limit``.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise ClassifierProtocolError("classifier response exceeded max_response_bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _verdict_from_body(body: bytes, *, stage: GuardrailCheckStage) -> ClassifierVerdict:
    """Parse and validate one ``ClassifierVerdict`` response body.

    Args:
        body: Raw response bytes already bounded by the adapter.
        stage: Inspection stage used to reject invalid replacements.

    Returns:
        The validated verdict.

    Raises:
        ClassifierProtocolError: The body is not a verdict or replacements
            do not match the stage contract.
    """
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ClassifierProtocolError("classifier response is not valid JSON") from exc
    try:
        verdict = ClassifierVerdict.model_validate(parsed)
    except ValidationError as exc:
        raise ClassifierProtocolError("classifier response drifted from ClassifierVerdict") from exc
    _reject_invalid_replacements(verdict, stage)
    return verdict


def _reject_invalid_replacements(verdict: ClassifierVerdict, stage: GuardrailCheckStage) -> None:
    """Reject replacements that do not belong to the inspected stage.

    Args:
        verdict: Parsed classifier result.
        stage: Inspection stage that produced the verdict.

    Raises:
        ClassifierProtocolError: A replacement is present on the wrong stage
            or on an unflagged verdict.
    """
    has_text = verdict.replacement_text is not None
    has_messages = verdict.replacement_messages is not None
    if not verdict.flagged and (has_text or has_messages):
        raise ClassifierProtocolError("unflagged verdict cannot include a replacement")
    if stage is GuardrailCheckStage.INPUT and has_text:
        raise ClassifierProtocolError("input verdict cannot include replacement_text")
    if stage is GuardrailCheckStage.OUTPUT and has_messages:
        raise ClassifierProtocolError("output verdict cannot include replacement_messages")
