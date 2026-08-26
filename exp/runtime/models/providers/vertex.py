"""Native Vertex AI adapter for Google-published models over the Gemini wire protocol.

Vertex serves the same ``generateContent`` and ``streamGenerateContent`` protocol as the
Gemini API, so request and response conversion is shared with the Gemini adapter. What
differs is identity: the endpoint root names one project and location, model routes live
under ``publishers/google/models/``, and authentication uses short-lived OAuth bearer
tokens minted from a service-account JSON credential instead of a static API key.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from typing import Protocol
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
    ProviderDeadlineExceeded,
    RequestDeadline,
)
from exp.runtime.models.providers.base import (
    DEFAULT_RETRY_POLICY,
    DEFAULT_TIMEOUT_SECONDS,
    GatewayWireProfile,
    ProviderHttpClient,
    completion_timeout_seconds,
)
from exp.runtime.models.providers.gemini import (
    gemini_generate_request,
    gemini_generate_response,
)
from exp.runtime.models.providers.gemini_streaming import start_gemini_generate_stream
from exp.runtime.models.providers.streaming import NormalizedProviderStream
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy
from exp.runtime.openai_protocol.model_adapter import model_request as gateway_model_request

VERTEX_TOKEN_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
"""OAuth scope requested for every Vertex access token."""

_MODEL_PATH_PREFIX = "publishers/google/models/"
_VERTEX_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?aiplatform\.googleapis\.com")


class VertexCredentialError(ValueError):
    """A Vertex service-account credential could not mint a usable OAuth access token."""


class VertexTokenProvider(Protocol):
    """Returns a currently valid OAuth bearer token for one Vertex connection.

    The runtime may call the provider more than once per request (an off-loop warm mint
    followed by header construction), so implementations return a cached token until it
    expires instead of minting on every call.
    """

    def __call__(self) -> str:
        """Return a non-empty bearer token that authorizes the next request."""
        ...


class VertexTokenProviderFactory(Protocol):
    """Builds one connection-bound token provider from its service-account credential."""

    def __call__(self, *, credentials_json: str) -> VertexTokenProvider:
        """Return a token provider for one connection's already-read credential value."""
        ...


class _RefreshableCredentials(Protocol):
    """Narrow google-auth credential surface used to mint and renew access tokens."""

    valid: bool
    token: str | None

    def refresh(self, request: object) -> None:
        """Mint or renew the access token through one blocking token-endpoint call."""


class ServiceAccountTokenProvider:
    """Mints short-lived Vertex bearer tokens from one service-account JSON credential.

    Construction parses and validates the credential without any network call. Minting is
    one blocking HTTPS call to Google's token endpoint roughly once per token lifetime
    (about an hour); it runs under a lock so concurrent requests share a single mint and
    every later call returns the cached token until it expires.
    """

    def __init__(
        self,
        credentials_json: str,
        *,
        credentials: _RefreshableCredentials | None = None,
    ) -> None:
        """Validate one service-account credential and prepare lazy token minting.

        Args:
            credentials_json: Full service-account JSON credential value, normally read from
                the connection's named environment variable.
            credentials: Optional deterministic credential object used by tests to observe
                refresh behavior without contacting Google.

        Raises:
            VertexCredentialError: The value is not a service-account JSON credential.
        """
        self._lock = threading.Lock()
        if credentials is not None:
            self._credentials: _RefreshableCredentials = credentials
            return
        try:
            info = json.loads(credentials_json)
        except ValueError as exc:
            raise VertexCredentialError(
                "the Vertex credential is not valid JSON; set the connection's environment "
                "variable to the full service-account JSON key file contents"
            ) from exc
        if not isinstance(info, dict):
            raise VertexCredentialError(
                "the Vertex credential must be a service-account JSON object, not a bare value"
            )
        # Official credential construction, kept lazy like the Bedrock boto3 boundary so the
        # SDK never loads for catalogs that hold no Vertex connection.
        from google.oauth2.service_account import Credentials

        try:
            self._credentials = Credentials.from_service_account_info(
                info, scopes=[VERTEX_TOKEN_SCOPE]
            )
        except ValueError as exc:
            raise VertexCredentialError(
                f"the Vertex service-account credential is incomplete: {exc}; paste the full "
                "JSON key file for a service account with Vertex AI access"
            ) from exc

    def __call__(self) -> str:
        """Return a currently valid bearer token, minting or renewing it when needed.

        Returns:
            A non-empty OAuth access token.

        Raises:
            VertexCredentialError: Google's token endpoint returned no usable token.
        """
        with self._lock:
            if not self._credentials.valid:
                from google.auth.transport.requests import Request

                self._credentials.refresh(Request())
            token = self._credentials.token
            if not token:
                raise VertexCredentialError(
                    "Google's token endpoint returned no access token; verify the service "
                    "account exists and has Vertex AI permission on the project"
                )
            return token


class VertexClient(ProviderHttpClient):
    """Calls one explicit Google-published Vertex model through its native REST protocol."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        token_provider: VertexTokenProvider | None = None,
        supports_temperature: bool = True,
        supports_top_p: bool = True,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create a client with explicit generation gates for one Vertex endpoint root.

        Args:
            model: Resolved configured model identity.
            api_key: Service-account JSON credential read from the connection's environment
                variable. It never travels on the wire; it mints each request's bearer token.
            base_url: Project-and-location endpoint root, such as
                ``https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1``.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Per-attempt timeout floor.
            token_provider: Optional deterministic bearer-token seam for tests and callers
                that own credential refresh themselves.
            supports_temperature: Whether the exact model accepts temperature.
            supports_top_p: Whether the exact model accepts top-p sampling.
            supports_top_k: Whether the exact model accepts top-k sampling.
            supports_logprobs: Whether the catalog reports logprob support.
            supports_reasoning: Whether the exact model accepts thinking configuration.
            reasoning_effort: Optional catalog-pinned reasoning effort.
        """
        _require_vertex_host(base_url)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
        )
        self._token_provider = token_provider or ServiceAccountTokenProvider(api_key)
        self._supports_temperature = supports_temperature
        self._supports_top_p = supports_top_p
        self._supports_top_k = supports_top_k
        self._supports_logprobs = supports_logprobs
        self._supports_reasoning = supports_reasoning
        self._reasoning_effort = reasoning_effort

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Warm the bearer token off the event loop, then run the shared completion flow.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.
            deadline: Optional request-wide deadline supplied by gateway execution.
            idempotency_key: Optional stable caller or gateway attempt identity.

        Returns:
            The typed completed response with observed request economics.
        """
        request_deadline = deadline or RequestDeadline.after(
            completion_timeout_seconds(self._timeout_seconds, request.maximum_output_tokens)
        )
        await self._warm_bearer_token(request_deadline)
        return await super().complete_async(
            request, deadline=request_deadline, idempotency_key=idempotency_key
        )

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> NormalizedProviderStream:
        """Start one native Vertex SSE stream under the gateway deadline.

        Args:
            request: Canonical streaming gateway request.
            deadline: Immutable request-wide deadline.
            idempotency_key: Stable identity for this deployment operation.
            retry_policy: Optional caller-owned physical dispatch limit.

        Returns:
            A cancellable provider-neutral event stream.
        """
        bearer_token = await self._warm_bearer_token(deadline)
        return await start_gemini_generate_stream(
            self._transport,
            f"{self._base_url}/{self._stream_path()}",
            headers={
                "authorization": f"Bearer {bearer_token}",
                "content-type": "application/json",
            },
            payload=gemini_generate_request(
                self._model.model_id,
                gateway_model_request(request),
                supports_temperature=self._supports_temperature,
                supports_top_p=self._supports_top_p,
                supports_top_k=self._supports_top_k,
                supports_logprobs=self._supports_logprobs,
                supports_reasoning=self._supports_reasoning,
                reasoning_effort=self._reasoning_effort,
            ),
            request=request,
            deadline=deadline,
            idempotency_key=idempotency_key,
            retry_policy=retry_policy or self._retry_policy,
            timeout_seconds=self._timeout_seconds,
        )

    async def _warm_bearer_token(self, deadline: RequestDeadline) -> str:
        """Mint or read the bearer token off the event loop, bounded by the request deadline.

        Token minting is one blocking HTTPS call to Google's token endpoint. It runs on a
        worker thread so the gateway event loop stays responsive, and the wait is bounded by
        the smaller of the remaining request budget and the per-attempt timeout floor.

        Args:
            deadline: Immutable request-wide deadline shared with the provider dispatch.

        Returns:
            A currently valid bearer token.

        Raises:
            ProviderDeadlineExceeded: The deadline expired before a token was available.
        """
        timeout_seconds = deadline.attempt_timeout(self._timeout_seconds)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await asyncio.to_thread(self._token_provider)
        except TimeoutError as exc:
            raise ProviderDeadlineExceeded(
                "Vertex token refresh exhausted the provider request deadline"
            ) from exc

    def _headers(self) -> dict[str, str]:
        """Build native Vertex headers carrying the provider's current bearer token.

        The async entry points warm the provider off the event loop first, so this call
        returns the cached token without blocking in the ordinary case.
        """
        return {
            "authorization": f"Bearer {self._token_provider()}",
            "content-type": "application/json",
        }

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the native Gemini-dialect profile for this Vertex connection.

        Vertex serves the shared Gemini wire format. Profile resolution runs on
        the native bridge's blocking callback thread, so the roughly-hourly
        OAuth refresh never blocks Rust's async dispatcher; the resulting
        bearer token is frozen only for this admitted request.
        """
        return GatewayWireProfile(
            dialect="gemini_generate_content",
            url=f"{self._base_url}/{self._stream_path()}",
            headers=self._headers(),
            model_id=self._model.model_id,
            timeout_seconds=self._timeout_seconds,
            supports_temperature=self._supports_temperature,
            maximum_temperature=2.0,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_wire_format="gemini_thinking",
            reasoning_effort=self._reasoning_effort,
        )

    def _completion_path(self) -> str:
        """Return the publisher-scoped native generateContent route."""
        return f"{_MODEL_PATH_PREFIX}{_vertex_model_id(self._model.model_id)}:generateContent"

    def _stream_path(self) -> str:
        """Return the publisher-scoped native SSE streaming route."""
        model_id = _vertex_model_id(self._model.model_id)
        return f"{_MODEL_PATH_PREFIX}{model_id}:streamGenerateContent?alt=sse"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a native generateContent payload."""
        return gemini_generate_request(
            self._model.model_id,
            request,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_effort=self._reasoning_effort,
        )

    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one completed generateContent payload into the shared response contract."""
        return gemini_generate_response(
            payload, configured_model=self._model, latency_seconds=latency_seconds
        )


def _require_vertex_host(base_url: str) -> None:
    """Refuse endpoint roots that would receive the OAuth token on a non-Vertex host.

    Catalog validation enforces the same rule with a configuration-time message; this
    check makes the guarantee hold for every direct construction of the client.

    Args:
        base_url: Candidate project-and-location endpoint root.

    Raises:
        ValueError: The URL is not an HTTPS Vertex AI service host.
    """
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not _VERTEX_HOST.fullmatch(host):
        raise ValueError(
            "Vertex clients only send OAuth tokens to HTTPS *.aiplatform.googleapis.com "
            f"hosts; got {base_url!r}"
        )


def _vertex_model_id(model_id: str) -> str:
    """Remove optional catalog spellings before placing a model in a Vertex route.

    Args:
        model_id: Catalog model identifier, with or without a resource-path prefix.

    Returns:
        The bare publisher model identifier.
    """
    return model_id.removeprefix(_MODEL_PATH_PREFIX).removeprefix("models/")
