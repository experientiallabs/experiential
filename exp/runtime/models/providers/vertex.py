"""Native Vertex AI adapter: Gemini wire for Google models, OpenAI wire for Model Garden MaaS.

Vertex serves Google-published models over the same ``generateContent`` and
``streamGenerateContent`` protocol as the Gemini API, so request and response conversion
is shared with the Gemini adapter. Third-party Model Garden models served as a managed API
(MaaS: DeepSeek, Qwen, Kimi, Grok, Llama, ...) are instead served over Vertex's
OpenAI-compatible ``endpoints/openapi/chat/completions`` route, addressed by a
``<publisher>/<model>`` id. Both wires share identity and authentication: the endpoint
root names one project and location, and every request carries a short-lived OAuth bearer
token minted from a service-account JSON credential instead of a static API key. The
catalog model id spelling picks the wire (:func:`vertex_wire_for_model`).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import replace
from typing import Literal, Protocol
from urllib.parse import urlsplit

from exp.common.core.artifacts import JsonObject
from exp.common.models import ChatMaxTokensField, ModelRequest, ModelResponse, ModelSnapshot
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
from exp.runtime.models.providers.openai_compatible import (
    OpenAICompatibleClient,
    openai_compatible_request,
)
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

VERTEX_TOKEN_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
"""OAuth scope requested for every Vertex access token."""

VERTEX_OPENAPI_PATH_PREFIX = "endpoints/openapi/"
"""Route prefix, below the project-and-location root, of Vertex's OpenAI-compatible surface."""

VertexWire = Literal["gemini_generate_content", "openai_compatible"]
"""The two wire dialects one Vertex connection serves, chosen per model id."""

_MODEL_PATH_PREFIX = "publishers/google/models/"
# A bare id (``gemini-2.5-pro``) or a Google resource path names a Google-published model
# on the Gemini wire; any OTHER ``<publisher>/<model>`` spelling is a Model Garden MaaS id
# for the OpenAI-compatible route (see ``vertex_wire_for_model``).
# Google's OWN Model Garden managed endpoints (``gemma-4-26b-a4b-it-maas``) carry this
# suffix in Vertex's listing; it is what separates them from the Gemini models that share
# the ``publishers/google/models/`` resource path.
_MAAS_SUFFIX = "-maas"
_PUBLISHER_RESOURCE_PATH = re.compile(r"^publishers/([^/]+)/models/(.+)$")
_VERTEX_HOST = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-)?aiplatform\.googleapis\.com")


def vertex_wire_for_model(model_id: str) -> VertexWire:
    """Choose the wire dialect one Vertex catalog model id is served over.

    Google-published models (``gemini-2.5-pro``, ``models/gemini-2.5-pro``,
    ``publishers/google/models/gemini-2.5-pro``) ride the native Gemini wire. A
    publisher-qualified id (``deepseek-ai/deepseek-v3.2-maas``, ``xai/grok-4.20-reasoning``,
    ``publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas``) names a Model Garden
    MaaS model, which Vertex serves only over its OpenAI-compatible route. Google's own
    MaaS-served open models follow the same rule under either spelling: the listing's
    ``publishers/google/models/gemma-4-26b-a4b-it-maas`` is recognized by Vertex's
    ``-maas`` endpoint suffix, and ``google/gemma-4-26b-a4b-it-maas`` IS the OpenAI-route
    address.

    Args:
        model_id: Catalog model identifier as spelled on the deployment record.

    Returns:
        The dialect the resolved client speaks for this model.
    """
    if model_id.startswith(_MODEL_PATH_PREFIX):
        return "openai_compatible" if model_id.endswith(_MAAS_SUFFIX) else "gemini_generate_content"
    if model_id.startswith("models/") or "/" not in model_id:
        return "gemini_generate_content"
    return "openai_compatible"


def vertex_openapi_model_id(model_id: str) -> str:
    """Return the ``<publisher>/<model>`` id Vertex's OpenAI-compatible route accepts.

    The route rejects the resource-path spelling (``publishers/X/models/Y``) with 400
    "expected '<publisher>/<model>'", so a catalog carrying the listing's resource path is
    collapsed onto the accepted form; an id already in that form passes through verbatim.

    Args:
        model_id: Catalog model identifier for a MaaS model.

    Returns:
        The publisher-qualified id to place on the wire.
    """
    match = _PUBLISHER_RESOURCE_PATH.match(model_id)
    if match is None:
        return model_id
    return f"{match.group(1)}/{match.group(2)}"


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
        supports_frequency_penalty: bool = False,
        supports_presence_penalty: bool = False,
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
        self._supports_frequency_penalty = supports_frequency_penalty
        self._supports_presence_penalty = supports_presence_penalty
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
        await _warm_vertex_bearer_token(
            self._token_provider, request_deadline, self._timeout_seconds
        )
        return await super().complete_async(
            request, deadline=request_deadline, idempotency_key=idempotency_key
        )

    def _headers(self) -> dict[str, str]:
        """Build native Vertex headers carrying the provider's current bearer token.

        The async entry points warm the provider off the event loop first, so this call
        returns the cached token without blocking in the ordinary case.
        """
        return _vertex_bearer_headers(self._token_provider)

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
            supports_frequency_penalty=self._supports_frequency_penalty,
            supports_presence_penalty=self._supports_presence_penalty,
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


class VertexOpenAIClient(OpenAICompatibleClient):
    """Calls one Model Garden MaaS model through Vertex's OpenAI-compatible route.

    The Chat Completions request and response conversion, the embeddings route, and the
    native ``openai_compatible`` wire profile are the compatible client's; this class only
    changes identity and authentication: every route sits below
    ``endpoints/openapi/`` on the project-and-location root, the model travels as its
    ``<publisher>/<model>`` id, and the bearer token is minted from the service-account
    credential exactly like :class:`VertexClient`.
    """

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
        supports_top_p: bool | None = None,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_frequency_penalty: bool = False,
        supports_presence_penalty: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
        chat_max_tokens_field: ChatMaxTokensField | None = None,
        sampling_requires_reasoning_none: bool = False,
    ) -> None:
        """Create one MaaS client with explicit generation gates for one Vertex endpoint root.

        Args:
            model: Resolved configured model identity (a ``<publisher>/<model>`` MaaS id).
            api_key: Service-account JSON credential read from the connection's environment
                variable. It never travels on the wire; it mints each request's bearer token.
            base_url: Project-and-location endpoint root, such as
                ``https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/global``.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Per-attempt timeout floor.
            token_provider: Optional deterministic bearer-token seam for tests and callers
                that own credential refresh themselves.
            supports_temperature: Whether the exact model accepts temperature.
            supports_top_p: Whether the exact model accepts top-p sampling.
            supports_top_k: Whether the exact model accepts top-k sampling.
            supports_logprobs: Whether the catalog reports logprob support.
            supports_frequency_penalty: Whether the exact model accepts frequency_penalty.
            supports_presence_penalty: Whether the exact model accepts presence_penalty.
            supports_reasoning: Whether the exact model accepts ``reasoning_effort``.
            reasoning_effort: Optional catalog-pinned reasoning effort.
            chat_max_tokens_field: Wire field carrying the output-token ceiling.
            sampling_requires_reasoning_none: Whether sampling controls are only accepted
                with reasoning disabled.
        """
        _require_vertex_host(base_url)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
            supports_frequency_penalty=supports_frequency_penalty,
            supports_presence_penalty=supports_presence_penalty,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
            chat_max_tokens_field=chat_max_tokens_field,
            sampling_requires_reasoning_none=sampling_requires_reasoning_none,
        )
        self._token_provider = token_provider or ServiceAccountTokenProvider(api_key)
        self._wire_model_id = vertex_openapi_model_id(model.model_id)

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
        await _warm_vertex_bearer_token(
            self._token_provider, request_deadline, self._timeout_seconds
        )
        return await super().complete_async(
            request, deadline=request_deadline, idempotency_key=idempotency_key
        )

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the compatible client's profile addressed with the MaaS wire id.

        Profile resolution runs on the native bridge's blocking callback thread,
        exactly like :meth:`VertexClient.gateway_wire_profile`, so the roughly-hourly
        OAuth refresh never blocks Rust's async dispatcher; the resulting bearer token
        is frozen only for this admitted request.
        """
        return replace(super().gateway_wire_profile(), model_id=self._wire_model_id)

    def _embedding_model_id(self) -> str:
        """Name the MaaS id on the embeddings wire too, never the resource-path spelling."""
        return self._wire_model_id

    async def _post_async(
        self,
        path: str,
        payload: JsonObject,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """Warm the bearer token off the event loop before the shared JSON post.

        The embeddings routes reach ``_headers`` through this path, so the same bounded
        off-loop mint the completion flow performs happens here too: a blocking token
        refresh never runs inline on the event loop, and a stalled token endpoint fails
        the request at its deadline instead of outliving it.

        Args:
            path: Provider route below the configured base URL.
            payload: Complete JSON request object.
            deadline: Optional request-wide deadline.
            idempotency_key: Optional stable identity for same-endpoint retries.

        Returns:
            The first successful decoded provider body.
        """
        request_deadline = deadline or RequestDeadline.after(self._timeout_seconds)
        await _warm_vertex_bearer_token(
            self._token_provider, request_deadline, self._timeout_seconds
        )
        return await super()._post_async(
            path, payload, deadline=request_deadline, idempotency_key=idempotency_key
        )

    def _headers(self) -> dict[str, str]:
        """Build headers carrying the provider's current bearer token (never the credential).

        Every async entry point (completions and the embeddings post) warms the provider
        off the event loop first, so this call returns the cached token without blocking
        in the ordinary case.
        """
        return _vertex_bearer_headers(self._token_provider)

    def _request_path(self, path: str) -> str:
        """Place every OpenAI-compatible route below Vertex's ``endpoints/openapi/`` prefix."""
        return f"{VERTEX_OPENAPI_PATH_PREFIX}{path}"

    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into a Chat Completions payload naming the MaaS id."""
        return openai_compatible_request(
            self._wire_model_id,
            request,
            token_limit_key=self._token_limit_key,
            supports_temperature=self._supports_temperature,
            supports_top_p=self._supports_top_p,
            supports_top_k=self._supports_top_k,
            supports_logprobs=self._supports_logprobs,
            supports_reasoning=self._supports_reasoning,
            reasoning_effort=self._reasoning_effort,
            reasoning_wire_format=self.reasoning_wire_format,
            sampling_requires_reasoning_none=self._sampling_requires_reasoning_none,
        )


async def _warm_vertex_bearer_token(
    token_provider: VertexTokenProvider, deadline: RequestDeadline, timeout_seconds: float
) -> str:
    """Mint or read the bearer token off the event loop, bounded by the request deadline.

    Token minting is one blocking HTTPS call to Google's token endpoint. It runs on a
    worker thread so the gateway event loop stays responsive, and the wait is bounded by
    the smaller of the remaining request budget and the per-attempt timeout floor.

    Args:
        token_provider: The connection's cached token provider.
        deadline: Immutable request-wide deadline shared with the provider dispatch.
        timeout_seconds: The client's per-attempt timeout floor.

    Returns:
        A currently valid bearer token.

    Raises:
        ProviderDeadlineExceeded: The deadline expired before a token was available.
    """
    attempt_timeout = deadline.attempt_timeout(timeout_seconds)
    try:
        async with asyncio.timeout(attempt_timeout):
            return await asyncio.to_thread(token_provider)
    except TimeoutError as exc:
        raise ProviderDeadlineExceeded(
            "Vertex token refresh exhausted the provider request deadline"
        ) from exc


def _vertex_bearer_headers(token_provider: VertexTokenProvider) -> dict[str, str]:
    """Authenticated JSON headers carrying the current bearer token, never the credential."""
    return {
        "authorization": f"Bearer {token_provider()}",
        "content-type": "application/json",
    }


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
