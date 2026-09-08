"""Vertex adapter routing, OAuth header, token-provider, and catalog resolution tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelFinishReason,
    ModelRecord,
)
from exp.common.models.setup import ProviderConnection
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayMessage, GatewayRequest
from exp.runtime.models.providers.async_transport import (
    ProviderDeadlineExceeded,
    RequestDeadline,
)
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.streaming_requests import dialect_stream_payload
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.providers.vertex import (
    ServiceAccountTokenProvider,
    VertexClient,
    VertexCredentialError,
    VertexOpenAIClient,
    VertexTokenProvider,
    _vertex_model_id,
    vertex_openapi_model_id,
    vertex_wire_for_model,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.openai_protocol.model_adapter import model_request

_BASE_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1"
    "/projects/fixture-project/locations/us-central1"
)


def _generate_response() -> JsonObject:
    """Return one minimal completed generateContent payload."""
    return {
        "modelVersion": "gemini-2.5-pro-001",
        "candidates": [{"content": {"parts": [{"text": "Working."}]}}],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 6,
            "cachedContentTokenCount": 0,
        },
    }


class _FakeCredentials:
    """Deterministic refreshable credential recording every mint."""

    def __init__(self, tokens: list[str | None]) -> None:
        """Store the token values handed out by successive refresh calls.

        Args:
            tokens: Token values consumed in order, one per refresh.
        """
        self.valid = False
        self.token: str | None = None
        self.refresh_calls = 0
        self._tokens = tokens

    def refresh(self, request: object) -> None:
        """Consume the next scripted token and mark the credential valid.

        Args:
            request: Transport object supplied by the provider; unused by the fake.
        """
        self.refresh_calls += 1
        self.token = self._tokens.pop(0)
        self.valid = True


def test_vertex_routes_publisher_models_with_a_bearer_token() -> None:
    """Completion posts to the publisher route with OAuth auth and Gemini payload shape."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=200, body=_generate_response())]
    )
    client = VertexClient(
        model=_snapshot("vertex", "gemini-2.5-pro"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=transport,
        token_provider=lambda: "fixture-bearer-token",
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert not isinstance(client, EmbeddingClient)
    assert response.model.model_id == "gemini-2.5-pro-001"
    assert response.output.content == "Working."
    url, headers, payload = transport.requests[0]
    assert url == f"{_BASE_URL}/publishers/google/models/gemini-2.5-pro:generateContent"
    assert headers["authorization"] == "Bearer fixture-bearer-token"
    assert "x-goog-api-key" not in headers
    assert payload["contents"]
    assert payload["systemInstruction"] == {"parts": [{"text": "You are precise."}]}


class _StatefulTokenProvider:
    """Cached-token fake matching the provider contract of repeatable per-request calls."""

    def __init__(self, token: str) -> None:
        """Start with one current token value.

        Args:
            token: Bearer token served until the test replaces it.
        """
        self.token = token
        self.calls = 0

    def __call__(self) -> str:
        """Count the read and return the current cached token."""
        self.calls += 1
        return self.token


def test_vertex_reads_the_token_provider_on_every_request() -> None:
    """A renewed bearer token reaches the wire without rebuilding the client."""
    transport = ScriptedJsonTransport(
        [
            JsonHttpResponse(status_code=200, body=_generate_response()),
            JsonHttpResponse(status_code=200, body=_generate_response()),
        ]
    )
    provider = _StatefulTokenProvider("token-first")
    client = VertexClient(
        model=_snapshot("vertex", "gemini-2.5-pro"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=transport,
        token_provider=provider,
    )

    client.complete(_request())
    provider.token = "token-second"
    client.complete(_request())

    assert transport.requests[0][1]["authorization"] == "Bearer token-first"
    assert transport.requests[1][1]["authorization"] == "Bearer token-second"
    assert provider.calls >= 2


def test_vertex_refuses_to_send_tokens_to_non_google_hosts() -> None:
    """Construction fails closed before any bearer token could leave for a foreign host."""
    for base_url in (
        "https://attacker.example.com/v1/projects/p/locations/us-central1",
        "https://aiplatform.googleapis.com.evil.example/v1/projects/p/locations/us",
        "http://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1",
    ):
        with pytest.raises(ValueError, match="aiplatform.googleapis.com"):
            VertexClient(
                model=_snapshot("vertex", "gemini-2.5-pro"),
                api_key='{"placeholder": true}',
                base_url=base_url,
                transport=ScriptedJsonTransport(),
                token_provider=lambda: "fixture-bearer-token",
            )
    with pytest.raises(ValueError, match="HTTPS Vertex AI host"):
        ConnectionConfig(
            provider="vertex",
            base_url="https://attacker.example.com/v1/projects/p/locations/us-central1",
            api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
        )


def test_vertex_stream_path_targets_the_sse_route() -> None:
    """Streaming reuses the Gemini SSE protocol on the publisher-scoped route."""
    client = VertexClient(
        model=_snapshot("vertex", "publishers/google/models/gemini-2.5-flash"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=lambda: "fixture-bearer-token",
    )

    path = client._stream_path()

    assert path == "publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"


def test_vertex_native_profile_and_payload_share_exact_generation_gates() -> None:
    """Vertex preserves Gemini parameters identically on Python and Rust dispatch paths."""
    client = VertexClient(
        model=_snapshot("vertex", "publishers/google/models/gemini-2.5-flash"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=lambda: "fixture-bearer-token",
        supports_temperature=False,
        supports_top_p=False,
        supports_top_k=True,
        supports_reasoning=True,
        reasoning_effort="high",
    )

    profile = client.gateway_wire_profile()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Preserve these controls."),),
        maximum_output_tokens=128,
        temperature=0.4,
        top_p=0.8,
        top_k=32,
        reasoning_effort="high",
    )
    payload = client._build_request(model_request(request))
    native_payload = dialect_stream_payload(profile, request)
    generation = payload["generationConfig"]

    assert profile.dialect == "gemini_generate_content"
    assert profile.url == (
        f"{_BASE_URL}/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert profile.headers == {
        "authorization": "Bearer fixture-bearer-token",
        "content-type": "application/json",
    }
    assert profile.model_id == "publishers/google/models/gemini-2.5-flash"
    assert not profile.supports_temperature
    assert profile.supports_top_p is False
    assert profile.supports_top_k
    assert not profile.supports_logprobs
    assert profile.supports_reasoning
    assert profile.reasoning_wire_format == "gemini_thinking"
    assert profile.reasoning_effort == "high"
    assert isinstance(generation, dict)
    assert "temperature" not in generation
    assert "topP" not in generation
    assert generation["topK"] == 32
    assert generation["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert generation["maxOutputTokens"] == 128
    assert native_payload == payload


def test_vertex_model_id_strips_resource_path_spellings() -> None:
    """Catalog spellings with resource prefixes collapse onto the bare publisher model."""
    assert _vertex_model_id("gemini-2.5-pro") == "gemini-2.5-pro"
    assert _vertex_model_id("models/gemini-2.5-pro") == "gemini-2.5-pro"
    assert _vertex_model_id("publishers/google/models/gemini-2.5-pro") == "gemini-2.5-pro"


def test_service_account_provider_mints_once_and_serves_the_cached_token() -> None:
    """A valid cached token is reused; the credential refreshes only when invalid."""
    credentials = _FakeCredentials(tokens=["minted-token"])
    provider = ServiceAccountTokenProvider("{}", credentials=credentials)

    first = provider()
    second = provider()

    assert first == "minted-token"
    assert second == "minted-token"
    assert credentials.refresh_calls == 1


def test_service_account_provider_renews_an_expired_token() -> None:
    """An invalidated credential mints again instead of serving the stale token."""
    credentials = _FakeCredentials(tokens=["minted-token", "renewed-token"])
    provider = ServiceAccountTokenProvider("{}", credentials=credentials)

    provider()
    credentials.valid = False

    assert provider() == "renewed-token"
    assert credentials.refresh_calls == 2


def test_service_account_provider_rejects_an_empty_minted_token() -> None:
    """A refresh that yields no token fails with an actionable credential error."""
    credentials = _FakeCredentials(tokens=[None])
    provider = ServiceAccountTokenProvider("{}", credentials=credentials)

    with pytest.raises(VertexCredentialError, match="no access token"):
        provider()


def test_service_account_provider_rejects_non_json_credentials() -> None:
    """A pasted API key or other non-JSON value fails before any network use."""
    with pytest.raises(VertexCredentialError, match="not valid JSON"):
        ServiceAccountTokenProvider("AIzaSyFixtureNotAServiceAccount")


def test_service_account_provider_rejects_a_non_object_credential() -> None:
    """A bare JSON scalar cannot stand in for a service-account object."""
    with pytest.raises(VertexCredentialError, match="JSON object"):
        ServiceAccountTokenProvider('"just-a-string"')


def test_catalog_resolution_builds_a_vertex_client_through_the_token_seam() -> None:
    """RuntimeModelCatalog constructs Vertex clients without contacting Google."""
    seen_credentials: list[str] = []

    def factory(*, credentials_json: str) -> VertexTokenProvider:
        """Record the routed credential and hand back a deterministic token provider."""
        seen_credentials.append(credentials_json)
        return lambda: "factory-token"

    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "vertex": ConnectionConfig(
                    provider="vertex",
                    base_url=_BASE_URL,
                    api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
                )
            },
            models={
                "gemini-pro": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="vertex",
                    model="gemini-2.5-pro",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                        supports_embeddings=False,
                    ),
                )
            },
        ),
        environment={"VERTEX_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}'},
        transport_factory=lambda: ScriptedJsonTransport(
            [JsonHttpResponse(status_code=200, body=_generate_response())]
        ),
        vertex_token_provider_factory=factory,
    )

    snapshot, _capabilities = catalog.snapshot("gemini-pro")
    resolved = catalog.resolve("gemini-pro")
    response = resolved.client.complete(_request())

    assert snapshot.provider == "vertex"
    assert isinstance(resolved.client, VertexClient)
    assert resolved.embedding_client is None
    assert response.finish_reason == ModelFinishReason.COMPLETED
    assert seen_credentials == ['{"type": "service_account"}']


def test_catalog_rejects_vertex_connections_missing_their_endpoint_or_credential() -> None:
    """Vertex connection metadata fails closed with actionable endpoint and credential errors."""
    with pytest.raises(ValueError, match="vertex requires base_url"):
        ConnectionConfig(provider="vertex", api_key_env="VERTEX_SERVICE_ACCOUNT_JSON")
    with pytest.raises(ValueError, match="vertex requires api_key_env"):
        ConnectionConfig(provider="vertex", base_url=_BASE_URL)
    with pytest.raises(ValueError, match="region is only accepted"):
        ConnectionConfig(
            provider="vertex",
            base_url=_BASE_URL,
            api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
            region="us-central1",
        )


def test_setup_accepts_a_complete_vertex_connection() -> None:
    """Programmatic setup collects Vertex with an endpoint root and a credential name."""
    connection = ProviderConnection(
        name="vertex-primary",
        provider="vertex",
        base_url=_BASE_URL,
        api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
    )

    config = connection.catalog_config()

    assert config.provider == "vertex"
    assert config.base_url == _BASE_URL
    with pytest.raises(ValueError, match="vertex requires an explicit project-and-location"):
        ProviderConnection(
            name="vertex-primary",
            provider="vertex",
            api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
        )


def test_vertex_token_refresh_is_bounded_by_the_request_deadline() -> None:
    """A hung token mint surfaces the runtime's deadline error instead of outliving it."""

    def stalled_provider() -> str:
        """Simulate a token endpoint that answers far too late."""
        time.sleep(0.5)
        return "too-late-token"

    client = VertexClient(
        model=_snapshot("vertex", "gemini-2.5-pro"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=stalled_provider,
    )

    async def scenario() -> float:
        """Time how quickly the deadline error reaches the caller inside the loop."""
        started = time.monotonic()
        with pytest.raises(ProviderDeadlineExceeded, match="token refresh"):
            await client.complete_async(_request(), deadline=RequestDeadline.after(0.05))
        return time.monotonic() - started

    # asyncio.run itself may wait for the orphaned worker thread at loop shutdown; the
    # bound under test is how quickly the caller inside the loop sees the deadline error.
    assert asyncio.run(scenario()) < 0.4


def test_vertex_token_refresh_fails_fast_on_an_already_spent_deadline() -> None:
    """No token mint starts once the request-wide budget is exhausted."""
    provider = _StatefulTokenProvider("unused-token")
    client = VertexClient(
        model=_snapshot("vertex", "gemini-2.5-pro"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=provider,
    )
    spent = RequestDeadline.after(10.0, now_monotonic=0.0)

    with pytest.raises(ProviderDeadlineExceeded):
        asyncio.run(client.complete_async(_request(), deadline=spent))

    assert provider.calls == 0


def test_vertex_error_status_surfaces_after_bounded_retries() -> None:
    """Non-2xx provider answers raise instead of parsing a partial body."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=403, body={"error": {"status": "PERMISSION_DENIED"}})]
    )
    client = VertexClient(
        model=_snapshot("vertex", "gemini-2.5-pro"),
        api_key='{"placeholder": true}',
        base_url=_BASE_URL,
        transport=transport,
        token_provider=lambda: "fixture-bearer-token",
    )

    with pytest.raises(Exception, match="403"):
        client.complete(_request())


_GLOBAL_BASE_URL = "https://aiplatform.googleapis.com/v1/projects/fixture-project/locations/global"


def _chat_completion_response() -> JsonObject:
    """Return one minimal Chat Completions payload as Vertex's MaaS route shapes it.

    ``prompt_tokens_details`` is ``null`` and ``usage`` carries Google's
    ``extra_properties`` on the live wire (2026-09-08), so the fixture keeps both.
    """
    return {
        "id": "04aa73bb-2800-4f51-985b-a6ef5b3b2ca9",
        "object": "chat.completion",
        "model": "deepseek-ai/deepseek-v3.2-maas",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Working.", "tool_calls": None},
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "total_tokens": 18,
            "prompt_tokens_details": None,
            "extra_properties": {"google": {"traffic_type": "ON_DEMAND"}},
        },
    }


def test_vertex_wire_follows_the_model_id_spelling() -> None:
    """Google ids ride the Gemini wire; any other publisher-qualified id is a MaaS route."""
    for google_id in (
        "gemini-2.5-pro",
        "models/gemini-2.5-pro",
        "publishers/google/models/gemini-2.5-flash",
    ):
        assert vertex_wire_for_model(google_id) == "gemini_generate_content"
    for maas_id in (
        "deepseek-ai/deepseek-v3.2-maas",
        "xai/grok-4.20-reasoning",
        "google/gemma-4-26b-a4b-it-maas",
        "publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas",
    ):
        assert vertex_wire_for_model(maas_id) == "openai_compatible"


def test_vertex_openapi_model_id_collapses_the_resource_path_spelling() -> None:
    """The route accepts only ``<publisher>/<model>``; the listing's path form is collapsed."""
    assert (
        vertex_openapi_model_id("deepseek-ai/deepseek-v3.2-maas")
        == "deepseek-ai/deepseek-v3.2-maas"
    )
    assert (
        vertex_openapi_model_id("publishers/qwen/models/qwen3-coder-480b-a35b-instruct-maas")
        == "qwen/qwen3-coder-480b-a35b-instruct-maas"
    )


def test_vertex_openapi_client_posts_chat_completions_with_a_bearer_token() -> None:
    """A MaaS completion posts to the openapi route with OAuth auth and the publisher id."""
    transport = ScriptedJsonTransport(
        [JsonHttpResponse(status_code=200, body=_chat_completion_response())]
    )
    client = VertexOpenAIClient(
        model=_snapshot("vertex", "deepseek-ai/deepseek-v3.2-maas"),
        api_key='{"placeholder": true}',
        base_url=_GLOBAL_BASE_URL,
        transport=transport,
        token_provider=lambda: "fixture-bearer-token",
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert isinstance(client, EmbeddingClient)
    assert response.output.content == "Working."
    assert response.model.model_id == "deepseek-ai/deepseek-v3.2-maas"
    assert response.economics.usage is not None
    assert response.economics.usage.input_tokens == 12
    assert response.economics.usage.cached_input_tokens is None
    url, headers, payload = transport.requests[0]
    assert url == f"{_GLOBAL_BASE_URL}/endpoints/openapi/chat/completions"
    assert headers["authorization"] == "Bearer fixture-bearer-token"
    assert "x-goog-api-key" not in headers
    assert payload["model"] == "deepseek-ai/deepseek-v3.2-maas"
    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "system", "content": "You are precise."}
    assert payload["stream"] is False


def test_vertex_openapi_profile_is_the_compatible_dialect_on_the_vertex_host() -> None:
    """The native profile speaks openai_compatible at the openapi route with the MaaS id."""
    provider = _StatefulTokenProvider("token-first")
    client = VertexOpenAIClient(
        model=_snapshot("vertex", "publishers/xai/models/grok-4.20-reasoning"),
        api_key='{"placeholder": true}',
        base_url=_GLOBAL_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=provider,
        supports_top_k=False,
        supports_reasoning=True,
    )

    profile = client.gateway_wire_profile()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Preserve these controls."),),
        maximum_output_tokens=128,
        temperature=0.4,
        top_k=32,
        reasoning_effort="high",
    )
    native_payload = dialect_stream_payload(profile, request)
    payload = client._build_request(model_request(request))

    assert profile.dialect == "openai_compatible"
    assert profile.url == f"{_GLOBAL_BASE_URL}/endpoints/openapi/chat/completions"
    assert profile.embeddings_url == f"{_GLOBAL_BASE_URL}/endpoints/openapi/embeddings"
    assert profile.headers == {
        "authorization": "Bearer token-first",
        "content-type": "application/json",
    }
    assert profile.model_id == "xai/grok-4.20-reasoning"
    assert profile.reasoning_wire_format == "reasoning_effort"
    assert profile.token_limit_key == "max_tokens"
    assert native_payload["model"] == "xai/grok-4.20-reasoning"
    assert native_payload["stream"] is True
    assert native_payload["stream_options"] == {"include_usage": True}
    assert native_payload["temperature"] == 0.4
    assert "top_k" not in native_payload
    assert native_payload["reasoning_effort"] == "high"
    assert native_payload["max_tokens"] == 128
    assert payload["model"] == "xai/grok-4.20-reasoning"
    assert payload["reasoning_effort"] == "high"
    # The token is re-read per profile so a renewed bearer reaches the wire.
    provider.token = "token-second"
    assert client.gateway_wire_profile().headers["authorization"] == "Bearer token-second"


def test_vertex_openapi_client_refuses_non_google_hosts() -> None:
    """The MaaS client fails closed before a bearer token could leave for a foreign host."""
    with pytest.raises(ValueError, match="aiplatform.googleapis.com"):
        VertexOpenAIClient(
            model=_snapshot("vertex", "deepseek-ai/deepseek-v3.2-maas"),
            api_key='{"placeholder": true}',
            base_url="https://attacker.example.com/v1/projects/p/locations/global",
            transport=ScriptedJsonTransport(),
            token_provider=lambda: "fixture-bearer-token",
        )


def test_vertex_openapi_token_refresh_is_bounded_by_the_request_deadline() -> None:
    """The MaaS client shares the Gemini-wire client's bounded off-loop token warm."""
    provider = _StatefulTokenProvider("unused-token")
    client = VertexOpenAIClient(
        model=_snapshot("vertex", "deepseek-ai/deepseek-v3.2-maas"),
        api_key='{"placeholder": true}',
        base_url=_GLOBAL_BASE_URL,
        transport=ScriptedJsonTransport(),
        token_provider=provider,
    )
    spent = RequestDeadline.after(10.0, now_monotonic=0.0)

    with pytest.raises(ProviderDeadlineExceeded):
        asyncio.run(client.complete_async(_request(), deadline=spent))

    assert provider.calls == 0


def test_catalog_resolution_builds_the_openapi_client_for_model_garden_ids() -> None:
    """One Vertex connection resolves Gemini ids and MaaS ids to their own clients."""
    seen_credentials: list[str] = []

    def factory(*, credentials_json: str) -> VertexTokenProvider:
        """Record the routed credential and hand back a deterministic token provider."""
        seen_credentials.append(credentials_json)
        return lambda: "factory-token"

    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "vertex": ConnectionConfig(
                    provider="vertex",
                    base_url=_GLOBAL_BASE_URL,
                    api_key_env="VERTEX_SERVICE_ACCOUNT_JSON",
                )
            },
            models={
                "deepseek": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="vertex",
                    model="deepseek-ai/deepseek-v3.2-maas",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                        supports_embeddings=False,
                    ),
                ),
                "gemini-pro": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="vertex",
                    model="gemini-2.5-pro",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                        supports_embeddings=False,
                    ),
                ),
            },
        ),
        environment={"VERTEX_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}'},
        transport_factory=lambda: ScriptedJsonTransport(
            [JsonHttpResponse(status_code=200, body=_chat_completion_response())]
        ),
        vertex_token_provider_factory=factory,
    )

    maas = catalog.resolve("deepseek")
    gemini = catalog.resolve("gemini-pro")
    response = maas.client.complete(_request())

    assert isinstance(maas.client, VertexOpenAIClient)
    assert maas.embedding_client is None
    assert type(gemini.client) is VertexClient
    assert response.finish_reason == ModelFinishReason.COMPLETED
    assert response.output.content == "Working."
    assert seen_credentials == ['{"type": "service_account"}', '{"type": "service_account"}']
