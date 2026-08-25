"""Vertex adapter routing, OAuth header, token-provider, and catalog resolution tests."""

from __future__ import annotations

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
from exp.runtime.models.providers.openai_compatible_test import _request, _snapshot
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.providers.vertex import (
    ServiceAccountTokenProvider,
    VertexClient,
    VertexCredentialError,
    VertexTokenProvider,
    _vertex_model_id,
)
from exp.runtime.models.registry import RuntimeModelCatalog

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
