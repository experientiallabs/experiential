"""Azure adapter, catalog pairing, and RuntimeModelCatalog resolution tests."""

from __future__ import annotations

import pytest

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    EmbeddingClient,
    ModelCapabilities,
    ModelCatalog,
    ModelClient,
    ModelFinishReason,
    ModelRecord,
    ModelRoles,
)
from wmo.runtime.models.conftest import ScriptedAsyncJsonTransport
from wmo.runtime.models.credentials import ModelCredentialError
from wmo.runtime.models.providers.azure import (
    AZURE_OPENAI_API_KEY_ENV,
    AZURE_OPENAI_ENDPOINT_ENV,
    AzureClient,
    bind_azure_api_key,
    same_azure_endpoint,
)
from wmo.runtime.models.providers.openai_compatible_test import _request, _snapshot
from wmo.runtime.models.providers.transport import JsonHttpResponse
from wmo.runtime.models.registry import RuntimeModelCatalog

_SECRET = "azure-secret-key-value"
_ENDPOINT = "https://resource.openai.azure.com"
_OTHER_ENDPOINT = "https://other.openai.azure.com"


def _completion_response() -> JsonObject:
    """Return one minimal Azure Chat Completions payload."""
    return {
        "model": "gpt-deployment",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
    }


def test_v1_route_sends_deployment_and_api_key_header() -> None:
    """Foundry and Azure OpenAI v1 keep the deployment in the body and authenticate with api-key."""
    transport = ScriptedAsyncJsonTransport(
        [JsonHttpResponse(status_code=200, body=_completion_response())]
    )
    client = AzureClient(
        model=_snapshot("azure", "gpt-deployment"),
        endpoint=_ENDPOINT,
        api_key=_SECRET,
        api_version="v1",
        transport=transport,
    )

    response = client.complete(_request())

    assert isinstance(client, ModelClient)
    assert response.model.model_id == "gpt-deployment"
    url, headers, payload = transport.requests[0]
    assert url == "https://resource.openai.azure.com/openai/v1/chat/completions"
    assert headers["api-key"] == _SECRET
    assert "Authorization" not in headers
    assert payload["model"] == "gpt-deployment"


def test_classic_route_puts_the_exact_deployment_in_the_path() -> None:
    """Dated Azure OpenAI versions keep the deployment in the URL and the body."""
    transport = ScriptedAsyncJsonTransport(
        [JsonHttpResponse(status_code=200, body=_completion_response())]
    )
    client = AzureClient(
        model=_snapshot("azure", "exact-deployment"),
        endpoint=_ENDPOINT,
        api_key=_SECRET,
        api_version="2024-10-21",
        transport=transport,
    )

    client.complete(_request())

    url, _headers, payload = transport.requests[0]
    assert url == (
        "https://resource.openai.azure.com/openai/deployments/exact-deployment/"
        "chat/completions?api-version=2024-10-21"
    )
    assert payload["model"] == "exact-deployment"


def test_embeddings_use_the_configured_deployment_alias() -> None:
    """Embedding aliases send their own deployment ID rather than a guessed base model."""
    transport = ScriptedAsyncJsonTransport(
        [
            JsonHttpResponse(
                status_code=200,
                body={"data": [{"index": 0, "embedding": [3.0, 4.0]}]},
            )
        ]
    )
    client = AzureClient(
        model=_snapshot("azure", "embed-deployment"),
        endpoint=_ENDPOINT,
        api_key=_SECRET,
        api_version="v1",
        transport=transport,
    )

    embeddings = client.embed(["hello"])

    assert isinstance(client, EmbeddingClient)
    assert embeddings[0].values == (0.6, 0.8)
    url, _headers, payload = transport.requests[0]
    assert url == "https://resource.openai.azure.com/openai/v1/embeddings"
    assert payload["model"] == "embed-deployment"


def test_trusted_azure_key_is_not_sent_to_a_different_endpoint() -> None:
    """AZURE_OPENAI_API_KEY stays bound to AZURE_OPENAI_ENDPOINT."""
    with pytest.raises(ModelCredentialError, match="AZURE_OPENAI_ENDPOINT") as captured:
        bind_azure_api_key(
            endpoint=_OTHER_ENDPOINT,
            api_key_env=AZURE_OPENAI_API_KEY_ENV,
            api_key=_SECRET,
            environment={AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT},
        )
    assert _SECRET not in str(captured.value)
    assert (
        bind_azure_api_key(
            endpoint=_ENDPOINT,
            api_key_env=AZURE_OPENAI_API_KEY_ENV,
            api_key=_SECRET,
            environment={AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT + "/"},
        )
        == _SECRET
    )
    assert (
        bind_azure_api_key(
            endpoint=_OTHER_ENDPOINT,
            api_key_env="AZURE_FOUNDRY_API_KEY",
            api_key=_SECRET,
            environment={AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT},
        )
        == _SECRET
    )


def test_canonical_endpoint_comparison_is_host_insensitive_and_path_sensitive() -> None:
    """Trailing slashes and host case do not split a resource; path case and port do."""
    assert same_azure_endpoint(
        "HTTPS://Resource.openai.azure.com/",
        "https://resource.openai.azure.com",
    )
    assert same_azure_endpoint(
        "https://resource.openai.azure.com:443",
        "https://resource.openai.azure.com",
    )
    assert not same_azure_endpoint(
        "https://resource.openai.azure.com/Azure",
        "https://resource.openai.azure.com/azure",
    )
    assert not same_azure_endpoint(
        "https://resource.openai.azure.com:8443",
        "https://resource.openai.azure.com",
    )
    assert not same_azure_endpoint(_ENDPOINT, None)
    with pytest.raises(ModelCredentialError, match="different Azure resource"):
        bind_azure_api_key(
            endpoint="https://resource.openai.azure.com:8443",
            api_key_env=AZURE_OPENAI_API_KEY_ENV,
            api_key=_SECRET,
            environment={AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT},
        )


def test_a_v1_root_endpoint_is_not_double_appended() -> None:
    """Operators may store either the resource root or the v1 root."""
    transport = ScriptedAsyncJsonTransport(
        [JsonHttpResponse(status_code=200, body=_completion_response())]
    )
    client = AzureClient(
        model=_snapshot("azure", "gpt-deployment"),
        endpoint="https://resource.openai.azure.com/openai/v1",
        api_key=_SECRET,
        api_version="v1",
        transport=transport,
    )

    client.complete(_request())

    url, _headers, _payload = transport.requests[0]
    assert url == "https://resource.openai.azure.com/openai/v1/chat/completions"


def test_catalog_resolution_pairs_one_endpoint_with_its_key() -> None:
    """RuntimeModelCatalog constructs Azure clients without contacting the provider."""
    catalog = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "azure": ConnectionConfig(
                    provider="azure",
                    base_url=_ENDPOINT,
                    api_key_env=AZURE_OPENAI_API_KEY_ENV,
                    api_version="v1",
                )
            },
            models={
                "gpt": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="azure",
                    model="gpt-deployment",
                    capabilities=ModelCapabilities(
                        supports_tools=True,
                        supports_completions=True,
                        supports_embeddings=True,
                    ),
                )
            },
            roles=ModelRoles(world_model="gpt", judge="gpt", embedder="gpt"),
        ),
        environment={
            AZURE_OPENAI_API_KEY_ENV: _SECRET,
            AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT,
        },
        transport_factory=lambda: ScriptedAsyncJsonTransport(
            [JsonHttpResponse(status_code=200, body=_completion_response())]
        ),
    )

    snapshot, _capabilities = catalog.snapshot("gpt")
    resolved = catalog.resolve("gpt")
    response = resolved.client.complete(_request())

    assert snapshot.provider == "azure"
    assert snapshot.model_id == "gpt-deployment"
    assert isinstance(resolved.client, AzureClient)
    assert resolved.embedding_client is resolved.client
    assert response.finish_reason == ModelFinishReason.COMPLETED
    mismatched = RuntimeModelCatalog(
        ModelCatalog(
            connections={
                "azure": ConnectionConfig(
                    provider="azure",
                    base_url=_OTHER_ENDPOINT,
                    api_key_env=AZURE_OPENAI_API_KEY_ENV,
                    api_version="v1",
                )
            },
            models={
                "gpt": ModelRecord(
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    connection="azure",
                    model="gpt-deployment",
                    capabilities=ModelCapabilities(supports_completions=True),
                )
            },
        ),
        environment={
            AZURE_OPENAI_API_KEY_ENV: _SECRET,
            AZURE_OPENAI_ENDPOINT_ENV: _ENDPOINT,
        },
        transport_factory=ScriptedAsyncJsonTransport,
    )
    with pytest.raises(ModelCredentialError, match="different Azure resource"):
        mismatched.resolve("gpt")
