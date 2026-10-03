"""Tests for explicit model construction, capability preflight, and resolved identity."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
from threading import TIMEOUT_MAX
from typing import Literal
from uuid import uuid4

import httpx
import pytest

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, StoredOAuthTokens
from exp.common.core.artifacts import sha256_json
from exp.common.models import (
    AssistantAction,
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelMessage,
    ModelRecord,
    ModelRequest,
    ModelRoles,
    ModelSnapshot,
    Usage,
)
from exp.common.models.catalog import GatewayDeploymentMetadata, GatewayTokenPrices
from exp.common.models.gateway_catalog import normalize_gateway_catalog
from exp.common.models.pricing import completion_cost_reservation
from exp.common.models.token_cost_test import prices
from exp.runtime.gateway.execution_resolution import _resolved_wire_profile
from exp.runtime.models.credentials import (
    CredentialResolution,
    DispatchCredentialReceipt,
    ModelCredentialError,
)
from exp.runtime.models.credentials_test import AtomicEnvironment
from exp.runtime.models.preflight import CapabilityRequirement, ModelCapabilityError
from exp.runtime.models.providers import async_transport
from exp.runtime.models.providers.anthropic import AnthropicClient
from exp.runtime.models.providers.anthropic_subscription import AnthropicSubscriptionClient
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    ScriptedAsyncJsonTransport,
)
from exp.runtime.models.providers.azure import AzureClient
from exp.runtime.models.providers.base import ProviderHttpClient
from exp.runtime.models.providers.chatgpt_subscription import ChatGptSubscriptionClient
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.tinker_sampling import (
    TinkerOptionalDependencyError,
    TinkerSample,
    TinkerSampler,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
    is_known_unbilled_failure,
)
from exp.runtime.models.providers.typesafe import TYPESAFE_BASE_URL, TypeSafeClient
from exp.runtime.models.providers.vertex import VertexTokenProvider
from exp.runtime.models.registry import CatalogRoleName, ModelConnectionError, RuntimeModelCatalog

_DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_embeddings=True,
    context_window_tokens=128_000,
    maximum_output_tokens=16_000,
)


def test_complete_price_metadata_does_not_rebind_transport_or_capacities() -> None:
    """A full tariff changes pricing authority only, preserving the explicit gateway route."""
    catalog = _catalog(provider="openai-compatible", base_url="https://gateway.example.test/v1")
    runtime = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    before = runtime.resolve("fixture-model")
    record = catalog.models["fixture-model"].model_copy(
        update={"gateway": GatewayDeploymentMetadata(prices=prices())}
    )
    after = runtime.with_catalog(
        catalog.model_copy(update={"models": {"fixture-model": record}})
    ).resolve("fixture-model")
    assert before.snapshot == after.snapshot and before.capabilities == after.capabilities
    assert isinstance(before.client, ProviderHttpClient) and isinstance(
        after.client, ProviderHttpClient
    )
    assert before.client.gateway_wire_profile() == after.client.gateway_wire_profile()
    assert before.token_prices is None and after.token_prices == prices()


@pytest.mark.parametrize(
    "timeout",
    [0.0, -1.0, float("inf"), float("nan"), True, 1e20, math.nextafter(TIMEOUT_MAX, math.inf)],
)
def test_runtime_rejects_invalid_http_timeout_before_resolution(timeout: float) -> None:
    """Invalid explicit deadlines fail before credentials or transports are resolved."""
    with pytest.raises(ValueError, match="http_timeout_seconds must be finite and positive"):
        RuntimeModelCatalog(_catalog(), environment={}, http_timeout_seconds=timeout)


@pytest.mark.parametrize("timeout", [None, 120.0, TIMEOUT_MAX])
@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", None),
        ("anthropic", None),
        ("gemini", None),
        ("openrouter", None),
        ("openai-compatible", "https://models.example.test/v1"),
        ("azure", "https://resource.example.test"),
    ],
)
def test_runtime_preserves_http_timeout_in_resolved_wire_profile(
    provider: str, base_url: str | None, timeout: float | None
) -> None:
    """Default and explicit bounds survive resolution and catalog replacement."""
    catalog = _catalog(
        provider=provider, base_url=base_url, api_version="v1" if provider == "azure" else None
    )
    if timeout is None:
        runtime = RuntimeModelCatalog(
            catalog,
            environment={"FIXTURE_API_KEY": "fixture-key"},
            transport_factory=ScriptedJsonTransport,
        )
    else:
        runtime = RuntimeModelCatalog(
            catalog,
            environment={"FIXTURE_API_KEY": "fixture-key"},
            transport_factory=ScriptedJsonTransport,
            http_timeout_seconds=timeout,
        )
    for selected in (runtime, runtime.with_catalog(catalog)):
        client = selected.resolve("fixture-model").client
        assert isinstance(client, ProviderHttpClient)
        assert client.gateway_wire_profile().timeout_seconds == (timeout or 60.0)


def test_embedding_call_uses_explicit_runtime_http_timeout() -> None:
    """An embedding dispatch receives the configured bound, including after catalog replacement."""
    wire = ScriptedAsyncJsonTransport(
        [
            JsonHttpResponse(
                200,
                {
                    "model": "fixture-model",
                    "data": [{"index": 0, "embedding": [0.3, 0.7]}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        ]
    )
    catalog = _catalog(provider="openai-compatible", base_url="https://models.example.test/v1")
    runtime = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=lambda: wire,
        http_timeout_seconds=120.0,
    ).with_catalog(catalog)
    client = runtime.resolve("fixture-model").embedding_client
    assert client is not None
    assert len(client.embed(("a short retrieval query",))) == 1
    assert len(wire.requests) == 1
    assert 119.0 < wire.timeouts[0] <= 120.0


def test_vertex_maas_uses_explicit_runtime_http_timeout() -> None:
    """The separate Vertex OpenAI constructor preserves the same configured bound."""
    catalog = _catalog(
        provider="vertex",
        base_url="https://aiplatform.googleapis.com/v1/projects/test/locations/global",
    )
    catalog.models["fixture-model"] = catalog.models["fixture-model"].model_copy(
        update={"model": "deepseek-ai/deepseek-v3.2-maas"}
    )
    runtime = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        vertex_token_provider_factory=lambda credentials_json: lambda: "fixture-bearer",
        http_timeout_seconds=123.0,
    )
    client = runtime.resolve("fixture-model").client
    assert isinstance(client, ProviderHttpClient)
    assert client.gateway_wire_profile().timeout_seconds == 123.0


def test_native_vertex_retains_atomic_source_receipt_across_bearer_refresh_and_rotation() -> None:
    """The service-account binding survives OAuth refresh, not credential rotation."""
    first, second = DispatchCredentialReceipt(uuid4()), DispatchCredentialReceipt(uuid4())
    environment = AtomicEnvironment(
        CredentialResolution("source-one", "environment", receipt=first)
    )
    tokens = {"source-one": "token-one", "source-two": "token-two"}

    def factory(*, credentials_json: str) -> VertexTokenProvider:
        """Resolve a synthetic bearer using the exact source supplied to this client."""
        return lambda: tokens[credentials_json]

    catalog = ModelCatalog(
        connections={
            "vertex": ConnectionConfig(
                provider="vertex",
                base_url="https://aiplatform.googleapis.com/v1/projects/fruit-project/locations/global",
                api_key_env="VERTEX_TEST",
            )
        },
        models={
            "gemini-test": ModelRecord(
                connection="vertex",
                model="gemini-2.5-pro",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True),
            )
        },
    )
    runtime = RuntimeModelCatalog(
        catalog,
        environment=environment,
        vertex_token_provider_factory=factory,
        http_timeout_seconds=123.0,
    )
    deployment = normalize_gateway_catalog(catalog).deployments[0]
    resolved = runtime.resolve("gemini-test")
    original = _resolved_wire_profile(deployment, resolved)
    assert original.credential_receipt is first
    assert original.timeout_seconds == 123.0
    assert original.headers["authorization"] == "Bearer token-one"
    tokens["source-one"] = "token-one-refreshed"
    refreshed = _resolved_wire_profile(deployment, resolved)
    assert refreshed.credential_receipt is first
    assert refreshed.headers["authorization"] == "Bearer token-one-refreshed"
    environment.resolved = CredentialResolution("source-two", "environment", receipt=second)
    rotated = _resolved_wire_profile(deployment, runtime.resolve("gemini-test"))
    assert rotated.credential_receipt is second
    assert rotated.headers["authorization"] == "Bearer token-two"
    assert original.credential_receipt is first
    assert original.headers["authorization"] == "Bearer token-one"
    assert environment.calls == 2
    assert "source-one" not in repr(original)
    assert str(first.binding_id) not in repr(original)


class _FakeTinkerSampler:
    """A completed-model sampler returned by the injected factory."""

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Return a fixed action without invoking a Tinker service."""
        del request
        return TinkerSample(
            output=AssistantAction(content="sampled"),
            usage=Usage(input_tokens=2, output_tokens=1),
        )


def _catalog(
    *,
    provider: str = "openai",
    base_url: str | None = None,
    api_key_env: str | None = "FIXTURE_API_KEY",
    api_version: str | None = None,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None,
    region: str | None = None,
    capabilities: ModelCapabilities | None = _DEFAULT_CAPABILITIES,
    served_model_id: str | None = None,
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED,
) -> ModelCatalog:
    """Build a minimum one-alias local catalog for deterministic resolution tests."""
    return ModelCatalog(
        connections={
            "primary": ConnectionConfig(
                provider=provider,
                base_url=base_url,
                api_key_env=api_key_env,
                api_version=api_version,
                azure_api_surface=azure_api_surface,
                region=region,
            )
        },
        models={
            "fixture-model": ModelRecord(
                connection="primary",
                model="fixture-model",
                served_model_id=served_model_id,
                billing_source=billing_source,
                capabilities=capabilities,
            )
        },
        roles=ModelRoles(candidates=("fixture-model",), incumbent="fixture-model"),
    )


@pytest.mark.parametrize("role", ["world_model", "judge", "candidate"])
@pytest.mark.parametrize(
    "base_url,explicit_transport,expected",
    [
        ("https://api.experientiallabs.ai/v1", False, True),
        ("https://api.experientiallabs.ai:443/v1", False, True),
        ("https://api.experientiallabs.ai:444/v1", False, False),
        ("https://api.experientiallabs.ai.attacker.test/v1", False, False),
        ("http://api.experientiallabs.ai/v1", False, False),
        ("https://api.experientiallabs.ai/v1", True, False),
    ],
)
def test_default_catalog_trusts_only_official_authenticated_gateway(
    monkeypatch: pytest.MonkeyPatch,
    role: CatalogRoleName,
    base_url: str,
    explicit_transport: bool,
    expected: bool,
) -> None:
    """All inference roles use exact origin trust while injected factories remain caller-owned."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a trusted-shaped refusal with a wait exceeding the request deadline."""
        requests.append(request)
        return httpx.Response(
            429, json={}, headers={"Retry-After": "10000", "x-gateway-admission-refused": "true"}
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(async_transport, "_pooled_client", lambda: http_client)
    catalog = _catalog(provider="openai-compatible", base_url=base_url)
    if explicit_transport:
        runtime = RuntimeModelCatalog(
            catalog,
            environment={"FIXTURE_API_KEY": "fixture"},
            transport_factory=lambda: HttpxAsyncJsonTransport(),
        )
    else:
        runtime = RuntimeModelCatalog(catalog, environment={"FIXTURE_API_KEY": "fixture"})
    with pytest.raises(ProviderTransportError) as caught:
        runtime.resolve("fixture-model", role=role).client.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="hello"),))
        )
    assert len(requests) == 1
    assert caught.value.known_unbilled is expected
    assert is_known_unbilled_failure(caught.value) is expected
    asyncio.run(http_client.aclose())


def test_typesafe_resolves_native_client_without_changing_capability_identity() -> None:
    """The registry adds a decision client, not conversational or embedding support."""
    declared = ModelCapabilities(supports_completions=False, supports_embeddings=False)
    runtime = RuntimeModelCatalog(
        _catalog(provider="typesafe", capabilities=declared),
        environment={"FIXTURE_API_KEY": "provider-secret-canary"},
        transport_factory=ScriptedJsonTransport,
    )
    resolved = runtime.resolve("fixture-model")
    assert isinstance(resolved.client, TypeSafeClient)
    assert resolved.client.gateway_wire_profile().url == f"{TYPESAFE_BASE_URL}/systemone"
    assert resolved.embedding_client is None
    assert resolved.capabilities == declared
    assert resolved.snapshot.capabilities_sha256 == declared.identity_sha256()
    assert "provider-secret-canary" not in repr(resolved)


def test_typesafe_preserves_fixed_origin_policy_and_trusted_https_override() -> None:
    """Custom endpoints stay denied unless the existing explicit trust opt-in is present."""
    with pytest.raises(ValueError, match="built-in official endpoint"):
        _catalog(provider="typesafe", base_url="https://example.test/v1")
    with pytest.raises(ValueError, match="https base_url"):
        ConnectionConfig(
            provider="typesafe",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="FIXTURE_API_KEY",
            trusted_custom_origin=True,
        )
    catalog = _catalog(provider="typesafe")
    connection = ConnectionConfig(
        provider="typesafe",
        base_url="https://example.test/v1",
        api_key_env="FIXTURE_API_KEY",
        trusted_custom_origin=True,
    )
    runtime = RuntimeModelCatalog(
        catalog.model_copy(update={"connections": {"primary": connection}}),
        environment={"FIXTURE_API_KEY": "provider-secret-canary"},
        transport_factory=ScriptedJsonTransport,
    )
    client = runtime.resolve("fixture-model").client
    assert isinstance(client, TypeSafeClient)
    assert client.gateway_wire_profile().url == "https://example.test/v1/systemone"


def test_snapshot_is_credential_free_and_records_capability_digest() -> None:
    """Static identity resolves before credential reads or provider construction.

    The regression proves snapshots contain only secret-free capability evidence.
    """
    catalog = RuntimeModelCatalog(
        _catalog(),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )

    snapshot, capabilities = catalog.snapshot("fixture-model")

    assert snapshot.model_id == "fixture-model"
    assert capabilities == ModelCapabilities(
        supports_tools=True,
        supports_embeddings=True,
        context_window_tokens=128_000,
        maximum_output_tokens=16_000,
    )
    assert snapshot.capabilities_sha256 == capabilities.identity_sha256()


def test_foundry_catalog_resolution_uses_model_inference_token_field() -> None:
    """A setup-shaped Foundry connection emits the API surface's max_tokens contract."""
    catalog = RuntimeModelCatalog(
        _catalog(
            provider="azure",
            base_url="https://resource.services.ai.azure.com",
            api_version="2024-05-01-preview",
            azure_api_surface="model_inference",
            capabilities=ModelCapabilities(supports_completions=True),
        ),
        environment={"FIXTURE_API_KEY": "azure-secret"},
        transport_factory=ScriptedJsonTransport,
    )

    resolved = catalog.resolve("fixture-model")

    assert isinstance(resolved.client, AzureClient)
    assert resolved.client.gateway_wire_profile().token_limit_key == "max_tokens"


def test_azure_foundry_routes_a_known_anthropic_model_over_the_native_messages_wire() -> None:
    """A known Anthropic model on an Azure (Foundry) connection dispatches over the
    NATIVE Anthropic Messages API at /anthropic/v1 with Bearer auth, not the
    OpenAI-deployments wire (which 404s api_not_supported for Claude on Foundry)."""
    # A `/models`-spelled Foundry endpoint must still collapse to the resource
    # root's /anthropic/v1, never `/models/anthropic/v1`.
    catalog = ModelCatalog(
        connections={
            "primary": ConnectionConfig(
                provider="azure",
                base_url="https://silen-resource.services.ai.azure.com/models",
                api_key_env="FIXTURE_API_KEY",
                api_version="2024-10-21",
            )
        },
        models={
            "opus": ModelRecord(
                connection="primary",
                model="claude-opus-4-6",
                billing_source=BillingSource.HOST_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True, supports_reasoning=True),
            )
        },
        roles=ModelRoles(candidates=("opus",), incumbent="opus"),
    )
    runtime = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "foundry-secret"},
        transport_factory=ScriptedJsonTransport,
        http_timeout_seconds=123.0,
    )

    resolved = runtime.resolve("opus")

    assert isinstance(resolved.client, AnthropicClient)
    profile = resolved.client.gateway_wire_profile()
    assert profile.dialect == "anthropic_messages"
    assert profile.timeout_seconds == 123.0
    assert profile.url == "https://silen-resource.services.ai.azure.com/anthropic/v1/messages"
    assert profile.headers["Authorization"] == "Bearer foundry-secret"
    assert "x-api-key" not in profile.headers


def test_azure_non_anthropic_model_keeps_the_openai_deployments_wire() -> None:
    """A non-Anthropic model on the SAME Azure connection still uses AzureClient,
    so the mixed Foundry connection (glm/kimi/deepseek + Claude) routes per model."""
    catalog = ModelCatalog(
        connections={
            "primary": ConnectionConfig(
                provider="azure",
                base_url="https://silen-resource.services.ai.azure.com",
                api_key_env="FIXTURE_API_KEY",
                api_version="2024-10-21",
            )
        },
        models={
            "glm": ModelRecord(
                connection="primary",
                model="FW-GLM-5.2",
                billing_source=BillingSource.HOST_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True),
            )
        },
        roles=ModelRoles(candidates=("glm",), incumbent="glm"),
    )
    runtime = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "foundry-secret"},
        transport_factory=ScriptedJsonTransport,
    )

    assert isinstance(runtime.resolve("glm").client, AzureClient)


def test_snapshots_preserve_per_model_billing_source_on_one_connection() -> None:
    """Do not infer one credential owner from a connection shared by two model aliases."""
    catalog = _catalog(billing_source=BillingSource.HOST_MANAGED)
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "customer-model": catalog.models["fixture-model"].model_copy(
                    update={
                        "model": "customer-model",
                        "billing_source": BillingSource.CUSTOMER_MANAGED,
                    }
                ),
            }
        }
    )
    runtime = RuntimeModelCatalog(catalog, environment={})

    host, _host_capabilities = runtime.snapshot("fixture-model")
    customer, _customer_capabilities = runtime.snapshot("customer-model")

    assert host.billing_source == BillingSource.HOST_MANAGED
    assert customer.billing_source == BillingSource.CUSTOMER_MANAGED
    assert host.connection_sha256 == customer.connection_sha256


def test_snapshot_identity_excludes_workflow_metadata_added_to_existing_catalogs() -> None:
    """Pricing and structured-output metadata do not invalidate frozen model identities.

    The regression preserves compatibility for snapshots frozen before workflow metadata exists.
    """
    original = ModelCapabilities(supports_tools=True, maximum_output_tokens=16_000)
    enriched = original.model_copy(
        update={
            "supports_structured_output": True,
            "input_cost_per_million_tokens_usd": 0.25,
        }
    )

    assert original.identity_sha256() == enriched.identity_sha256()
    assert original.identity_sha256() == sha256_json(
        {
            "supports_tools": True,
            "supports_embeddings": None,
            "context_window_tokens": None,
            "maximum_output_tokens": 16_000,
        }
    )


def test_snapshot_connection_digest_is_normalized_and_endpoint_specific() -> None:
    """Resolved identity changes for another endpoint but excludes credential metadata."""
    first = RuntimeModelCatalog(
        _catalog(
            provider="openai-compatible",
            base_url="HTTPS://Models.Example.test:443/v1/",
        ),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )
    equivalent = RuntimeModelCatalog(
        _catalog(provider="openai-compatible", base_url="https://models.example.test/v1"),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )
    distinct = RuntimeModelCatalog(
        _catalog(provider="openai-compatible", base_url="https://models.example.test/v2"),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )

    first_snapshot, _ = first.snapshot("fixture-model")
    equivalent_snapshot, _ = equivalent.snapshot("fixture-model")
    distinct_snapshot, _ = distinct.snapshot("fixture-model")

    assert first_snapshot == equivalent_snapshot
    assert first_snapshot.connection_sha256 != distinct_snapshot.connection_sha256
    assert first_snapshot != distinct_snapshot
    assert (
        first_snapshot.connection_sha256
        == ConnectionConfig(
            provider="openai-compatible",
            base_url="https://models.example.test/v1",
            api_key_env="ANOTHER_API_KEY",
        ).identity_sha256()
    )
    serialized = first_snapshot.model_dump_json()
    assert "models.example.test" not in serialized
    assert "FIXTURE_API_KEY" not in serialized


def test_preflight_rejects_capability_before_reading_missing_credentials() -> None:
    """A known unsupported requirement fails locally and cannot trigger a paid request."""
    catalog = RuntimeModelCatalog(
        _catalog(provider="anthropic", capabilities=ModelCapabilities(supports_embeddings=False)),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )

    with pytest.raises(ModelCapabilityError, match="declares no embedding support"):
        catalog.preflight(
            "fixture-model",
            CapabilityRequirement(requires_embeddings=True),
        )


def test_resolution_requires_named_credential_without_exposing_its_value() -> None:
    """A missing local key reports only the configured environment variable name."""
    catalog = RuntimeModelCatalog(
        _catalog(),
        environment={},
        transport_factory=ScriptedJsonTransport,
    )

    with pytest.raises(ModelCredentialError, match="FIXTURE_API_KEY"):
        catalog.resolve("fixture-model")


def test_resolution_rejects_unsupported_connection_and_incomplete_compatible_url() -> None:
    """The runtime allows only the focused provider set and explicit compatible endpoints."""
    unsupported = RuntimeModelCatalog(
        _catalog(provider="waterfall"),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    compatible = RuntimeModelCatalog(
        _catalog(provider="openai-compatible"),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )

    with pytest.raises(ModelConnectionError, match="unsupported provider"):
        unsupported.snapshot("fixture-model")
    with pytest.raises(ModelConnectionError, match="needs connection.base_url"):
        compatible.resolve("fixture-model")


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", None),
        ("anthropic", None),
        ("gemini", None),
        ("openrouter", None),
        ("openai-compatible", "https://models.example.test/v1"),
        ("azure", "https://resource.example.test"),
    ],
)
def test_resolution_threads_catalog_served_model_pin_to_every_http_provider(
    provider: str,
    base_url: str | None,
) -> None:
    """A cataloged served-model pin reaches the resolved identity for each HTTP provider.

    Args:
        provider: Catalog connection provider under test.
        base_url: Explicit endpoint for providers that require one.
    """
    pinned = RuntimeModelCatalog(
        _catalog(
            provider=provider,
            base_url=base_url,
            api_version="v1" if provider == "azure" else None,
            served_model_id="fixture-model-served",
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    unpinned = RuntimeModelCatalog(
        _catalog(
            provider=provider, base_url=base_url, api_version="v1" if provider == "azure" else None
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )

    assert pinned.resolve("fixture-model").served_model_id == "fixture-model-served"
    assert unpinned.resolve("fixture-model").served_model_id is None


def test_resolution_carries_the_cataloged_served_model_identity() -> None:
    """An openai-compatible alias exposes its declared served identity for response checks.

    A vLLM endpoint may publish an alias in /models yet echo the canonical served name in every
    completion; the resolved identity must carry that declared exception.
    """
    catalog = RuntimeModelCatalog(
        _catalog(
            provider="openai-compatible",
            base_url="https://models.example.test/v1",
            served_model_id="served-canonical-name",
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )

    resolved = catalog.resolve("fixture-model")

    assert resolved.served_model_id == "served-canonical-name"
    assert resolved.snapshot.model_id == "fixture-model"

    default = RuntimeModelCatalog(
        _catalog(),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )

    assert default.resolve("fixture-model").served_model_id is None


def test_model_capability_snapshot_has_exact_limits_and_stays_permissive_when_absent() -> None:
    """Explicit declarations bound preflight while an undeclared model stays usable."""
    capabilities = ModelCapabilities(
        supports_tools=True,
        context_window_tokens=32_768,
        maximum_output_tokens=16_000,
    )
    catalog = RuntimeModelCatalog(
        _catalog(capabilities=capabilities),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )

    catalog.preflight(
        "fixture-model",
        CapabilityRequirement(
            requires_tools=True,
            minimum_context_window_tokens=32_768,
            minimum_output_tokens=16_000,
        ),
    )
    with pytest.raises(ModelCapabilityError, match="below required 16001"):
        catalog.preflight(
            "fixture-model",
            CapabilityRequirement(minimum_output_tokens=16_001),
        )
    with pytest.raises(ModelCapabilityError, match="below required 32769"):
        catalog.preflight(
            "fixture-model",
            CapabilityRequirement(minimum_context_window_tokens=32_769),
        )

    unknown = RuntimeModelCatalog(
        _catalog(capabilities=None),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    assert unknown.snapshot("fixture-model")[1] == ModelCapabilities()
    assert unknown.resolve("fixture-model").embedding_client is not None
    unknown.preflight("fixture-model", CapabilityRequirement(requires_tools=True))
    unknown.preflight(
        "fixture-model",
        CapabilityRequirement(minimum_context_window_tokens=1),
    )

    declared_unsupported = RuntimeModelCatalog(
        _catalog(capabilities=ModelCapabilities(supports_tools=False)),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    with pytest.raises(ModelCapabilityError, match="declares no tool call support"):
        declared_unsupported.preflight("fixture-model", CapabilityRequirement(requires_tools=True))


def test_tinker_resolution_uses_runtime_owned_default_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cataloged Tinker handle resolves normally while tests can replace only its SDK seam."""
    constructed: list[tuple[str, str]] = []

    def runtime_sampler(
        model: ModelSnapshot,
        api_key: str,
        base_url: str | None,
    ) -> TinkerSampler:
        constructed.append((model.model_id, api_key))
        assert base_url is None
        return _FakeTinkerSampler()

    monkeypatch.setattr("exp.runtime.models.registry.create_tinker_sampler", runtime_sampler)
    catalog = RuntimeModelCatalog(
        _catalog(provider="tinker"),
        environment={"FIXTURE_API_KEY": "fixture-tinker-key"},
        transport_factory=ScriptedJsonTransport,
    )

    resolved = catalog.resolve("fixture-model")

    assert constructed == [("fixture-model", "fixture-tinker-key")]
    assert resolved.embedding_client is None
    assert (
        resolved.client.complete(
            ModelRequest(messages=(ModelMessage(role="user", content="hello"),))
        ).output.content
        == "sampled"
    )


def test_tinker_resolution_reports_a_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal catalog construction explains how to install the sampling-only dependency."""

    def missing_tinker(
        model: ModelSnapshot,
        api_key: str,
        base_url: str | None,
    ) -> TinkerSampler:
        del model, api_key, base_url
        raise TinkerOptionalDependencyError("install with uv sync --extra sft")

    monkeypatch.setattr("exp.runtime.models.registry.create_tinker_sampler", missing_tinker)
    catalog = RuntimeModelCatalog(
        _catalog(provider="tinker"),
        environment={"FIXTURE_API_KEY": "fixture-tinker-key"},
        transport_factory=ScriptedJsonTransport,
    )

    with pytest.raises(ModelConnectionError, match="uv sync --extra sft"):
        catalog.resolve("fixture-model")


def test_resolution_threads_reasoning_content_native_to_compatible_rungs() -> None:
    """A flagged openai-compatible rung on any origin resolves a preserved-thinking route."""
    catalog = RuntimeModelCatalog(
        _catalog(
            provider="openai-compatible",
            base_url="https://hy4-preview--serve.modal.run/v1",
            capabilities=ModelCapabilities(
                supports_reasoning=True,
                reasoning_output_exposed=True,
                reasoning_content_native=True,
            ),
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    resolved = catalog.resolve("fixture-model")
    assert isinstance(resolved.client, OpenAICompatibleClient)
    profile = resolved.client.gateway_wire_profile()
    assert profile.hunyuan_reasoning_route_sha256 is not None
    assert profile.reasoning_output_exposed is True
    assert profile.forwards_prompt_cache_key is False


def test_openrouter_resolution_forwards_the_prompt_cache_key_hint() -> None:
    """The catalog's openrouter provider resolves to a rung that routes by the hint."""
    catalog = RuntimeModelCatalog(
        _catalog(provider="openrouter"),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    resolved = catalog.resolve("fixture-model")
    assert isinstance(resolved.client, OpenAICompatibleClient)
    assert resolved.client.gateway_wire_profile().forwards_prompt_cache_key is True


def test_resolution_threads_system_messages_leading_only_to_compatible_rungs() -> None:
    """A flagged openai-compatible rung resolves a profile that folds non-leading system turns."""
    catalog = RuntimeModelCatalog(
        _catalog(
            provider="openai-compatible",
            base_url="https://gateway.xplabs.ai/qwen/v1",
            capabilities=ModelCapabilities(system_messages_leading_only=True),
        ),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    )
    resolved = catalog.resolve("fixture-model")
    assert isinstance(resolved.client, OpenAICompatibleClient)
    assert resolved.client.gateway_wire_profile().system_messages_leading_only is True


def test_anthropic_connection_geography_reaches_both_client_paths() -> None:
    """Catalog construction supplies both gateway and ordinary completion constraints."""
    catalog = _catalog(provider="anthropic")
    catalog = catalog.model_copy(
        update={
            "connections": {
                "primary": ConnectionConfig(
                    provider="anthropic", api_key_env="FIXTURE_API_KEY", inference_geo="us"
                )
            }
        }
    )
    runtime = RuntimeModelCatalog(
        catalog, environment={"FIXTURE_API_KEY": "fixture"}, transport_factory=ScriptedJsonTransport
    )
    resolved = runtime.resolve(next(iter(catalog.models)))
    assert isinstance(resolved.client, AnthropicClient)
    assert resolved.client.gateway_wire_profile().inference_geo == "us"
    payload = resolved.client._build_request(
        ModelRequest(messages=(ModelMessage(role="user", content="hi"),))
    )
    assert payload["inference_geo"] == "us"


def test_subscription_connection_resolves_to_a_plan_client_that_signs_each_dispatch(
    tmp_path: Path,
) -> None:
    """A chatgpt plan connection needs no environment key; its client mints bearers itself."""
    store = ProviderAuthStore(tmp_path / "auth.json")
    plan = ConnectionConfig(provider="openai", subscription="chatgpt")
    access = _plan_jwt(exp_seconds=4_000_000_000, account="acct-plan")
    store.put_oauth(
        "plan",
        StoredOAuthTokens(
            access_token=access,
            refresh_token="refresh",
            expires_at_ms=4_000_000_000_000,
            account_id="acct-plan",
        ),
        binding=StoredCredentialBinding(provider="openai", endpoint_sha256=plan.identity_sha256()),
    )
    catalog = ModelCatalog(
        connections={"plan": plan},
        models={
            "codex": ModelRecord(
                connection="plan",
                model="gpt-5.6-sol",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=_DEFAULT_CAPABILITIES,
            )
        },
        roles=ModelRoles(candidates=("codex",), incumbent="codex"),
    )

    resolved = RuntimeModelCatalog(
        catalog,
        environment={},
        transport_factory=lambda: ScriptedJsonTransport([]),
        auth_store=store,
        http_timeout_seconds=123.0,
    ).resolve("codex")

    assert isinstance(resolved.client, ChatGptSubscriptionClient)
    profile = resolved.client.gateway_wire_profile()
    assert profile.url == "https://chatgpt.com/backend-api/codex/responses"
    assert profile.timeout_seconds == 123.0
    assert profile.headers["chatgpt-account-id"] == "acct-plan"
    assert resolved.client.sign_gateway_dispatch(url=profile.url, body="{}") == {
        "Authorization": f"Bearer {access}"
    }
    assert resolved.embedding_client is None


def _plan_jwt(*, exp_seconds: int, account: str) -> str:
    """Return an unsigned JWT carrying a ChatGPT account claim and ``exp``."""
    payload = json.dumps(
        {"exp": exp_seconds, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    ).encode()
    segment = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"header.{segment}.signature"


def test_claude_plan_connection_needs_the_operator_app_and_then_signs_each_dispatch() -> None:
    """A Claude plan refuses without the operator's Anthropic app.

    With one, the injected token source mints the bearer and the app's headers ride the wire
    profile.
    """
    plan = ConnectionConfig(provider="anthropic", subscription="anthropic")
    catalog = ModelCatalog(
        connections={"claude-plan": plan},
        models={
            "claude": ModelRecord(
                connection="claude-plan",
                model="claude-sonnet-5",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=_DEFAULT_CAPABILITIES,
            )
        },
        roles=ModelRoles(candidates=("claude",), incumbent="claude"),
    )
    tokens = StoredOAuthTokens(access_token="injected", refresh_token="r", expires_at_ms=1)

    class _Source:
        """An embedder-supplied token source."""

        connection_id = "claude-plan"

        def current(self) -> StoredOAuthTokens:
            """Return the injected sign-in."""
            return tokens

    built: list[str] = []

    def factory(*, connection_id: str, connection: ConnectionConfig) -> _Source:
        """Record which connection asked for a source."""
        built.append(f"{connection_id}:{connection.subscription}")
        return _Source()

    with pytest.raises(ModelConnectionError, match="OAuth app Anthropic issued"):
        RuntimeModelCatalog(catalog, environment={}).resolve("claude")

    resolved = RuntimeModelCatalog(
        catalog,
        environment={
            "EXP_ANTHROPIC_OAUTH_CLIENT_ID": "experiential-issued",
            "EXP_ANTHROPIC_OAUTH_REDIRECT_URI": "https://platform.example.test/cb",
            "EXP_ANTHROPIC_OAUTH_BETA": "oauth-2025-04-20",
        },
        transport_factory=lambda: ScriptedJsonTransport([]),
        subscription_token_source_factory=factory,
        http_timeout_seconds=123.0,
    ).resolve("claude")

    assert isinstance(resolved.client, AnthropicSubscriptionClient)
    profile = resolved.client.gateway_wire_profile()
    assert profile.headers["anthropic-beta"] == "oauth-2025-04-20"
    assert profile.timeout_seconds == 123.0
    assert "Authorization" not in profile.headers
    assert resolved.client.sign_gateway_dispatch(url=profile.url, body="{}") == {
        "Authorization": "Bearer injected"
    }
    assert built == ["claude-plan:anthropic"]


@pytest.mark.parametrize("card", [None, GatewayTokenPrices()])
def test_runtime_preserves_absent_or_explicit_unknown_tariffs(
    card: GatewayTokenPrices | None,
) -> None:
    """Runtime binding shares catalog pricing authority after complete JSON persistence."""
    catalog = _catalog()
    metadata = (
        GatewayDeploymentMetadata() if card is None else GatewayDeploymentMetadata(prices=card)
    )
    record = catalog.models["fixture-model"].model_copy(update={"gateway": metadata})
    catalog = ModelCatalog.model_validate_json(
        catalog.model_copy(update={"models": {"fixture-model": record}}).model_dump_json()
    )
    resolved = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    ).resolve("fixture-model")
    assert resolved.token_prices == card


@pytest.mark.parametrize("authored", [False, True])
def test_capability_only_metadata_keeps_flat_budget_pricing(authored: bool) -> None:
    """An absent tariff permits flat reservations; an explicit unknown card cannot be masked."""
    catalog = _catalog()
    metadata = (
        GatewayDeploymentMetadata(prices=GatewayTokenPrices())
        if authored
        else GatewayDeploymentMetadata()
    )
    record = catalog.models["fixture-model"].model_copy(
        update={
            "gateway": metadata,
            "capabilities": ModelCapabilities(
                input_cost_per_million_tokens_usd=1,
                output_cost_per_million_tokens_usd=2,
                cached_input_cost_per_million_tokens_usd=0.25,
                cache_write_cost_per_million_tokens_usd=1.25,
            ),
        }
    )
    catalog = ModelCatalog.model_validate_json(
        catalog.model_copy(update={"models": {"fixture-model": record}}).model_dump_json()
    )
    resolved = RuntimeModelCatalog(
        catalog,
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=ScriptedJsonTransport,
    ).resolve("fixture-model")

    def reserve() -> float:
        """Exercise the real pricing boundary without any provider dispatch."""
        reservation = completion_cost_reservation(
            model=resolved.snapshot,
            input_usd_per_million_tokens=1,
            output_usd_per_million_tokens=2,
            cached_input_usd_per_million_tokens=0.25,
            cache_write_usd_per_million_tokens=1.25,
            maximum_attempts=1,
            maximum_input_tokens=100,
            maximum_output_tokens=20,
            token_prices=resolved.token_prices,
        )
        return reservation.estimated_maximum_call_cost_usd

    if authored:
        with pytest.raises(
            ValueError, match="flat prices differ from the complete token schedule base"
        ):
            reserve()
    else:
        assert reserve() == pytest.approx(0.000165)
