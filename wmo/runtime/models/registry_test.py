"""Tests for explicit model construction, capability preflight, and resolved identity."""

from __future__ import annotations

import pytest

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
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
from wmo.runtime.models.credentials import ModelCredentialError
from wmo.runtime.models.preflight import CapabilityRequirement, ModelCapabilityError
from wmo.runtime.models.providers.tinker_sampling import (
    TinkerOptionalDependencyError,
    TinkerSample,
    TinkerSampler,
)
from wmo.runtime.models.providers.transport import ScriptedJsonTransport
from wmo.runtime.models.registry import ModelConnectionError, RuntimeModelCatalog

_DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_tools=True,
    supports_embeddings=True,
    context_window_tokens=128_000,
    maximum_output_tokens=16_000,
)


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
            "supports_embeddings": False,
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

    monkeypatch.setattr("wmo.runtime.models.registry.create_tinker_sampler", runtime_sampler)
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

    monkeypatch.setattr("wmo.runtime.models.registry.create_tinker_sampler", missing_tinker)
    catalog = RuntimeModelCatalog(
        _catalog(provider="tinker"),
        environment={"FIXTURE_API_KEY": "fixture-tinker-key"},
        transport_factory=ScriptedJsonTransport,
    )

    with pytest.raises(ModelConnectionError, match="uv sync --extra sft"):
        catalog.resolve("fixture-model")
