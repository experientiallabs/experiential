"""Tests for explicit model construction, capability preflight, and resolved identity."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from wmo.common.core.artifacts import JsonObject, sha256_json
from wmo.common.models import (
    AssistantAction,
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
from wmo.runtime.models.providers.tinker_sampling import TinkerSample, TinkerSampler
from wmo.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from wmo.runtime.models.registry import ModelConnectionError, RuntimeModelCatalog


class _UnusedTransport(JsonHttpTransport):
    """Fails if construction unexpectedly tries to contact a provider."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Reject every attempted network-shaped call from this construction fixture."""
        del url, headers, payload, timeout_seconds
        raise AssertionError("catalog construction must not make a provider request")


class _FakeTinkerSampler:
    """A completed-model sampler returned by the injected factory."""

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Return a fixed action without invoking a Tinker service."""
        del request
        return TinkerSample(
            output=AssistantAction(content="sampled"),
            usage=Usage(input_tokens=2, output_tokens=1),
        )


def _catalog(*, provider: str = "openai", base_url: str | None = None) -> ModelCatalog:
    """Build a minimum one-alias local catalog for deterministic resolution tests."""
    return ModelCatalog(
        connections={
            "primary": ConnectionConfig(
                provider=provider,
                base_url=base_url,
                api_key_env="FIXTURE_API_KEY",
            )
        },
        models={"fixture-model": ModelRecord(connection="primary", model="fixture-model")},
        roles=ModelRoles(candidates=("fixture-model",), incumbent="fixture-model"),
    )


def test_snapshot_is_credential_free_and_records_capability_digest() -> None:
    """Static identity resolves before credential reads or provider construction."""
    catalog = RuntimeModelCatalog(
        _catalog(),
        environment={},
        transport_factory=_UnusedTransport,
    )

    snapshot, capabilities = catalog.snapshot("fixture-model")

    assert snapshot.model_id == "fixture-model"
    assert capabilities == ModelCapabilities(supports_tools=True, supports_embeddings=True)
    assert snapshot.capabilities_sha256 == sha256_json(capabilities)


def test_preflight_rejects_capability_before_reading_missing_credentials() -> None:
    """A known unsupported requirement fails locally and cannot trigger a paid request."""
    catalog = RuntimeModelCatalog(
        _catalog(provider="anthropic"),
        environment={},
        transport_factory=_UnusedTransport,
    )

    with pytest.raises(ModelCapabilityError, match="does not support embeddings"):
        catalog.preflight(
            "fixture-model",
            CapabilityRequirement(requires_embeddings=True),
        )


def test_resolution_requires_named_credential_without_exposing_its_value() -> None:
    """A missing local key reports only the configured environment variable name."""
    catalog = RuntimeModelCatalog(
        _catalog(),
        environment={},
        transport_factory=_UnusedTransport,
    )

    with pytest.raises(ModelCredentialError, match="FIXTURE_API_KEY"):
        catalog.resolve("fixture-model")


def test_resolution_rejects_unsupported_connection_and_incomplete_compatible_url() -> None:
    """The runtime allows only the focused provider set and explicit compatible endpoints."""
    unsupported = RuntimeModelCatalog(
        _catalog(provider="bedrock"),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=_UnusedTransport,
    )
    compatible = RuntimeModelCatalog(
        _catalog(provider="openai-compatible"),
        environment={"FIXTURE_API_KEY": "fixture-key"},
        transport_factory=_UnusedTransport,
    )

    with pytest.raises(ModelConnectionError, match="unsupported provider"):
        unsupported.snapshot("fixture-model")
    with pytest.raises(ModelConnectionError, match="needs connection.base_url"):
        compatible.resolve("fixture-model")


def test_tinker_resolution_uses_only_an_explicit_completed_model_sampler_factory() -> None:
    """Tinker construction never discovers, trains, or promotes a model handle."""
    constructed: list[tuple[str, str]] = []

    def sampler_factory(model: ModelSnapshot, api_key: str) -> TinkerSampler:
        constructed.append((model.model_id, api_key))
        return _FakeTinkerSampler()

    catalog = RuntimeModelCatalog(
        _catalog(provider="tinker"),
        environment={"FIXTURE_API_KEY": "fixture-tinker-key"},
        transport_factory=_UnusedTransport,
        tinker_sampler_factory=sampler_factory,
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
