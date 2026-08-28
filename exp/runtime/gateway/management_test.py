"""Behavior tests for content-free gateway management operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.models import GATEWAY_EXCLUDED_PROVIDERS, ConnectionConfig, ModelCatalog
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite.store import GatewayStoreError
from exp.runtime.models import SUPPORTED_PROVIDERS


def test_upsert_provider_connection_rejects_an_unsupported_provider(tmp_path: Path) -> None:
    """An unknown provider identifier fails closed before it can reach SQLite authority."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    with pytest.raises(GatewayStoreError, match="unsupported provider 'openai_compatible'"):
        manager.upsert_provider_connection(
            connection_id="mistyped",
            config=ConnectionConfig(
                provider="openai_compatible",
                base_url="http://127.0.0.1:9/v1",
                api_key_env="TEST_PROVIDER_KEY",
            ),
        )

    assert manager.provider_connections() == ()


def test_upsert_provider_connection_error_lists_every_servable_provider(tmp_path: Path) -> None:
    """The rejection names every gateway-servable provider and no excluded one."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    with pytest.raises(GatewayStoreError) as excinfo:
        manager.upsert_provider_connection(
            connection_id="mistyped",
            config=ConnectionConfig(provider="not-a-provider"),
        )

    message = str(excinfo.value)
    for provider in SUPPORTED_PROVIDERS - GATEWAY_EXCLUDED_PROVIDERS:
        assert provider in message
    for provider in GATEWAY_EXCLUDED_PROVIDERS:
        assert provider not in message


def test_upsert_provider_connection_rejects_gateway_excluded_providers(tmp_path: Path) -> None:
    """Runtime-only providers whose records never serve are rejected at configuration."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    with pytest.raises(GatewayStoreError, match="unsupported provider 'tinker'"):
        manager.upsert_provider_connection(
            connection_id="training",
            config=ConnectionConfig(provider="tinker"),
        )

    assert manager.provider_connections() == ()
    assert "tinker" in SUPPORTED_PROVIDERS
    assert "tinker" in GATEWAY_EXCLUDED_PROVIDERS


def test_upsert_provider_connection_accepts_registry_supported_providers(
    tmp_path: Path,
) -> None:
    """Registry-supported identifiers remain configurable for native and custom endpoints."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    custom_changed, custom_authority = manager.upsert_provider_connection(
        connection_id="custom",
        config=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="TEST_PROVIDER_KEY",
        ),
    )
    native_changed, native_authority = manager.upsert_provider_connection(
        connection_id="native",
        config=ConnectionConfig(provider="anthropic", api_key_env="TEST_PROVIDER_KEY"),
    )

    assert custom_changed
    assert custom_authority.config.provider == "openai-compatible"
    assert native_changed
    assert native_authority.config.provider == "anthropic"
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "openai-compatible" in SUPPORTED_PROVIDERS


def test_legacy_bedrock_snapshot_binds_to_canonical_pair_authority(tmp_path: Path) -> None:
    """An authored pre-mode snapshot remains bindable after canonical persistence."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    legacy = ConnectionConfig(
        provider="bedrock",
        api_key_env="AWS_SECRET_ACCESS_KEY",
        aws_access_key_id_env="AWS_ACCESS_KEY_ID",
        region="us-west-2",
    )
    changed, authority = manager.upsert_provider_connection(
        connection_id="bedrock",
        config=legacy,
    )

    bindings = manager.provider_bindings(ModelCatalog(connections={"bedrock": legacy}, models={}))

    assert changed
    assert authority.config.bedrock_auth_mode == "access_key_pair"
    assert bindings[0].connection_sha256 == authority.connection_sha256
