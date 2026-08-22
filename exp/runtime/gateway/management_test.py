"""Behavior tests for content-free gateway management operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.models import ConnectionConfig
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


def test_upsert_provider_connection_error_lists_every_supported_provider(tmp_path: Path) -> None:
    """The rejection names the full registry-supported provider set for remediation."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()

    with pytest.raises(GatewayStoreError) as excinfo:
        manager.upsert_provider_connection(
            connection_id="mistyped",
            config=ConnectionConfig(provider="not-a-provider"),
        )

    message = str(excinfo.value)
    for provider in SUPPORTED_PROVIDERS:
        assert provider in message


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
