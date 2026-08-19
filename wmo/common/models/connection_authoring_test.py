"""Tests for provider authoring without optimizer role assignment."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ProviderConnection,
    ProviderConnectionAuthoringError,
    ProviderSetup,
    configure_provider_connections,
    load_model_catalog,
    write_model_catalog,
)


def test_role_free_connection_authoring_creates_connections_only_catalog(tmp_path: Path) -> None:
    """Gateway setup can persist one real BYOK connection before choosing a model or role."""
    path = tmp_path / "models.toml"

    configured = configure_provider_connections(
        path,
        (
            ProviderConnection(
                name="openai",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
            ),
        ),
    )

    assert configured.models == {}
    assert configured.roles.candidates == ()
    assert load_model_catalog(path) == configured


def test_role_free_authoring_preserves_models_and_rejects_connection_rebinding(
    tmp_path: Path,
) -> None:
    """Connection updates preserve unrelated state and cannot move an existing model endpoint."""
    path = tmp_path / "models.toml"
    original = ModelCatalog(
        connections={
            "openai": ProviderConnection(
                name="openai", provider="openai", api_key_env="OPENAI_API_KEY"
            ).catalog_config()
        },
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=ModelCapabilities(supports_completions=True),
            )
        },
    )
    write_model_catalog(path, original)

    configure_provider_connections(
        path,
        (
            ProviderConnection(
                name="anthropic",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
            ),
        ),
    )
    with pytest.raises(ProviderConnectionAuthoringError, match="used by model aliases"):
        configure_provider_connections(
            path,
            (
                ProviderConnection(
                    name="openai",
                    provider="openai-compatible",
                    api_key_env="COMPATIBLE_API_KEY",
                    base_url="https://models.example.test/v1",
                ),
            ),
            replace=True,
        )

    loaded = load_model_catalog(path)
    assert loaded.models == original.models
    assert set(loaded.connections) == {"anthropic", "openai"}


def test_optimizer_provider_setup_still_requires_all_build_roles() -> None:
    """The new runtime authoring seam does not weaken build and optimize role validation."""
    with pytest.raises(ValidationError, match="world_model"):
        ProviderSetup.model_validate(
            {
                "connections": (
                    ProviderConnection(
                        name="openai",
                        provider="openai",
                        api_key_env="OPENAI_API_KEY",
                    ),
                )
            }
        )
