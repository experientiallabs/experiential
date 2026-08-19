"""Tests for local model-catalog TOML loading and credential boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelCatalogError,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)


def _catalog() -> ModelCatalog:
    return ModelCatalog(
        connections={
            "openrouter": ConnectionConfig(
                provider="openrouter",
                api_key_env="OPENROUTER_API_KEY",
            )
        },
        models={
            "candidate-economy": ModelRecord(
                connection="openrouter",
                model="deepseek/deepseek-v4-flash",
                capabilities=ModelCapabilities(
                    supports_tools=True,
                    context_window_tokens=128_000,
                    maximum_output_tokens=16_000,
                ),
            )
        },
        roles=ModelRoles(candidates=("candidate-economy",), incumbent="candidate-economy"),
    )


def test_model_catalog_round_trip_preserves_aliases_and_environment_name(tmp_path: Path) -> None:
    """Models TOML records aliases and an environment variable name, never its value."""
    path = tmp_path / "models.toml"
    catalog = _catalog()

    write_model_catalog(path, catalog)

    assert load_model_catalog(path) == catalog
    assert "OPENROUTER_API_KEY" in path.read_text(encoding="utf-8")
    assert "api_key =" not in path.read_text(encoding="utf-8")


def test_model_record_round_trips_an_alternate_served_identity(tmp_path: Path) -> None:
    """A declared served identity persists so vLLM alias endpoints stay resolvable."""
    path = tmp_path / "models.toml"
    catalog = _catalog()
    record = catalog.models["candidate-economy"]
    catalog = catalog.model_copy(
        update={
            "models": {
                "candidate-economy": record.model_copy(
                    update={"served_model_id": "deepseek-v4-flash"}
                )
            }
        }
    )

    write_model_catalog(path, catalog)

    loaded = load_model_catalog(path)
    assert loaded.models["candidate-economy"].served_model_id == "deepseek-v4-flash"
    assert loaded == catalog


def test_model_catalog_rejects_credential_values_and_embedded_url_credentials(
    tmp_path: Path,
) -> None:
    """The catalog permits a credential environment name but no secret value or URL credentials."""
    raw_key_path = tmp_path / "raw-key.toml"
    raw_key_path.write_text(
        """
[connections.openrouter]
provider = "openrouter"
api_key = "sk-abcdefghijklmnopqrstuvwxyz123456"

[models.candidate-economy]
connection = "openrouter"
model = "deepseek/deepseek-v4-flash"
""".strip(),
        encoding="utf-8",
    )
    embedded_credential_path = tmp_path / "embedded.toml"
    embedded_credential_path.write_text(
        """
[connections.private-world-model]
provider = "openai-compatible"
base_url = "https://user:password@models.example.com/v1"

[models.world-model]
connection = "private-world-model"
model = "private-world-model"
""".strip(),
        encoding="utf-8",
    )
    query_credential_path = tmp_path / "query-credential.toml"
    query_credential_path.write_text(
        """
[connections.private-world-model]
provider = "openai-compatible"
base_url = "https://models.example.com/v1?api_key=sk-abcdefghijklmnopqrstuvwxyz123456"

[models.world-model]
connection = "private-world-model"
model = "private-world-model"
""".strip(),
        encoding="utf-8",
    )
    model_credential_path = tmp_path / "model-credential.toml"
    model_credential_path.write_text(
        """
[connections.openrouter]
provider = "openrouter"

[models.candidate-economy]
connection = "openrouter"
model = "sk-abcdefghijklmnopqrstuvwxyz123456"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="api_key"):
        load_model_catalog(raw_key_path)
    with pytest.raises(ModelCatalogError, match="embed credentials"):
        load_model_catalog(embedded_credential_path)
    with pytest.raises(ModelCatalogError, match="query parameters"):
        load_model_catalog(query_credential_path)
    with pytest.raises(ModelCatalogError, match="model identity"):
        load_model_catalog(model_credential_path)


def test_openai_compatible_records_require_explicit_capabilities(tmp_path: Path) -> None:
    """Private compatible endpoints cannot inherit unproven provider-wide capability claims."""
    path = tmp_path / "compatible.toml"
    path.write_text(
        """
[connections.private-world-model]
provider = "openai-compatible"
base_url = "https://models.example.com/v1"

[models.world-model]
connection = "private-world-model"
model = "private-world-model"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="explicit capabilities"):
        load_model_catalog(path)


@pytest.mark.parametrize("provider", ("anthropic", "gemini", "openai", "openrouter", "tinker"))
def test_native_provider_rejects_a_custom_endpoint_that_could_receive_its_key(
    provider: str,
) -> None:
    """Native provider keys cannot follow a catalog-controlled custom endpoint."""
    with pytest.raises(ValueError, match="openai-compatible"):
        ConnectionConfig(
            provider=provider,
            base_url="https://untrusted.example.test/v1",
            api_key_env="FIXTURE_API_KEY",
        )


def test_azure_connection_requires_endpoint_key_and_api_version() -> None:
    """Azure catalog records pair one resource endpoint with one key name and API version."""
    connection = ConnectionConfig(
        provider="azure",
        base_url="HTTPS://Resource.openai.azure.com/",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    )

    assert (
        connection.identity_sha256()
        == ConnectionConfig(
            provider="azure",
            base_url="https://resource.openai.azure.com",
            api_key_env="OTHER_AZURE_KEY",
            api_version="v1",
        ).identity_sha256()
    )
    assert (
        connection.identity_sha256()
        != ConnectionConfig(
            provider="azure",
            base_url="https://resource.openai.azure.com",
            api_key_env="AZURE_OPENAI_API_KEY",
            api_version="2024-10-21",
        ).identity_sha256()
    )
    with pytest.raises(ValueError, match="resource endpoint"):
        ConnectionConfig(provider="azure", api_key_env="AZURE_OPENAI_API_KEY", api_version="v1")
    with pytest.raises(ValueError, match="api_version"):
        ConnectionConfig(
            provider="azure",
            base_url="https://resource.openai.azure.com",
            api_key_env="AZURE_OPENAI_API_KEY",
        )
    with pytest.raises(ValueError, match="embed credentials"):
        ConnectionConfig(
            provider="azure",
            base_url="https://user:secret@resource.openai.azure.com",
            api_key_env="AZURE_OPENAI_API_KEY",
            api_version="v1",
        )
    with pytest.raises(ValueError, match="query parameters or fragments"):
        ConnectionConfig(
            provider="azure",
            base_url="https://resource.openai.azure.com?api-key=secret",
            api_key_env="AZURE_OPENAI_API_KEY",
            api_version="v1",
        )
    with pytest.raises(ValueError, match="query parameters or fragments"):
        ConnectionConfig(
            provider="azure",
            base_url="https://resource.openai.azure.com#frag",
            api_key_env="AZURE_OPENAI_API_KEY",
            api_version="v1",
        )


def test_bedrock_connection_rejects_api_keys_and_includes_region_in_identity() -> None:
    """Bedrock catalog records use the AWS chain and keep region in connection identity."""
    with_region = ConnectionConfig(provider="bedrock", region="us-east-1")
    without_region = ConnectionConfig(provider="bedrock")

    assert with_region.identity_sha256() != without_region.identity_sha256()
    assert (
        with_region.identity_sha256()
        == ConnectionConfig(
            provider="bedrock",
            region="us-east-1",
        ).identity_sha256()
    )
    with pytest.raises(ValueError, match="api_key_env"):
        ConnectionConfig(provider="bedrock", api_key_env="AWS_ACCESS_KEY_ID")
    with pytest.raises(ValueError, match="base_url"):
        ConnectionConfig(provider="bedrock", base_url="https://bedrock.example.test")


def test_azure_and_bedrock_models_require_explicit_capabilities(tmp_path: Path) -> None:
    """Provider names do not imply Azure or Bedrock protocol support or prices."""
    azure_path = tmp_path / "azure.toml"
    azure_path.write_text(
        """
[connections.azure]
provider = "azure"
base_url = "https://resource.openai.azure.com"
api_key_env = "AZURE_OPENAI_API_KEY"
api_version = "v1"

[models.gpt]
connection = "azure"
model = "gpt-deployment"
""".strip(),
        encoding="utf-8",
    )
    bedrock_path = tmp_path / "bedrock.toml"
    bedrock_path.write_text(
        """
[connections.bedrock]
provider = "bedrock"
region = "us-east-1"

[models.claude]
connection = "bedrock"
model = "anthropic.claude-sonnet-4-5"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="explicit capabilities"):
        load_model_catalog(azure_path)
    with pytest.raises(ModelCatalogError, match="explicit capabilities"):
        load_model_catalog(bedrock_path)
