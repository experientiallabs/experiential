"""Tests for local model-catalog TOML loading and credential boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import ArtifactInput, sha256_json
from wmo.common.models import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    ModelCatalogError,
    ModelRecord,
    ModelRoles,
    ModelSnapshot,
    SFTModelProvenance,
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
                billing_source=BillingSource.HOST_MANAGED,
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
    assert 'billing_source = "host_managed"' in path.read_text(encoding="utf-8")
    assert "OPENROUTER_API_KEY" in path.read_text(encoding="utf-8")
    assert "api_key =" not in path.read_text(encoding="utf-8")


def test_current_model_records_require_explicit_billing_source() -> None:
    """New catalog construction cannot silently infer who owns a provider credential."""
    with pytest.raises(ValidationError, match="billing_source"):
        ModelRecord(  # ty: ignore[missing-argument]
            connection="openrouter",
            model="candidate",
        )


def test_legacy_catalog_migrates_but_current_missing_billing_source_fails(
    tmp_path: Path,
) -> None:
    """Only the schema-v1 decode boundary supplies the conservative legacy source."""
    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """
schema_version = 1

[connections.provider]
provider = "openai"

[models.candidate]
connection = "provider"
model = "candidate"
""".strip(),
        encoding="utf-8",
    )
    migrated = load_model_catalog(legacy)
    assert migrated.schema_version == 2
    assert migrated.models["candidate"].billing_source == BillingSource.CUSTOMER_MANAGED

    current = tmp_path / "current.toml"
    current.write_text(
        legacy.read_text(encoding="utf-8").replace("schema_version = 1", "schema_version = 2"),
        encoding="utf-8",
    )
    with pytest.raises(ModelCatalogError, match="billing_source"):
        load_model_catalog(current)


def test_legacy_catalog_rejects_current_billing_source_injection(tmp_path: Path) -> None:
    """A schema-v1 label cannot smuggle host-paid attribution through migration."""
    path = tmp_path / "injected-v1.toml"
    write_model_catalog(path, _catalog())
    path.write_text(
        path.read_text(encoding="utf-8").replace("schema_version = 2", "schema_version = 1"),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="schema-v1 model record"):
        load_model_catalog(path)


@pytest.mark.parametrize("schema_version", ["true", "1.0"])
def test_legacy_catalog_rejects_noninteger_schema_one_lookalikes(
    tmp_path: Path,
    schema_version: str,
) -> None:
    """Boolean and floating-point values cannot enter the schema-v1 migration path."""
    path = tmp_path / f"schema-{schema_version}.toml"
    write_model_catalog(path, _catalog())
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "schema_version = 2", f"schema_version = {schema_version}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelCatalogError, match="schema_version must be an integer"):
        load_model_catalog(path)


def test_legacy_catalog_recursively_migrates_sft_base_model_billing(tmp_path: Path) -> None:
    """A schema-v1 SFT record conservatively migrates both registered and base models."""
    handle = "tinker://sampling-handle"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256="a" * 64),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run",
        model_id="model",
        model_sha256="d" * 64,
        result_id="result",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base-model",
            billing_source=BillingSource.HOST_MANAGED,
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": handle}),
    )
    catalog = ModelCatalog(
        connections={"tinker": ConnectionConfig(provider="tinker")},
        models={
            "fine-tuned": ModelRecord(
                connection="tinker",
                model=handle,
                billing_source=BillingSource.HOST_MANAGED,
                sft_provenance=provenance,
            )
        },
    )
    path = tmp_path / "legacy-sft.toml"
    write_model_catalog(path, catalog)
    current_text = path.read_text(encoding="utf-8")
    legacy_text = "\n".join(
        line for line in current_text.splitlines() if not line.startswith("billing_source =")
    ).replace("schema_version = 2", "schema_version = 1")
    path.write_text(legacy_text, encoding="utf-8")

    migrated = load_model_catalog(path)

    record = migrated.models["fine-tuned"]
    assert record.billing_source == BillingSource.CUSTOMER_MANAGED
    assert record.sft_provenance is not None
    assert record.sft_provenance.base_model.billing_source == BillingSource.CUSTOMER_MANAGED


def test_legacy_catalog_rejects_nested_sft_billing_source_injection(tmp_path: Path) -> None:
    """A schema-v1 nested SFT base-model label cannot assert host-paid attribution."""
    handle = "tinker://nested-injection"
    provenance = SFTModelProvenance(
        source_dataset=ArtifactInput(artifact_id="dataset", sha256="a" * 64),
        optimization_config=ArtifactInput(artifact_id="config", sha256="b" * 64),
        training_spec_sha256="c" * 64,
        run_id="run",
        model_id="model",
        model_sha256="d" * 64,
        result_id="result",
        result_sha256="e" * 64,
        base_model=ModelSnapshot(
            provider="tinker",
            model_id="base-model",
            billing_source=BillingSource.HOST_MANAGED,
            capabilities_sha256="f" * 64,
            connection_sha256="0" * 64,
        ),
        connection_config_sha256="1" * 64,
        sampling_handle_sha256=sha256_json({"sampling_handle": handle}),
    )
    path = tmp_path / "nested-injection.toml"
    write_model_catalog(
        path,
        ModelCatalog(
            connections={"tinker": ConnectionConfig(provider="tinker")},
            models={
                "fine-tuned": ModelRecord(
                    connection="tinker",
                    model=handle,
                    billing_source=BillingSource.HOST_MANAGED,
                    sft_provenance=provenance,
                )
            },
        ),
    )
    current_text = path.read_text(encoding="utf-8")
    legacy_text = current_text.replace("schema_version = 2", "schema_version = 1")
    legacy_text = legacy_text.replace('billing_source = "host_managed"\n', "", 1)
    path.write_text(legacy_text, encoding="utf-8")

    with pytest.raises(ModelCatalogError, match="schema-v1 SFT base model"):
        load_model_catalog(path)


def test_connections_only_catalog_round_trips_without_optimizer_roles(tmp_path: Path) -> None:
    """A gateway may author a real provider connection before it creates any model records."""
    path = tmp_path / "models.toml"
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={},
    )

    write_model_catalog(path, catalog)

    assert load_model_catalog(path) == catalog
    assert load_model_catalog(path).roles == ModelRoles()


def test_gateway_metadata_is_deployment_local_and_secret_free(tmp_path: Path) -> None:
    """Gateway protocol and integer pricing metadata persist outside frozen capabilities."""
    path = tmp_path / "models.toml"
    capabilities = ModelCapabilities(supports_tools=True)
    original_identity = capabilities.identity_sha256()
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                capabilities=capabilities,
                gateway=GatewayDeploymentMetadata(
                    exact_model_id="coding-exact-v1",
                    capabilities=GatewayDeploymentCapabilities(
                        supports_streaming=True,
                        supports_streaming_tool_arguments=True,
                    ),
                    prices=GatewayTokenPrices(
                        input_micro_usd_per_million_tokens=1_250_000,
                    ),
                    pricing_source="operator",
                ),
            )
        },
    )

    write_model_catalog(path, catalog)
    loaded = load_model_catalog(path)

    assert loaded == catalog
    assert loaded.models["coding"].capabilities is not None
    assert loaded.models["coding"].capabilities.identity_sha256() == original_identity
    assert "input_micro_usd_per_million_tokens = 1250000" in path.read_text(encoding="utf-8")


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
