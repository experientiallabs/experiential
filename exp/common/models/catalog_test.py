"""Tests for local model-catalog TOML loading and credential boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import (
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
    ReasoningEffort,
    SFTModelProvenance,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.catalog import (
    SANE_MAX_MODEL_CATALOG_SCHEMA_VERSION,
    infer_azure_api_surface,
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


def test_model_record_rejects_empty_and_oversized_served_model_id() -> None:
    """The served-model pin keeps the same bounded shape as the requested model ID."""
    for invalid in ("", "x" * 2_049):
        with pytest.raises(ValidationError, match="served_model_id"):
            ModelRecord(
                connection="openrouter",
                model="deepseek/deepseek-v4-flash",
                served_model_id=invalid,
                billing_source=BillingSource.HOST_MANAGED,
            )


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


def test_authored_schema_version_window_and_shape_are_pinned() -> None:
    """Change-detector for the authored catalog contract every hydration parses first.

    The version is a sane-ranged int, never a ``Literal``: a changed literal on
    a known field raises ``literal_error``, which the forward-compatible read
    path cannot drop, so a literal bump would warm-fail every older pod (the
    09-02 incident class on the authored side). If this test fails you changed
    the authored contract: an additive field is fine (old read-tolerant parsers
    drop it) — update the fingerprint; narrowing the version window back to a
    literal, or a revision that reinterprets existing fields, needs a new field
    name or a fleet-first tolerance release instead.
    """
    authored = _catalog().model_dump(mode="json")
    for accepted in (2, 3, SANE_MAX_MODEL_CATALOG_SCHEMA_VERSION):
        parsed = ModelCatalog.model_validate({**authored, "schema_version": accepted})
        assert parsed.schema_version == accepted
    for rejected in (0, 1, SANE_MAX_MODEL_CATALOG_SCHEMA_VERSION + 1):
        with pytest.raises(ValidationError):
            ModelCatalog.model_validate({**authored, "schema_version": rejected})
    assert sorted(ModelCatalog.model_fields) == [
        "connections",
        "gateway_pools",
        "models",
        "roles",
        "schema_version",
    ]


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
    capabilities = ModelCapabilities(supports_tools=True, supports_reasoning=True)
    original_identity = capabilities.identity_sha256()
    catalog = ModelCatalog(
        connections={"openai": ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")},
        models={
            "coding": ModelRecord(
                connection="openai",
                model="gpt-coding",
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities=capabilities,
                gateway=GatewayDeploymentMetadata(
                    exact_model_id="coding-exact-v1",
                    capabilities=GatewayDeploymentCapabilities(
                        supports_streaming=True,
                        supports_streaming_tool_arguments=True,
                        supported_reasoning_efforts=("low", "high", "max"),
                        reasoning_default_effort="max",
                        reasoning_effort_required=True,
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
    assert loaded.models["coding"].gateway is not None
    assert loaded.models["coding"].gateway.capabilities.supported_reasoning_efforts == (
        "low",
        "high",
        "max",
    )
    assert "input_micro_usd_per_million_tokens = 1250000" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "values",
    (("high", "low"), ("high", "high")),
)
def test_gateway_reasoning_efforts_require_unique_canonical_order(
    values: tuple[ReasoningEffort, ...],
) -> None:
    """Ambiguous provider effort sets fail when the catalog is authored."""
    with pytest.raises(ValueError):
        GatewayDeploymentCapabilities(supported_reasoning_efforts=values)


def test_required_gateway_reasoning_effort_needs_supported_values() -> None:
    """A mandatory wire parameter cannot omit its provider value domain."""
    with pytest.raises(ValueError, match="at least one supported reasoning effort"):
        GatewayDeploymentCapabilities(reasoning_effort_required=True)


def test_required_gateway_reasoning_effort_needs_an_explicit_default() -> None:
    """A mandatory wire parameter cannot force admission to guess its value."""
    with pytest.raises(ValueError, match="needs reasoning_default_effort"):
        GatewayDeploymentCapabilities(
            supported_reasoning_efforts=("low", "high"),
            reasoning_effort_required=True,
        )


def test_gateway_reasoning_default_must_be_supported() -> None:
    """A provider default outside the exact domain fails catalog loading."""
    with pytest.raises(ValueError, match="must be one of the supported"):
        GatewayDeploymentCapabilities(
            supported_reasoning_efforts=("low", "high"),
            reasoning_default_effort="max",
        )


@pytest.mark.parametrize("capabilities", (None, ModelCapabilities()))
def test_gateway_reasoning_metadata_requires_model_reasoning_support(
    capabilities: ModelCapabilities | None,
) -> None:
    """Authored reasoning metadata cannot contradict the model capability contract."""
    with pytest.raises(ValueError, match="supports_reasoning=true"):
        ModelRecord(
            connection="openai",
            model="gpt-coding",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities=capabilities,
            gateway=GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supported_reasoning_efforts=("medium",),
                )
            ),
        )


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


def test_azure_surface_inference_follows_the_resource_host() -> None:
    """Foundry hosts serve model inference, Azure OpenAI hosts serve deployments."""
    assert infer_azure_api_surface("https://resource.openai.azure.com") == "openai_deployments"
    assert (
        infer_azure_api_surface("https://Resource.Services.AI.Azure.com/models")
        == "model_inference"
    )
    assert infer_azure_api_surface("https://resource.inference.ai.azure.com") == "model_inference"
    assert infer_azure_api_surface("https://gateway.example.test/tenant-a") is None


def test_undeclared_foundry_endpoint_keeps_its_stored_identity() -> None:
    """Inference never moves the identity digest of a connection whose surface was not declared."""
    undeclared = ConnectionConfig(
        provider="azure",
        base_url="https://resource.services.ai.azure.com/models",
        api_key_env="AZURE_FOUNDRY_API_KEY",
        api_version="2024-05-01-preview",
    )
    declared = undeclared.model_copy(update={"azure_api_surface": "model_inference"})

    assert undeclared.identity_sha256() != declared.identity_sha256()
    assert undeclared.identity_sha256() == sha256_json(
        {
            "provider": "azure",
            "base_url": "https://resource.services.ai.azure.com/models",
            "api_version": "2024-05-01-preview",
        }
    )


def test_azure_connection_requires_endpoint_key_and_api_version() -> None:
    """Azure catalog records pair one resource endpoint with one key name and API version."""
    connection = ConnectionConfig(
        provider="azure",
        base_url="HTTPS://Resource.openai.azure.com/",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version="v1",
    )
    assert connection.azure_api_surface is None
    inference = ConnectionConfig(
        provider="azure",
        base_url="https://resource.services.ai.azure.com/models",
        api_key_env="AZURE_FOUNDRY_API_KEY",
        api_version="2024-05-01-preview",
        azure_api_surface="model_inference",
    )
    inference_root = inference.model_copy(
        update={"base_url": "https://resource.services.ai.azure.com"}
    )
    assert inference.identity_sha256() == inference_root.identity_sha256()
    inference_redundant_terminal = inference.model_copy(
        update={"base_url": "https://resource.services.ai.azure.com//models"}
    )
    assert inference_redundant_terminal.identity_sha256() == inference_root.identity_sha256()
    inference_internal_separator = inference.model_copy(
        update={"base_url": "https://gateway.example.test/tenant-a//azure/models"}
    )
    inference_collapsed_separator = inference_internal_separator.model_copy(
        update={"base_url": "https://gateway.example.test/tenant-a/azure/models"}
    )
    assert (
        inference_internal_separator.identity_sha256()
        != inference_collapsed_separator.identity_sha256()
    )
    assert inference.identity_sha256() != connection.identity_sha256()
    explicit_classic = connection.model_copy(update={"azure_api_surface": "openai_deployments"})
    assert explicit_classic.identity_sha256() == connection.identity_sha256()
    assert connection.identity_sha256() == sha256_json(
        {
            "provider": "azure",
            "base_url": "https://resource.openai.azure.com",
            "api_version": "v1",
        }
    )
    with pytest.raises(ValueError, match="azure_api_surface"):
        ConnectionConfig(provider="openai", azure_api_surface="model_inference")
    with pytest.raises(ValueError, match="dated api_version"):
        ConnectionConfig(
            provider="azure",
            base_url="https://resource.services.ai.azure.com",
            api_key_env="AZURE_FOUNDRY_API_KEY",
            api_version="v1",
            azure_api_surface="model_inference",
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


def test_bedrock_connection_accepts_complete_key_pair_without_hashing_secret_pointers() -> None:
    """Bedrock accepts ambient or paired auth while identity remains endpoint-only."""
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
    explicit = ConnectionConfig(
        provider="bedrock",
        api_key_env="AWS_SECRET_ACCESS_KEY",
        aws_access_key_id_env="BEDROCK_ACCESS_KEY_ID",
        region="us-east-1",
    )
    assert explicit.identity_sha256() != with_region.identity_sha256()
    assert (
        explicit.identity_sha256()
        == ConnectionConfig(
            provider="bedrock",
            api_key_env="OTHER_SECRET_ACCESS_KEY",
            aws_access_key_id_env="OTHER_ACCESS_KEY_ID",
            bedrock_auth_mode="access_key_pair",
            region="us-east-1",
        ).identity_sha256()
    )
    api_key = ConnectionConfig(
        provider="bedrock",
        api_key_env="BEDROCK_API_KEY",
        bedrock_auth_mode="api_key",
        region="us-east-1",
    )
    assert api_key.identity_sha256() != explicit.identity_sha256()
    with pytest.raises(ValueError, match="bedrock_auth_mode"):
        ConnectionConfig(provider="openai", bedrock_auth_mode="api_key")
    with pytest.raises(ValueError, match="base_url"):
        ConnectionConfig(provider="bedrock", base_url="https://bedrock.example.test")


def test_absent_bedrock_fields_preserve_legacy_connection_payload_shape() -> None:
    """Optional Bedrock metadata does not rewrite unrelated canonical payloads."""
    legacy = ConnectionConfig(provider="openai", api_key_env="OPENAI_API_KEY")
    payload = legacy.model_dump(mode="json", exclude_none=False)

    assert "aws_access_key_id_env" not in payload
    assert "bedrock_auth_mode" not in payload

    explicit = ConnectionConfig(
        provider="bedrock",
        api_key_env="BEDROCK_SECRET_ACCESS_KEY",
        aws_access_key_id_env="BEDROCK_ACCESS_KEY_ID",
        bedrock_auth_mode="access_key_pair",
    ).model_dump(mode="json", exclude_none=False)
    assert explicit["aws_access_key_id_env"] == "BEDROCK_ACCESS_KEY_ID"
    assert explicit["bedrock_auth_mode"] == "access_key_pair"


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


def test_declared_foundry_endpoint_spellings_keep_their_stored_identity() -> None:
    """Identity folds only the terminal /models segment, so no stored digest moves on upgrade."""
    root = ConnectionConfig(
        provider="azure",
        base_url="https://resource.services.ai.azure.com",
        api_key_env="AZURE_FOUNDRY_API_KEY",
        api_version="2024-05-01-preview",
        azure_api_surface="model_inference",
    )
    with_models = root.model_copy(
        update={"base_url": "https://resource.services.ai.azure.com/models"}
    )
    with_v1_root = root.model_copy(
        update={"base_url": "https://resource.services.ai.azure.com/openai/v1"}
    )

    assert root.identity_sha256() == with_models.identity_sha256()
    assert root.identity_sha256() != with_v1_root.identity_sha256()


def test_astra_responses_capability_slots_default_off() -> None:
    """The three GPT-6 Astra Responses capability slots exist and default off.

    These are declaration slots for async function calling, mid-turn steering,
    and mid-conversation reasoning-effort updates. They default False (no
    deployment advertises a behavior the decoder/turn lifecycle does not yet
    honor) and, being defaulted, stay identity-invisible (see
    gateway_catalog_test's identity-digest pin). The platform's
    generation-capability vocabulary is drift-locked to these field names, so
    they must remain present for that projection to admit the keys.
    """
    caps = GatewayDeploymentCapabilities()
    assert caps.supports_async_tools is False
    assert caps.supports_mid_turn_steering is False
    assert caps.supports_reasoning_effort_update is False
    # Defaulted addition contributes zero identity bytes.
    assert caps.model_dump(mode="json", by_alias=True, exclude_defaults=True) == {}
