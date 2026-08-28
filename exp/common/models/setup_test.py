"""Collection-first provider model setup tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    ProviderSetupError,
    catalog_state_sha256,
    configure_provider_catalog,
    write_model_catalog,
)


def _setup(*, judge_model: str = "judge-id") -> ProviderSetup:
    """Return three independent aliases on two explicitly available connections.

    Args:
        judge_model: Provider model ID assigned to the judge alias.

    Returns:
        Complete provider setup fixture.
    """
    return ProviderSetup(
        connections=(
            ProviderConnection(
                name="openai-main",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
            ),
            ProviderConnection(
                name="gemini-embed",
                provider="gemini",
                api_key_env="GEMINI_API_KEY",
            ),
        ),
        models=(
            ProviderModelSelection(
                alias="world",
                connection="openai-main",
                model="world-id",
                capabilities=ModelCapabilities(supports_tools=True),
            ),
            ProviderModelSelection(
                alias="judge",
                connection="openai-main",
                model=judge_model,
                capabilities=ModelCapabilities(supports_structured_output=True),
            ),
            ProviderModelSelection(
                alias="embed",
                connection="gemini-embed",
                model="embedding-id",
                capabilities=ModelCapabilities(
                    supports_embeddings=True,
                    input_cost_per_million_tokens_usd=0.25,
                ),
            ),
        ),
        world_model="world",
        judge="judge",
        embedder="embed",
    )


def test_setup_writes_named_models_and_independent_roles(tmp_path: Path) -> None:
    """Collected aliases remain reusable instead of becoming fixed role-named records.

    Args:
        tmp_path: Temporary root receiving the shared catalog.
    """
    path = tmp_path / "models.toml"

    first = configure_provider_catalog(path, _setup())
    second = configure_provider_catalog(path, _setup())

    assert second == first
    assert first.roles == ModelRoles(world_model="world", judge="judge", embedder="embed")
    assert first.models["world"].capabilities == ModelCapabilities(supports_tools=True)
    assert first.models["judge"].capabilities == ModelCapabilities(supports_structured_output=True)
    assert first.models["embed"].capabilities == ModelCapabilities(
        supports_embeddings=True,
        input_cost_per_million_tokens_usd=0.25,
    )
    assert "OPENAI_API_KEY" in path.read_text(encoding="utf-8")


def test_setup_preserves_unrelated_models_and_router_candidates(tmp_path: Path) -> None:
    """Build setup cannot silently choose or replace router candidate state.

    Args:
        tmp_path: Temporary root containing preserved catalog roles.
    """
    path = tmp_path / "models.toml"
    candidate = ModelRecord(
        billing_source=BillingSource.CUSTOMER_MANAGED, connection="router", model="vendor/candidate"
    )
    existing = ModelCatalog(
        connections={
            "router": ConnectionConfig(
                provider="openrouter",
                api_key_env="OPENROUTER_API_KEY",
            )
        },
        models={"candidate-a": candidate},
        roles=ModelRoles(candidates=("candidate-a",), incumbent="candidate-a"),
    )
    write_model_catalog(path, existing)

    configured = configure_provider_catalog(path, _setup())

    assert configured.models["candidate-a"] == candidate
    assert configured.roles.candidates == ("candidate-a",)
    assert configured.roles.incumbent == "candidate-a"


def test_conflicting_alias_requires_replace_and_protected_alias_never_replaces(
    tmp_path: Path,
) -> None:
    """Conflicts fail atomically, including when a router role protects the alias.

    Args:
        tmp_path: Temporary root containing conflicting model aliases.
    """
    path = tmp_path / "models.toml"
    original = configure_provider_catalog(path, _setup())
    payload = path.read_bytes()

    with pytest.raises(ProviderSetupError, match="--replace"):
        configure_provider_catalog(path, _setup(judge_model="other"))
    assert path.read_bytes() == payload

    protected = original.model_copy(
        update={
            "roles": original.roles.model_copy(
                update={"candidates": ("judge",), "incumbent": "judge"}
            )
        }
    )
    write_model_catalog(path, protected)
    with pytest.raises(ProviderSetupError, match="router or training role"):
        configure_provider_catalog(path, _setup(judge_model="other"), replace=True)


def test_prompt_session_digest_rejects_concurrent_catalog_change(tmp_path: Path) -> None:
    """A completed prompt session cannot overwrite catalog edits made during collection.

    Args:
        tmp_path: Temporary root whose catalog changes during simulated collection.
    """
    path = tmp_path / "models.toml"
    starting = catalog_state_sha256(path)
    configure_provider_catalog(path, _setup())

    with pytest.raises(ProviderSetupError, match="changed while setup"):
        configure_provider_catalog(path, _setup(), expected_state_sha256=starting)


def test_non_role_alias_collision_fails_without_writing(tmp_path: Path) -> None:
    """Every confirmed model alias is either saved exactly or rejected before commit.

    Args:
        tmp_path: Temporary root containing a conflicting non-role alias.
    """
    path = tmp_path / "models.toml"
    setup = _setup()
    first = setup.model_copy(
        update={
            "models": (
                *setup.models,
                ProviderModelSelection(
                    alias="available",
                    connection="openai-main",
                    model="first-id",
                ),
            )
        }
    )
    configure_provider_catalog(path, first)
    payload = path.read_bytes()
    second = setup.model_copy(
        update={
            "models": (
                *setup.models,
                ProviderModelSelection(
                    alias="available",
                    connection="openai-main",
                    model="second-id",
                ),
            )
        }
    )

    with pytest.raises(ProviderSetupError, match="available.*already differs"):
        configure_provider_catalog(path, second)
    assert path.read_bytes() == payload


def test_embedder_role_requires_explicit_embedding_capability() -> None:
    """Provider identity never implies per-model embedding support.

    The regression validates the complete setup model without writing catalog state.
    """
    setup = _setup()
    replacement = ModelCapabilities(input_cost_per_million_tokens_usd=0.25)
    models = tuple(
        model.model_copy(update={"capabilities": replacement}) if model.alias == "embed" else model
        for model in setup.models
    )
    with pytest.raises(ValueError, match="must declare embedding support"):
        ProviderSetup(
            connections=setup.connections,
            models=models,
            world_model="world",
            judge="judge",
            embedder="embed",
        )


def test_embedding_model_requires_explicit_input_price() -> None:
    """Build consent never relies on an invented or silently absent embedding price.

    The regression covers both rejected unknown pricing and explicit zero pricing.
    """
    with pytest.raises(ValueError, match="explicit input cost"):
        ProviderModelSelection(
            alias="embed",
            connection="gemini",
            model="embedding-id",
            capabilities=ModelCapabilities(supports_embeddings=True),
        )


@pytest.mark.parametrize(
    ("provider", "base_url", "api_key_env", "api_version", "region"),
    [
        ("openai", None, "SELECTED_API_KEY", None, None),
        ("openrouter", None, "SELECTED_API_KEY", None, None),
        ("anthropic", None, "SELECTED_API_KEY", None, None),
        ("gemini", None, "SELECTED_API_KEY", None, None),
        ("openai-compatible", "https://models.example.test/v1", "SELECTED_API_KEY", None, None),
        ("azure", "https://resource.openai.azure.com", "AZURE_OPENAI_API_KEY", "v1", None),
        ("bedrock", None, None, None, "us-east-1"),
    ],
)
def test_setup_accepts_each_supported_connection(
    provider: str,
    base_url: str | None,
    api_key_env: str | None,
    api_version: str | None,
    region: str | None,
) -> None:
    """Every supported provider remains an explicit user-selected connection.

    Args:
        provider: Parameterized supported provider kind.
        base_url: Required Azure or compatible endpoint, otherwise ``None``.
        api_key_env: Named credential variable, or ``None`` for Bedrock.
        api_version: Required Azure API version, otherwise ``None``.
        region: Optional Bedrock region.
    """
    connection = ProviderConnection(
        name="selected",
        provider=provider,
        api_key_env=api_key_env,
        base_url=base_url,
        api_version=api_version,
        region=region,
    )

    assert connection.provider == provider
    assert connection.base_url == base_url
    assert connection.api_key_env == api_key_env
    assert connection.api_version == api_version
    assert connection.region == region


def test_setup_rejects_an_incomplete_bedrock_access_key_pair() -> None:
    """Bedrock setup never accepts a secret reference without its access-key identifier."""
    with pytest.raises(ValueError, match="api_key_env"):
        ProviderConnection(
            name="bedrock",
            provider="bedrock",
            api_key_env="AWS_ACCESS_KEY_ID",
            region="us-east-1",
        )


def test_setup_accepts_an_explicit_bedrock_access_key_pair() -> None:
    """Stored Bedrock connections may pair a named secret with a non-secret key ID."""
    connection = ProviderConnection(
        name="bedrock",
        provider="bedrock",
        api_key_env="AWS_SECRET_ACCESS_KEY",
        aws_access_key_id_env="BEDROCK_ACCESS_KEY_ID",
        bedrock_auth_mode="access_key_pair",
        region="us-east-1",
    )

    assert connection.catalog_config().aws_access_key_id_env == "BEDROCK_ACCESS_KEY_ID"
    assert connection.catalog_config().bedrock_auth_mode == "access_key_pair"


def test_setup_accepts_a_bedrock_api_key_without_an_access_key_id() -> None:
    """Bedrock bearer setup preserves its explicit authentication mode."""
    connection = ProviderConnection(
        name="bedrock",
        provider="bedrock",
        api_key_env="BEDROCK_API_KEY",
        bedrock_auth_mode="api_key",
        region="us-east-1",
    )

    assert connection.catalog_config().bedrock_auth_mode == "api_key"
    assert connection.catalog_config().aws_access_key_id_env is None


def test_absent_bedrock_fields_preserve_legacy_setup_payload_shape() -> None:
    """New setup metadata remains absent from pre-Bedrock connection shapes."""
    connection = ProviderConnection(
        name="openai-main",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    )

    payload = connection.model_dump(mode="json", exclude_none=False)

    assert "aws_access_key_id_env" not in payload
    assert "bedrock_auth_mode" not in payload
