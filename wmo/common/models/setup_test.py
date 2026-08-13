"""Provider-first model catalog setup tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmo.common.models import (
    ConnectionConfig,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    ProviderConnection,
    ProviderModelSelection,
    ProviderSetup,
    ProviderSetupError,
    configure_provider_catalog,
    load_model_catalog,
    write_model_catalog,
)


def _openai_setup(*, judge: str = "judge-id") -> ProviderSetup:
    """Return one exact native-provider setup without model defaults."""
    return ProviderSetup(
        connections=(
            ProviderConnection(
                name="openai-main",
                provider="openai",
                api_key_env="OPENAI_API_KEY",
            ),
        ),
        world_model=ProviderModelSelection(
            connection="openai-main",
            model="world-id",
            supports_tools=True,
        ),
        judge=ProviderModelSelection(connection="openai-main", model=judge),
        embedder=ProviderModelSelection(connection="openai-main", model="embedding-id"),
    )


def test_setup_creates_secret_free_build_roles_and_is_idempotent(tmp_path: Path) -> None:
    """Setup records exact role IDs and credential names, never values."""
    path = tmp_path / "models.toml"

    first = configure_provider_catalog(path, _openai_setup())
    second = configure_provider_catalog(path, _openai_setup())

    assert second == first
    assert first.roles.world_model == "world-model"
    assert first.roles.judge == "judge"
    assert first.roles.embedder == "embedder"
    assert first.models["world-model"].capabilities == ModelCapabilities(supports_tools=True)
    assert first.models["embedder"].capabilities == ModelCapabilities(supports_embeddings=True)
    payload = path.read_text()
    assert "OPENAI_API_KEY" in payload
    assert "sk-secret-value" not in payload


def test_setup_preserves_existing_catalog_and_router_candidates(tmp_path: Path) -> None:
    """Adding build-time roles does not select or mutate router candidates."""
    path = tmp_path / "models.toml"
    existing = ModelCatalog(
        connections={
            "router-provider": ConnectionConfig(
                provider="openrouter", api_key_env="OPENROUTER_API_KEY"
            )
        },
        models={"candidate-a": ModelRecord(connection="router-provider", model="vendor/candidate")},
        roles=ModelRoles(
            candidates=("candidate-a",),
            incumbent="candidate-a",
            teacher="candidate-a",
        ),
    )
    write_model_catalog(path, existing)

    configured = configure_provider_catalog(path, _openai_setup())

    assert configured.connections["router-provider"] == existing.connections["router-provider"]
    assert configured.models["candidate-a"] == existing.models["candidate-a"]
    assert configured.roles.candidates == ("candidate-a",)
    assert configured.roles.incumbent == "candidate-a"
    assert configured.roles.teacher == "candidate-a"


def test_conflict_fails_without_changing_catalog_until_replace(tmp_path: Path) -> None:
    """A conflicting setup needs explicit replacement and failed writes leave state intact."""
    path = tmp_path / "models.toml"
    original = configure_provider_catalog(path, _openai_setup())
    original_payload = path.read_bytes()

    with pytest.raises(ProviderSetupError, match="--replace"):
        configure_provider_catalog(path, _openai_setup(judge="different-judge"))

    assert path.read_bytes() == original_payload
    assert load_model_catalog(path) == original
    replaced = configure_provider_catalog(
        path, _openai_setup(judge="different-judge"), replace=True
    )
    assert replaced.models["judge"].model == "different-judge"


def test_replace_does_not_mutate_a_connection_used_by_router_candidates(tmp_path: Path) -> None:
    """Even explicit build-role replacement cannot change an existing router candidate."""
    path = tmp_path / "models.toml"
    existing = ModelCatalog(
        connections={
            "openai-main": ConnectionConfig(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
        },
        models={"candidate-a": ModelRecord(connection="openai-main", model="vendor/candidate")},
        roles=ModelRoles(candidates=("candidate-a",), incumbent="candidate-a"),
    )
    write_model_catalog(path, existing)

    with pytest.raises(ProviderSetupError, match="use a new connection name"):
        configure_provider_catalog(path, _openai_setup(), replace=True)

    assert load_model_catalog(path) == existing


def test_replace_cannot_drift_connection_behind_an_identical_preserved_alias(
    tmp_path: Path,
) -> None:
    """Connection changes are blocked even when a preserved fixed-alias record stays equal."""
    path = tmp_path / "models.toml"
    existing = ModelCatalog(
        connections={
            "openai-main": ConnectionConfig(provider="openrouter", api_key_env="OPENROUTER_API_KEY")
        },
        models={
            "judge": ModelRecord(
                connection="openai-main",
                model="judge-id",
                capabilities=ModelCapabilities(),
            )
        },
        roles=ModelRoles(candidates=("judge",), incumbent="judge", judge="judge"),
    )
    write_model_catalog(path, existing)

    with pytest.raises(ProviderSetupError, match="use a new connection name"):
        configure_provider_catalog(path, _openai_setup(), replace=True)

    assert load_model_catalog(path) == existing


def test_anthropic_primary_can_use_a_separate_gemini_embedder(tmp_path: Path) -> None:
    """Provider setup represents roles on separate current-runtime connections."""
    setup = ProviderSetup(
        connections=(
            ProviderConnection(
                name="anthropic-main",
                provider="anthropic",
                api_key_env="ANTHROPIC_API_KEY",
            ),
            ProviderConnection(
                name="gemini-embed",
                provider="gemini",
                api_key_env="GEMINI_API_KEY",
            ),
        ),
        world_model=ProviderModelSelection(connection="anthropic-main", model="world-id"),
        judge=ProviderModelSelection(connection="anthropic-main", model="judge-id"),
        embedder=ProviderModelSelection(connection="gemini-embed", model="embedding-id"),
    )

    configured = configure_provider_catalog(tmp_path / "models.toml", setup)

    assert configured.models["embedder"].connection == "gemini-embed"


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", None),
        ("openrouter", None),
        ("anthropic", None),
        ("gemini", None),
        ("openai-compatible", "https://models.example.test/v1"),
    ],
)
def test_setup_accepts_each_supported_provider_connection(
    provider: str, base_url: str | None
) -> None:
    """The setup contract covers every current provider without choosing one."""
    connection = ProviderConnection(
        name="selected-provider",
        provider=provider,
        api_key_env="SELECTED_PROVIDER_API_KEY",
        base_url=base_url,
    )

    assert connection.provider == provider
    assert connection.base_url == base_url


@pytest.mark.parametrize(
    ("connection", "message"),
    [
        (
            {
                "name": "native",
                "provider": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://example.test/v1",
            },
            "base_url is only accepted",
        ),
        (
            {
                "name": "custom",
                "provider": "openai-compatible",
                "api_key_env": "CUSTOM_API_KEY",
            },
            "requires an explicit base_url",
        ),
    ],
)
def test_connection_endpoint_rules_are_explicit(connection: dict[str, str], message: str) -> None:
    """Native endpoints stay fixed while compatible endpoints name their URL."""
    with pytest.raises(ValueError, match=message):
        ProviderConnection.model_validate(connection)


def test_anthropic_cannot_be_declared_as_the_embedder() -> None:
    """Setup rejects a role the current Anthropic runtime cannot satisfy."""
    connection = ProviderConnection(
        name="anthropic-main",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
    )
    with pytest.raises(ValueError, match="does not expose embeddings"):
        ProviderSetup(
            connections=(connection,),
            world_model=ProviderModelSelection(connection="anthropic-main", model="world-id"),
            judge=ProviderModelSelection(connection="anthropic-main", model="judge-id"),
            embedder=ProviderModelSelection(connection="anthropic-main", model="embed-id"),
        )
