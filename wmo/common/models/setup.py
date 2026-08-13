"""Provider-first model catalog setup with atomic, secret-free updates."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from wmo.common.core.artifacts import ContractModel
from wmo.common.core.locks import file_write_lock
from wmo.common.models.catalog import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.models.model import ModelCapabilities

SETUP_PROVIDERS = frozenset({"anthropic", "gemini", "openai", "openai-compatible", "openrouter"})
_BUILD_ROLE_ALIASES = frozenset({"world-model", "judge", "embedder"})


class ProviderSetupError(ValueError):
    """Provider setup cannot be applied without replacing existing catalog state."""


class ProviderConnection(ContractModel):
    """One named, secret-free provider connection collected during setup."""

    name: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=128)
    api_key_env: str = Field(min_length=1, max_length=256)
    base_url: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def _require_supported_connection_shape(self) -> ProviderConnection:
        if self.provider not in SETUP_PROVIDERS:
            choices = ", ".join(sorted(SETUP_PROVIDERS))
            raise ValueError(f"provider must be one of: {choices}")
        if self.provider == "openai-compatible" and self.base_url is None:
            raise ValueError("openai-compatible requires an explicit base_url")
        if self.provider != "openai-compatible" and self.base_url is not None:
            raise ValueError(
                "base_url is only accepted for provider='openai-compatible'; native providers "
                "use their official endpoint"
            )
        ConnectionConfig(
            provider=self.provider,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
        )
        return self

    def catalog_config(self) -> ConnectionConfig:
        """Return the validated catalog connection represented by this input."""
        return ConnectionConfig(
            provider=self.provider,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
        )


class ProviderModelSelection(ContractModel):
    """One exact provider-side model selected for a build-time role."""

    connection: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=2_048)
    supports_tools: bool = False
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)

    def capabilities(self, *, embeddings: bool = False) -> ModelCapabilities:
        """Return the explicit capabilities captured by provider setup."""
        return ModelCapabilities(
            supports_tools=self.supports_tools,
            supports_embeddings=embeddings,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=self.maximum_output_tokens,
        )


class ProviderSetup(ContractModel):
    """Complete provider connections and exact build-time role selections."""

    connections: tuple[ProviderConnection, ...] = Field(min_length=1)
    world_model: ProviderModelSelection
    judge: ProviderModelSelection
    embedder: ProviderModelSelection

    @model_validator(mode="after")
    def _require_unique_referenced_connections(self) -> ProviderSetup:
        names = tuple(connection.name for connection in self.connections)
        if len(set(names)) != len(names):
            raise ValueError("provider connection names must be unique")
        unknown = {
            selection.connection
            for selection in (self.world_model, self.judge, self.embedder)
            if selection.connection not in names
        }
        if unknown:
            raise ValueError(
                f"role selections name unknown connections: {', '.join(sorted(unknown))}"
            )
        providers = {connection.name: connection.provider for connection in self.connections}
        if providers[self.embedder.connection] == "anthropic":
            raise ValueError(
                "anthropic does not expose embeddings through the current runtime; "
                "select an OpenAI, OpenRouter, Gemini, or OpenAI-compatible embedder connection "
                "with --embedder-provider, --embedder-connection, and --embedder-api-key-env"
            )
        return self


def configure_provider_catalog(
    path: Path,
    setup: ProviderSetup,
    *,
    replace: bool = False,
) -> ModelCatalog:
    """Atomically add connections and assign build-time roles in ``models.toml``.

    Existing connections, model aliases, and router candidate roles are preserved. Fixed role
    aliases are replaced only when ``replace`` is explicit. The function never reads credential
    values and stores only their environment-variable names.

    Args:
        path: Local ``.wmo/models.toml`` path.
        setup: Fully collected connections and exact role model IDs.
        replace: Whether conflicting connection or fixed role aliases may be replaced.

    Returns:
        The complete validated catalog written to ``path``.

    Raises:
        ProviderSetupError: Existing state conflicts and replacement was not authorized.
        ModelCatalogError: Existing catalog content is invalid.
    """
    with file_write_lock(path, what="provider model configuration"):
        existing = load_model_catalog(path) if path.exists() else None
        catalog = _merge_provider_setup(existing, setup, replace=replace)
        write_model_catalog(path, catalog)
        return catalog


def _merge_provider_setup(
    existing: ModelCatalog | None,
    setup: ProviderSetup,
    *,
    replace: bool,
) -> ModelCatalog:
    """Merge one setup into existing catalog state without changing router roles."""
    connections = dict(existing.connections) if existing is not None else {}
    models = dict(existing.models) if existing is not None else {}
    roles = existing.roles if existing is not None else ModelRoles()
    preserved_role_aliases = set(roles.candidates)
    preserved_role_aliases.update(
        role_alias
        for role_alias in (
            roles.incumbent,
            roles.rubric_proposer,
            roles.teacher,
        )
        if role_alias is not None
    )

    for selected in setup.connections:
        proposed = selected.catalog_config()
        current = connections.get(selected.name)
        if current is not None and current != proposed and not replace:
            raise ProviderSetupError(
                f"connection {selected.name!r} already differs; rerun with --replace"
            )
        preserved_aliases = tuple(
            alias
            for alias, record in models.items()
            if record.connection == selected.name
            and (alias not in _BUILD_ROLE_ALIASES or alias in preserved_role_aliases)
        )
        if current is not None and current != proposed and preserved_aliases:
            raise ProviderSetupError(
                f"connection {selected.name!r} is used by preserved model aliases "
                f"{', '.join(sorted(preserved_aliases))}; use a new connection name"
            )
        connections[selected.name] = proposed

    selections = {
        "world-model": (setup.world_model, False),
        "judge": (setup.judge, False),
        "embedder": (setup.embedder, True),
    }
    for alias, (selection, embeddings) in selections.items():
        proposed = ModelRecord(
            connection=selection.connection,
            model=selection.model,
            capabilities=selection.capabilities(embeddings=embeddings),
        )
        current = models.get(alias)
        if current is not None and current != proposed and not replace:
            raise ProviderSetupError(f"model alias {alias!r} already differs; rerun with --replace")
        if current is not None and current != proposed and alias in preserved_role_aliases:
            raise ProviderSetupError(
                f"model alias {alias!r} is assigned to a preserved router or training role; "
                "rename that alias before provider setup"
            )
        models[alias] = proposed

    role_values = roles.model_dump()
    role_values.update(world_model="world-model", judge="judge", embedder="embedder")
    updated_roles = ModelRoles.model_validate(role_values)
    return ModelCatalog(
        schema_version=existing.schema_version if existing is not None else 1,
        connections=connections,
        models=models,
        roles=updated_roles,
    )
