"""Collection-first model catalog setup with atomic, secret-free updates."""

from __future__ import annotations

import hashlib
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
EMPTY_CATALOG_SHA256 = hashlib.sha256(b"").hexdigest()


class ProviderSetupError(ValueError):
    """Provider setup cannot be applied without replacing or losing catalog state."""


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
    """One stable alias and exact provider-side model selected during setup."""

    alias: str = Field(min_length=1, max_length=128)
    connection: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=2_048)
    supports_tools: bool = False
    supports_embeddings: bool = False
    supports_structured_output: bool = False
    context_window_tokens: int | None = Field(default=None, gt=0)
    maximum_output_tokens: int | None = Field(default=None, gt=0)
    input_cost_per_million_tokens_usd: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _require_embedding_price(self) -> ProviderModelSelection:
        """Require explicit input pricing for models selected as embedders.

        Returns:
            The validated model selection.

        Raises:
            ValueError: The model supports embeddings but omits its input price.
        """
        if self.supports_embeddings and self.input_cost_per_million_tokens_usd is None:
            raise ValueError(
                "embedding-capable models require explicit input cost per million tokens; "
                "use 0 for a model with no input charge"
            )
        return self

    def capabilities(self) -> ModelCapabilities:
        """Return the explicit per-model capabilities captured by setup.

        Returns:
            Canonical capability record for the selected model alias.
        """
        return ModelCapabilities(
            supports_tools=self.supports_tools,
            supports_embeddings=self.supports_embeddings,
            supports_structured_output=self.supports_structured_output,
            context_window_tokens=self.context_window_tokens,
            maximum_output_tokens=self.maximum_output_tokens,
            input_cost_per_million_tokens_usd=self.input_cost_per_million_tokens_usd,
        )

    def catalog_record(self) -> ModelRecord:
        """Return the exact catalog record represented by this selection.

        Returns:
            Model-catalog entry that preserves connection, model ID, and capabilities.
        """
        return ModelRecord(
            connection=self.connection,
            model=self.model,
            capabilities=self.capabilities(),
        )


class ProviderSetup(ContractModel):
    """Collected connections, available model aliases, and independent build roles."""

    connections: tuple[ProviderConnection, ...] = ()
    models: tuple[ProviderModelSelection, ...] = ()
    known_existing_connections: tuple[str, ...] = ()
    known_existing_aliases: tuple[str, ...] = ()
    world_model: str = Field(min_length=1, max_length=128)
    judge: str = Field(min_length=1, max_length=128)
    embedder: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def _require_unique_references_and_role_capabilities(self) -> ProviderSetup:
        """Validate collected identities and the three independently selected roles.

        Returns:
            The complete validated provider setup.

        Raises:
            ValueError: Connections, aliases, references, or role capabilities are invalid.
        """
        connection_names = tuple(connection.name for connection in self.connections)
        if len(set(connection_names)) != len(connection_names):
            raise ValueError("provider connection names must be unique")
        aliases = tuple(model.alias for model in self.models)
        if len(set(aliases)) != len(aliases):
            raise ValueError("provider model aliases must be unique")
        known_connections = set(connection_names).union(self.known_existing_connections)
        if not known_connections:
            raise ValueError("provider setup needs at least one available connection")
        if not set(aliases).union(self.known_existing_aliases):
            raise ValueError("provider setup needs at least one available model alias")
        unknown_connections = {
            model.connection for model in self.models if model.connection not in known_connections
        }
        if unknown_connections:
            raise ValueError(
                "model aliases name unknown connections: " + ", ".join(sorted(unknown_connections))
            )
        known_aliases = set(aliases).union(self.known_existing_aliases)
        unknown_roles = {self.world_model, self.judge, self.embedder}.difference(known_aliases)
        if unknown_roles:
            raise ValueError(
                "build roles name unknown model aliases: " + ", ".join(sorted(unknown_roles))
            )
        by_alias = {model.alias: model for model in self.models}
        if self.embedder in by_alias and not by_alias[self.embedder].supports_embeddings:
            raise ValueError(f"embedder alias {self.embedder!r} must declare embedding support")
        return self


def catalog_state_sha256(path: Path) -> str:
    """Return the exact catalog-file digest, or the empty-state digest when absent.

    Args:
        path: Shared model-catalog path.

    Returns:
        SHA-256 digest of the current bytes or the canonical empty state.
    """
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        payload = b""
    return hashlib.sha256(payload).hexdigest()


def configure_provider_catalog(
    path: Path,
    setup: ProviderSetup,
    *,
    replace: bool = False,
    expected_state_sha256: str | None = None,
) -> ModelCatalog:
    """Atomically merge collected connections, model aliases, and build roles.

    Existing unrelated aliases and router candidate assignments are preserved. The optional
    expected digest protects a prompt session from overwriting catalog edits made while answers
    were being collected.

    Args:
        path: Local ``.wmo/models.toml`` path.
        setup: Fully collected connections, model aliases, and role selections.
        replace: Whether conflicting collected entries may be replaced.
        expected_state_sha256: Exact catalog state observed before interactive collection.

    Returns:
        The complete validated catalog written to ``path``.

    Raises:
        ProviderSetupError: Existing state conflicts, changed, or protects a router candidate.
        ModelCatalogError: Existing catalog content is invalid.
    """
    with file_write_lock(path, what="provider model configuration"):
        current_state = catalog_state_sha256(path)
        if expected_state_sha256 is not None and current_state != expected_state_sha256:
            raise ProviderSetupError(
                "models.toml changed while setup was in progress; review the new catalog and retry"
            )
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
    """Merge one collected setup while retaining unrelated models and role assignments.

    Args:
        existing: Existing catalog, or ``None`` for first setup.
        setup: Fully validated collected provider setup.
        replace: Whether unequal unprotected records may be replaced.

    Returns:
        Complete merged catalog with updated build roles.

    Raises:
        ProviderSetupError: A collected record conflicts with preserved or protected state.
    """
    connections = dict(existing.connections) if existing is not None else {}
    models = dict(existing.models) if existing is not None else {}
    roles = existing.roles if existing is not None else ModelRoles()
    proposed_models = {selection.alias: selection.catalog_record() for selection in setup.models}
    protected_aliases = set(roles.candidates)
    protected_aliases.update(
        alias
        for alias in (roles.incumbent, roles.rubric_proposer, roles.teacher)
        if alias is not None
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
            and (alias not in proposed_models or proposed_models[alias] == record)
        )
        if current is not None and current != proposed and preserved_aliases:
            raise ProviderSetupError(
                f"connection {selected.name!r} is used by preserved model aliases "
                f"{', '.join(sorted(preserved_aliases))}; use a new connection name"
            )
        connections[selected.name] = proposed

    for alias, proposed in proposed_models.items():
        current = models.get(alias)
        if current == proposed:
            continue
        if current is not None and current != proposed and not replace:
            raise ProviderSetupError(f"model alias {alias!r} already differs; rerun with --replace")
        if current is not None and current != proposed and alias in protected_aliases:
            raise ProviderSetupError(
                f"model alias {alias!r} is assigned to a router or training role and cannot be "
                "replaced during provider setup"
            )
        models[alias] = proposed

    role_values = roles.model_dump()
    role_values.update(
        world_model=setup.world_model,
        judge=setup.judge,
        embedder=setup.embedder,
    )
    catalog = ModelCatalog(
        schema_version=existing.schema_version if existing is not None else 1,
        connections=connections,
        models=models,
        roles=ModelRoles.model_validate(role_values),
    )
    embedder = catalog.models[catalog.roles.embedder or ""]
    if embedder.capabilities is None or not embedder.capabilities.supports_embeddings:
        raise ProviderSetupError(
            f"embedder alias {catalog.roles.embedder!r} must declare embedding support"
        )
    return catalog
