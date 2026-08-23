"""Role-free provider connection authoring for gateway and other runtime consumers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from exp.common.core.artifacts import validate_artifact_id
from exp.common.core.locks import file_write_lock
from exp.common.models.catalog import (
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.setup import ProviderConnection, catalog_state_sha256


class ProviderConnectionAuthoringError(ValueError):
    """A role-free provider update conflicts with existing catalog state."""


def configure_provider_connections(
    path: Path,
    connections: tuple[ProviderConnection, ...],
    *,
    replace: bool = False,
    expected_state_sha256: str | None = None,
) -> ModelCatalog:
    """Atomically add provider connections without assigning optimizer roles.

    This authoring path is for runtime consumers such as the gateway. The existing
    ``ProviderSetup`` path remains responsible for build-role validation and keeps requiring a
    world model, judge, and embedding-capable embedder.

    Args:
        path: Local ``.exp/models.toml`` path.
        connections: Named, validated provider connections to add.
        replace: Whether an unused conflicting connection may be replaced.
        expected_state_sha256: Exact catalog state observed before collecting input.

    Returns:
        Complete validated catalog after the connection update.

    Raises:
        ProviderConnectionAuthoringError: Input repeats, state changed, or replacement is unsafe.
        ModelCatalogError: Existing catalog content is invalid.
    """
    names = tuple(connection.name for connection in connections)
    if not names:
        raise ProviderConnectionAuthoringError("at least one provider connection is required")
    if len(set(names)) != len(names):
        raise ProviderConnectionAuthoringError("provider connection names must be unique")
    with file_write_lock(path, what="provider connection configuration"):
        current_state = catalog_state_sha256(path)
        if expected_state_sha256 is not None and current_state != expected_state_sha256:
            raise ProviderConnectionAuthoringError(
                "models.toml changed while provider input was collected; review and retry"
            )
        existing = load_model_catalog(path) if path.exists() else None
        current_connections = dict(existing.connections) if existing is not None else {}
        models = dict(existing.models) if existing is not None else {}
        for selected in connections:
            proposed = selected.catalog_config()
            current = current_connections.get(selected.name)
            if current is not None and current != proposed and not replace:
                raise ProviderConnectionAuthoringError(
                    f"connection {selected.name!r} already differs; rerun with --replace"
                )
            dependent_aliases = tuple(
                alias
                for alias, record in models.items()
                if record.connection == selected.name and current != proposed
            )
            if dependent_aliases:
                raise ProviderConnectionAuthoringError(
                    f"connection {selected.name!r} is used by model aliases "
                    f"{', '.join(sorted(dependent_aliases))}; use a new connection name"
                )
            current_connections[selected.name] = proposed
        catalog = ModelCatalog(
            schema_version=existing.schema_version if existing is not None else 2,
            connections=current_connections,
            models=models,
            roles=existing.roles if existing is not None else ModelRoles(),
        )
        write_model_catalog(path, catalog)
        return catalog


def sync_provider_models(
    path: Path,
    *,
    connection: ProviderConnection,
    models: Mapping[str, ModelRecord],
    protected_connections: Mapping[str, ConnectionConfig] | None = None,
    replace: bool = True,
) -> ModelCatalog:
    """Atomically register one provider and its authenticated model identities.

    This role-free path is used by account login synchronization. It keeps every discovered
    model visible without forcing the optimizer's world-model, judge, or embedder roles.

    Args:
        path: Local ``models.toml`` path.
        connection: Secret-free provider connection to register.
        models: Secret-free records keyed by their local catalog aliases.
        protected_connections: Active SQLite gateway connections keyed by connection name. A
            changed endpoint cannot replace one of these authorities during account sync.
        replace: Whether changed non-serving model metadata may be refreshed.

    Returns:
        Complete catalog after the provider and model update.

    Raises:
        ProviderConnectionAuthoringError: Input is empty, inconsistent, or conflicts with
            protected serving state.
    """
    if not models:
        raise ProviderConnectionAuthoringError("provider model sync needs at least one model")
    aliases = tuple(models)
    try:
        for alias in aliases:
            validate_artifact_id(alias)
    except ValueError as exc:
        raise ProviderConnectionAuthoringError(str(exc)) from exc
    if any(record.connection != connection.name for record in models.values()):
        raise ProviderConnectionAuthoringError(
            "provider model records must reference the synchronized connection"
        )
    with file_write_lock(path, what="provider model synchronization"):
        existing = load_model_catalog(path) if path.exists() else None
        current_connections = dict(existing.connections) if existing is not None else {}
        current_models = dict(existing.models) if existing is not None else {}
        current = current_connections.get(connection.name)
        proposed_connection = connection.catalog_config()
        protected = (protected_connections or {}).get(connection.name)
        if protected is not None and protected != proposed_connection:
            raise ProviderConnectionAuthoringError(
                f"connection {connection.name!r} differs from active gateway authority; use the "
                "existing endpoint or explicitly reconfigure the gateway"
            )
        if current is not None and current != proposed_connection:
            protected_aliases = tuple(
                alias
                for alias, record in current_models.items()
                if record.connection == connection.name
                and (record.gateway is not None or protected is not None)
            )
            if protected_aliases:
                raise ProviderConnectionAuthoringError(
                    f"connection {connection.name!r} differs from active gateway deployments "
                    f"{', '.join(sorted(protected_aliases))}; use the existing endpoint or "
                    "explicitly reconfigure the gateway"
                )
            if not replace:
                raise ProviderConnectionAuthoringError(
                    f"connection {connection.name!r} already differs; rerun with replacement"
                )
        current_connections[connection.name] = proposed_connection
        for alias, proposed in models.items():
            previous = current_models.get(alias)
            if previous == proposed:
                continue
            if previous is not None and previous.gateway is not None:
                # A gateway deployment owns its immutable serving record. The account sync still
                # keeps the identity visible, but must not erase the active deployment metadata.
                continue
            if previous is not None and not replace:
                raise ProviderConnectionAuthoringError(
                    f"model alias {alias!r} already differs; rerun with replacement"
                )
            current_models[alias] = proposed
        catalog = ModelCatalog(
            schema_version=existing.schema_version if existing is not None else 2,
            connections=current_connections,
            models=current_models,
            gateway_pools=existing.gateway_pools if existing is not None else {},
            roles=existing.roles if existing is not None else ModelRoles(),
        )
        write_model_catalog(path, catalog)
        return catalog
