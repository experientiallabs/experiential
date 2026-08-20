"""Role-free provider connection authoring for gateway and other runtime consumers."""

from __future__ import annotations

from pathlib import Path

from exp.common.core.locks import file_write_lock
from exp.common.models.catalog import (
    ModelCatalog,
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
