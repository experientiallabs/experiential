"""Role-free provider connection authoring for gateway and other runtime consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from exp.common.core.artifacts import validate_artifact_id
from exp.common.core.files import resolve_write_target, write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.common.models.catalog import (
    MODEL_CATALOG_SCHEMA_VERSION,
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.setup import ProviderConnection


class ProviderConnectionAuthoringError(ValueError):
    """A role-free provider update conflicts with existing catalog state."""


def sync_provider_models(
    path: Path,
    *,
    connection: ProviderConnection,
    models: Mapping[str, ModelRecord],
    protected_connections: Mapping[str, ConnectionConfig] | None = None,
    replace: bool = True,
    on_commit: Callable[[], None] | None = None,
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
        on_commit: Optional final persistence step called with the catalog write lock held.
            It must leave its own state unchanged when raising and must not reacquire the
            catalog lock. Ordinary callback exceptions restore the previous catalog before
            propagating. This recovery does not provide crash atomicity across files.

    Returns:
        Complete catalog after the provider and model update.

    Raises:
        ProviderConnectionAuthoringError: Input is empty, inconsistent, or conflicts with
            protected serving state.
        RuntimeError: The callback failed and the previous catalog could not be restored.
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
        target = resolve_write_target(path)
        existing = load_model_catalog(target) if target.exists() else None
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
            schema_version=(
                existing.schema_version if existing is not None else MODEL_CATALOG_SCHEMA_VERSION
            ),
            connections=current_connections,
            models=current_models,
            gateway_pools=existing.gateway_pools if existing is not None else {},
            roles=existing.roles if existing is not None else ModelRoles(),
        )
        previous_bytes = target.read_bytes() if on_commit is not None and target.exists() else None
        write_model_catalog(target, catalog)
        if on_commit is not None:
            try:
                on_commit()
            except Exception:
                try:
                    if previous_bytes is None:
                        target.unlink()
                    else:
                        write_bytes_atomic(target, previous_bytes)
                except OSError:
                    raise RuntimeError(
                        "Provider model commit failed and the previous catalog could not be "
                        "restored. Check models.toml and credential configuration before retrying."
                    ) from None
                raise
        return catalog
