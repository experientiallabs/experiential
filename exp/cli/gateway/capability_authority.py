"""Read exact prior endpoint capability declarations during CLI reconfiguration."""

from __future__ import annotations

from exp.common.models import ConnectionConfig, ModelCatalog
from exp.runtime.gateway.catalog_authority import authored_snapshot_path
from exp.runtime.gateway.management import GatewayManagement


def retained_streaming_tool_arguments(
    manager: GatewayManagement,
    *,
    alias_id: str,
    connection_id: str,
    connection: ConnectionConfig,
) -> bool | None:
    """Return a prior declaration only for the same frozen endpoint identity.

    The mutable connection name is insufficient authority: an operator may
    replace its endpoint while retaining the name. The active alias revision's
    immutable snapshot and provider binding must both match the candidate
    connection before its endpoint-specific declaration can be reused.
    """
    active = next(
        (item for item in manager.aliases() if item.alias_id == alias_id and item.active),
        None,
    )
    if active is None or active.revision_id is None or active.snapshot_ref is None:
        return None
    bound = next(
        (
            item
            for item in manager.alias_provider_connections(
                alias_id=alias_id,
                alias_revision_id=active.revision_id,
            )
            if item.connection_id == connection_id
        ),
        None,
    )
    if bound is None or bound.config.identity_sha256() != connection.identity_sha256():
        return None
    normalized_snapshot = manager.root / "gateway" / active.snapshot_ref
    snapshot = ModelCatalog.model_validate_json(
        authored_snapshot_path(normalized_snapshot).read_bytes()
    )
    record = snapshot.models.get(alias_id)
    if (
        record is None
        or record.connection != connection_id
        or record.gateway is None
        or snapshot.connections.get(connection_id) != connection
    ):
        return None
    return record.gateway.capabilities.supports_streaming_tool_arguments
