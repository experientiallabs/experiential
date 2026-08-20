"""SQLite store facade for versioned serving provider connections."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from typing import Protocol, cast

from exp.common.models import ConnectionConfig
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionAuthority,
    ProviderConnectionBinding,
    active_provider_connections,
    bind_alias_provider_connections,
    bound_provider_connections,
    disable_provider_connection,
    upsert_provider_connection,
)


class _ProviderStore(Protocol):
    """Structural view of store facilities used by the provider facade."""

    _clock: GatewayClock
    _store_error: type[ValueError]

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        """Open one configured SQLite connection."""
        ...

    def _transaction(self) -> AbstractContextManager[sqlite3.Connection]:
        """Open one immediate SQLite transaction."""
        ...


class ProviderConnectionStoreMixin:
    """Expose provider connection authority through the gateway store facade."""

    def upsert_provider_connection(
        self,
        *,
        organization_id: str,
        connection_id: str,
        revision_id: str,
        config: ConnectionConfig,
        replace: bool = False,
    ) -> tuple[bool, ProviderConnectionAuthority]:
        """Create or explicitly revise one secret-free serving connection."""
        store = cast(_ProviderStore, self)
        with store._transaction() as connection:
            return upsert_provider_connection(
                connection,
                organization_id=organization_id,
                connection_id=connection_id,
                revision_id=revision_id,
                config=config,
                replace=replace,
                now=utc_text(store._clock.now()),
            )

    def provider_connections(
        self,
        *,
        organization_id: str,
    ) -> tuple[ProviderConnectionAuthority, ...]:
        """Return active provider connections from SQLite authority."""
        store = cast(_ProviderStore, self)
        with store._connect() as connection:
            return active_provider_connections(
                connection,
                organization_id=organization_id,
            )

    def alias_provider_connections(
        self,
        *,
        organization_id: str,
        alias_id: str,
        alias_revision_id: str,
    ) -> tuple[ProviderConnectionAuthority, ...]:
        """Return exact provider revisions frozen into one alias revision."""
        store = cast(_ProviderStore, self)
        with store._connect() as connection:
            return bound_provider_connections(
                connection,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_revision_id=alias_revision_id,
            )

    def bind_existing_alias_provider_connections(
        self,
        *,
        organization_id: str,
        alias_id: str,
        alias_revision_id: str,
        provider_connections: tuple[ProviderConnectionBinding, ...],
    ) -> bool:
        """Migrate one legacy alias revision to explicit provider bindings."""
        store = cast(_ProviderStore, self)
        with store._transaction() as connection:
            existing = bound_provider_connections(
                connection,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_revision_id=alias_revision_id,
            )
            if existing:
                expected = tuple(
                    (
                        item.connection_id,
                        item.connection_revision_id,
                        item.connection_sha256,
                    )
                    for item in provider_connections
                )
                actual = tuple(
                    (item.connection_id, item.revision_id, item.connection_sha256)
                    for item in existing
                )
                if actual != expected:
                    raise store._store_error(
                        "legacy alias provider bindings differ from SQLite authority"
                    )
                return False
            revision = connection.execute(
                """
                SELECT 1 FROM alias_revisions
                WHERE organization_id = ? AND alias_id = ? AND revision_id = ?
                """,
                (organization_id, alias_id, alias_revision_id),
            ).fetchone()
            if revision is None:
                raise store._store_error("legacy alias revision is not registered")
            bind_alias_provider_connections(
                connection,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_revision_id=alias_revision_id,
                bindings=provider_connections,
                now=utc_text(store._clock.now()),
            )
            return True

    def disable_provider_connection(
        self,
        *,
        organization_id: str,
        connection_id: str,
    ) -> bool:
        """Disable an unreferenced serving provider connection."""
        store = cast(_ProviderStore, self)
        with store._transaction() as connection:
            return disable_provider_connection(
                connection,
                organization_id=organization_id,
                connection_id=connection_id,
                now=utc_text(store._clock.now()),
            )
