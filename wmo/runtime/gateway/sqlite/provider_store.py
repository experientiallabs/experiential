"""SQLite store facade for versioned serving provider connections."""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from typing import Protocol, cast

from wmo.common.models import ConnectionConfig
from wmo.runtime.gateway.auth import utc_text
from wmo.runtime.gateway.interfaces import GatewayClock
from wmo.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionAuthority,
    active_provider_connections,
    bound_provider_connections,
    disable_provider_connection,
    upsert_provider_connection,
)


class _ProviderStore(Protocol):
    """Structural view of store facilities used by the provider facade."""

    _clock: GatewayClock

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
