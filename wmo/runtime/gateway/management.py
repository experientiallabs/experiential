"""Content-free management and status reads for the local gateway authority."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from wmo.common.core.artifacts import ContractModel, stable_id
from wmo.common.models import ConnectionConfig, ModelCatalog, load_model_catalog
from wmo.runtime.gateway.auth import IssuedVirtualKey, utc_text
from wmo.runtime.gateway.contracts import DirectTarget, ProjectTarget
from wmo.runtime.gateway.sqlite import key_delivery
from wmo.runtime.gateway.sqlite.migrations import connect_database
from wmo.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionAuthority,
    ProviderConnectionBinding,
)
from wmo.runtime.gateway.sqlite.store import GatewayStoreError, SQLiteGatewayStore


class GatewayIdentityView(ContractModel):
    """One content-free local identity."""

    identity_id: str
    display_name: str
    description: str | None = None
    active: bool


class GatewayKeyView(ContractModel):
    """One virtual-key record without secret material or fingerprints."""

    key_id: str
    identity_id: str
    prefix: str
    active: bool
    expires_at: datetime | None = None
    created_at: datetime
    last_used_at: datetime | None = None


class GatewayAliasView(ContractModel):
    """One public alias and its active immutable revision."""

    alias_id: str
    alias_name: str
    active: bool
    revision_id: str | None = None
    target_kind: str | None = None
    pool_id: str | None = None
    project_ref: str | None = None
    activation_ref: str | None = None
    snapshot_ref: str | None = None
    catalog_sha256: str | None = None
    refusal_failover: bool = False


class GatewayGrantView(ContractModel):
    """One deny-by-default identity-to-alias grant."""

    identity_id: str
    alias_id: str
    alias_name: str


class GatewayStatus(ContractModel):
    """Content-free summary of initialized local gateway state."""

    initialized: bool
    organization_id: str
    active_identities: int = Field(default=0, ge=0)
    active_keys: int = Field(default=0, ge=0)
    active_aliases: int = Field(default=0, ge=0)
    active_provider_connections: int = Field(default=0, ge=0)
    grants: int = Field(default=0, ge=0)


class GatewayManagement:
    """Own additive management operations around one SQLite gateway store."""

    def __init__(self, root: Path, *, organization_id: str = "local") -> None:
        """Bind one explicit local organization under a WMO root.

        Args:
            root: WMO root containing private gateway state.
            organization_id: Stable local organization identifier.
        """
        self.root = root
        self.organization_id = organization_id
        self.state_dir = root / "gateway"
        self.database_path = self.state_dir / "gateway.db"

    @property
    def initialized(self) -> bool:
        """Return whether the gateway database already exists."""
        return self.database_path.is_file()

    def initialize(self, *, display_name: str = "Local") -> GatewayStatus:
        """Create the explicit local organization without identities, keys, or aliases.

        Args:
            display_name: Operator-facing organization name.

        Returns:
            Current content-free gateway status.
        """
        store = self.store()
        connection = connect_database(self.database_path)
        try:
            row = connection.execute(
                "SELECT display_name FROM organizations WHERE organization_id = ?",
                (self.organization_id,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            store.create_organization(
                organization_id=self.organization_id,
                slug=self.organization_id,
                display_name=display_name,
            )
        elif str(row["display_name"]) != display_name:
            raise GatewayStoreError("gateway organization already exists with another display name")
        self.migrate_legacy_provider_connections()
        return self.status()

    def store(self) -> SQLiteGatewayStore:
        """Open the authoritative SQLite store and private pepper state."""
        return SQLiteGatewayStore(self.database_path)

    def require_initialized(self) -> SQLiteGatewayStore:
        """Return the store or fail with an actionable initialization command."""
        if not self.initialized:
            raise GatewayStoreError(
                "gateway is not initialized; run 'wmo config gateway init' first"
            )
        return self.store()

    def upsert_provider_connection(
        self,
        *,
        connection_id: str,
        config: ConnectionConfig,
        replace: bool = False,
    ) -> tuple[bool, ProviderConnectionAuthority]:
        """Create or revise one SQLite-authoritative serving connection."""
        revision_id = stable_id(
            "provider-connection-revision",
            {
                "connection_id": connection_id,
                "config": config.model_dump(mode="json", exclude_none=False),
            },
        )
        return self.require_initialized().upsert_provider_connection(
            organization_id=self.organization_id,
            connection_id=connection_id,
            revision_id=revision_id,
            config=config,
            replace=replace,
        )

    def provider_connections(self) -> tuple[ProviderConnectionAuthority, ...]:
        """Return active SQLite-authoritative serving connections."""
        if not self.initialized:
            return ()
        return self.require_initialized().provider_connections(
            organization_id=self.organization_id,
        )

    def disable_provider_connection(self, *, connection_id: str) -> bool:
        """Disable one provider connection not used by an active alias revision."""
        return self.require_initialized().disable_provider_connection(
            organization_id=self.organization_id,
            connection_id=connection_id,
        )

    def migrate_legacy_provider_connections(self) -> int:
        """Import a legacy models.toml only when SQLite has no serving connections.

        Existing equal records are stable replays. Existing different records fail closed so a
        catalog file can never silently override current serving authority.
        """
        if self.provider_connections():
            return 0
        path = self.root / "models.toml"
        if not path.is_file():
            return 0
        imported = 0
        for connection_id, config in sorted(load_model_catalog(path).connections.items()):
            changed, _authority = self.upsert_provider_connection(
                connection_id=connection_id,
                config=config,
                replace=False,
            )
            imported += int(changed)
        return imported

    def provider_bindings(self, catalog: ModelCatalog) -> tuple[ProviderConnectionBinding, ...]:
        """Resolve an authored snapshot to exact active SQLite connection revisions."""
        authorities = {item.connection_id: item for item in self.provider_connections()}
        bindings: list[ProviderConnectionBinding] = []
        for connection_id, config in sorted(catalog.connections.items()):
            authority = authorities.get(connection_id)
            if authority is None or authority.config != config:
                raise GatewayStoreError(
                    f"provider connection {connection_id!r} differs from SQLite authority"
                )
            bindings.append(
                ProviderConnectionBinding(
                    connection_id=connection_id,
                    connection_revision_id=authority.revision_id,
                    connection_sha256=authority.connection_sha256,
                )
            )
        return tuple(bindings)

    def ensure_alias_provider_bindings(
        self,
        *,
        alias_id: str,
        alias_revision_id: str,
        catalog: ModelCatalog,
    ) -> tuple[ProviderConnectionAuthority, ...]:
        """Migrate and return exact provider revisions for one legacy alias."""
        store = self.require_initialized()
        existing = store.alias_provider_connections(
            organization_id=self.organization_id,
            alias_id=alias_id,
            alias_revision_id=alias_revision_id,
        )
        if existing:
            return existing
        store.bind_existing_alias_provider_connections(
            organization_id=self.organization_id,
            alias_id=alias_id,
            alias_revision_id=alias_revision_id,
            provider_connections=self.provider_bindings(catalog),
        )
        return store.alias_provider_connections(
            organization_id=self.organization_id,
            alias_id=alias_id,
            alias_revision_id=alias_revision_id,
        )

    def create_identity(
        self,
        *,
        identity_id: str,
        display_name: str,
        description: str | None = None,
        operation_id: str | None = None,
    ) -> str:
        """Create one retry-safe identity.

        Args:
            identity_id: Stable identity identifier.
            display_name: Operator-facing name.
            description: Optional content-free description.
            operation_id: Optional idempotent mutation identifier.

        Returns:
            Created or replayed identity identifier.
        """
        return self.require_initialized().create_identity(
            organization_id=self.organization_id,
            identity_id=identity_id,
            display_name=display_name,
            description=description,
            operation_id=operation_id,
        )

    def update_identity(
        self,
        *,
        identity_id: str,
        display_name: str,
        description: str | None = None,
    ) -> bool:
        """Update display-only identity metadata without changing authority.

        Args:
            identity_id: Stable identity identifier.
            display_name: Replacement operator-facing name.
            description: Optional replacement description.

        Returns:
            Whether the identity exists.
        """
        self.require_initialized()
        connection = connect_database(self.database_path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                """
                UPDATE identities SET display_name = ?, description = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now')
                WHERE organization_id = ? AND identity_id = ?
                """,
                (display_name, description, self.organization_id, identity_id),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return result.rowcount == 1

    def disable_identity(self, *, identity_id: str) -> bool:
        """Disable one identity and every key attached to it."""
        return self.require_initialized().disable_identity(
            organization_id=self.organization_id,
            identity_id=identity_id,
        )

    def identities(self) -> tuple[GatewayIdentityView, ...]:
        """Return identities in stable identifier order."""
        if not self.initialized:
            return ()
        rows = self._rows(
            """
            SELECT identity_id, display_name, description, active
            FROM identities WHERE organization_id = ? ORDER BY identity_id
            """
        )
        return tuple(
            GatewayIdentityView(
                identity_id=str(row["identity_id"]),
                display_name=str(row["display_name"]),
                description=None if row["description"] is None else str(row["description"]),
                active=bool(row["active"]),
            )
            for row in rows
        )

    def issue_key(
        self,
        *,
        identity_id: str,
        key_id: str,
        expires_at: datetime | None = None,
        operation_id: str | None = None,
        secret_delivery: key_delivery.KeyDeliverySink | None = None,
    ) -> IssuedVirtualKey:
        """Issue one virtual key and optionally deliver it before durable commit."""
        return self.require_initialized().issue_virtual_key(
            organization_id=self.organization_id,
            identity_id=identity_id,
            key_id=key_id,
            expires_at=expires_at,
            operation_id=operation_id,
            secret_delivery=secret_delivery,
        )

    def revoke_key(self, *, key_id: str) -> bool:
        """Revoke one virtual key idempotently."""
        return self.require_initialized().revoke_virtual_key(
            organization_id=self.organization_id,
            key_id=key_id,
        )

    def keys(self, *, identity_id: str | None = None) -> tuple[GatewayKeyView, ...]:
        """Return key metadata without fingerprints or raw material."""
        if not self.initialized:
            return ()
        predicate = "organization_id = ?"
        parameters: tuple[str, ...] = (self.organization_id,)
        if identity_id is not None:
            predicate += " AND identity_id = ?"
            parameters = (self.organization_id, identity_id)
        rows = self._rows(
            f"""
            SELECT key_id, identity_id, prefix, expires_at, revoked_at,
                   created_at, last_used_at
            FROM virtual_keys WHERE {predicate} ORDER BY key_id
            """,
            parameters,
        )
        now = datetime.now().astimezone()
        views: list[GatewayKeyView] = []
        for row in rows:
            expires_at = _datetime(row["expires_at"])
            views.append(
                GatewayKeyView(
                    key_id=str(row["key_id"]),
                    identity_id=str(row["identity_id"]),
                    prefix=str(row["prefix"]),
                    active=row["revoked_at"] is None and (expires_at is None or expires_at > now),
                    expires_at=expires_at,
                    created_at=_required_datetime(row["created_at"]),
                    last_used_at=_datetime(row["last_used_at"]),
                )
            )
        return tuple(views)

    def aliases(self) -> tuple[GatewayAliasView, ...]:
        """Return public aliases and their active revision targets."""
        if not self.initialized:
            return ()
        rows = self._rows(
            """
            SELECT a.alias_id, a.alias_name, a.active, a.active_revision_id,
                   r.target_kind, r.pool_id, r.project_ref, r.activation_ref,
                   r.snapshot_ref, r.catalog_sha256, r.refusal_failover
            FROM gateway_aliases AS a
            LEFT JOIN alias_revisions AS r
              ON r.organization_id = a.organization_id
             AND r.revision_id = a.active_revision_id
            WHERE a.organization_id = ? ORDER BY a.alias_name
            """
        )
        return tuple(
            GatewayAliasView(
                alias_id=str(row["alias_id"]),
                alias_name=str(row["alias_name"]),
                active=bool(row["active"]),
                revision_id=(
                    None if row["active_revision_id"] is None else str(row["active_revision_id"])
                ),
                target_kind=None if row["target_kind"] is None else str(row["target_kind"]),
                pool_id=None if row["pool_id"] is None else str(row["pool_id"]),
                project_ref=None if row["project_ref"] is None else str(row["project_ref"]),
                activation_ref=(
                    None if row["activation_ref"] is None else str(row["activation_ref"])
                ),
                snapshot_ref=(None if row["snapshot_ref"] is None else str(row["snapshot_ref"])),
                catalog_sha256=(
                    None if row["catalog_sha256"] is None else str(row["catalog_sha256"])
                ),
                refusal_failover=bool(row["refusal_failover"]),
            )
            for row in rows
        )

    def activate_direct_alias(
        self,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        pool_id: str,
        snapshot_ref: str,
        catalog_sha256: str,
        provider_connections: tuple[ProviderConnectionBinding, ...] = (),
        refusal_failover: bool = False,
    ) -> bool:
        """Activate one singleton direct alias against an immutable catalog snapshot."""
        return self._activate_alias(
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=DirectTarget(pool_id=pool_id),
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            refusal_failover=refusal_failover,
        )

    def preflight_direct_alias_activation(
        self,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        pool_id: str,
        snapshot_ref: str,
        catalog_sha256: str,
        refusal_failover: bool = False,
    ) -> bool:
        """Validate direct activation invariants without changing SQLite authority.

        Args:
            alias_id: Stable alias resource identifier.
            alias_name: Public model name.
            revision_id: Immutable revision identifier.
            pool_id: Direct target pool identifier.
            snapshot_ref: Content-addressed catalog snapshot reference.
            catalog_sha256: Exact normalized catalog digest.
            refusal_failover: Whether typed precommit refusals may advance.

        Returns:
            Whether activation would create a new immutable revision.
        """
        changed, _snapshot_registered = self._alias_activation_preflight(
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=DirectTarget(pool_id=pool_id),
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            refusal_failover=refusal_failover,
        )
        return changed

    def activate_project_alias(
        self,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        project_ref: str,
        activation_ref: str,
        snapshot_ref: str,
        catalog_sha256: str,
        provider_connections: tuple[ProviderConnectionBinding, ...] = (),
        refusal_failover: bool = False,
    ) -> bool:
        """Activate one verified frozen project as exactly one public alias."""
        return self._activate_alias(
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=ProjectTarget(
                project_ref=project_ref,
                activation_ref=activation_ref,
                catalog_sha256=catalog_sha256,
            ),
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            refusal_failover=refusal_failover,
        )

    def disable_alias(self, *, alias_id: str) -> bool:
        """Disable one public alias and release its project binding."""
        return self.require_initialized().disable_alias(
            organization_id=self.organization_id,
            alias_id=alias_id,
        )

    def grants(self, *, identity_id: str | None = None) -> tuple[GatewayGrantView, ...]:
        """Return grants without exposing keys or fingerprints."""
        if not self.initialized:
            return ()
        predicate = "g.organization_id = ?"
        parameters: tuple[str, ...] = (self.organization_id,)
        if identity_id is not None:
            predicate += " AND g.identity_id = ?"
            parameters = (self.organization_id, identity_id)
        rows = self._rows(
            f"""
            SELECT g.identity_id, g.alias_id, a.alias_name
            FROM identity_alias_grants AS g
            JOIN gateway_aliases AS a
              ON a.organization_id = g.organization_id AND a.alias_id = g.alias_id
            WHERE {predicate} ORDER BY g.identity_id, a.alias_name
            """,
            parameters,
        )
        return tuple(
            GatewayGrantView(
                identity_id=str(row["identity_id"]),
                alias_id=str(row["alias_id"]),
                alias_name=str(row["alias_name"]),
            )
            for row in rows
        )

    def add_grant(self, *, identity_id: str, alias_id: str) -> bool:
        """Grant one identity access to one alias."""
        return self.require_initialized().grant_alias(
            organization_id=self.organization_id,
            identity_id=identity_id,
            alias_id=alias_id,
        )

    def remove_grant(self, *, identity_id: str, alias_id: str) -> bool:
        """Remove one identity-to-alias grant idempotently."""
        return self.require_initialized().revoke_alias_grant(
            organization_id=self.organization_id,
            identity_id=identity_id,
            alias_id=alias_id,
        )

    def status(self) -> GatewayStatus:
        """Return content-free resource counts without creating state."""
        if not self.initialized:
            return GatewayStatus(initialized=False, organization_id=self.organization_id)
        connection = connect_database(self.database_path)
        try:
            organization = connection.execute(
                "SELECT 1 FROM organizations WHERE organization_id = ? AND active = 1",
                (self.organization_id,),
            ).fetchone()
            counts = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM identities
                     WHERE organization_id = ? AND active = 1),
                    (SELECT COUNT(*) FROM virtual_keys
                     WHERE organization_id = ? AND revoked_at IS NULL
                       AND (expires_at IS NULL OR expires_at > ?)),
                    (SELECT COUNT(*) FROM gateway_aliases
                     WHERE organization_id = ? AND active = 1),
                    (SELECT COUNT(*) FROM provider_connections
                     WHERE organization_id = ? AND active = 1),
                    (SELECT COUNT(*) FROM identity_alias_grants
                     WHERE organization_id = ?)
                """,
                (
                    self.organization_id,
                    self.organization_id,
                    utc_text(datetime.now(UTC)),
                    self.organization_id,
                    self.organization_id,
                    self.organization_id,
                ),
            ).fetchone()
        finally:
            connection.close()
        return GatewayStatus(
            initialized=organization is not None,
            organization_id=self.organization_id,
            active_identities=int(counts[0]),
            active_keys=int(counts[1]),
            active_aliases=int(counts[2]),
            active_provider_connections=int(counts[3]),
            grants=int(counts[4]),
        )

    def _activate_alias(
        self,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        target: DirectTarget | ProjectTarget,
        snapshot_ref: str,
        catalog_sha256: str,
        provider_connections: tuple[ProviderConnectionBinding, ...],
        refusal_failover: bool,
    ) -> bool:
        """Register one snapshot and activate an idempotent immutable alias revision."""
        store = self.require_initialized()
        changed, snapshot_registered = self._alias_activation_preflight(
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=target,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            refusal_failover=refusal_failover,
        )
        if not changed:
            return False
        if not snapshot_registered:
            store.register_catalog_snapshot(
                organization_id=self.organization_id,
                snapshot_ref=snapshot_ref,
                catalog_sha256=catalog_sha256,
            )
        store.activate_alias_revision(
            organization_id=self.organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=target,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            refusal_failover=refusal_failover,
        )
        return True

    def _alias_activation_preflight(
        self,
        *,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        target: DirectTarget | ProjectTarget,
        snapshot_ref: str,
        catalog_sha256: str,
        refusal_failover: bool,
    ) -> tuple[bool, bool]:
        """Return activation and snapshot change status after read-only validation."""
        self.require_initialized()
        connection = connect_database(self.database_path)
        try:
            existing = connection.execute(
                """
                SELECT r.organization_id, r.alias_id, a.alias_name,
                       r.target_kind, r.pool_id, r.project_ref,
                       r.activation_ref, r.snapshot_ref, r.catalog_sha256,
                       r.refusal_failover
                FROM alias_revisions AS r
                JOIN gateway_aliases AS a
                  ON a.organization_id = r.organization_id AND a.alias_id = r.alias_id
                WHERE r.revision_id = ?
                """,
                (revision_id,),
            ).fetchone()
            alias_by_id = connection.execute(
                """
                SELECT organization_id, alias_name FROM gateway_aliases
                WHERE alias_id = ?
                """,
                (alias_id,),
            ).fetchone()
            alias_by_name = connection.execute(
                """
                SELECT alias_id FROM gateway_aliases
                WHERE organization_id = ? AND alias_name = ?
                """,
                (self.organization_id, alias_name),
            ).fetchone()
            snapshot_by_digest = connection.execute(
                """
                SELECT snapshot_ref FROM catalog_snapshot_refs
                WHERE organization_id = ? AND catalog_sha256 = ?
                """,
                (self.organization_id, catalog_sha256),
            ).fetchone()
            snapshot_by_ref = connection.execute(
                """
                SELECT organization_id, catalog_sha256 FROM catalog_snapshot_refs
                WHERE snapshot_ref = ?
                """,
                (snapshot_ref,),
            ).fetchone()
        finally:
            connection.close()
        if existing is not None:
            expected = (
                self.organization_id,
                alias_id,
                alias_name,
                target.kind,
                target.pool_id if isinstance(target, DirectTarget) else None,
                target.project_ref if isinstance(target, ProjectTarget) else None,
                target.activation_ref if isinstance(target, ProjectTarget) else None,
                snapshot_ref,
                catalog_sha256,
                int(refusal_failover),
            )
            actual = tuple(existing[index] for index in range(10))
            if actual != expected:
                raise GatewayStoreError("alias revision ID was reused with different input")
            return False, True
        if alias_by_id is not None and (
            str(alias_by_id["organization_id"]) != self.organization_id
            or str(alias_by_id["alias_name"]) != alias_name
        ):
            raise GatewayStoreError("alias ID cannot be renamed or reused across organizations")
        if alias_by_name is not None and str(alias_by_name["alias_id"]) != alias_id:
            raise GatewayStoreError("alias name is already assigned to another alias ID")
        if (
            snapshot_by_digest is not None
            and str(snapshot_by_digest["snapshot_ref"]) != snapshot_ref
        ):
            raise GatewayStoreError("catalog digest is already registered under another snapshot")
        if snapshot_by_ref is not None and (
            str(snapshot_by_ref["organization_id"]) != self.organization_id
            or str(snapshot_by_ref["catalog_sha256"]) != catalog_sha256
        ):
            raise GatewayStoreError("catalog snapshot reference was reused with another digest")
        return True, snapshot_by_digest is not None

    def _rows(
        self, query: str, parameters: tuple[str, ...] | None = None
    ) -> tuple[sqlite3.Row, ...]:
        """Execute one fixed content-free read query."""
        connection = connect_database(self.database_path)
        try:
            rows = connection.execute(
                query,
                (self.organization_id,) if parameters is None else parameters,
            ).fetchall()
        finally:
            connection.close()
        return tuple(rows)


def _datetime(value: object) -> datetime | None:
    """Parse one optional stored UTC timestamp."""
    return None if value is None else datetime.fromisoformat(str(value))


def _required_datetime(value: object) -> datetime:
    """Parse one required stored UTC timestamp."""
    parsed = _datetime(value)
    if parsed is None:
        raise GatewayStoreError("gateway timestamp is missing")
    return parsed
