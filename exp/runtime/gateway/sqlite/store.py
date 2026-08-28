"""Transactional SQLite authority for organizations, keys, grants, and aliases."""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import Sha256, sha256_json
from exp.runtime.gateway.auth import (
    FingerprintPepperFile,
    GatewayAuthError,
    IssuedVirtualKey,
    fingerprint_virtual_key,
    issue_key_material,
    key_prefix,
    utc_text,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayRequest,
    GatewayTarget,
    ProjectTarget,
    canonical_request_sha256,
)
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.sqlite import key_delivery
from exp.runtime.gateway.sqlite.alias_activation import (
    activate_alias_revision_in_transaction,
    alias_activation_transaction,
)
from exp.runtime.gateway.sqlite.migrations import initialize_database, persistent_connection
from exp.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionBinding,
    ProviderConnectionMutation,
)
from exp.runtime.gateway.sqlite.provider_store import ProviderConnectionStoreMixin
from exp.runtime.gateway.sqlite.setup_authority import (
    configure_direct_alias_with_identity,
    upsert_provider_connections_and_activate_direct_alias,
)

_LAST_USED_REFRESH_SECONDS = 60.0


class GatewayStoreError(ValueError):
    """A control-store mutation or lookup violates gateway authority invariants."""


class InvalidVirtualKeyError(GatewayStoreError):
    """A virtual key is unknown, expired, revoked, or attached to disabled authority."""


class AliasNotGrantedError(GatewayStoreError):
    """The authenticated identity has no active grant for the requested alias."""


class OperationConflictError(GatewayStoreError):
    """An operation ID was reused for different mutation input."""


class OperationReplayUnavailableError(GatewayStoreError):
    """A completed one-time key operation cannot reveal its raw key again."""


class OperationOutcomeUnknownError(GatewayStoreError):
    """A one-time key operation could not prove its durable commit outcome."""

    def __init__(self, *, issued: IssuedVirtualKey) -> None:
        """Create a content-free recovery error for one non-secret key identifier."""
        self.issued = issued
        super().__init__(
            "operation_outcome_unknown: preserve the delivered secret and inspect key status "
            f"for {issued.key_id!r} before retrying"
        )


class KeyIssuanceCommitError(GatewayStoreError):
    """A virtual-key transaction was proven not to have committed."""


class KeyIssuanceConflictError(GatewayStoreError):
    """Virtual-key issuance conflicts with existing gateway authority."""


class SystemGatewayClock:
    """Production wall and monotonic clock implementation."""

    def now(self) -> datetime:
        """Return the current UTC wall-clock time."""
        return datetime.now(UTC)

    def monotonic(self) -> float:
        """Return the process monotonic clock."""
        return time.monotonic()


class SQLiteGatewayStore(ProviderConnectionStoreMixin):
    """SQLite implementation of gateway authority and management operations."""

    _store_error = GatewayStoreError

    def __init__(
        self,
        database_path: Path,
        *,
        pepper_path: Path | None = None,
        clock: GatewayClock | None = None,
        busy_timeout_ms: int = 5_000,
    ) -> None:
        """Initialize private database and fingerprint-pepper state.

        Args:
            database_path: Local gateway SQLite path.
            pepper_path: Optional HMAC pepper path outside SQLite.
            clock: Injectable time source.
            busy_timeout_ms: Maximum SQLite lock wait.
        """
        self.database_path = database_path
        self._busy_timeout_ms = busy_timeout_ms
        self._clock = SystemGatewayClock() if clock is None else clock
        self._pepper = FingerprintPepperFile(
            pepper_path or database_path.with_name("gateway-key-pepper.json")
        )
        initialize_database(database_path, busy_timeout_ms=busy_timeout_ms)
        self._pepper.current()

    @property
    def pepper_path(self) -> Path:
        """Return the user-only virtual-key pepper path."""
        return self._pepper.path

    @property
    def busy_timeout_ms(self) -> int:
        """Return the configured SQLite lock-wait bound."""
        return self._busy_timeout_ms

    def create_organization(
        self,
        *,
        organization_id: str,
        slug: str,
        display_name: str,
    ) -> None:
        """Create one explicit tenant without adding identities or seed data.

        Args:
            organization_id: Stable tenant ID.
            slug: Unique operator-facing slug.
            display_name: Operator-facing tenant name.
        """
        now = utc_text(self._clock.now())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO organizations (
                    organization_id, slug, display_name, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (organization_id, slug, display_name, now, now),
            )

    def create_identity(
        self,
        *,
        organization_id: str,
        identity_id: str,
        display_name: str,
        description: str | None = None,
        operation_id: str | None = None,
    ) -> str:
        """Create one identity with optional mutation idempotency.

        Args:
            organization_id: Owning tenant.
            identity_id: Stable identity ID.
            display_name: Operator-facing identity name.
            description: Optional content-free identity description.
            operation_id: Optional retry-safe operation ID.

        Returns:
            The created or previously receipted identity ID.
        """
        request_sha256 = sha256_json(
            {
                "organization_id": organization_id,
                "identity_id": identity_id,
                "display_name": display_name,
                "description": description,
            }
        )
        now = utc_text(self._clock.now())
        with self._transaction() as connection:
            replay = self._operation_replay(
                connection,
                organization_id=organization_id,
                operation_id=operation_id,
                operation_kind="create_identity",
                request_sha256=request_sha256,
            )
            if replay is not None:
                return replay
            connection.execute(
                """
                INSERT INTO identities (
                    identity_id, organization_id, display_name, description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (identity_id, organization_id, display_name, description, now, now),
            )
            self._record_operation(
                connection,
                organization_id=organization_id,
                operation_id=operation_id,
                operation_kind="create_identity",
                request_sha256=request_sha256,
                resource_kind="identity",
                resource_id=identity_id,
                created_at=now,
            )
        return identity_id

    def issue_virtual_key(
        self,
        *,
        organization_id: str,
        identity_id: str,
        key_id: str,
        expires_at: datetime | None = None,
        operation_id: str | None = None,
        secret_delivery: key_delivery.KeyDeliverySink | None = None,
    ) -> IssuedVirtualKey:
        """Issue a 256-bit virtual key and persist only its HMAC fingerprint.

        Args:
            organization_id: Owning tenant.
            identity_id: Credential owner.
            key_id: Stable non-secret credential ID.
            expires_at: Optional timezone-aware expiry.
            operation_id: Optional retry-safe operation ID.
            secret_delivery: Optional transactional one-time secret sink. The sink
                returns cleanup that removes its published output if commit fails.

        Returns:
            One-time receipt containing the raw key.

        Raises:
            KeyIssuanceConflictError: The requested key conflicts with existing authority.
            OperationOutcomeUnknownError: Commit outcome cannot be proven after delivery.
            OperationReplayUnavailableError: A retry names an already issued key operation.
        """
        expires_text = None if expires_at is None else utc_text(expires_at)
        if expires_at is not None and expires_at <= self._clock.now():
            raise GatewayStoreError("virtual key expiry must be in the future")
        request_sha256 = sha256_json(
            {
                "organization_id": organization_id,
                "identity_id": identity_id,
                "key_id": key_id,
                "expires_at": expires_text,
            }
        )
        prefix, raw_key = issue_key_material()
        pepper = self._pepper.current()
        fingerprint = fingerprint_virtual_key(raw_key, pepper)
        created_at = self._clock.now()
        created_text = utc_text(created_at)
        issued = IssuedVirtualKey(
            key_id=key_id,
            organization_id=organization_id,
            identity_id=identity_id,
            prefix=prefix,
            raw_key=raw_key,
            expires_at=expires_at,
            created_at=created_at,
        )
        delivery_hooks: key_delivery.KeyDeliveryHooks | None = None
        commit_error: BaseException | None = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                replay = self._operation_replay(
                    connection,
                    organization_id=organization_id,
                    operation_id=operation_id,
                    operation_kind="issue_virtual_key",
                    request_sha256=request_sha256,
                )
                if replay is not None:
                    raise OperationReplayUnavailableError(
                        f"virtual key {replay!r} was already issued and cannot be revealed again"
                    )
                connection.execute(
                    """
                    INSERT INTO virtual_keys (
                        key_id, organization_id, identity_id, prefix, fingerprint_version,
                        fingerprint_sha256, expires_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key_id,
                        organization_id,
                        identity_id,
                        prefix,
                        pepper.version,
                        fingerprint,
                        expires_text,
                        created_text,
                    ),
                )
                if secret_delivery is not None:
                    delivery_hooks = secret_delivery(
                        raw_key,
                        key_delivery.KeyDeliveryEvidence(
                            organization_id=organization_id,
                            identity_id=identity_id,
                            key_id=key_id,
                            operation_id=operation_id,
                            request_sha256=request_sha256,
                            prefix=prefix,
                            fingerprint_version=pepper.version,
                            fingerprint_sha256=fingerprint,
                            expires_at=expires_text,
                            created_at=created_text,
                        ),
                    )
                self._record_operation(
                    connection,
                    organization_id=organization_id,
                    operation_id=operation_id,
                    operation_kind="issue_virtual_key",
                    request_sha256=request_sha256,
                    resource_kind="virtual_key",
                    resource_id=key_id,
                    created_at=created_text,
                )
                try:
                    connection.execute("COMMIT")
                except BaseException as error:  # noqa: BLE001 - COMMIT outcome is ambiguous
                    commit_error = error
            except BaseException as body_error:
                try:
                    connection.execute("ROLLBACK")
                except BaseException as rollback_error:
                    if delivery_hooks is not None:
                        raise OperationOutcomeUnknownError(issued=issued) from rollback_error
                    raise body_error from rollback_error
                if delivery_hooks is not None:
                    delivery_hooks.rollback()
                if isinstance(body_error, sqlite3.IntegrityError):
                    raise KeyIssuanceConflictError(
                        "virtual key issuance conflicts with existing gateway authority"
                    ) from None
                raise

        if commit_error is None:
            return issued
        outcome = key_delivery.reconcile_key_issue(
            self.database_path,
            busy_timeout_ms=self._busy_timeout_ms,
            organization_id=organization_id,
            identity_id=identity_id,
            key_id=key_id,
            prefix=prefix,
            fingerprint_version=pepper.version,
            fingerprint=fingerprint,
            expires_at=expires_text,
            created_at=created_text,
            operation_id=operation_id,
            request_sha256=request_sha256,
        )
        if outcome is True:
            return issued
        if outcome is False:
            if delivery_hooks is not None:
                delivery_hooks.rollback()
            raise KeyIssuanceCommitError("virtual key issuance did not commit") from commit_error
        raise OperationOutcomeUnknownError(issued=issued) from commit_error

    def revoke_virtual_key(self, *, organization_id: str, key_id: str) -> bool:
        """Revoke a key idempotently.

        Args:
            organization_id: Owning tenant.
            key_id: Stable key ID.

        Returns:
            True when an active key changed state.
        """
        with self._transaction() as connection:
            result = connection.execute(
                """
                UPDATE virtual_keys SET revoked_at = ?
                WHERE organization_id = ? AND key_id = ? AND revoked_at IS NULL
                """,
                (utc_text(self._clock.now()), organization_id, key_id),
            )
        return result.rowcount == 1

    def disable_identity(self, *, organization_id: str, identity_id: str) -> bool:
        """Disable an identity and therefore all of its keys.

        Args:
            organization_id: Owning tenant.
            identity_id: Stable identity ID.

        Returns:
            True when an active identity changed state.
        """
        with self._transaction() as connection:
            result = connection.execute(
                """
                UPDATE identities SET active = 0, updated_at = ?
                WHERE organization_id = ? AND identity_id = ? AND active = 1
                """,
                (utc_text(self._clock.now()), organization_id, identity_id),
            )
        return result.rowcount == 1

    def register_catalog_snapshot(
        self,
        *,
        organization_id: str,
        snapshot_ref: str,
        catalog_sha256: Sha256,
    ) -> None:
        """Register an immutable catalog snapshot reference without catalog content.

        Args:
            organization_id: Owning tenant.
            snapshot_ref: Content-addressed external snapshot reference.
            catalog_sha256: Normalized secret-free catalog digest.
        """
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO catalog_snapshot_refs (
                    snapshot_ref, organization_id, catalog_sha256, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (snapshot_ref, organization_id, catalog_sha256, utc_text(self._clock.now())),
            )

    def activate_alias_revision(
        self,
        *,
        organization_id: str,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        target: GatewayTarget,
        snapshot_ref: str,
        catalog_sha256: Sha256,
        provider_connections: tuple[ProviderConnectionBinding, ...] = (),
        refusal_failover: bool = False,
    ) -> None:
        """Create and atomically activate one immutable alias revision.

        Args:
            organization_id: Owning tenant.
            alias_id: Stable alias resource ID.
            alias_name: Public model string.
            revision_id: Immutable revision ID.
            target: Direct pool or frozen project activation target.
            snapshot_ref: Registered catalog snapshot reference.
            catalog_sha256: Exact normalized catalog digest.
            provider_connections: Exact active connection revisions used by the snapshot.
            refusal_failover: Whether typed precommit refusals may advance within the pool.
        """
        if isinstance(target, ProjectTarget) and target.catalog_sha256 != catalog_sha256:
            raise GatewayStoreError("project target catalog digest differs from alias activation")
        now = utc_text(self._clock.now())
        with alias_activation_transaction(
            connect=self._connect,
            organization_id=organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            target=target,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            refusal_failover=refusal_failover,
        ) as connection:
            activate_alias_revision_in_transaction(
                connection,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_name=alias_name,
                revision_id=revision_id,
                target=target,
                snapshot_ref=snapshot_ref,
                catalog_sha256=catalog_sha256,
                provider_connections=provider_connections,
                refusal_failover=refusal_failover,
                now=now,
                store_error=GatewayStoreError,
            )

    def upsert_provider_connections_and_activate_direct_alias(
        self,
        *,
        organization_id: str,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        pool_id: str,
        snapshot_ref: str,
        catalog_sha256: Sha256,
        provider_connections: tuple[ProviderConnectionMutation, ...],
        replace: bool,
        refusal_failover: bool = False,
    ) -> None:
        """Atomically revise providers, register a snapshot, and activate one direct alias."""
        upsert_provider_connections_and_activate_direct_alias(
            self,
            organization_id=organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            pool_id=pool_id,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            replace=replace,
            refusal_failover=refusal_failover,
            activate_alias_revision=activate_alias_revision_in_transaction,
        )

    def configure_direct_alias_with_identity(
        self,
        *,
        organization_id: str,
        alias_id: str,
        alias_name: str,
        revision_id: str,
        pool_id: str,
        snapshot_ref: str,
        catalog_sha256: Sha256,
        provider_connections: tuple[ProviderConnectionMutation, ...],
        replace: bool,
        identity_id: str,
        identity_display_name: str,
        key_id: str,
    ) -> tuple[bool, IssuedVirtualKey]:
        """Atomically configure serving authority and setup caller credentials."""
        return configure_direct_alias_with_identity(
            self,
            organization_id=organization_id,
            alias_id=alias_id,
            alias_name=alias_name,
            revision_id=revision_id,
            pool_id=pool_id,
            snapshot_ref=snapshot_ref,
            catalog_sha256=catalog_sha256,
            provider_connections=provider_connections,
            replace=replace,
            identity_id=identity_id,
            identity_display_name=identity_display_name,
            key_id=key_id,
            activate_alias_revision=activate_alias_revision_in_transaction,
        )

    def disable_alias(self, *, organization_id: str, alias_id: str) -> bool:
        """Disable one alias and release its active project binding.

        Args:
            organization_id: Owning tenant.
            alias_id: Stable alias resource ID.

        Returns:
            True when an active alias changed state.
        """
        now = utc_text(self._clock.now())
        with self._transaction() as connection:
            connection.execute(
                """
                DELETE FROM project_activation_bindings
                WHERE organization_id = ? AND alias_id = ?
                """,
                (organization_id, alias_id),
            )
            result = connection.execute(
                """
                UPDATE gateway_aliases SET active = 0, updated_at = ?
                WHERE organization_id = ? AND alias_id = ? AND active = 1
                """,
                (now, organization_id, alias_id),
            )
        return result.rowcount == 1

    def grant_alias(self, *, organization_id: str, identity_id: str, alias_id: str) -> bool:
        """Grant one identity one alias, returning whether a row was added.

        Args:
            organization_id: Owning tenant.
            identity_id: Granted identity.
            alias_id: Granted alias resource.

        Returns:
            True when the grant did not already exist.
        """
        with self._transaction() as connection:
            result = connection.execute(
                """
                INSERT OR IGNORE INTO identity_alias_grants (
                    organization_id, identity_id, alias_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (organization_id, identity_id, alias_id, utc_text(self._clock.now())),
            )
        return result.rowcount == 1

    def revoke_alias_grant(self, *, organization_id: str, identity_id: str, alias_id: str) -> bool:
        """Remove one grant idempotently.

        Args:
            organization_id: Owning tenant.
            identity_id: Previously granted identity.
            alias_id: Previously granted alias.

        Returns:
            True when a grant was removed.
        """
        with self._transaction() as connection:
            result = connection.execute(
                """
                DELETE FROM identity_alias_grants
                WHERE organization_id = ? AND identity_id = ? AND alias_id = ?
                """,
                (organization_id, identity_id, alias_id),
            )
        return result.rowcount == 1

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
        app_referer: str | None = None,
        app_title: str | None = None,
    ) -> AuthorizationSnapshot:
        """Authenticate and authorize before any model or provider work.

        Args:
            raw_key: Caller virtual key.
            alias: Requested public model alias.
            request: Canonical content-bearing request used only for its digest.
            deadline_monotonic: Absolute request-wide monotonic deadline.
            app_referer: Caller ``HTTP-Referer`` and ``app_title`` its ``X-Title`` app identity.

        Returns:
            Immutable content-free authority snapshot.

        Raises:
            InvalidVirtualKeyError: Authentication fails.
            AliasNotGrantedError: No active grant resolves the alias.
        """
        if deadline_monotonic <= self._clock.monotonic():
            raise GatewayStoreError("request deadline has already expired")
        with self._transaction() as connection:
            organization_id, identity_id, key_id = self._authenticate_in_transaction(
                connection, raw_key
            )
            row = connection.execute(
                """
                SELECT a.alias_id, a.alias_name, a.active_revision_id,
                       r.target_kind, r.pool_id, r.project_ref, r.activation_ref,
                       r.catalog_sha256, r.refusal_failover
                FROM identity_alias_grants AS g
                JOIN identities AS i
                  ON i.organization_id = g.organization_id AND i.identity_id = g.identity_id
                JOIN gateway_aliases AS a
                  ON a.organization_id = g.organization_id AND a.alias_id = g.alias_id
                JOIN alias_revisions AS r
                  ON r.organization_id = a.organization_id
                 AND r.alias_id = a.alias_id
                 AND r.revision_id = a.active_revision_id
                WHERE g.organization_id = ? AND g.identity_id = ?
                  AND a.alias_name = ? AND i.active = 1 AND a.active = 1
                """,
                (organization_id, identity_id, alias),
            ).fetchone()
        if row is None:
            raise AliasNotGrantedError("requested model alias is not granted")
        target: GatewayTarget
        if str(row["target_kind"]) == "direct":
            target = DirectTarget(pool_id=str(row["pool_id"]))
        else:
            target = ProjectTarget(
                project_ref=str(row["project_ref"]),
                activation_ref=str(row["activation_ref"]),
                catalog_sha256=str(row["catalog_sha256"]),
            )
        caller_operation = _caller_operation_sha256(request)
        return AuthorizationSnapshot(
            request_id=f"request-{uuid.uuid4().hex}",
            organization_id=organization_id,
            identity_id=identity_id,
            virtual_key_id=key_id,
            alias=alias,
            alias_revision_id=str(row["active_revision_id"]),
            target=target,
            catalog_sha256=str(row["catalog_sha256"]),
            canonical_request_sha256=canonical_request_sha256(request),
            deadline_monotonic=deadline_monotonic,
            surface=request.surface,
            caller_operation_sha256=caller_operation,
            refusal_failover=bool(row["refusal_failover"]),
            app_referer=app_referer,
            app_title=app_title,
        )

    def authenticate_key(self, *, raw_key: str) -> None:
        """Validate one virtual key without loading grants or request content.

        Args:
            raw_key: Presented virtual key.

        Raises:
            InvalidVirtualKeyError: The key is unknown, expired, or revoked.
        """
        with self._transaction() as connection:
            self._authenticate_in_transaction(connection, raw_key)

    def authenticated_identity(self, *, raw_key: str) -> tuple[str, str]:
        """Return the organization and identity IDs owning one valid key."""
        with self._transaction() as connection:
            organization_id, identity_id, _ = self._authenticate_in_transaction(connection, raw_key)
        return organization_id, identity_id

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """List only active aliases granted to the key-derived identity.

        Args:
            raw_key: Caller virtual key.

        Returns:
            Granted public alias names in stable order.
        """
        return tuple(
            alias for alias, _revision, _digest in self.granted_alias_authorities(raw_key=raw_key)
        )

    def granted_alias_authorities(self, *, raw_key: str) -> tuple[tuple[str, str, str], ...]:
        """Return exact active authorities granted to the key-derived identity.

        Args:
            raw_key: Caller virtual key.

        Returns:
            Alias name, active revision, and catalog digest tuples in stable order.
        """
        with self._transaction() as connection:
            organization_id, identity_id, _ = self._authenticate_in_transaction(connection, raw_key)
            rows = connection.execute(
                """
                SELECT a.alias_name, a.active_revision_id, r.catalog_sha256
                FROM identity_alias_grants AS g
                JOIN gateway_aliases AS a
                  ON a.organization_id = g.organization_id AND a.alias_id = g.alias_id
                JOIN alias_revisions AS r
                  ON r.organization_id = a.organization_id
                 AND r.alias_id = a.alias_id
                 AND r.revision_id = a.active_revision_id
                WHERE g.organization_id = ? AND g.identity_id = ?
                  AND a.active = 1 AND a.active_revision_id IS NOT NULL
                ORDER BY a.alias_name
                """,
                (organization_id, identity_id),
            ).fetchall()
        return tuple(
            (
                str(row["alias_name"]),
                str(row["active_revision_id"]),
                str(row["catalog_sha256"]),
            )
            for row in rows
        )

    def rotate_fingerprint_pepper(self) -> int:
        """Rotate future key fingerprints while retaining old key authentication."""
        return self._pepper.rotate()

    def _authenticate_in_transaction(
        self, connection: sqlite3.Connection, raw_key: str
    ) -> tuple[str, str, str]:
        """Authenticate one key inside the caller's authority transaction.

        The key's last-used timestamp is operator telemetry with deliberate
        coarse granularity: it is rewritten at most once per refresh interval
        so hot keys do not dirty a page and pay a durable write per request.

        Args:
            connection: Immediate transaction retained through the authority read.
            raw_key: Caller key that must never enter SQLite or logs.

        Returns:
            Organization, identity, and virtual-key IDs for active authority.

        Raises:
            InvalidVirtualKeyError: The key or its owning authority is inactive.
        """
        try:
            prefix = key_prefix(raw_key)
        except GatewayAuthError as exc:
            raise InvalidVirtualKeyError("virtual key is invalid") from exc
        rows = connection.execute(
            """
            SELECT k.organization_id, k.identity_id, k.key_id,
                   k.fingerprint_version, k.fingerprint_sha256,
                   k.expires_at, k.revoked_at, k.last_used_at,
                   i.active AS identity_active,
                   o.active AS organization_active
            FROM virtual_keys AS k
            JOIN identities AS i
              ON i.organization_id = k.organization_id AND i.identity_id = k.identity_id
            JOIN organizations AS o ON o.organization_id = k.organization_id
            WHERE k.prefix = ?
            """,
            (prefix,),
        ).fetchall()
        now = self._clock.now()
        selected: sqlite3.Row | None = None
        for row in rows:
            try:
                pepper = self._pepper.key(int(row["fingerprint_version"]))
            except GatewayAuthError:
                continue
            fingerprint = fingerprint_virtual_key(raw_key, pepper)
            if hmac.compare_digest(fingerprint, str(row["fingerprint_sha256"])):
                selected = row
        if selected is None:
            raise InvalidVirtualKeyError("virtual key is invalid")
        expires_at = selected["expires_at"]
        expired = expires_at is not None and datetime.fromisoformat(str(expires_at)) <= now
        if (
            selected["revoked_at"] is not None
            or expired
            or int(selected["identity_active"]) != 1
            or int(selected["organization_active"]) != 1
        ):
            raise InvalidVirtualKeyError("virtual key is invalid")
        organization_id = str(selected["organization_id"])
        identity_id = str(selected["identity_id"])
        key_id = str(selected["key_id"])
        last_used = selected["last_used_at"]
        stale = (
            last_used is None
            or (now - datetime.fromisoformat(str(last_used))).total_seconds()
            >= _LAST_USED_REFRESH_SECONDS
        )
        if stale:
            connection.execute(
                """
                UPDATE virtual_keys SET last_used_at = ?
                WHERE organization_id = ? AND key_id = ?
                """,
                (utc_text(now), organization_id, key_id),
            )
        return organization_id, identity_id, key_id

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Check out one reusable configured connection."""
        with persistent_connection(
            self.database_path, busy_timeout_ms=self._busy_timeout_ms
        ) as connection:
            yield connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one explicit immediate transaction with rollback on failure."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")

    @staticmethod
    def _operation_replay(
        connection: sqlite3.Connection,
        *,
        organization_id: str,
        operation_id: str | None,
        operation_kind: str,
        request_sha256: Sha256,
    ) -> str | None:
        """Return a matching receipted resource or reject operation-ID drift."""
        if operation_id is None:
            return None
        row = connection.execute(
            """
            SELECT operation_kind, request_sha256, resource_id
            FROM operation_receipts
            WHERE organization_id = ? AND operation_id = ?
            """,
            (organization_id, operation_id),
        ).fetchone()
        if row is None:
            return None
        if (
            str(row["operation_kind"]) != operation_kind
            or str(row["request_sha256"]) != request_sha256
        ):
            raise OperationConflictError("operation ID was reused with different input")
        return str(row["resource_id"])

    @staticmethod
    def _record_operation(
        connection: sqlite3.Connection,
        *,
        organization_id: str,
        operation_id: str | None,
        operation_kind: str,
        request_sha256: Sha256,
        resource_kind: str,
        resource_id: str,
        created_at: str,
    ) -> None:
        """Persist a content-free operation receipt inside its mutation transaction."""
        if operation_id is None:
            return
        connection.execute(
            """
            INSERT INTO operation_receipts (
                organization_id, operation_id, operation_kind, request_sha256,
                resource_kind, resource_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                organization_id,
                operation_id,
                operation_kind,
                request_sha256,
                resource_kind,
                resource_id,
                created_at,
            ),
        )


def _caller_operation_sha256(request: GatewayRequest) -> Sha256 | None:
    """Hash an opted-in caller operation without retaining the raw identifier.

    Args:
        request: Canonical gateway request.

    Returns:
        Namespaced caller-operation digest, or ``None`` for ordinary requests.

    Raises:
        GatewayStoreError: Both supported headers name different operations.
    """
    if (
        request.idempotency_key is not None
        and request.client_request_id is not None
        and request.idempotency_key != request.client_request_id
    ):
        raise GatewayStoreError("idempotency and client request IDs must match when both are set")
    value = request.idempotency_key or request.client_request_id
    if value is None:
        return None
    return hashlib.sha256(f"gateway-caller-operation-v1\0{value}".encode()).hexdigest()
