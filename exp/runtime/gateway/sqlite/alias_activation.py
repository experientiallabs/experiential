"""Typed SQLite transaction boundary for immutable alias activation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from exp.common.core.artifacts import Sha256
from exp.runtime.gateway.contracts import DirectTarget, GatewayTarget, ProjectTarget
from exp.runtime.gateway.sqlite.provider_authority import (
    ProviderConnectionBinding,
    bind_alias_provider_connections,
)

ConnectionFactory = Callable[[], AbstractContextManager[sqlite3.Connection]]


class AliasActivationOutcomeUnknownError(ValueError):
    """An alias activation could not prove whether its COMMIT landed."""

    def __init__(self, *, alias_id: str, revision_id: str) -> None:
        """Create a content-free recovery error for one immutable alias revision.

        Args:
            alias_id: Stable public alias identifier.
            revision_id: Immutable revision whose commit outcome is unknown.
        """
        self.alias_id = alias_id
        self.revision_id = revision_id
        super().__init__(
            "operation_outcome_unknown: inspect alias "
            f"{alias_id!r} revision {revision_id!r} before retrying"
        )


@contextmanager
def alias_activation_transaction(
    *,
    connect: ConnectionFactory,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    target: GatewayTarget,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    refusal_failover: bool,
) -> Iterator[sqlite3.Connection]:
    """Run activation while typing only ambiguous COMMIT acknowledgements.

    Args:
        connect: Factory for configured SQLite connection contexts.
        organization_id: Owning tenant.
        alias_id: Stable alias resource identifier.
        alias_name: Public model name.
        revision_id: Immutable revision identifier.
        target: Direct pool or project target.
        snapshot_ref: Registered catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        refusal_failover: Whether typed precommit refusals may advance.

    Yields:
        Open immediate transaction for the activation body.

    Raises:
        AliasActivationOutcomeUnknownError: COMMIT or teardown failed after dispatch and a fresh
            read was inconclusive.
        ValueError: An unacknowledged COMMIT was proven not to contain the desired activation.
    """
    commit_started = False
    commit_acknowledged = False
    try:
        with connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            commit_started = True
            connection.execute("COMMIT")
            commit_acknowledged = True
    except BaseException as activation_error:  # noqa: BLE001 - teardown may follow a durable COMMIT
        if not commit_started:
            raise
        try:
            outcome = reconcile_alias_activation(
                connect=connect,
                organization_id=organization_id,
                alias_id=alias_id,
                alias_name=alias_name,
                revision_id=revision_id,
                target=target,
                snapshot_ref=snapshot_ref,
                catalog_sha256=catalog_sha256,
                refusal_failover=refusal_failover,
            )
        except BaseException as reconciliation_error:  # noqa: BLE001 - never undo possible COMMIT
            raise AliasActivationOutcomeUnknownError(
                alias_id=alias_id,
                revision_id=revision_id,
            ) from reconciliation_error
        if outcome is True:
            return
        if outcome is False and not commit_acknowledged:
            raise ValueError("alias activation did not commit") from activation_error
        raise AliasActivationOutcomeUnknownError(
            alias_id=alias_id,
            revision_id=revision_id,
        ) from activation_error


def reconcile_alias_activation(
    *,
    connect: ConnectionFactory,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    target: GatewayTarget,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    refusal_failover: bool,
) -> bool | None:
    """Return whether the exact activation committed, or ``None`` if unreadable.

    Args:
        connect: Factory for a fresh configured SQLite connection context.
        organization_id: Owning tenant.
        alias_id: Stable alias resource identifier.
        alias_name: Public model name.
        revision_id: Immutable revision identifier.
        target: Direct pool or project target.
        snapshot_ref: Registered catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        refusal_failover: Whether typed precommit refusals may advance.

    Returns:
        ``True`` for the exact committed revision, ``False`` for a proven absence or mismatch,
        and ``None`` when a fresh SQLite read is unavailable.
    """
    try:
        with connect() as connection:
            row = connection.execute(
                """
                SELECT a.alias_name, r.target_kind, r.pool_id, r.project_ref, r.activation_ref,
                       r.snapshot_ref, r.catalog_sha256, r.refusal_failover
                FROM alias_revisions AS r
                JOIN gateway_aliases AS a
                  ON a.organization_id = r.organization_id AND a.alias_id = r.alias_id
                WHERE r.organization_id = ? AND r.alias_id = ? AND r.revision_id = ?
                """,
                (organization_id, alias_id, revision_id),
            ).fetchone()
    except Exception:  # noqa: BLE001 - every ordinary read failure leaves COMMIT ambiguous
        return None
    if row is None:
        return False
    expected = (
        alias_name,
        target.kind,
        target.pool_id if isinstance(target, DirectTarget) else None,
        target.project_ref if isinstance(target, ProjectTarget) else None,
        target.activation_ref if isinstance(target, ProjectTarget) else None,
        snapshot_ref,
        catalog_sha256,
        int(refusal_failover),
    )
    return tuple(row[index] for index in range(8)) == expected


def register_catalog_snapshot_in_transaction(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    now: str,
    store_error: type[ValueError],
) -> None:
    """Register one catalog snapshot idempotently inside a caller-owned transaction.

    Args:
        connection: Open SQLite transaction.
        organization_id: Owning tenant.
        snapshot_ref: Content-addressed external snapshot reference.
        catalog_sha256: Normalized secret-free catalog digest.
        now: Canonical transaction timestamp.
        store_error: Gateway-specific error type for authority conflicts.

    Raises:
        ValueError: The snapshot reference or digest was previously assigned differently.
    """
    by_ref = connection.execute(
        """
        SELECT organization_id, catalog_sha256 FROM catalog_snapshot_refs
        WHERE snapshot_ref = ?
        """,
        (snapshot_ref,),
    ).fetchone()
    if by_ref is not None:
        if (
            str(by_ref["organization_id"]) != organization_id
            or str(by_ref["catalog_sha256"]) != catalog_sha256
        ):
            raise store_error("catalog snapshot reference was reused with another digest")
        return
    by_digest = connection.execute(
        """
        SELECT snapshot_ref FROM catalog_snapshot_refs
        WHERE organization_id = ? AND catalog_sha256 = ?
        """,
        (organization_id, catalog_sha256),
    ).fetchone()
    if by_digest is not None:
        raise store_error("catalog digest is already registered under another snapshot")
    connection.execute(
        """
        INSERT INTO catalog_snapshot_refs (
            snapshot_ref, organization_id, catalog_sha256, created_at
        ) VALUES (?, ?, ?, ?)
        """,
        (snapshot_ref, organization_id, catalog_sha256, now),
    )


def activate_alias_revision_in_transaction(
    connection: sqlite3.Connection,
    *,
    organization_id: str,
    alias_id: str,
    alias_name: str,
    revision_id: str,
    target: GatewayTarget,
    snapshot_ref: str,
    catalog_sha256: Sha256,
    provider_connections: tuple[ProviderConnectionBinding, ...],
    refusal_failover: bool,
    now: str,
    store_error: type[ValueError],
) -> None:
    """Create and activate one immutable alias revision in an open transaction.

    Args:
        connection: Open SQLite transaction.
        organization_id: Owning tenant.
        alias_id: Stable alias resource ID.
        alias_name: Public model string.
        revision_id: Immutable alias revision ID.
        target: Direct pool or frozen project target.
        snapshot_ref: Registered catalog snapshot reference.
        catalog_sha256: Exact normalized catalog digest.
        provider_connections: Exact active connection revisions for the snapshot.
        refusal_failover: Whether typed precommit refusals may advance.
        now: Canonical transaction timestamp.
        store_error: Gateway-specific error type for authority conflicts.

    Raises:
        ValueError: The snapshot or alias invariants conflict.
    """
    snapshot = connection.execute(
        """
        SELECT 1 FROM catalog_snapshot_refs
        WHERE organization_id = ? AND snapshot_ref = ? AND catalog_sha256 = ?
        """,
        (organization_id, snapshot_ref, catalog_sha256),
    ).fetchone()
    if snapshot is None:
        raise store_error("catalog snapshot reference is not registered")
    alias_row = connection.execute(
        """
        SELECT alias_name FROM gateway_aliases
        WHERE organization_id = ? AND alias_id = ?
        """,
        (organization_id, alias_id),
    ).fetchone()
    if alias_row is None:
        connection.execute(
            """
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (alias_id, organization_id, alias_name, now, now),
        )
        revision_number = 1
    else:
        if str(alias_row["alias_name"]) != alias_name:
            raise store_error("alias ID cannot be renamed")
        revision_number = int(
            connection.execute(
                """
                SELECT COALESCE(MAX(revision_number), 0) + 1
                FROM alias_revisions WHERE organization_id = ? AND alias_id = ?
                """,
                (organization_id, alias_id),
            ).fetchone()[0]
        )
    pool_id: str | None = None
    project_ref: str | None = None
    activation_ref: str | None = None
    if isinstance(target, DirectTarget):
        pool_id = target.pool_id
    else:
        project_ref = target.project_ref
        activation_ref = target.activation_ref
    connection.execute(
        """
        INSERT INTO alias_revisions (
            revision_id, organization_id, alias_id, revision_number, target_kind,
            pool_id, project_ref, activation_ref, catalog_sha256, snapshot_ref,
            refusal_failover, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            revision_id,
            organization_id,
            alias_id,
            revision_number,
            target.kind,
            pool_id,
            project_ref,
            activation_ref,
            catalog_sha256,
            snapshot_ref,
            int(refusal_failover),
            now,
        ),
    )
    bind_alias_provider_connections(
        connection,
        organization_id=organization_id,
        alias_id=alias_id,
        alias_revision_id=revision_id,
        bindings=provider_connections,
        now=now,
    )
    connection.execute(
        """
        DELETE FROM project_activation_bindings
        WHERE organization_id = ? AND alias_id = ?
        """,
        (organization_id, alias_id),
    )
    if isinstance(target, ProjectTarget):
        connection.execute(
            """
            INSERT INTO project_activation_bindings (
                organization_id, project_ref, activation_ref,
                alias_id, revision_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                organization_id,
                target.project_ref,
                target.activation_ref,
                alias_id,
                revision_id,
                now,
            ),
        )
    connection.execute(
        """
        UPDATE gateway_aliases
        SET active_revision_id = ?, active = 1, updated_at = ?
        WHERE organization_id = ? AND alias_id = ?
        """,
        (revision_id, now, organization_id, alias_id),
    )
