"""Read and fence request alias authority for the local SQLite gateway."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager

from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.model_chain_authority import (
    SnapshotClassificationMemo,
    prepare_sqlite_chain_authority,
)


def authorize_sqlite_alias(
    *,
    raw_key: str,
    alias: str,
    deadline_monotonic: float,
    clock: GatewayClock,
    connect: Callable[[], AbstractContextManager[sqlite3.Connection]],
    transaction: Callable[..., AbstractContextManager[sqlite3.Connection]],
    authenticate: Callable[[sqlite3.Connection, str], tuple[str, str, str]],
    classification_memo: SnapshotClassificationMemo,
    serving_snapshot_max_bytes: int,
    alias_not_granted_error: type[Exception],
) -> tuple[str, str, str, sqlite3.Row, str]:
    """Authenticate, resolve one alias and fence its local snapshot before routing.

    Args:
        raw_key: Caller virtual key.
        alias: Requested public model alias.
        deadline_monotonic: Absolute request-wide monotonic deadline.
        clock: Gateway clock used for the remaining preparation budget.
        connect: Store connection factory.
        transaction: Store transaction factory supporting deferred reads.
        authenticate: Store key-authentication operation.
        classification_memo: Bounded snapshot parser memo owned by the store.
        serving_snapshot_max_bytes: Serving limit for each local snapshot file.
        alias_not_granted_error: Store-specific error raised for an inactive grant.

    Returns:
        Organization, identity, key and alias row, plus the bound request ID.
    """

    def read_alias(
        connection: sqlite3.Connection,
    ) -> tuple[str, str, str, sqlite3.Row | None]:
        """Authenticate and resolve the active row within one SQLite snapshot."""
        organization_id, identity_id, key_id = authenticate(connection, raw_key)
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
        return organization_id, identity_id, key_id, row

    try:
        # Fresh keys only read authority. Deferred mode avoids serializing
        # concurrent admissions on SQLite's writer lock.
        with transaction(immediate=False) as connection:
            organization_id, identity_id, key_id, row = read_alias(connection)
    except sqlite3.OperationalError as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        if code is None or code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
            raise
        # A concurrent stale-key refresh can invalidate a deferred read's
        # write upgrade. Retry that rare case with the normal write lock.
        with transaction() as connection:
            organization_id, identity_id, key_id, row = read_alias(connection)
    if row is None:
        raise alias_not_granted_error("requested model alias is not granted")

    request_id = f"request-{uuid.uuid4().hex}"
    with (
        connect() as reader,
        prepare_sqlite_chain_authority(
            reader,
            organization_id,
            str(row["active_revision_id"]),
            request_id=request_id,
            operation="authorize",
            maximum_bytes=serving_snapshot_max_bytes,
            remaining_seconds=deadline_monotonic - clock.monotonic(),
            classification_memo=classification_memo,
        ) as proof,
        transaction(connection=reader, immediate=False) as connection,
    ):
        proof.validate(
            connection,
            request_id=request_id,
            organization_id=organization_id,
            alias_revision_id=str(row["active_revision_id"]),
            operation="authorize",
        )
    return organization_id, identity_id, key_id, row, request_id
