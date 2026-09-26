"""Read and fence request alias authority for the local SQLite gateway."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager

from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.model_chain_authority import (
    SnapshotClassificationMemo,
    SQLiteChainWitness,
    observe_sqlite_chain_authority,
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
) -> tuple[str, str, str, sqlite3.Row, str, SQLiteChainWitness]:
    """Authenticate, resolve one alias and capture its classified request witness.

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
        Organization, identity, key and alias row, plus the request ID and chain witness.
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
                   r.catalog_sha256, r.refusal_failover,
                   r.organization_id AS authority_organization_id,
                   r.alias_id AS authority_alias_id,
                   r.revision_id AS authority_revision_id,
                   r.catalog_sha256 AS authority_catalog_sha256,
                   r.snapshot_ref AS authority_snapshot_ref,
                   a.active_revision_id AS authority_active_revision_id,
                   active.catalog_sha256 AS active_catalog_sha256,
                   active.snapshot_ref AS active_snapshot_ref
            FROM identity_alias_grants AS g
            JOIN identities AS i
              ON i.organization_id = g.organization_id AND i.identity_id = g.identity_id
            JOIN gateway_aliases AS a
              ON a.organization_id = g.organization_id AND a.alias_id = g.alias_id
            JOIN alias_revisions AS r
              ON r.organization_id = a.organization_id
             AND r.alias_id = a.alias_id
             AND r.revision_id = a.active_revision_id
            LEFT JOIN alias_revisions AS active
              ON active.organization_id = a.organization_id
             AND active.revision_id = a.active_revision_id
            WHERE g.organization_id = ? AND g.identity_id = ?
              AND a.alias_name = ? AND i.active = 1 AND a.active = 1
            """,
            (organization_id, identity_id, alias),
        ).fetchone()
        return organization_id, identity_id, key_id, row

    with connect() as reader:
        try:
            # Fresh keys only read authority. Deferred mode avoids serializing
            # concurrent admissions on SQLite's writer lock.
            with transaction(connection=reader, immediate=False) as connection:
                organization_id, identity_id, key_id, row = read_alias(connection)
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            if code is None or code & 0xFF not in (
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            ):
                raise
            if reader.in_transaction:
                reader.execute("ROLLBACK")
            # A concurrent stale-key refresh can invalidate a deferred read's
            # write upgrade. Retry that rare case with the normal write lock.
            with transaction(connection=reader) as connection:
                organization_id, identity_id, key_id, row = read_alias(connection)
        if row is None:
            raise alias_not_granted_error("requested model alias is not granted")

        alias_revision_id = str(row["active_revision_id"])
        authority_row = tuple(
            None if row[field] is None else str(row[field])
            for field in (
                "authority_organization_id",
                "authority_alias_id",
                "authority_revision_id",
                "authority_catalog_sha256",
                "authority_snapshot_ref",
                "authority_active_revision_id",
                "active_catalog_sha256",
                "active_snapshot_ref",
            )
        )
        observation = observe_sqlite_chain_authority(
            reader,
            organization_id,
            alias_revision_id,
            rows=(authority_row,),
        )
        request_id = f"request-{uuid.uuid4().hex}"
        with prepare_sqlite_chain_authority(
            None,
            organization_id,
            alias_revision_id,
            request_id=request_id,
            operation="authorize",
            maximum_bytes=serving_snapshot_max_bytes,
            remaining_seconds=deadline_monotonic - clock.monotonic(),
            classification_memo=classification_memo,
            observation=observation,
        ) as proof:
            witness = proof.authority_witness()
    return organization_id, identity_id, key_id, row, request_id, witness
