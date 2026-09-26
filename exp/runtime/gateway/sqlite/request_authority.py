"""Read and fence request alias authority for the local SQLite gateway."""

from __future__ import annotations

import hmac
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime

from exp.runtime.gateway.auth import (
    GatewayAuthError,
    PepperKey,
    fingerprint_virtual_key,
    key_prefix,
    utc_text,
)
from exp.runtime.gateway.interfaces import GatewayClock
from exp.runtime.gateway.model_chain_authority import (
    SnapshotClassificationMemo,
    SQLiteChainWitness,
    observe_sqlite_chain_authority,
    prepare_sqlite_chain_authority,
)


def authorize_preflight_alias_in_transaction(
    connection: sqlite3.Connection,
    raw_key: str,
    preauthenticated_key: tuple[str, str, str],
    alias: str,
    *,
    pepper_key: Callable[[int], PepperKey],
    clock: GatewayClock,
    invalid_key_error: type[Exception],
    last_used_refresh_seconds: float,
    update_last_used: bool = True,
    timing_recorder: Callable[[str, float], None] | None = None,
) -> tuple[str, str, str, sqlite3.Row | None]:
    """Revalidate the exact preflight key and resolve its alias in one query."""
    organization_id, identity_id, key_id = preauthenticated_key
    try:
        prefix = key_prefix(raw_key)
    except GatewayAuthError as exc:
        raise invalid_key_error("virtual key is invalid") from exc

    alias_lookup_started = time.monotonic()
    selected = connection.execute(
        """
        SELECT k.organization_id, k.identity_id, k.key_id,
               k.fingerprint_version, k.fingerprint_sha256,
               k.expires_at, k.revoked_at, k.last_used_at,
               i.active AS identity_active,
               o.active AS organization_active,
               g.alias_id AS granted_alias_id,
               a.alias_id, a.alias_name, a.active_revision_id,
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
        FROM virtual_keys AS k
        JOIN identities AS i
          ON i.organization_id = k.organization_id AND i.identity_id = k.identity_id
        JOIN organizations AS o ON o.organization_id = k.organization_id
        LEFT JOIN gateway_aliases AS a
          ON a.organization_id = k.organization_id
         AND a.alias_name = ? AND a.active = 1
        LEFT JOIN identity_alias_grants AS g
          ON g.organization_id = k.organization_id
         AND g.identity_id = k.identity_id AND g.alias_id = a.alias_id
        LEFT JOIN alias_revisions AS r
          ON r.organization_id = a.organization_id
         AND r.alias_id = a.alias_id
         AND r.revision_id = a.active_revision_id
        LEFT JOIN alias_revisions AS active
          ON active.organization_id = a.organization_id
         AND active.alias_id = a.alias_id
         AND active.revision_id = a.active_revision_id
        WHERE k.organization_id = ? AND k.identity_id = ? AND k.key_id = ?
          AND k.prefix = ?
        """,
        (alias, organization_id, identity_id, key_id, prefix),
    ).fetchone()
    if timing_recorder is not None:
        timing_recorder("sqlite_alias_lookup_ms", alias_lookup_started)
    if selected is None:
        raise invalid_key_error("virtual key is invalid")

    authentication_started = time.monotonic()
    try:
        pepper = pepper_key(int(selected["fingerprint_version"]))
    except GatewayAuthError as exc:
        raise invalid_key_error("virtual key is invalid") from exc
    fingerprint = fingerprint_virtual_key(raw_key, pepper)
    if not hmac.compare_digest(fingerprint, str(selected["fingerprint_sha256"])):
        raise invalid_key_error("virtual key is invalid")
    expires_at = selected["expires_at"]
    expired = expires_at is not None and datetime.fromisoformat(str(expires_at)) <= clock.now()
    if (
        selected["revoked_at"] is not None
        or expired
        or int(selected["identity_active"]) != 1
        or int(selected["organization_active"]) != 1
    ):
        raise invalid_key_error("virtual key is invalid")
    if timing_recorder is not None:
        timing_recorder("sqlite_key_authentication_ms", authentication_started)

    last_used = selected["last_used_at"]
    now = clock.now()
    stale = (
        last_used is None
        or (now - datetime.fromisoformat(str(last_used))).total_seconds()
        >= last_used_refresh_seconds
    )
    if stale and update_last_used:
        connection.execute(
            """
            UPDATE virtual_keys SET last_used_at = ?
            WHERE organization_id = ? AND key_id = ?
            """,
            (utc_text(now), organization_id, key_id),
        )

    alias_row = (
        selected
        if selected["granted_alias_id"] is not None
        and selected["active_revision_id"] is not None
        and selected["target_kind"] is not None
        else None
    )
    return organization_id, identity_id, key_id, alias_row


def authorize_sqlite_alias(
    *,
    raw_key: str,
    alias: str,
    deadline_monotonic: float,
    clock: GatewayClock,
    connect: Callable[[], AbstractContextManager[sqlite3.Connection]],
    transaction: Callable[..., AbstractContextManager[sqlite3.Connection]],
    authenticate: Callable[[sqlite3.Connection, str], tuple[str, str, str]],
    authenticate_readonly: Callable[[sqlite3.Connection, str], tuple[str, str, str]],
    authorize_preflight_alias: Callable[..., tuple[str, str, str, sqlite3.Row | None]],
    preauthenticated_key: tuple[str, str, str] | None,
    busy_timeout_ms: int,
    classification_memo: SnapshotClassificationMemo,
    serving_snapshot_max_bytes: int,
    alias_not_granted_error: type[Exception],
    timing_recorder: Callable[[str, float], None] | None = None,
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
        authorize_preflight_alias: Exact-key revalidation and alias lookup for Chat preflight.
        classification_memo: Bounded snapshot parser memo owned by the store.
        serving_snapshot_max_bytes: Serving limit for each local snapshot file.
        alias_not_granted_error: Store-specific error raised for an inactive grant.

    Returns:
        Organization, identity, key and alias row, plus the request ID and chain witness.
    """

    def read_alias(
        connection: sqlite3.Connection,
        *,
        readonly: bool = False,
    ) -> tuple[str, str, str, sqlite3.Row | None]:
        """Authenticate and resolve the active row within one SQLite snapshot."""
        if preauthenticated_key is not None:
            organization_id, identity_id, key_id, row = authorize_preflight_alias(
                connection,
                raw_key,
                preauthenticated_key,
                alias,
                update_last_used=not readonly,
                timing_recorder=timing_recorder,
            )
            return organization_id, identity_id, key_id, row

        authentication_started = time.monotonic()
        authenticate_key = authenticate_readonly if readonly else authenticate
        organization_id, identity_id, key_id = authenticate_key(connection, raw_key)
        if timing_recorder is not None:
            timing_recorder("sqlite_key_authentication_ms", authentication_started)
        alias_lookup_started = time.monotonic()
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
        if timing_recorder is not None:
            timing_recorder("sqlite_alias_lookup_ms", alias_lookup_started)
        return organization_id, identity_id, key_id, row

    authority_transaction_started = time.monotonic()
    with connect() as reader:
        try:
            try:
                # Fresh keys only read authority. Deferred mode avoids serializing
                # concurrent admissions on SQLite's writer lock.
                with transaction(
                    connection=reader,
                    immediate=False,
                    busy_timeout_ms=busy_timeout_ms,
                ) as connection:
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
                # write upgrade. Re-read authority from a fresh snapshot without
                # refreshing coarse last-used telemetry instead of waiting for a
                # long writer timeout.
                with transaction(connection=reader, immediate=False) as connection:
                    organization_id, identity_id, key_id, row = read_alias(
                        connection,
                        readonly=True,
                    )
        finally:
            if timing_recorder is not None:
                timing_recorder("sqlite_authority_transaction_ms", authority_transaction_started)
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
        chain_snapshot_started = time.monotonic()
        try:
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
        finally:
            if timing_recorder is not None:
                timing_recorder("sqlite_chain_snapshot_preparation_ms", chain_snapshot_started)
    return organization_id, identity_id, key_id, row, request_id, witness
