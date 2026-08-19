"""Content-free reconciliation for one-time virtual-key delivery commits."""

from __future__ import annotations

import hmac
import sqlite3
from pathlib import Path

from wmo.common.core.artifacts import Sha256
from wmo.runtime.gateway.sqlite.migrations import connect_database


def reconcile_key_issue(
    database_path: Path,
    *,
    busy_timeout_ms: int,
    organization_id: str,
    identity_id: str,
    key_id: str,
    prefix: str,
    fingerprint_version: int,
    fingerprint: Sha256,
    expires_at: str | None,
    created_at: str,
    operation_id: str | None,
    request_sha256: Sha256,
) -> bool | None:
    """Reconcile an ambiguous key commit from exact content-free authority rows.

    Args:
        database_path: Gateway authority database.
        busy_timeout_ms: SQLite lock wait bound.
        organization_id: Owning tenant.
        identity_id: Credential owner.
        key_id: Stable non-secret key identifier.
        prefix: Public key prefix.
        fingerprint_version: Pepper version used by the key fingerprint.
        fingerprint: HMAC fingerprint persisted for authentication.
        expires_at: Optional canonical expiry.
        created_at: Canonical creation time.
        operation_id: Optional retry-safe operation identifier.
        request_sha256: Content-free operation request digest.

    Returns:
        ``True`` for the exact committed key and receipt, ``False`` when both
        are absent, or ``None`` when the durable outcome cannot be proven.
    """
    try:
        connection = connect_database(database_path, busy_timeout_ms=busy_timeout_ms)
        try:
            key = connection.execute(
                """
                SELECT identity_id, prefix, fingerprint_version, fingerprint_sha256,
                       expires_at, created_at
                FROM virtual_keys
                WHERE organization_id = ? AND key_id = ?
                """,
                (organization_id, key_id),
            ).fetchone()
            receipt = None
            if operation_id is not None:
                receipt = connection.execute(
                    """
                    SELECT operation_kind, request_sha256, resource_kind, resource_id
                    FROM operation_receipts
                    WHERE organization_id = ? AND operation_id = ?
                    """,
                    (organization_id, operation_id),
                ).fetchone()
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return None
    if key is None:
        return False if receipt is None else None
    key_matches = (
        str(key["identity_id"]) == identity_id
        and str(key["prefix"]) == prefix
        and int(key["fingerprint_version"]) == fingerprint_version
        and hmac.compare_digest(str(key["fingerprint_sha256"]), fingerprint)
        and key["expires_at"] == expires_at
        and str(key["created_at"]) == created_at
    )
    if not key_matches:
        return None
    if operation_id is None:
        return True
    if receipt is None:
        return None
    receipt_matches = (
        str(receipt["operation_kind"]) == "issue_virtual_key"
        and str(receipt["request_sha256"]) == request_sha256
        and str(receipt["resource_kind"]) == "virtual_key"
        and str(receipt["resource_id"]) == key_id
    )
    return True if receipt_matches else None
