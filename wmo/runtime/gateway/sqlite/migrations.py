"""Forward-only SQLite schema initialization and guarded migration."""

from __future__ import annotations

import os
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 4


class GatewaySchemaError(RuntimeError):
    """The gateway database schema cannot be opened safely."""


_MIGRATION_1 = (
    """
    CREATE TABLE organizations (
        organization_id TEXT PRIMARY KEY,
        slug TEXT NOT NULL UNIQUE,
        display_name TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE identities (
        identity_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        display_name TEXT NOT NULL,
        description TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, identity_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
    """
    CREATE TABLE virtual_keys (
        key_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        prefix TEXT NOT NULL,
        fingerprint_version INTEGER NOT NULL CHECK (fingerprint_version > 0),
        fingerprint_sha256 TEXT NOT NULL CHECK (length(fingerprint_sha256) = 64),
        expires_at TEXT,
        revoked_at TEXT,
        last_used_at TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (fingerprint_version, fingerprint_sha256),
        UNIQUE (organization_id, prefix),
        UNIQUE (organization_id, key_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id)
    ) STRICT
    """,
    """
    CREATE TABLE catalog_snapshot_refs (
        snapshot_ref TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, catalog_sha256),
        UNIQUE (organization_id, snapshot_ref),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
    """
    CREATE TABLE gateway_aliases (
        alias_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        alias_name TEXT NOT NULL,
        active_revision_id TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, alias_name),
        UNIQUE (organization_id, alias_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id),
        FOREIGN KEY (organization_id, alias_id, active_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            DEFERRABLE INITIALLY DEFERRED
    ) STRICT
    """,
    """
    CREATE TABLE alias_revisions (
        revision_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        revision_number INTEGER NOT NULL CHECK (revision_number > 0),
        target_kind TEXT NOT NULL CHECK (target_kind IN ('direct', 'project')),
        pool_id TEXT,
        project_ref TEXT,
        activation_ref TEXT,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        snapshot_ref TEXT NOT NULL,
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, alias_id, revision_number),
        UNIQUE (organization_id, alias_id, revision_id),
        CHECK (
            (target_kind = 'direct' AND pool_id IS NOT NULL
                AND project_ref IS NULL AND activation_ref IS NULL)
            OR
            (target_kind = 'project' AND pool_id IS NULL
                AND project_ref IS NOT NULL AND activation_ref IS NOT NULL)
        ),
        FOREIGN KEY (organization_id, alias_id)
            REFERENCES gateway_aliases (organization_id, alias_id),
        FOREIGN KEY (organization_id, snapshot_ref)
            REFERENCES catalog_snapshot_refs (organization_id, snapshot_ref)
    ) STRICT
    """,
    """
    CREATE TABLE project_activation_bindings (
        organization_id TEXT NOT NULL,
        project_ref TEXT NOT NULL,
        activation_ref TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        revision_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, project_ref, activation_ref),
        UNIQUE (organization_id, alias_id),
        UNIQUE (organization_id, revision_id),
        FOREIGN KEY (organization_id, alias_id, revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            ON DELETE CASCADE
    ) STRICT
    """,
    """
    CREATE TABLE identity_alias_grants (
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, identity_id, alias_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, alias_id)
            REFERENCES gateway_aliases (organization_id, alias_id)
    ) STRICT
    """,
    """
    CREATE TABLE operation_receipts (
        organization_id TEXT NOT NULL,
        operation_id TEXT NOT NULL,
        operation_kind TEXT NOT NULL,
        request_sha256 TEXT NOT NULL CHECK (length(request_sha256) = 64),
        resource_kind TEXT NOT NULL,
        resource_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (organization_id, operation_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id)
    ) STRICT
    """,
)

_MIGRATION_2 = (
    """
    CREATE TABLE gateway_requests (
        request_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        identity_id TEXT NOT NULL,
        key_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        alias_revision_id TEXT NOT NULL,
        api_surface TEXT NOT NULL CHECK (api_surface IN ('chat_completions', 'responses')),
        canonical_request_sha256 TEXT NOT NULL CHECK (length(canonical_request_sha256) = 64),
        caller_operation_sha256 TEXT,
        accepted_at TEXT NOT NULL,
        deadline_at TEXT NOT NULL,
        terminal_state TEXT CHECK (
            terminal_state IS NULL OR terminal_state IN (
                'completed', 'failed', 'cancelled', 'incomplete',
                'expired_before_dispatch', 'unknown_after_crash'
            )
        ),
        terminal_at TEXT,
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        UNIQUE (organization_id, request_id),
        FOREIGN KEY (organization_id, identity_id)
            REFERENCES identities (organization_id, identity_id),
        FOREIGN KEY (organization_id, key_id)
            REFERENCES virtual_keys (organization_id, key_id),
        FOREIGN KEY (organization_id, alias_id, alias_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
    ) STRICT
    """,
    """
    CREATE INDEX gateway_requests_caller_operation
    ON gateway_requests (
        organization_id, identity_id, alias_revision_id, api_surface, caller_operation_sha256
    ) WHERE caller_operation_sha256 IS NOT NULL
    """,
    """
    CREATE TABLE gateway_attempts (
        attempt_id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        organization_id TEXT NOT NULL,
        route_depth INTEGER NOT NULL CHECK (route_depth >= 0),
        deployment_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        exact_model_id TEXT NOT NULL,
        pool_id TEXT NOT NULL,
        catalog_sha256 TEXT NOT NULL CHECK (length(catalog_sha256) = 64),
        pricing_source TEXT,
        pricing_effective_at TEXT,
        route_reason TEXT,
        fallback_reason TEXT,
        input_rate INTEGER CHECK (input_rate IS NULL OR input_rate >= 0),
        cached_input_rate INTEGER CHECK (cached_input_rate IS NULL OR cached_input_rate >= 0),
        output_rate INTEGER CHECK (output_rate IS NULL OR output_rate >= 0),
        reasoning_rate INTEGER CHECK (reasoning_rate IS NULL OR reasoning_rate >= 0),
        state TEXT NOT NULL CHECK (state IN (
            'dispatched', 'completed', 'failed', 'cancelled',
            'incomplete', 'unknown_after_crash'
        )),
        started_at TEXT NOT NULL,
        terminal_at TEXT,
        failure_class TEXT,
        input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
        cached_input_tokens INTEGER CHECK (
            cached_input_tokens IS NULL OR cached_input_tokens >= 0
        ),
        output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
        reasoning_tokens INTEGER CHECK (reasoning_tokens IS NULL OR reasoning_tokens >= 0),
        usage_source TEXT CHECK (
            usage_source IS NULL OR usage_source IN ('observed', 'estimated', 'unknown')
        ),
        estimated_cost_micro_usd INTEGER CHECK (
            estimated_cost_micro_usd IS NULL OR estimated_cost_micro_usd >= 0
        ),
        content_retained INTEGER NOT NULL DEFAULT 0 CHECK (content_retained = 0),
        UNIQUE (organization_id, attempt_id),
        UNIQUE (request_id, route_depth),
        FOREIGN KEY (organization_id, request_id)
            REFERENCES gateway_requests (organization_id, request_id)
    ) STRICT
    """,
    """
    CREATE INDEX gateway_attempts_usage
    ON gateway_attempts (organization_id, terminal_at, state)
    """,
    """
    CREATE INDEX gateway_requests_identity
    ON gateway_requests (organization_id, identity_id, accepted_at)
    """,
)

_MIGRATION_3 = (
    """
    ALTER TABLE gateway_attempts
    ADD COLUMN billing_source TEXT NOT NULL DEFAULT 'customer_managed'
    CHECK (billing_source IN ('customer_managed', 'host_managed'))
    """,
)

_MIGRATION_4 = (
    """
    CREATE TABLE provider_connections (
        connection_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        active_revision_id TEXT,
        active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (organization_id, connection_id),
        FOREIGN KEY (organization_id) REFERENCES organizations (organization_id),
        FOREIGN KEY (organization_id, connection_id, active_revision_id)
            REFERENCES provider_connection_revisions (
                organization_id, connection_id, revision_id
            )
            DEFERRABLE INITIALLY DEFERRED
    ) STRICT
    """,
    """
    CREATE TABLE provider_connection_revisions (
        revision_id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL,
        connection_id TEXT NOT NULL,
        revision_number INTEGER NOT NULL CHECK (revision_number > 0),
        provider TEXT NOT NULL,
        base_url TEXT,
        api_key_env TEXT,
        api_version TEXT,
        region TEXT,
        connection_sha256 TEXT NOT NULL CHECK (length(connection_sha256) = 64),
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, connection_id, revision_number),
        UNIQUE (organization_id, connection_id, revision_id),
        FOREIGN KEY (organization_id, connection_id)
            REFERENCES provider_connections (organization_id, connection_id)
    ) STRICT
    """,
    """
    CREATE TABLE alias_revision_provider_connections (
        organization_id TEXT NOT NULL,
        alias_id TEXT NOT NULL,
        alias_revision_id TEXT NOT NULL,
        connection_id TEXT NOT NULL,
        connection_revision_id TEXT NOT NULL,
        connection_sha256 TEXT NOT NULL CHECK (length(connection_sha256) = 64),
        created_at TEXT NOT NULL,
        PRIMARY KEY (
            organization_id, alias_id, alias_revision_id, connection_id
        ),
        FOREIGN KEY (organization_id, alias_id, alias_revision_id)
            REFERENCES alias_revisions (organization_id, alias_id, revision_id)
            ON DELETE CASCADE,
        FOREIGN KEY (
            organization_id, connection_id, connection_revision_id
        ) REFERENCES provider_connection_revisions (
            organization_id, connection_id, revision_id
        )
    ) STRICT
    """,
)

_MIGRATIONS = {
    1: _MIGRATION_1,
    2: _MIGRATION_2,
    3: _MIGRATION_3,
    4: _MIGRATION_4,
}


def connect_database(
    path: Path, *, busy_timeout_ms: int = 5_000, enable_wal: bool = True
) -> sqlite3.Connection:
    """Open one configured SQLite connection with mandatory safety pragmas.

    Args:
        path: Gateway database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.
        enable_wal: Whether to assert the supported journal mode after version checks.

    Returns:
        Configured connection with row-name access.
    """
    connection = sqlite3.connect(path, timeout=busy_timeout_ms / 1_000, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    if enable_wal:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
    return connection


def initialize_database(path: Path, *, busy_timeout_ms: int = 5_000) -> Path | None:
    """Create or forward-migrate one database without deleting incompatible state.

    Args:
        path: Gateway database path.
        busy_timeout_ms: Bounded lock wait in milliseconds.

    Returns:
        Backup path when an existing older schema was migrated, otherwise ``None``.

    Raises:
        GatewaySchemaError: State is corrupt, newer, or cannot migrate atomically.
    """
    _create_private_database_file(path)
    try:
        connection = connect_database(path, busy_timeout_ms=busy_timeout_ms, enable_wal=False)
    except sqlite3.DatabaseError as exc:
        raise GatewaySchemaError("gateway database is corrupt or unreadable") from exc
    backup: Path | None = None
    try:
        connection.execute("BEGIN EXCLUSIVE")
        try:
            _supported_schema_version(connection)
            connection.execute("COMMIT")
        except GatewaySchemaError:
            connection.execute("ROLLBACK")
            raise
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN EXCLUSIVE")
        try:
            version = _supported_schema_version(connection)
            if 0 < version < SCHEMA_VERSION:
                backup = _backup_database(path, version)
            for next_version in range(version + 1, SCHEMA_VERSION + 1):
                for statement in _MIGRATIONS[next_version]:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version = {next_version}")
            _require_schema_objects(connection)
            connection.execute("COMMIT")
        except GatewaySchemaError:
            connection.execute("ROLLBACK")
            raise
        except (sqlite3.DatabaseError, OSError) as exc:
            connection.execute("ROLLBACK")
            raise GatewaySchemaError("gateway database migration failed") from exc
    except sqlite3.DatabaseError as exc:
        raise GatewaySchemaError("gateway database is corrupt or unreadable") from exc
    finally:
        connection.close()
    os.chmod(path, 0o600)
    return backup


def _supported_schema_version(connection: sqlite3.Connection) -> int:
    """Read and validate schema state while the caller holds an exclusive transaction.

    Args:
        connection: Database connection inside an exclusive transaction.

    Returns:
        Supported current schema version.

    Raises:
        GatewaySchemaError: Integrity fails or the schema is newer than this code.
    """
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise GatewaySchemaError("gateway database failed integrity check")
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise GatewaySchemaError(
            f"gateway database schema {version} is newer than supported {SCHEMA_VERSION}"
        )
    return version


def _create_private_database_file(path: Path) -> None:
    """Create a missing database as a user-only regular file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise GatewaySchemaError("gateway database path must be a regular file")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise GatewaySchemaError("gateway database must not be group or world accessible")


def _backup_database(path: Path, version: int) -> Path:
    """Create a consistent mode-0600 SQLite backup before forward migration.

    The caller retains the exclusive migration transaction while this separate read
    connection copies the last committed WAL snapshot.

    Args:
        path: Database protected by the caller's exclusive transaction.
        version: Last committed schema version represented by the backup.

    Returns:
        Private path containing the consistent pre-migration snapshot.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup = path.with_name(f"{path.name}.backup-v{version}-{stamp}")
    descriptor = os.open(backup, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    source = sqlite3.connect(path)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    os.chmod(backup, 0o600)
    return backup


def _require_schema_objects(connection: sqlite3.Connection) -> None:
    """Refuse a version marker that does not have every required table."""
    expected = {
        "alias_revisions",
        "catalog_snapshot_refs",
        "gateway_aliases",
        "gateway_attempts",
        "gateway_requests",
        "identities",
        "identity_alias_grants",
        "operation_receipts",
        "organizations",
        "provider_connection_revisions",
        "provider_connections",
        "project_activation_bindings",
        "alias_revision_provider_connections",
        "virtual_keys",
    }
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    actual = {str(row[0]) for row in rows}
    if not expected.issubset(actual):
        raise GatewaySchemaError("gateway database schema marker does not match required tables")
