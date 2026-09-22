"""Tests for preserving durable accounting across cache-write schema extension."""

import sqlite3
from pathlib import Path

import pytest

from exp.common.sqlite.connection import connect_database
from exp.runtime.gateway.sqlite import migrations
from exp.runtime.gateway.sqlite.cache_write_migration import (
    CACHE_WRITE_COLUMNS,
    CACHE_WRITE_MIGRATION,
    migrate_cache_write,
)
from exp.runtime.gateway.sqlite.migrations import (
    GatewaySchemaError,
    initialize_database,
)
from exp.runtime.gateway.sqlite.migrations_test import _replay_history


def test_failed_cache_extension_rolls_back_columns_and_keeps_prior_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial DDL failure preserves schema 22 and the original committed data."""
    path = tmp_path / "gateway.db"
    path.touch(mode=0o600)
    connection = connect_database(path)
    try:
        _replay_history(connection, upto=23)
        connection.execute(
            "INSERT INTO organizations VALUES ('org', 'org', 'Original', 1, 't', 't')"
        )
        connection.execute("PRAGMA user_version = 22")
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setitem(migrations._MIGRATIONS, 23, (migrate_cache_write, "INVALID SQL"))
    with pytest.raises(GatewaySchemaError, match="migration failed") as failure:
        initialize_database(path)
    assert isinstance(failure.value.__cause__, sqlite3.OperationalError)
    connection = connect_database(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 22
        assert (
            connection.execute("SELECT display_name FROM organizations").fetchone()[0] == "Original"
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")}
        assert columns.isdisjoint(CACHE_WRITE_COLUMNS)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


@pytest.mark.parametrize("cache_layout", [False, True], ids=["stable-zdr", "cache-prerelease"])
def test_both_published_v22_layouts_preserve_attempt_evidence(
    tmp_path: Path, cache_layout: bool
) -> None:
    """Both released layouts gain the missing extension without repricing rows.

    Args:
        tmp_path: Isolated ledger directory.
        cache_layout: Whether the source is the cache-write prerelease layout.
    """
    path = tmp_path / "gateway.db"
    path.touch(mode=0o600)
    connection = connect_database(path)
    try:
        _replay_history(connection, upto=22)
        statements = CACHE_WRITE_MIGRATION if cache_layout else migrations._MIGRATIONS[22]
        for statement in statements:
            assert isinstance(statement, str)
            connection.execute(statement)
        _seed_attempt(connection)
        if cache_layout:
            connection.execute(
                "UPDATE gateway_attempts SET cache_creation_input_rate = 3750000000, "
                "cache_creation_1h_input_rate = 6000000000, cache_creation_input_tokens = 600, "
                "cache_creation_1h_input_tokens = 200"
            )
        else:
            connection.execute("UPDATE gateway_attempts SET upstream_provider = 'Azure'")
        connection.execute("PRAGMA user_version = 22")
        before = dict(connection.execute("SELECT * FROM gateway_attempts").fetchone())
    finally:
        connection.close()

    backup = initialize_database(path)
    assert backup is not None
    with sqlite3.connect(backup) as original:
        original.row_factory = sqlite3.Row
        assert original.execute("PRAGMA user_version").fetchone()[0] == 22
        assert dict(original.execute("SELECT * FROM gateway_attempts").fetchone()) == before
    connection = connect_database(path)
    try:
        after = dict(connection.execute("SELECT * FROM gateway_attempts").fetchone())
        assert {column: after[column] for column in before} == before
        added = {"upstream_provider"} if cache_layout else set(CACHE_WRITE_COLUMNS)
        assert after.keys() - before.keys() == added
        assert all(after[column] is None for column in added)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 23
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
    assert initialize_database(path) is None


def _seed_attempt(connection: sqlite3.Connection) -> None:
    """Seed a valid durable attempt with a frozen price and no external calls.

    Args:
        connection: Isolated test ledger containing the source schema.
    """
    digest = "b" * 64
    connection.executescript(
        f"""
        INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
        INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
        INSERT INTO virtual_keys (
            key_id, organization_id, identity_id, prefix,
            fingerprint_version, fingerprint_sha256, created_at
        ) VALUES ('key', 'org', 'id', 'pfx', 1, '{"a" * 64}', 't');
        INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
        INSERT INTO gateway_aliases (
            alias_id, organization_id, alias_name, created_at, updated_at
        ) VALUES ('alias', 'org', 'alias', 't', 't');
        INSERT INTO alias_revisions (
            revision_id, organization_id, alias_id, revision_number,
            target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
        ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
        INSERT INTO gateway_requests (
            request_id, organization_id, identity_id, key_id, alias_id,
            alias_revision_id, api_surface, canonical_request_sha256, accepted_at, deadline_at
        ) VALUES ('req', 'org', 'id', 'key', 'alias', 'rev',
                  'chat_completions', '{digest}', 't', 't');
        INSERT INTO gateway_attempts (
            attempt_id, request_id, organization_id, attempt_ordinal, route_depth,
            deployment_id, provider, exact_model_id, pool_id, catalog_sha256,
            state, started_at, budget_period_start, input_rate
        ) VALUES ('att', 'req', 'org', 0, 0, 'dep', 'anthropic', 'model', 'pool',
                  '{digest}', 'completed', 't', '2026-09-01T00:00:00+00:00', 3000000000);
        """  # noqa: S608 - fixed test-only values.
    )


def test_partial_cache_schema_is_rejected_without_further_mutation(tmp_path: Path) -> None:
    """An incomplete extension fails closed and retains its original schema marker.

    Args:
        tmp_path: Isolated ledger directory.
    """
    path = tmp_path / "gateway.db"
    path.touch(mode=0o600)
    connection = connect_database(path)
    try:
        _replay_history(connection, upto=22)
        connection.execute(CACHE_WRITE_MIGRATION[0])
        connection.execute("PRAGMA user_version = 22")
        before = connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    finally:
        connection.close()
    with pytest.raises(GatewaySchemaError, match="migration failed") as failure:
        initialize_database(path)
    assert "cache-write schema is incomplete" in str(failure.value.__cause__)
    connection = connect_database(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 22
        assert (
            connection.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == before
        )
    finally:
        connection.close()


def test_fresh_extension_and_reinitialization_have_one_nullable_column_set(tmp_path: Path) -> None:
    """Fresh setup and repeated initialization preserve one bounded schema extension."""
    path = tmp_path / "gateway.db"
    initialize_database(path)
    assert initialize_database(path) is None
    connection = connect_database(path)
    try:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(gateway_attempts)")]
        assert all(columns.count(name) == 1 for name in CACHE_WRITE_COLUMNS)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
