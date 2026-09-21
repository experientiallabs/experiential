"""Tests for the v20 micro-USD to nano-USD ledger migration."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from exp.runtime.gateway.sqlite import migrations
from exp.runtime.gateway.sqlite.migrations import (
    SCHEMA_VERSION,
    GatewaySchemaError,
    connect_database,
    initialize_database,
)
from exp.runtime.gateway.sqlite.nano_usd_migration import (
    MONEY_COLUMNS,
    NANO_USD_PER_MICRO_USD,
    RATE_COLUMNS,
    nano_usd_column_name,
)


def _replay_history(connection: sqlite3.Connection, *, upto: int) -> None:
    """Replay the plain-SQL migrations ``1..upto-1`` onto one raw connection."""
    for version in range(1, upto):
        for step in migrations._MIGRATIONS[version]:
            assert isinstance(step, str), f"migration {version} is not plain SQL"
            connection.execute(step)


def test_every_v19_money_column_is_named_and_renamed_by_rule() -> None:
    """The column inventory is the migration's contract with the readers."""
    assert NANO_USD_PER_MICRO_USD == 1_000
    assert len(RATE_COLUMNS) == 12 and all(column.endswith("_rate") for column in RATE_COLUMNS)
    for _table, columns in MONEY_COLUMNS:
        for column in columns:
            assert column.endswith("_micro_usd")
            assert nano_usd_column_name(column) == column[: -len("_micro_usd")] + "_nano_usd"


def _seed_v19_money_rows(path: Path, *, estimated_cost: int) -> None:
    """Build one v19 (micro-USD) database holding one attempt, one monthly
    budget, and one budget charge, each with money in every column."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    connection = connect_database(path)
    try:
        connection.execute("BEGIN EXCLUSIVE")
        _replay_history(connection, upto=20)
        connection.execute("PRAGMA user_version = 19")
        seed_statements = """
            INSERT INTO organizations VALUES ('org', 'org', 'Org', 1, 't', 't');
            INSERT INTO identities VALUES ('id', 'org', 'Identity', NULL, 1, 't', 't');
            INSERT INTO virtual_keys (
                key_id, organization_id, identity_id, prefix,
                fingerprint_version, fingerprint_sha256, created_at
            ) VALUES ('key', 'org', 'id', 'pfx', 1, '{fingerprint}', 't');
            INSERT INTO catalog_snapshot_refs VALUES ('snap', 'org', '{digest}', 't');
            INSERT INTO gateway_aliases (
                alias_id, organization_id, alias_name, active_revision_id,
                created_at, updated_at
            ) VALUES ('alias', 'org', 'alias', NULL, 't', 't');
            INSERT INTO alias_revisions (
                revision_id, organization_id, alias_id, revision_number,
                target_kind, pool_id, catalog_sha256, snapshot_ref, created_at
            ) VALUES ('rev', 'org', 'alias', 1, 'direct', 'pool', '{digest}', 'snap', 't');
            INSERT INTO gateway_requests (
                request_id, organization_id, identity_id, key_id, alias_id,
                alias_revision_id, api_surface, canonical_request_sha256,
                accepted_at, deadline_at
            ) VALUES (
                'req-1', 'org', 'id', 'key', 'alias', 'rev', 'chat_completions',
                '{digest}', 't', 't'
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start,
                estimated_cost_micro_usd, budget_reserved_micro_usd,
                budget_settled_micro_usd, counterfactual_cost_micro_usd,
                input_rate, cached_input_rate, output_rate, reasoning_rate,
                long_context_input_rate, long_context_output_rate,
                preferred_input_rate, preferred_output_rate
            ) VALUES (
                'att-1', 'req-1', 'org', 0, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'completed', 't', '2026-08-01T00:00:00+00:00',
                {estimated_cost}, 9, 7, 5,
                1000000, 100000, 2000000, NULL,
                2500000, 15000000,
                500000, 1000000
            );
            INSERT INTO gateway_attempts (
                attempt_id, request_id, organization_id, attempt_ordinal,
                route_depth, deployment_id, provider, exact_model_id, pool_id,
                catalog_sha256, state, started_at, budget_period_start
            ) VALUES (
                'att-2', 'req-1', 'org', 1, 0, 'deploy', 'provider', 'exact',
                'pool', '{digest}', 'failed', 't', '2026-08-01T00:00:00+00:00'
            );
            INSERT INTO gateway_monthly_budgets (
                budget_id, organization_id, period_start, scope_kind, scope_key,
                limit_micro_usd, reserved_micro_usd, settled_micro_usd, created_at, updated_at
            ) VALUES (
                'budget-1', 'org', '2026-08-01T00:00:00+00:00', 'team', 'team',
                1000, 30, 20, 't', 't'
            );
            INSERT INTO gateway_attempt_budget_charges (
                budget_id, attempt_id, reserved_micro_usd, settled_micro_usd
            ) VALUES ('budget-1', 'att-1', NULL, 7);
            """.format(fingerprint="a" * 64, digest="b" * 64, estimated_cost=estimated_cost)
        for statement in seed_statements.split(";"):
            if statement.strip():
                connection.execute(statement)
        connection.execute("COMMIT")
    finally:
        connection.close()


def test_v20_migration_renames_every_money_column_to_nano_usd_and_scales_rows(
    tmp_path: Path,
) -> None:
    """The v20 migration moves the local ledger from integer micro-USD to
    integer nano-USD in place: every money column is renamed ``*_nano_usd``,
    every stored amount is multiplied by exactly 1000 (NULL stays NULL), the
    CHECK constraints follow the rename, and no ``micro_usd`` column survives
    on any table."""
    path = tmp_path / "gateway.db"
    _seed_v19_money_rows(path, estimated_cost=1_234)

    backup = initialize_database(path)

    assert backup is not None and backup.exists()
    migrated = connect_database(path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert SCHEMA_VERSION >= 20
        assert migrated.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert migrated.execute("PRAGMA foreign_key_check").fetchall() == []
        for table in (
            "gateway_attempts",
            "gateway_monthly_budgets",
            "gateway_attempt_budget_charges",
        ):
            names = {str(row[1]) for row in migrated.execute(f"PRAGMA table_info({table})")}
            assert not {name for name in names if "micro_usd" in name}, table
        assert tuple(
            migrated.execute(
                """
                SELECT estimated_cost_nano_usd, budget_reserved_nano_usd,
                       budget_settled_nano_usd, counterfactual_cost_nano_usd
                FROM gateway_attempts WHERE attempt_id = 'att-1'
                """
            ).fetchone()
        ) == (1_234_000, 9_000, 7_000, 5_000)
        assert tuple(
            migrated.execute(
                """
                SELECT estimated_cost_nano_usd, budget_reserved_nano_usd,
                       budget_settled_nano_usd, counterfactual_cost_nano_usd
                FROM gateway_attempts WHERE attempt_id = 'att-2'
                """
            ).fetchone()
        ) == (None, None, None, None)
        assert tuple(
            migrated.execute(
                "SELECT limit_nano_usd, reserved_nano_usd, settled_nano_usd "
                "FROM gateway_monthly_budgets WHERE budget_id = 'budget-1'"
            ).fetchone()
        ) == (1_000_000, 30_000, 20_000)
        assert tuple(
            migrated.execute(
                "SELECT reserved_nano_usd, settled_nano_usd "
                "FROM gateway_attempt_budget_charges WHERE attempt_id = 'att-1'"
            ).fetchone()
        ) == (None, 7_000)
        # The frozen per-million rates (unit-free names, micro values at v19)
        # are scaled with the amounts, NULL staying NULL.
        assert tuple(
            migrated.execute(
                """
                SELECT input_rate, cached_input_rate, output_rate, reasoning_rate,
                       long_context_input_rate, long_context_cached_input_rate,
                       long_context_output_rate, long_context_reasoning_rate,
                       preferred_input_rate, preferred_cached_input_rate,
                       preferred_output_rate, preferred_reasoning_rate
                FROM gateway_attempts WHERE attempt_id = 'att-1'
                """
            ).fetchone()
        ) == (
            1_000_000_000,
            100_000_000,
            2_000_000_000,
            None,
            2_500_000_000,
            None,
            15_000_000_000,
            None,
            500_000_000,
            None,
            1_000_000_000,
            None,
        )
        # Every stored amount is still a typed integer (STRICT refuses a REAL).
        assert tuple(
            migrated.execute(
                "SELECT typeof(estimated_cost_nano_usd) FROM gateway_attempts "
                "WHERE attempt_id = 'att-1'"
            ).fetchone()
        ) == ("integer",)
        # The renamed CHECK still refuses a negative amount.
        with pytest.raises(sqlite3.IntegrityError):
            migrated.execute(
                "UPDATE gateway_monthly_budgets SET settled_nano_usd = -1 "
                "WHERE budget_id = 'budget-1'"
            )
    finally:
        migrated.close()
    prior = sqlite3.connect(backup)
    try:
        assert prior.execute("PRAGMA user_version").fetchone() == (19,)
        assert prior.execute(
            "SELECT estimated_cost_micro_usd FROM gateway_attempts WHERE attempt_id = 'att-1'"
        ).fetchone() == (1_234,)
    finally:
        prior.close()


def test_v20_migration_refuses_a_micro_usd_amount_that_overflows_int64_at_nano(
    tmp_path: Path,
) -> None:
    """A stored micro-USD amount whose nano-USD value does not fit a signed
    64-bit integer fails the migration closed (the live database is left at
    v19, untouched) instead of being wrapped, coerced to a REAL, or dropped."""
    path = tmp_path / "gateway.db"
    _seed_v19_money_rows(path, estimated_cost=2**63 // 1000 + 1)

    with pytest.raises(GatewaySchemaError, match="nano-USD"):
        initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (19,)
        assert connection.execute(
            "SELECT estimated_cost_micro_usd FROM gateway_attempts WHERE attempt_id = 'att-1'"
        ).fetchone() == (2**63 // 1000 + 1,)
    finally:
        connection.close()
