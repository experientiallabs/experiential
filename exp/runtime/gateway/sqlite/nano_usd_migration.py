"""SQLite v20: the ledger's money unit moves from integer micro-USD to integer nano-USD.

Every money column is renamed ``*_nano_usd`` in place (SQLite rewrites the CHECK
constraints with the column) and every stored amount, plus every frozen
per-million-token rate on ``gateway_attempts``, is multiplied by exactly 1000,
NULL staying NULL. Amounts are guarded BEFORE the multiply: a micro value
whose nano value would not fit a signed 64-bit integer fails the whole migration
closed (the database stays at v19) instead of being wrapped or, as SQLite
arithmetic would otherwise do, coerced to a REAL that the STRICT tables then
refuse mid-statement. Migrations 1-19 keep their historical micro-USD column
names verbatim: a fresh database replays them and is renamed here, so there is
exactly one authoring of every statement and no ``micro`` alias survives.
"""

from __future__ import annotations

import sqlite3

NANO_USD_PER_MICRO_USD = 1_000
_MAXIMUM_NANO_USD = 9_223_372_036_854_775_807

MONEY_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gateway_attempts",
        (
            "estimated_cost_micro_usd",
            "budget_reserved_micro_usd",
            "budget_settled_micro_usd",
            "counterfactual_cost_micro_usd",
        ),
    ),
    ("gateway_monthly_budgets", ("limit_micro_usd", "reserved_micro_usd", "settled_micro_usd")),
    ("gateway_attempt_budget_charges", ("reserved_micro_usd", "settled_micro_usd")),
)
"""Every v19 money column by table, under its historical micro-USD name."""

RATE_COLUMNS: tuple[str, ...] = (
    "input_rate",
    "cached_input_rate",
    "output_rate",
    "reasoning_rate",
    "long_context_input_rate",
    "long_context_cached_input_rate",
    "long_context_output_rate",
    "long_context_reasoning_rate",
    "preferred_input_rate",
    "preferred_cached_input_rate",
    "preferred_output_rate",
    "preferred_reasoning_rate",
)
"""The frozen per-million-token rates on ``gateway_attempts``.

Their names carry no unit, so they keep them, but their VALUES were micro-USD
per million tokens at v19 and settle/retry read them as nano-USD per million
after v20, so they are scaled by the same 1000 (a migrated dispatched row then
settles from usage at the right value, and a retried reservation compares its
frozen rates with the nano catalog rates and stays idempotent).
"""


class NanoUsdMigrationError(RuntimeError):
    """A stored micro-USD amount cannot be carried into the nano-USD ledger."""


def nano_usd_column_name(micro_column: str) -> str:
    """Return the v20 name of one v19 money column."""
    return micro_column.replace("_micro_usd", "_nano_usd")


def migrate_money_to_nano_usd(connection: sqlite3.Connection) -> None:
    """Rename every micro-USD money column to nano-USD and scale every amount and rate by 1000.

    Args:
        connection: Connection inside the migration's exclusive transaction.

    Raises:
        NanoUsdMigrationError: A stored amount would exceed the int8 column in nano-USD.
    """
    scaled: list[tuple[str, str]] = [
        (table, column) for table, columns in MONEY_COLUMNS for column in columns
    ]
    scaled.extend(("gateway_attempts", column) for column in RATE_COLUMNS)
    limit = _MAXIMUM_NANO_USD // NANO_USD_PER_MICRO_USD
    for table, column in scaled:
        over = connection.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} > ?",  # noqa: S608 - fixed names
            (limit,),
        ).fetchone()
        if over is not None and int(over[0]) > 0:
            raise NanoUsdMigrationError(
                f"{table}.{column} holds {int(over[0])} amount(s) that do not fit "
                "a signed 64-bit integer in nano-USD; refusing to migrate the ledger"
            )
    for table, columns in MONEY_COLUMNS:
        for micro_column in columns:
            nano_column = nano_usd_column_name(micro_column)
            connection.execute(f"ALTER TABLE {table} RENAME COLUMN {micro_column} TO {nano_column}")
    for table, column in scaled:
        column = column if column in RATE_COLUMNS else nano_usd_column_name(column)
        connection.execute(
            f"UPDATE {table} SET {column} = {column} * {NANO_USD_PER_MICRO_USD} "  # noqa: S608 - fixed names
            f"WHERE {column} IS NOT NULL"
        )
