"""SQLite v20: the ledger's money unit moves from integer micro-USD to integer nano-USD.

Every money column is renamed ``*_nano_usd`` in place (SQLite rewrites the CHECK
constraints with the column) and every stored amount is multiplied by exactly
1000, NULL staying NULL. Amounts are guarded BEFORE the multiply: a micro value
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


class NanoUsdMigrationError(RuntimeError):
    """A stored micro-USD amount cannot be carried into the nano-USD ledger."""


def nano_usd_column_name(micro_column: str) -> str:
    """Return the v20 name of one v19 money column."""
    return micro_column.replace("_micro_usd", "_nano_usd")


def migrate_money_to_nano_usd(connection: sqlite3.Connection) -> None:
    """Rename every micro-USD money column to nano-USD and scale its rows by 1000.

    Args:
        connection: Connection inside the migration's exclusive transaction.

    Raises:
        NanoUsdMigrationError: A stored amount would exceed the int8 column in nano-USD.
    """
    limit = _MAXIMUM_NANO_USD // NANO_USD_PER_MICRO_USD
    for table, columns in MONEY_COLUMNS:
        for micro_column in columns:
            over = connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {micro_column} > ?",  # noqa: S608 - fixed names
                (limit,),
            ).fetchone()
            if over is not None and int(over[0]) > 0:
                raise NanoUsdMigrationError(
                    f"{table}.{micro_column} holds {int(over[0])} amount(s) that do not fit "
                    "a signed 64-bit integer in nano-USD; refusing to migrate the ledger"
                )
    for table, columns in MONEY_COLUMNS:
        for micro_column in columns:
            nano_column = nano_usd_column_name(micro_column)
            connection.execute(f"ALTER TABLE {table} RENAME COLUMN {micro_column} TO {nano_column}")
            connection.execute(
                f"UPDATE {table} SET {nano_column} = {nano_column} * {NANO_USD_PER_MICRO_USD} "  # noqa: S608 - fixed names
                f"WHERE {nano_column} IS NOT NULL"
            )
