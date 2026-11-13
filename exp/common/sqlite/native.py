"""Local SQL access through the same SQLite library as native gateway accounting.

Separate SQLite libraries in one process do not share descriptor and WAL lock
bookkeeping. This facade keeps initial authority reads and native request writes
on one implementation while retaining the local SQL connection interface.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Literal, cast, overload

from exp_gateway_native import NativeSQLiteConnection, NativeSQLiteCursor

SQLValue = str | bytes | int | float | None
Parameters = Sequence[SQLValue] | Mapping[str, SQLValue]


def native_connection(connection: sqlite3.Connection) -> NativeSQLiteConnection:
    """Require a connection backed by the gateway's shared native SQLite library."""
    if not isinstance(connection, Connection):
        raise TypeError("Accounting requires a native SQLite connection")
    return connection.native


class Row(Sequence[SQLValue]):
    """Named, case-insensitive immutable SQL values with positional iteration."""

    def __init__(self, names: tuple[str, ...], values: tuple[SQLValue, ...]) -> None:
        """Keep column order and duplicate-name behavior identical to SQLite rows."""
        self._names = names
        self._values = values

    @overload
    def __getitem__(self, key: int | str) -> SQLValue: ...

    @overload
    def __getitem__(self, key: slice) -> tuple[SQLValue, ...]: ...

    def __getitem__(self, key: int | str | slice) -> SQLValue | tuple[SQLValue, ...]:
        """Read a position, slice or the first matching column name."""
        if isinstance(key, str):
            lowered = key.lower()
            for index, name in enumerate(self._names):
                if name.lower() == lowered:
                    return self._values[index]
            raise IndexError("No item with that key")
        return self._values[key]

    def __len__(self) -> int:
        """Return the number of selected columns."""
        return len(self._values)

    def keys(self) -> list[str]:
        """Expose column names for mapping conversion without changing iteration."""
        return list(self._names)

    def __eq__(self, other: object) -> bool:
        """Compare both column names and values, as sqlite3.Row does."""
        return (
            isinstance(other, Row) and self._names == other._names and self._values == other._values
        )

    def __hash__(self) -> int:
        """Hash the same immutable fields used by equality."""
        return hash((self._names, self._values))


class Cursor(Iterator[object]):
    """A bounded cursor that materializes only requested rows."""

    def __init__(self, connection: Connection) -> None:
        """Retain the connection and its row factory for this cursor."""
        self.connection = connection
        self.row_factory = connection.row_factory
        self.arraysize = 1
        self._native: NativeSQLiteCursor | None = None
        self._closed = False
        self._rowcount: int | None = None

    def execute(self, sql: str, parameters: Parameters = ()) -> Cursor:
        """Prepare one bound statement through the shared native library."""
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed cursor.")
        if self._native is not None:
            self._native.close()
        self._native = self.connection._native.execute(sql, parameters)
        self._rowcount = None
        return self

    def executemany(self, sql: str, parameters: Iterable[Parameters]) -> Cursor:
        """Apply a sequence inside the caller's existing transaction."""
        count = 0
        for arguments in parameters:
            self.execute(sql, arguments)
            count += max(0, self.rowcount)
        self._rowcount = count
        return self

    @property
    def description(self) -> tuple[tuple[object, ...], ...] | None:
        """Return DB-API column descriptors, or None without a result set."""
        names = () if self._native is None else tuple(self._native.columns)
        return tuple((name, None, None, None, None, None, None) for name in names) or None

    @property
    def rowcount(self) -> int:
        """Return changed rows for completed writes, or -1 for a result set."""
        if self._rowcount is not None:
            return self._rowcount
        return -1 if self._native is None else self._native.rowcount

    @property
    def lastrowid(self) -> int | None:
        """Return the connection's last successful inserted row identifier."""
        return None if self._native is None else self._native.lastrowid

    def fetchone(self) -> object | None:
        """Fetch one row and apply the cursor's frozen row factory."""
        if self._closed:
            raise sqlite3.ProgrammingError("Cannot operate on a closed cursor.")
        if self._native is None:
            return None
        row = self._native.fetchone()
        if row is None or self.row_factory is None:
            return row
        if self.row_factory is sqlite3.Row:
            return Row(tuple(self._native.columns), row)
        # sqlite3.Row is handled above; all remaining factories use the facade cursor.
        factory = cast("Callable[[Cursor, tuple[SQLValue, ...]], object]", self.row_factory)
        return factory(self, row)

    def fetchall(self) -> list[object]:
        """Consume the remaining result set when the caller explicitly requests it."""
        return list(self)

    def fetchmany(self, size: int | None = None) -> list[object]:
        """Read at most size rows without materializing the remaining result set."""
        count = self.arraysize if size is None else size
        if count < 0:
            raise ValueError("fetchmany size must be nonnegative")
        result = []
        for _ in range(count):
            row = self.fetchone()
            if row is None:
                break
            result.append(row)
        return result

    def __next__(self) -> object:
        """Yield one requested row or terminate iteration."""
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def close(self) -> None:
        """Finalize any prepared statement exactly once."""
        if self._native is not None:
            self._native.close()
        self._closed = True


class Connection:
    """Autocommit local SQL with explicit durable transaction boundaries."""

    def __init__(self, path: Path, busy_timeout_ms: int, mode: Literal["ro", "rw", "rwc"]) -> None:
        """Open one native connection without altering journal or schema policy."""
        self._native = NativeSQLiteConnection(path, busy_timeout_ms, mode)
        self.row_factory: (
            Callable[[Cursor, tuple[SQLValue, ...]], object] | type[sqlite3.Row] | None
        ) = None

    @property
    def native(self) -> NativeSQLiteConnection:
        """Expose the shared native handle to cold transactional accounting callers."""
        return self._native

    def cursor(self) -> Cursor:
        """Create a lazily prepared cursor with this connection's row factory."""
        return Cursor(self)

    def execute(self, sql: str, parameters: Parameters = ()) -> Cursor:
        """Execute one SQL statement with positional or named parameters."""
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[Parameters]) -> Cursor:
        """Execute repeated bound statements in the caller's transaction."""
        return self.cursor().executemany(sql, parameters)

    def executescript(self, sql: str) -> Cursor:
        """Execute trusted schema SQL; caller-owned transactions remain explicit."""
        self._native.executescript(sql)
        return self.cursor()

    @property
    def in_transaction(self) -> bool:
        """Report whether an explicit transaction remains open."""
        return self._native.in_transaction

    @property
    def total_changes(self) -> int:
        """Return this connection's cumulative changed-row count."""
        return self._native.total_changes

    def commit(self) -> None:
        """Commit a live transaction; autocommit connections need no write."""
        if self.in_transaction:
            self.execute("COMMIT")

    def rollback(self) -> None:
        """Roll back a live transaction without inventing a transaction."""
        if self.in_transaction:
            self.execute("ROLLBACK")

    def backup(self, target: Connection) -> None:
        """Copy a consistent database through SQLite's native backup API."""
        self._native.backup(target._native)

    def close(self) -> None:
        """Finalize every cursor and release the native database handle."""
        self._native.close()

    def __enter__(self) -> Connection:
        """Retain DB-API transaction context semantics without beginning a transaction."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Commit a successful transaction or roll it back on failure."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def connect(
    path: Path, *, busy_timeout_ms: int = 5000, mode: Literal["ro", "rw", "rwc"] = "rwc"
) -> sqlite3.Connection:
    """Retain the existing SQL interface at one checked adapter boundary.

    The native facade implements the connection, cursor and row operations used
    by local stores. Existing sqlite3 exception classes remain the error contract;
    the implementation never opens a second SQLite library on the same file.
    """
    return cast("sqlite3.Connection", Connection(path, busy_timeout_ms, mode))
