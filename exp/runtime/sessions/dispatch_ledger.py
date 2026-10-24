"""Global SQLite provider-invocation ledger: one atomic slot per actual provider invocation.

Each invocation claims one slot in a ``BEGIN IMMEDIATE`` transaction before the network call. The
slot records a fresh attempt identifier, the phase configuration digest, and the sequence number
used to name receipts. The transaction counts every existing slot, including slots that later
failed or stayed ambiguous, so the cap holds across all sessions and processes.

The ledger does not identify sessions or steps and does not key on request bodies. Two
byte-identical requests are two independent invocations, each with its own slot. Slots are never
deleted, and a slot survives a crash after claim, so a crashed invocation still consumes the cap.
SQL triggers reject slot edits and deletions. This guards accidental database mutations; an
operator with write access to the file can still replace it or remove its guards.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final
from uuid import uuid4

from exp.common.sqlite.connection import connect_database

SCHEMA_VERSION: Final = 3
"""Ledger schema identity stored in ``PRAGMA user_version``."""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DispatchCapExhaustedError(RuntimeError):
    """The global provider cap is spent, so no further invocation may start."""


class LedgerSchemaError(RuntimeError):
    """The database is not a ledger of this schema, so it is refused rather than migrated."""


class DispatchLedger:
    """Durable, cross-process invocation counter with a hard cap.

    Args:
        path: SQLite file. Created with its schema when absent.
        cap: Maximum number of invocation slots ever claimed in this database.
        phase_config_sha256: Digest of the phase configuration every slot is recorded under.

    Raises:
        ValueError: If ``cap`` is not positive or the digest is malformed.
        LedgerSchemaError: If an existing file has a different schema.
    """

    def __init__(self, path: Path, cap: int, phase_config_sha256: str) -> None:
        """Create or open the ledger, validating the cap and the phase digest.

        Args:
            path: SQLite file path.
            cap: Global provider-call cap.
            phase_config_sha256: Lowercase hex SHA-256 of the phase configuration.

        Raises:
            ValueError: If ``cap`` is not positive or the digest is malformed.
            LedgerSchemaError: If the schema version does not match.
        """
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError("global provider cap must be a positive integer")
        if not _SHA256.fullmatch(phase_config_sha256):
            raise ValueError("phase_config_sha256 must be a lowercase hex SHA-256 digest")
        self._path = path
        self._cap = cap
        self._phase = phase_config_sha256
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = connect_database(path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                existing = connection.execute(
                    "SELECT COUNT(*) FROM sqlite_master"
                    " WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                ).fetchone()[0]
                if existing:
                    raise LedgerSchemaError(
                        "database has tables but no ledger schema; use a fresh ledger path"
                    )
                connection.execute(
                    """CREATE TABLE registration (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        cap INTEGER NOT NULL CHECK (cap > 0),
                        phase_config_sha256 TEXT NOT NULL
                    ) STRICT"""
                )
                connection.execute(
                    "INSERT INTO registration VALUES (1, ?, ?)", (cap, phase_config_sha256)
                )
                connection.execute(
                    """CREATE TRIGGER registration_no_update BEFORE UPDATE ON registration
                    BEGIN SELECT RAISE(ABORT, 'ledger registration is immutable'); END"""
                )
                connection.execute(
                    """CREATE TRIGGER registration_no_delete BEFORE DELETE ON registration
                    BEGIN SELECT RAISE(ABORT, 'ledger registration is immutable'); END"""
                )
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS invocation_slots (
                        seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        attempt_id TEXT NOT NULL UNIQUE,
                        phase_config_sha256 TEXT NOT NULL,
                        claimed_at TEXT NOT NULL,
                        receipt_state TEXT NOT NULL CHECK (receipt_state IN ('claimed'))
                    ) STRICT"""
                )
                connection.execute(
                    """CREATE TRIGGER invocation_slots_no_update BEFORE UPDATE ON invocation_slots
                    BEGIN SELECT RAISE(ABORT, 'invocation slots are append-only'); END"""
                )
                connection.execute(
                    """CREATE TRIGGER invocation_slots_no_delete BEFORE DELETE ON invocation_slots
                    BEGIN SELECT RAISE(ABORT, 'invocation slots are append-only'); END"""
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION:
                raise LedgerSchemaError(
                    f"ledger schema {version} is not {SCHEMA_VERSION}; use a fresh ledger path"
                )
            registration = connection.execute(
                "SELECT cap, phase_config_sha256 FROM registration WHERE singleton = 1"
            ).fetchone()
            if registration is None or tuple(registration) != (cap, phase_config_sha256):
                raise LedgerSchemaError(
                    "ledger registration differs from the requested cap or phase; "
                    "use its original configuration or a fresh registered ledger path"
                )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @property
    def cap(self) -> int:
        """Return the configured global cap.

        Returns:
            The cap.
        """
        return self._cap

    def used(self) -> int:
        """Return how many invocation slots have been claimed.

        Returns:
            The count of claimed slots, including failed and ambiguous ones.
        """
        connection = connect_database(self._path)
        try:
            return int(connection.execute("SELECT COUNT(*) FROM invocation_slots").fetchone()[0])
        finally:
            connection.close()

    def claim(self, claimed_at: str) -> tuple[int, str]:
        """Claim one slot before an invocation, or refuse without writing.

        The count and the insert happen in one ``BEGIN IMMEDIATE`` transaction. The caller makes
        the network call only after this returns.

        Args:
            claimed_at: ISO-8601 timestamp recorded with the slot.

        Returns:
            The slot sequence number, which names the invocation's receipts, and the fresh
            attempt identifier.

        Raises:
            DispatchCapExhaustedError: If the cap is already reached.
        """
        attempt_id = uuid4().hex
        connection = connect_database(self._path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                used = int(
                    connection.execute("SELECT COUNT(*) FROM invocation_slots").fetchone()[0]
                )
                if used >= self._cap:
                    raise DispatchCapExhaustedError(
                        f"global provider cap {self._cap} is spent ({used} slots claimed)"
                    )
                cursor = connection.execute(
                    "INSERT INTO invocation_slots"
                    " (attempt_id, phase_config_sha256, claimed_at, receipt_state)"
                    " VALUES (?, ?, ?, 'claimed')",
                    (attempt_id, self._phase, claimed_at),
                )
                seq = int(cursor.lastrowid or 0)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
            return seq, attempt_id
        finally:
            connection.close()
