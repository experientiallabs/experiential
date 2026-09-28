"""Versioned mutable records and append-only execution events scoped to a project."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from exp.common.project.database import has_project_schema, project_connection


class ProjectRecordError(ValueError):
    """A durable record is corrupt or a conditional mutation conflicts."""


def verified_payload(payload: bytes, digest: str) -> bytes:
    """Return exact stored bytes only when they match their recorded SHA-256."""
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ProjectRecordError(
            "Project record digest mismatch; restore verified evidence before resuming."
        )
    return payload


class ProjectRecords:
    """Atomic project records and ordered event streams.

    Attributes:
        root: EXP root containing the shared database.
        project_id: Explicit project namespace.
        namespace: Domain-owned record or event namespace.
    """

    def __init__(self, root: Path, project_id: str, namespace: str) -> None:
        """Bind a project domain without creating storage."""
        self.root = root
        self.project_id = project_id
        self.namespace = namespace

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize a short read-modify-write operation without holding provider calls."""
        with project_connection(self.root, write=True):
            yield

    def read(self, record_id: str) -> bytes | None:
        """Read verified state or return absence without creating storage."""
        if not has_project_schema(self.root):
            return None
        with project_connection(self.root) as connection:
            row = connection.execute(
                "SELECT payload, sha256 FROM project_state_records "
                "WHERE project_id=? AND namespace=? AND record_id=?",
                (self.project_id, self.namespace, record_id),
            ).fetchone()
            return None if row is None else verified_payload(row[0], row[1])

    def write(self, record_id: str, payload: bytes, *, exclusive: bool = False) -> None:
        """Commit one record; exclusive creation refuses an existing identity."""
        digest = hashlib.sha256(payload).hexdigest()
        with project_connection(self.root, write=True) as connection:
            current = self.read(record_id)
            if current is not None and exclusive:
                raise ProjectRecordError(f"Project record already exists: {record_id}")
            connection.execute(
                "INSERT INTO project_state_records VALUES (?, ?, ?, 1, ?, ?) "
                "ON CONFLICT(project_id, namespace, record_id) DO UPDATE SET "
                "revision=revision+1, payload=excluded.payload, sha256=excluded.sha256",
                (self.project_id, self.namespace, record_id, payload, digest),
            )

    def replace(self, record_id: str, *, expected: bytes, replacement: bytes | None) -> None:
        """Replace or delete only an unchanged exact record inside one transaction."""
        with self.transaction():
            if self.read(record_id) != expected:
                raise ProjectRecordError(f"Project record changed before mutation: {record_id}")
            if replacement is None:
                with project_connection(self.root, write=True) as connection:
                    connection.execute(
                        "DELETE FROM project_state_records WHERE project_id=? AND namespace=? "
                        "AND record_id=?",
                        (self.project_id, self.namespace, record_id),
                    )
            else:
                self.write(record_id, replacement)

    def list_ids(self) -> tuple[str, ...]:
        """Return deterministic domain record identities without filesystem scans."""
        if not has_project_schema(self.root):
            return ()
        with project_connection(self.root) as connection:
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT record_id FROM project_state_records WHERE project_id=? AND "
                    "namespace=? ORDER BY record_id",
                    (self.project_id, self.namespace),
                )
            )

    def append(self, event_id: str, payload: bytes) -> None:
        """Append an immutable event once, rejecting reuse with different bytes."""
        digest = hashlib.sha256(payload).hexdigest()
        with project_connection(self.root, write=True) as connection:
            previous = connection.execute(
                "SELECT payload, sha256 FROM project_state_events WHERE project_id=? AND "
                "namespace=? AND event_id=?",
                (self.project_id, self.namespace, event_id),
            ).fetchone()
            if previous is not None:
                if verified_payload(previous[0], previous[1]) != payload:
                    raise ProjectRecordError(f"Execution event identity was reused: {event_id}")
                return
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0)+1 FROM project_state_events WHERE "
                "project_id=? AND namespace=?",
                (self.project_id, self.namespace),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO project_state_events VALUES (?, ?, ?, ?, ?, ?)",
                (self.project_id, self.namespace, sequence, event_id, payload, digest),
            )

    def events(self) -> tuple[bytes, ...]:
        """Read the complete verified execution history in committed order."""
        if not has_project_schema(self.root):
            return ()
        with project_connection(self.root) as connection:
            rows = connection.execute(
                "SELECT sequence, payload, sha256 FROM project_state_events "
                "WHERE project_id=? AND namespace=? ORDER BY sequence",
                (self.project_id, self.namespace),
            ).fetchall()
            if tuple(row[0] for row in rows) != tuple(range(1, len(rows) + 1)):
                raise ProjectRecordError("Execution event history is not contiguous.")
            return tuple(verified_payload(row[1], row[2]) for row in rows)
