"""Content-addressed trace staging and atomic publication in the shared traffic database."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from exp.common.core.artifacts import (
    ContractModel,
    JsonObject,
    SourceIdentity,
    canonical_json_bytes,
    sha256_json,
    stable_id,
    validate_artifact_id,
)
from exp.common.traces.sqlite_import import ImportRecord, PreparedImport, prepare_import
from exp.common.traces.sqlite_schema import TraceStoreError, initialize_schema, validate_schema
from exp.common.traces.trace import Trace, TraceSource

_JSON_OBJECT = TypeAdapter(JsonObject)


def _require_private_file(path: Path, *, optional: bool = False) -> None:
    """Reject unsafe storage without following symlinks or changing existing files.

    Args:
        path: Database or sidecar in an operator-controlled directory.
        optional: Allow an absent sidecar before SQLite creates it.

    Raises:
        OSError: Storage metadata cannot be read, or a required file is absent.
        TraceStoreError: Storage is not a regular file or grants group/other access.
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        if optional:
            return
        raise
    if not stat.S_ISREG(mode):
        raise TraceStoreError(
            "Trace storage must be a regular file; use a database path without a file symlink."
        )
    if os.name == "posix" and mode & 0o077:
        raise TraceStoreError(
            "Trace storage requires owner-only file permissions; correct its permissions first."
        )


def _enable_wal(connection: sqlite3.Connection) -> None:
    """Retry WAL startup contention within the connection's existing busy timeout.

    SQLite can return BUSY immediately while promoting the journal lock, bypassing
    its busy handler. No import transaction has begun, so retrying this setup is safe.
    Ordinary statements retain the caller's busy timeout after setup completes.

    Args:
        connection: Validated database connection with no active transaction.

    Raises:
        sqlite3.OperationalError: The lock deadline expires or SQLite rejects setup.
        TraceStoreError: SQLite does not enable write-ahead logging.
    """
    timeout_ms = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    deadline = monotonic() + timeout_ms / 1000
    connection.execute("PRAGMA busy_timeout=0")
    try:
        while True:
            try:
                if connection.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
                    raise TraceStoreError("Cannot enable WAL for the trace database.")
                return
            except sqlite3.OperationalError as exc:
                remaining = deadline - monotonic()
                if exc.sqlite_errorcode & 0xFF != sqlite3.SQLITE_BUSY or remaining <= 0:
                    raise
                sleep(min(0.01, remaining))
    finally:
        connection.execute(f"PRAGMA busy_timeout={timeout_ms}")


class StoredTraceImport(ContractModel):
    """Immutable normalization evidence shared by every project selecting this import.

    Attributes:
        import_id: Content identity of the complete normalized import.
        source_format: Declared normalizer, such as chat-json or gateway.
        source: Original file or scoped capture provenance.
        traces: Exact normalized records in source order.
        metadata: Producer-owned normalization issues and model-identity provenance.
        created_at: First successful persistence time, unchanged on repeated imports.
    """

    import_id: str
    source_format: str = Field(min_length=1)
    source: SourceIdentity
    traces: tuple[Trace, ...]
    metadata: JsonObject
    created_at: AwareDatetime


class TraceImportReceipt(ContractModel):
    """Committed import identity and counts for one project association.

    Attributes:
        project_id: Local project namespace, independent of gateway authentication.
        import_id: Immutable import selected by this project.
        trace_count: Number of accepted trace records in the import.
        new_records: Number of canonical records newly added to the shared store.
        already_linked: Whether this project already selected the same complete import.
    """

    project_id: str
    import_id: str
    trace_count: int = Field(ge=0)
    new_records: int = Field(ge=0)
    already_linked: bool


def _record(trace: Trace) -> JsonObject:
    """Separate per-import provenance from reusable canonical trace content."""
    return trace.model_dump(mode="json", exclude={"source"})


def _import_id(
    source_format: str, source: SourceIdentity, traces: Sequence[Trace], metadata: JsonObject
) -> str:
    """Bind an import to its source, ordered records, and complete normalization metadata."""
    return stable_id(
        "import",
        {
            "source_format": source_format,
            "source": source.model_dump(mode="json"),
            "records": [
                {
                    "sha256": sha256_json(_record(trace)),
                    "source": trace.source.model_dump(mode="json"),
                }
                for trace in traces
            ],
            "metadata": metadata,
        },
    )


class SQLiteTraceStore:
    """Shared immutable trace storage with atomic, idempotent project membership.

    Construction and reads create no files. Only write_import obtains write authority.
    Native capture retains ownership of gateway_captures and its retention policy.
    Canonical imports preserve selected evidence independently of capture expiration.
    Project association is the publication boundary; unlinked staging is never readable.
    """

    def __init__(self, database_path: Path) -> None:
        """Bind the content database path without opening or creating it."""
        # Resolve directory aliases without hiding a symlink at the final filename.
        self.path = database_path.parent.resolve() / database_path.name

    @contextmanager
    def _connect(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Own one connection and report storage errors without exposing trace payloads."""
        try:
            # Reject unsafe orphaned sidecars before even creating an empty database.
            for suffix in ("-wal", "-shm", "-journal"):
                _require_private_file(Path(f"{self.path}{suffix}"), optional=True)
            if write:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                try:
                    descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
            _require_private_file(self.path)
            mode = "rw" if write else "ro"
            connection = sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True, timeout=5)
            try:
                connection.execute("PRAGMA foreign_keys=ON")
                validate_schema(connection)
                if write:
                    _enable_wal(connection)
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.execute("BEGIN IMMEDIATE")
                    initialize_schema(connection)
                else:
                    connection.execute("BEGIN")
                yield connection
                if write:
                    connection.commit()
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            raise TraceStoreError(
                "Cannot access trace database; check its path, permissions, and free space."
            ) from exc

    def write_import(
        self,
        project_id: str,
        *,
        source_format: str,
        source: SourceIdentity,
        traces: Iterable[Trace],
        metadata: JsonObject,
    ) -> TraceImportReceipt:
        """Stage normalized evidence and atomically publish its complete project association.

        Args:
            project_id: Local namespace receiving this import.
            source_format: Explicit source loader name.
            source: Immutable source identity supplied by normalization.
            traces: Ordered accepted records; their original sources remain intact.
            metadata: All exclusions and model-span identity evidence from normalization.

        Returns:
            A receipt after atomic publication; interrupted staging remains hidden and retryable.

        Raises:
            TraceStoreError: Stored evidence conflicts or a write cannot complete.
            ValueError: A project or source name is invalid.
        """
        validate_artifact_id(project_id)
        if not source_format.strip():
            raise ValueError("source_format must not be empty")
        with prepare_import(source_format, source, traces, metadata) as prepared:
            return self._publish(project_id, source_format, source, metadata, prepared)

    def _publish(
        self,
        project_id: str,
        source_format: str,
        source: SourceIdentity,
        metadata: JsonObject,
        prepared: PreparedImport,
    ) -> TraceImportReceipt:
        """Stage short transactions and atomically publish through project membership.

        Unlinked imports are private staging. Interrupted attempts can reuse their
        verified prefix, but read_import and list_imports never expose them. Published
        evidence is checked without filling in missing or corrupt records.
        """
        header = (
            source_format,
            canonical_json_bytes(source).decode(),
            canonical_json_bytes(metadata).decode(),
        )
        with self._connect(write=True) as connection:
            saved = connection.execute(
                "SELECT source_format,source,metadata FROM trace_imports WHERE import_id=?",
                (prepared.import_id,),
            ).fetchone()
            if saved is not None and saved != header:
                raise TraceStoreError("Stored import evidence differs from its immutable identity.")
            published = (
                connection.execute(
                    "SELECT 1 FROM trace_project_imports WHERE import_id=? LIMIT 1",
                    (prepared.import_id,),
                ).fetchone()
                is not None
            )
            if saved is None:
                connection.execute(
                    "INSERT INTO trace_imports VALUES (?, ?, ?, ?, ?)",
                    (prepared.import_id, *header, datetime.now(UTC).isoformat()),
                )
        new_records = 0
        ordinal = 0
        for batch in prepared.batches():
            with self._connect(write=not published) as connection:
                for record in batch:
                    new_records += self._stage_record(
                        connection, prepared.import_id, ordinal, record, published=published
                    )
                    ordinal += 1
        with self._connect(write=True) as connection:
            count, last = connection.execute(
                "SELECT COUNT(*),MAX(ordinal) FROM trace_import_records WHERE import_id=?",
                (prepared.import_id,),
            ).fetchone()
            if count != prepared.count or last != (prepared.count - 1 if prepared.count else None):
                raise TraceStoreError("Stored import evidence differs from its immutable identity.")
            if not connection.execute(
                "SELECT 1 FROM trace_project_imports WHERE import_id=? LIMIT 1",
                (prepared.import_id,),
            ).fetchone():
                connection.execute(
                    "UPDATE trace_imports SET created_at=? WHERE import_id=?",
                    (datetime.now(UTC).isoformat(), prepared.import_id),
                )
            linked = connection.execute(
                "INSERT OR IGNORE INTO trace_project_imports(project_id,import_id) VALUES (?, ?)",
                (project_id, prepared.import_id),
            ).rowcount
        return TraceImportReceipt(
            project_id=project_id,
            import_id=prepared.import_id,
            trace_count=prepared.count,
            new_records=new_records,
            already_linked=linked == 0,
        )

    @staticmethod
    def _stage_record(
        connection: sqlite3.Connection,
        import_id: str,
        ordinal: int,
        record: ImportRecord,
        *,
        published: bool,
    ) -> int:
        """Verify reusable bytes and membership, adding only unpublished evidence."""
        inserted = 0
        if not published:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO trace_records VALUES (?, ?, ?)",
                (record.digest, record.trace_id, record.payload),
            ).rowcount
            connection.execute(
                "INSERT OR IGNORE INTO trace_import_records VALUES (?, ?, ?, ?)",
                (import_id, ordinal, record.digest, record.source),
            )
        saved = connection.execute(
            "SELECT r.trace_id,r.payload,m.record_sha256,m.source "
            "FROM trace_import_records m JOIN trace_records r "
            "ON r.record_sha256=m.record_sha256 WHERE m.import_id=? AND m.ordinal=?",
            (import_id, ordinal),
        ).fetchone()
        if saved != (record.trace_id, record.payload, record.digest, record.source):
            raise TraceStoreError("Stored import evidence differs from its immutable identity.")
        return inserted

    def list_imports(self, project_id: str) -> tuple[str, ...]:
        """Read project import identities in selection order without creating missing storage."""
        validate_artifact_id(project_id)
        if not self.path.exists() and not self.path.is_symlink():
            return ()
        with self._connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trace_project_imports'"
            ).fetchone():
                return ()
            return tuple(
                row[0]
                for row in connection.execute(
                    "SELECT import_id FROM trace_project_imports "
                    "WHERE project_id=? ORDER BY sequence",
                    (project_id,),
                )
            )

    def read_import(self, import_id: str) -> StoredTraceImport:
        """Read and verify a frozen import, including every source and normalization exclusion."""
        try:
            with self._connect() as connection:
                result = self._read_import(connection, import_id)
                if (
                    _import_id(result.source_format, result.source, result.traces, result.metadata)
                    != import_id
                ):
                    raise TraceStoreError(
                        "Stored import evidence differs from its immutable identity."
                    )
                return result
        except ValidationError as exc:
            raise TraceStoreError(
                "Stored import is corrupt; restore it from a verified source."
            ) from exc

    @staticmethod
    def _read_import(connection: sqlite3.Connection, import_id: str) -> StoredTraceImport:
        """Reconstruct one import through an already owned transaction."""
        row = connection.execute(
            "SELECT source_format,source,metadata,created_at FROM trace_imports i "
            "WHERE import_id=? "
            "AND EXISTS(SELECT 1 FROM trace_project_imports p WHERE p.import_id=i.import_id)",
            (import_id,),
        ).fetchone()
        if row is None:
            raise TraceStoreError("Trace import was not found; select a saved import identity.")
        traces = []
        for ordinal, trace_id, digest, payload, source in connection.execute(
            "SELECT m.ordinal,r.trace_id,r.record_sha256,r.payload,m.source "
            "FROM trace_import_records m "
            "JOIN trace_records r ON r.record_sha256=m.record_sha256 "
            "WHERE m.import_id=? ORDER BY m.ordinal",
            (import_id,),
        ):
            content = _JSON_OBJECT.validate_json(payload)
            if (
                ordinal != len(traces)
                or content.get("trace_id") != trace_id
                or sha256_json(content) != digest
            ):
                raise TraceStoreError("Stored trace bytes do not match their content digest.")
            traces.append(
                Trace.model_validate({**content, "source": TraceSource.model_validate_json(source)})
            )
        return StoredTraceImport(
            import_id=import_id,
            source_format=row[0],
            source=SourceIdentity.model_validate_json(row[1]),
            metadata=_JSON_OBJECT.validate_json(row[2]),
            created_at=row[3],
            traces=tuple(traces),
        )
