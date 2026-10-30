"""Crash, concurrency, schema, and portable snapshot proofs for project SQLite state."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import exp.common.project.database as database_module
from exp.common.core.artifacts import ArtifactEnvelope, sha256_json
from exp.common.project import ProjectConfig, ProjectStore, ProjectStoreError, write_project_config
from exp.common.project.database import (
    ProjectDatabaseBusyError,
    content_database_path,
    project_connection,
)
from exp.common.project.records import ProjectRecordError, ProjectRecords
from exp.common.project.sqlite_schema import initialize_project_schema
from exp.common.sqlite import connection as sqlite_connection
from exp.common.sqlite.schema import ContentSchemaError

if TYPE_CHECKING:
    from sqlite3 import _Parameters


def test_read_only_discovery_creates_no_database_or_project_directory(tmp_path: Path) -> None:
    """Inspecting an absent project never initializes a database or folder."""
    root = tmp_path / "absent"
    store = ProjectStore(root, "project-a")
    assert not store.exists()
    assert store.artifacts.list_ids() == ()
    assert store.records.read("review") is None
    assert not root.exists()


def test_configuration_versions_and_review_commit_or_rollback_together(tmp_path: Path) -> None:
    """A failed selection retains the old head and exact immutable configuration bytes."""
    store = ProjectStore(tmp_path, "project-a")
    original = ProjectConfig(project_id="project-a")
    store.initialize(original)
    frozen = store.snapshot(sha256_json(original))
    changed = original.model_copy(update={"redacted_field_names": ("email",)})
    with pytest.raises(RuntimeError, match="before commit"):
        with project_connection(tmp_path, write=True):
            write_project_config(store.paths, changed)
            store.write_review({"selected": "new"})
            raise RuntimeError("before commit")
    assert store.load_project() == original
    assert store.read_review() is None
    with project_connection(tmp_path, write=True):
        write_project_config(store.paths, changed)
        store.write_review({"selected": "new"})
    assert store.load_project() == changed
    assert store.read_review() == {"selected": "new"}
    assert frozen.load_project() == original
    with pytest.raises(ProjectStoreError, match="snapshot"):
        frozen.initialize(changed)


def test_nested_failed_mutation_cannot_commit_a_partial_record(tmp_path: Path) -> None:
    """An intercepted inner failure rolls back its savepoint while the outer change commits."""
    store = ProjectStore(tmp_path, "project-a")
    with store.records.transaction():
        store.records.write("outer", b"retained")
        with pytest.raises(ProjectRecordError):
            with store.records.transaction():
                store.records.write("partial", b"discard")
                store.records.write("outer", b"conflict", exclusive=True)
    assert store.records.read("outer") == b"retained"
    assert store.records.read("partial") is None


def test_separate_processes_serialize_read_modify_write(tmp_path: Path) -> None:
    """Independent writers cannot lose committed increments on one project record."""
    script = """
import sys
from pathlib import Path
from exp.common.project.records import ProjectRecords
state = ProjectRecords(Path(sys.argv[1]), "project-a", "project")
for _ in range(20):
    with state.transaction():
        state.write("counter", str(int(state.read("counter") or b"0") + 1).encode())
"""
    workers = [
        subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], stderr=subprocess.PIPE)
        for _ in range(2)
    ]
    try:
        for worker in workers:
            _, error = worker.communicate(timeout=30)
            assert worker.returncode == 0, error.decode()
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    assert ProjectStore(tmp_path, "project-a").records.read("counter") == b"40"


def test_incompatible_project_definition_is_rejected_before_wal_or_ddl(tmp_path: Path) -> None:
    """Known names and version do not authorize a malformed project's table definitions."""
    database = content_database_path(tmp_path)
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        initialize_project_schema(connection)
        connection.execute("ALTER TABLE project_config_heads ADD COLUMN unexpected TEXT")
    os.chmod(database, 0o600)
    original = database.read_bytes()
    with pytest.raises(ContentSchemaError, match="Incompatible project table"):
        ProjectStore(tmp_path, "project-a").records.write("run", b"unchanged")
    assert database.read_bytes() == original
    assert not database.with_name("traffic.db-wal").exists()


def test_large_files_and_html_have_relative_digest_references(tmp_path: Path) -> None:
    """Database records own metadata while external immutable bytes remain portable."""
    store = ProjectStore(tmp_path, "project-a")
    large = b"large artifact\n" * 100_000
    files = {"large.bin": large, "report.html": b"<p>export</p>", "small.json": b"{}"}
    store.artifacts.write(
        artifact_id="result-a",
        artifact_type="result",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime(2026, 9, 22, tzinfo=UTC), code_revision="test"
        ),
        files=files,
    )
    with project_connection(tmp_path) as connection:
        rows = connection.execute(
            "SELECT path, payload, blob_path FROM project_artifact_files ORDER BY path"
        ).fetchall()
    assert rows == [
        ("large.bin", None, f"projects/project-a/blobs/{hashlib.sha256(large).hexdigest()}"),
        (
            "report.html",
            None,
            f"projects/project-a/blobs/{hashlib.sha256(files['report.html']).hexdigest()}",
        ),
        ("small.json", b"{}", None),
    ]
    assert dict(store.artifacts.read("result-a").payloads) == files
    assert not (store.paths.project_directory / "artifacts").exists()


def test_blob_publication_rejects_symlinked_project_ancestor(tmp_path: Path) -> None:
    """No metadata commits and no outside bytes are created through a directory redirect."""
    root, outside = tmp_path / "root", tmp_path / "outside"
    outside.mkdir()
    (root / "projects").mkdir(parents=True)
    (root / "projects" / "project-a").symlink_to(outside, target_is_directory=True)
    store = ProjectStore(root, "project-a")
    with pytest.raises(OSError):
        store.artifacts.write(
            artifact_id="result-a",
            artifact_type="result",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime(2026, 9, 22, tzinfo=UTC), code_revision="test"
            ),
            files={"report.html": b"<p>export</p>"},
        )
    assert list(outside.iterdir()) == []
    assert not store.artifacts.exists("result-a")


def test_unsupported_folder_state_is_preserved_without_silent_cutover(tmp_path: Path) -> None:
    """A new database cannot conceal a preexisting project's file-based evidence."""
    project_directory = tmp_path / "projects" / "project-a"
    project_directory.mkdir(parents=True)
    original = b'project_id = "project-a"\n'
    (project_directory / "project.toml").write_bytes(original)
    with pytest.raises(ValueError, match="unsupported folder layout"):
        ProjectStore(tmp_path, "project-a").initialize(ProjectConfig(project_id="project-a"))
    assert (project_directory / "project.toml").read_bytes() == original


def test_wal_initialization_and_begin_share_one_admission_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BEGIN cannot restart the busy timeout already consumed by WAL initialization."""
    now = [100.0]
    original = database_module.enable_wal_mode

    def initialize_wal(connection: sqlite3.Connection, *, deadline: float) -> None:
        """Charge initialization time to the same finite transaction admission budget."""
        original(connection, deadline=deadline)
        now[0] += 0.4

    monkeypatch.setattr(database_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(database_module, "enable_wal_mode", initialize_wal)
    with project_connection(tmp_path, write=True, timeout_s=0.5) as connection:
        remaining_ms = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        assert 0 < remaining_ms <= 100


def _busy() -> sqlite3.OperationalError:
    """Build the error SQLite raises when another writer holds the lock."""
    error = sqlite3.OperationalError("database is locked")
    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
    error.sqlite_errorname = "SQLITE_BUSY"
    return error


def _refusing(statement: str, refusals: int) -> type[sqlite3.Connection]:
    """Return a project connection class whose first ``statement`` runs report busy."""

    class Refusing(database_module._ProjectConnection):
        """Project connection that refuses a statement as if another writer held the lock."""

        remaining = refusals

        def execute(self, sql: str, parameters: _Parameters = (), /) -> sqlite3.Cursor:
            """Refuse ``statement`` before SQLite runs it while refusals remain."""
            if sql == statement and type(self).remaining > 0:
                type(self).remaining -= 1
                raise _busy()
            return super().execute(sql, parameters)

    return Refusing


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff pauses instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(database_module.time, "sleep", recorded.append)
    return recorded


def test_operations_reuse_one_connection_validated_once_per_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated reads and writes open one connection and validate only after schema changes."""
    opened: list[Path] = []
    validated: list[int] = []
    connect, validate = sqlite_connection.connect_database, database_module.validate_content_tables

    def counting_connect(
        path: Path,
        *,
        busy_timeout_ms: int,
        enable_wal: bool,
        create: bool,
        factory: type[sqlite3.Connection],
    ) -> sqlite3.Connection:
        """Count database opens."""
        opened.append(path)
        return connect(
            path,
            busy_timeout_ms=busy_timeout_ms,
            enable_wal=enable_wal,
            create=create,
            factory=factory,
        )

    def counting_validate(connection: sqlite3.Connection) -> frozenset[str]:
        """Count content-table validations."""
        validated.append(1)
        return validate(connection)

    monkeypatch.setattr(sqlite_connection, "connect_database", counting_connect)
    monkeypatch.setattr(database_module, "validate_content_tables", counting_validate)
    records = ProjectRecords(tmp_path, "project-a", "project")
    records.write("first", b"1")
    setup = len(validated)
    for index in range(50):
        records.write(f"record-{index}", b"x")
        assert records.read(f"record-{index}") == b"x"
    assert len(opened) == 1
    assert len(validated) == setup + 1, "only the schema created by the first write revalidates"
    with sqlite3.connect(content_database_path(tmp_path)) as other:
        other.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(ContentSchemaError, match="Unrecognized"):
        records.read("first")


def test_a_replaced_database_file_is_reopened_not_written_through_a_stale_handle(
    tmp_path: Path,
) -> None:
    """A database deleted and recreated at the same path receives the next write."""
    records = ProjectRecords(tmp_path, "project-a", "project")
    records.write("before", b"old")
    database = content_database_path(tmp_path)
    for suffix in ("", "-wal", "-shm"):
        database.with_name(database.name + suffix).unlink(missing_ok=True)
    records.write("after", b"new")
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT record_id FROM project_state_records").fetchall()
    assert rows == [("after",)]


def test_a_read_snapshot_on_a_reused_connection_refuses_writes(tmp_path: Path) -> None:
    """Reads stay read-only although the connection is shared with write transactions."""
    records = ProjectRecords(tmp_path, "project-a", "project")
    records.write("kept", b"1")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with project_connection(tmp_path) as connection:
            connection.execute("DELETE FROM project_state_records")
    records.write("next", b"2")
    assert records.list_ids() == ("kept", "next")


def test_an_attached_database_does_not_outlive_its_transaction(tmp_path: Path) -> None:
    """A reused connection starts every transaction with only its own databases."""
    other = tmp_path / "other.db"
    sqlite3.connect(other).close()
    ProjectRecords(tmp_path, "project-a", "project").write("seed", b"1")
    for _ in range(2):
        with project_connection(tmp_path, write=True) as connection:
            connection.execute("ATTACH DATABASE ? AS restored", (str(other),))
    with project_connection(tmp_path) as connection:
        names = [row[1] for row in connection.execute("PRAGMA database_list")]
    assert names == ["main"]


def test_busy_admission_retries_without_rerunning_the_transaction_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A lock refusal before the body is retried; the body and its write happen once."""
    ProjectRecords(tmp_path, "project-a", "project").write("seed", b"1")
    monkeypatch.setattr(database_module, "_ProjectConnection", _refusing("BEGIN IMMEDIATE", 2))
    bodies: list[int] = []
    log = ProjectRecords(tmp_path, "project-a", "log")
    with project_connection(tmp_path, write=True):
        bodies.append(1)
        log.append("event", b"once")
    assert bodies == [1]
    assert len(pauses) == 2
    assert 0 <= pauses[0] <= database_module.PROJECT_LOCK_BACKOFF_S
    assert log.events() == (b"once",)


def test_a_busy_commit_is_retried_only_while_uncommitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """A COMMIT refused for a lock is retried; the committed change appears exactly once."""
    ProjectRecords(tmp_path, "project-a", "project").write("seed", b"1")
    monkeypatch.setattr(database_module, "_ProjectConnection", _refusing("COMMIT", 1))
    bodies: list[int] = []
    log = ProjectRecords(tmp_path, "project-a", "log")
    with project_connection(tmp_path, write=True):
        bodies.append(1)
        log.append("event", b"once")
    assert bodies == [1]
    assert len(pauses) == 1
    assert log.events() == (b"once",)


def test_exhausted_lock_attempts_report_a_clear_busy_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
) -> None:
    """After the bound the caller gets a busy error naming what to do, and nothing commits."""
    records = ProjectRecords(tmp_path, "project-a", "project")
    records.write("seed", b"1")
    monkeypatch.setattr(database_module, "_ProjectConnection", _refusing("BEGIN IMMEDIATE", 99))
    with pytest.raises(ProjectDatabaseBusyError, match="stayed locked.*Retry") as raised:
        with project_connection(tmp_path, write=True):
            pytest.fail("the transaction body must not run without the write lock")
    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
    assert len(pauses) == database_module.PROJECT_LOCK_ATTEMPTS - 1
    monkeypatch.undo()
    assert records.list_ids() == ("seed",)


def test_a_writer_outlasting_the_lock_wait_retries_instead_of_crashing(tmp_path: Path) -> None:
    """A real lock held past one attempt's wait is survived by the bounded retry."""
    records = ProjectRecords(tmp_path, "project-a", "project")
    records.write("seed", b"1")
    holder = sqlite3.connect(
        content_database_path(tmp_path), isolation_level=None, check_same_thread=False
    )
    holder.execute("BEGIN IMMEDIATE")
    release = threading.Timer(0.2, holder.rollback)
    release.start()
    try:
        with project_connection(tmp_path, write=True, timeout_s=0.1):
            records.write("after", b"2")
    finally:
        release.join()
        holder.close()
    assert records.list_ids() == ("after", "seed")


def test_concurrent_processes_write_every_record_exactly_once(tmp_path: Path) -> None:
    """Writers in several processes on one store commit every write once, never crashing."""
    script = """
import sys, time
from pathlib import Path
from exp.common.project.records import ProjectRecords
root, worker = Path(sys.argv[1]), sys.argv[2]
events = ProjectRecords(root, "project-a", "events")
state = ProjectRecords(root, "project-a", "state")
for index in range(25):
    with state.transaction():
        events.append(f"{worker}-{index}", f"{worker}-{index}".encode())
        state.write("counter", str(int(state.read("counter") or b"0") + 1).encode())
        time.sleep(0.005)
    assert events.events()
"""
    workers = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), str(worker)], stderr=subprocess.PIPE
        )
        for worker in range(8)
    ]
    try:
        for worker in workers:
            _, error = worker.communicate(timeout=120)
            assert worker.returncode == 0, error.decode()
            assert b"database is locked" not in error
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
    events = ProjectRecords(tmp_path, "project-a", "events").events()
    expected = {f"{worker}-{index}".encode() for worker in range(8) for index in range(25)}
    assert len(events) == len(expected) and set(events) == expected
    assert ProjectRecords(tmp_path, "project-a", "state").read("counter") == b"200"
