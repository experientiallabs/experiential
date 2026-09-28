"""Crash, concurrency, schema, and portable snapshot proofs for project SQLite state."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

import exp.common.project.database as database_module
from exp.common.core.artifacts import ArtifactEnvelope, sha256_json
from exp.common.project import ProjectConfig, ProjectStore, ProjectStoreError, write_project_config
from exp.common.project.database import content_database_path, project_connection
from exp.common.project.records import ProjectRecordError
from exp.common.project.sqlite_schema import initialize_project_schema
from exp.common.sqlite.schema import ContentSchemaError


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
