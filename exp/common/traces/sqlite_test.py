"""Immutable import identity, atomicity, and evidence integrity in real SQLite."""

import os
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from exp.common.core.artifacts import SourceIdentity
from exp.common.traces.sqlite import SQLiteTraceStore, TraceImportReceipt
from exp.common.traces.sqlite_import import ImportRecord, PreparedImport
from exp.common.traces.sqlite_schema import TraceStoreError
from exp.common.traces.trace import Trace, TraceSource
from exp.common.traces.trace_test import _trace


def _save(
    store: SQLiteTraceStore, traces: tuple[Trace, ...], project: str = "powerset"
) -> TraceImportReceipt:
    """Persist a synthetic source with retained normalization metadata."""
    return store.write_import(
        project,
        source_format="otlp",
        source=traces[0].source.identity,
        traces=traces,
        metadata={"issues": [{"source_record": "line-2", "message": "invalid record"}]},
    )


def test_repeat_reopen_overlap_and_project_membership(tmp_path: Path) -> None:
    """Repeats reuse evidence, overlaps share content, and projects stay scoped."""
    path = tmp_path / "traffic.db"
    store = SQLiteTraceStore(path)
    assert store.list_imports("powerset") == ()
    assert not path.exists()
    trace = _trace()
    first = _save(store, (trace,))
    saved = store.read_import(first.import_id)
    assert saved.traces == (trace,)
    assert saved.metadata["issues"]
    repeated = _save(SQLiteTraceStore(path), (trace,))
    assert repeated.import_id == first.import_id
    assert repeated.already_linked and repeated.new_records == 0
    assert store.read_import(first.import_id).created_at == saved.created_at
    assert path.stat().st_mode & 0o777 == 0o600
    other_source = TraceSource(
        identity=SourceIdentity(kind="file", source_id="other-export"),
        semantic_convention_version="1.37.0",
    )
    overlap = trace.model_copy(update={"source": other_source})
    second = _save(store, (overlap,))
    assert second.import_id != first.import_id and second.new_records == 0
    assert store.read_import(second.import_id).traces[0].source == other_source
    other = _save(store, (trace,), project="other")
    assert other.import_id == first.import_id and not other.already_linked
    assert store.list_imports("powerset") == (first.import_id, second.import_id)
    assert store.list_imports("other") == (first.import_id,)
    changed = trace.model_copy(update={"task": "Changed evidence with the same source trace ID"})
    third = _save(store, (changed,))
    assert third.new_records == 1 and third.import_id != first.import_id
    assert store.read_import(first.import_id) == saved


@pytest.mark.skipif(os.name != "posix", reason="Unix mode bits require owner-only storage")
@pytest.mark.parametrize("mode", [0o644, 0o620, 0o604])
def test_import_rejects_nonprivate_database_without_changing_it(tmp_path: Path, mode: int) -> None:
    """Import cannot expose prompt content through an existing nonprivate file."""
    path = tmp_path / "traffic.db"
    path.touch(mode=mode)
    path.chmod(mode)
    with pytest.raises(TraceStoreError, match="owner-only"):
        SQLiteTraceStore(path).list_imports("powerset")
    with pytest.raises(TraceStoreError, match="owner-only"):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.read_bytes() == b""
    assert path.stat().st_mode & 0o777 == mode
    assert not Path(f"{path}-wal").exists()


@pytest.mark.skipif(os.name != "posix", reason="Symlink fixtures require Unix support")
@pytest.mark.parametrize("target_exists", [False, True])
def test_import_rejects_final_symlink_without_touching_target(
    tmp_path: Path, target_exists: bool
) -> None:
    """Resolving directory aliases never erases evidence of a final file symlink."""
    target = tmp_path / "target.db"
    if target_exists:
        target.touch(mode=0o600)
    path = tmp_path / "traffic.db"
    path.symlink_to(target)
    with pytest.raises(TraceStoreError, match="regular file"):
        SQLiteTraceStore(path).list_imports("powerset")
    with pytest.raises(TraceStoreError, match="regular file"):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.is_symlink()
    assert target.exists() == target_exists
    if target_exists:
        assert target.read_bytes() == b""


@pytest.mark.skipif(os.name != "posix", reason="Private sidecar fixtures require Unix support")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
@pytest.mark.parametrize("database_exists", [False, True])
@pytest.mark.parametrize("symlink", [False, True])
def test_import_rejects_unsafe_sidecar_before_database_creation(
    tmp_path: Path, suffix: str, database_exists: bool, symlink: bool
) -> None:
    """Unsafe orphaned and existing sidecars survive rejection without shared writes."""
    path = tmp_path / "traffic.db"
    if database_exists:
        path.touch(mode=0o600)
    sidecar = Path(f"{path}{suffix}")
    target = tmp_path / "sidecar-target"
    if symlink:
        target.write_bytes(b"unchanged sidecar target")
        sidecar.symlink_to(target)
    else:
        sidecar.write_bytes(b"unchanged sidecar fixture")
        sidecar.chmod(0o644)
    before = sidecar.read_bytes()
    with pytest.raises(TraceStoreError, match="regular file" if symlink else "owner-only"):
        _save(SQLiteTraceStore(path), (_trace(),))
    assert path.exists() == database_exists
    if database_exists:
        assert path.read_bytes() == b""
    assert sidecar.read_bytes() == before
    assert sidecar.is_symlink() == symlink


@pytest.mark.skipif(os.name != "posix", reason="Directory aliases require Unix symlinks")
def test_private_import_accepts_parent_alias_and_creates_private_sidecars(tmp_path: Path) -> None:
    """Normal directory aliases remain usable and SQLite sidecars inherit private mode."""
    directory = tmp_path / "real"
    directory.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    store = SQLiteTraceStore(alias / "traffic.db")
    with store._connect(write=True) as connection:
        connection.execute("CREATE TABLE gateway_captures (sequence INTEGER PRIMARY KEY)")
        for suffix in ("", "-wal", "-shm"):
            assert Path(f"{store.path}{suffix}").stat().st_mode & 0o777 == 0o600
    assert store.path == directory / "traffic.db"
    assert _save(store, (_trace(),)).trace_count == 1


def test_write_failure_keeps_staging_hidden_and_retryable(tmp_path: Path) -> None:
    """Publication failure keeps staged evidence hidden; retry verifies and completes it."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    first = _save(store, (_trace(),))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_link BEFORE INSERT ON trace_project_imports "
            "BEGIN SELECT RAISE(ABORT, 'private payload error'); END"
        )
    with pytest.raises(TraceStoreError) as raised:
        _save(store, (_trace().model_copy(update={"task": "Different task"}),))
    assert "private payload" not in str(raised.value)
    assert store.list_imports("powerset") == (first.import_id,)
    with sqlite3.connect(store.path) as connection:
        staged = connection.execute(
            "SELECT import_id FROM trace_imports WHERE import_id<>?", (first.import_id,)
        ).fetchone()[0]
    with pytest.raises(TraceStoreError, match="not found"):
        store.read_import(staged)
    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP TRIGGER reject_link")
    retry = _save(store, (_trace().model_copy(update={"task": "Different task"}),))
    assert retry.import_id == staged and retry.new_records == 0
    assert store.list_imports("powerset") == (first.import_id, staged)
    assert store.read_import(staged).traces[0].task == "Different task"


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE trace_records SET payload='{}'",
        "UPDATE trace_records SET payload='not json private-content'",
        "UPDATE trace_records SET trace_id='other'",
        "UPDATE trace_imports SET metadata='{}'",
        "UPDATE trace_import_records SET ordinal=9",
        "DELETE FROM trace_import_records",
    ],
)
def test_corrupt_evidence_is_rejected_on_read_and_repeat(tmp_path: Path, mutation: str) -> None:
    """Neither reading nor idempotent reuse can bless changed bytes or missing memberships."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    first = _save(store, (_trace(),))
    with sqlite3.connect(store.path) as connection:
        connection.execute(mutation)
    with pytest.raises(TraceStoreError) as raised:
        store.read_import(first.import_id)
    assert "private-content" not in str(raised.value)
    with pytest.raises(TraceStoreError):
        _save(store, (_trace(),))


def test_batches_release_writer_and_keep_partial_import_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A separate capture writer commits between batches while consumers see no partial import."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    original = PreparedImport.batches
    observed: list[str] = []

    def paced(prepared: PreparedImport) -> Iterator[tuple[ImportRecord, ...]]:
        """After each committed batch, exercise an independent short-timeout SQLite writer."""
        for batch in original(prepared):
            yield batch
            with sqlite3.connect(store.path, timeout=0.01) as writer:
                writer.execute(
                    "CREATE TABLE IF NOT EXISTS gateway_captures (sequence INTEGER PRIMARY KEY)"
                )
                writer.execute("INSERT INTO gateway_captures DEFAULT VALUES")
            assert store.list_imports("powerset") == ()
            with pytest.raises(TraceStoreError, match="not found"):
                store.read_import(prepared.import_id)
            observed.append(prepared.import_id)

    monkeypatch.setattr(PreparedImport, "batches", paced)
    traces = tuple(
        _trace().model_copy(update={"task": f"{index}:" + "x" * 50000}) for index in range(90)
    )
    receipt = _save(store, traces)
    assert len(observed) >= 4
    assert store.read_import(receipt.import_id).traces == traces


def test_interrupted_batches_resume_without_exposing_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation after a committed prefix remains invisible and reimport completes it."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    original = PreparedImport.batches
    staged: list[str] = []

    def interrupted(prepared: PreparedImport) -> Iterator[tuple[ImportRecord, ...]]:
        """Cancel after the first committed batch, as a process interruption would."""
        for batch in original(prepared):
            yield batch
            staged.append(prepared.import_id)
            raise KeyboardInterrupt

    traces = tuple(_trace().model_copy(update={"task": f"Task {index}"}) for index in range(100))
    with monkeypatch.context() as patch:
        patch.setattr(PreparedImport, "batches", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _save(store, traces)
    assert store.list_imports("powerset") == ()
    with pytest.raises(TraceStoreError, match="not found"):
        store.read_import(staged[0])
    receipt = _save(store, traces)
    assert receipt.import_id == staged[0]
    assert 0 < receipt.new_records < len(traces)
    assert store.read_import(receipt.import_id).traces == traces


def test_concurrent_identical_imports_publish_one_complete_identity(tmp_path: Path) -> None:
    """Two independent import writers deduplicate safely across interleaved transactions."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    traces = tuple(_trace().model_copy(update={"task": f"Task {index}"}) for index in range(180))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_save, SQLiteTraceStore(store.path), traces) for _ in range(2)]
        receipts = [future.result(timeout=10) for future in futures]
    assert receipts[0].import_id == receipts[1].import_id
    assert sum(receipt.new_records for receipt in receipts) == len(traces)
    assert sum(receipt.already_linked for receipt in receipts) == 1
    assert store.read_import(receipts[0].import_id).traces == traces


def test_wal_startup_retries_real_contention_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real rollback-journal reader blocks WAL until the import releases and retries."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    store.path.touch(mode=0o600)
    reader = sqlite3.connect(store.path)
    retries: list[float] = []
    try:
        reader.execute("CREATE TABLE gateway_captures(payload TEXT)")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM gateway_captures").fetchall()

        def release_reader(delay: float) -> None:
            """Release the real shared lock only after SQLite has returned BUSY."""
            retries.append(delay)
            reader.rollback()

        monkeypatch.setattr("exp.common.traces.sqlite.sleep", release_reader)
        receipt = _save(store, (_trace(),))
        assert len(retries) == 1 and 0 < retries[0] <= 0.01
        assert store.read_import(receipt.import_id).traces == (_trace(),)
        with closing(sqlite3.connect(store.path)) as reopened:
            assert reopened.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    finally:
        reader.close()


def test_wal_startup_contention_stops_at_the_existing_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent lock exhausts the setup deadline without publishing partial evidence."""
    store = SQLiteTraceStore(tmp_path / "traffic.db")
    store.path.touch(mode=0o600)
    reader = sqlite3.connect(store.path)
    ticks = iter((0.0, 5.0))
    try:
        reader.execute("CREATE TABLE gateway_captures(payload TEXT)")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM gateway_captures").fetchall()

        def elapsed() -> float:
            """Advance the setup clock to its deadline after the first real busy result."""
            return next(ticks)

        monkeypatch.setattr("exp.common.traces.sqlite.monotonic", elapsed)
        with pytest.raises(TraceStoreError) as raised:
            _save(store, (_trace(),))
        assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
        assert raised.value.__cause__.sqlite_errorcode == sqlite3.SQLITE_BUSY
        assert store.list_imports("powerset") == ()
        assert reader.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    finally:
        reader.close()
