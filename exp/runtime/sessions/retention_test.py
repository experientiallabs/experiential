"""Tests for retention pins and resumable collection of unowned historical state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.sqlite.connection import connect_database
from exp.runtime.sessions.edits import SetValue, StateEdit
from exp.runtime.sessions.errors import (
    RetentionConflictError,
    SessionNotFoundError,
    StateUnavailableError,
)
from exp.runtime.sessions.projection import fetch_subtree
from exp.runtime.sessions.records import SessionBinding
from exp.runtime.sessions.retention import collect
from exp.runtime.sessions.store import SessionStore

BINDING = SessionBinding(
    deadline=datetime(2099, 1, 1, tzinfo=UTC),
    target_id="retention@v1",
    initialization_receipt_sha256=sha256_json({"receipt": 1}),
    knowledge_sha256=sha256_json({"knowledge": 1}),
    corpus_sha256=sha256_json({"corpus": 1}),
    representation_sha256=sha256_json({"representation": 1}),
    execution_policy_sha256=sha256_json({"policy": "retention-test"}),
)


def _world(version: int) -> JsonObject:
    """Return a world whose ``rows`` subtree differs per version and ``static`` never changes."""
    return {
        "static": {f"k{i}": {"v": i} for i in range(20)},
        "rows": {f"r{i}": {"version": version, "i": i} for i in range(10)},
    }


def _commit(store: SessionStore, step: str, edits: tuple[StateEdit, ...]) -> None:
    """Claim, dispatch, record and commit one step of session ``s``.

    Args:
        store: Store under test.
        step: Distinct step identifier.
        edits: Structural edits to publish.
    """
    version = store.read_head("s").version
    claim = store.begin_step("s", step, sha256_json(step), version, BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 10_000)
    store.record_proposal(claim, {"step": step})
    store.complete_step(claim, edits, {"step": step})


def _history(tmp_path: Path, revisions: int = 5) -> SessionStore:
    """Create session ``s`` whose rows subtree is replaced at every revision.

    Args:
        tmp_path: Pytest temporary directory.
        revisions: Committed revisions after genesis.

    Returns:
        The store.
    """
    store = SessionStore(tmp_path / "s.sqlite")
    store.create_session("s", BINDING, _world(0))
    for version in range(1, revisions + 1):
        _commit(store, f"v{version}", (SetValue(path=("rows",), value=_world(version)["rows"]),))
    return store


def _node_count(path: Path) -> int:
    """Count stored nodes."""
    with sqlite3.connect(path) as raw:
        return raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]


def test_unpinned_history_is_reclaimed_while_heads_and_events_survive(tmp_path: Path) -> None:
    """Collection removes only unowned revisions; the event log still replays them."""
    store = _history(tmp_path)
    old = store.ref_at("s", 2)
    assert store.read_state(old, ("rows", "r0", "version")) == 2
    report = store.collect_garbage(batch_size=7)
    assert report.complete and report.reclaimed_nodes == 5 * 2 and report.reclaimed_bytes > 0
    assert store.read_session("s").state == _world(5)
    with pytest.raises(StateUnavailableError):
        store.read_state(old)
    assert len(store.verify_history("s")) == 6
    assert store.collect_garbage().reclaimed_nodes == 0


def test_overlapping_owners_keep_exactly_the_union_of_their_ranges(tmp_path: Path) -> None:
    """Releasing one owner leaves revisions another owner still holds restorable."""
    store = _history(tmp_path)
    store.pin_revisions("checkpoint-a", "s", 0, 3)
    store.pin_revisions("fork-b", "s", 2, 5)
    assert store.collect_garbage().reclaimed_nodes == 0
    assert store.release_pin("checkpoint-a", "s")
    assert not store.release_pin("checkpoint-a", "s")
    store.collect_garbage()
    for revision in (2, 3, 4):
        state = store.read_state(store.ref_at("s", revision))
        assert state == _world(revision)
    for revision in (0, 1):
        with pytest.raises(StateUnavailableError):
            store.read_state(store.ref_at("s", revision))
    with pytest.raises(StateUnavailableError):
        store.pin_revisions("late", "s", 1, 3)


def test_pins_refuse_conflicts_future_revisions_and_unknown_sessions(tmp_path: Path) -> None:
    """A pin is exact: same owner, same range, within committed revisions."""
    store = _history(tmp_path, revisions=2)
    pin = store.pin_revisions("owner", "s", 0, 2)
    assert store.pin_revisions("owner", "s", 0, 2) == pin
    with pytest.raises(RetentionConflictError):
        store.pin_revisions("owner", "s", 0, 3)
    with pytest.raises(RetentionConflictError):
        store.pin_revisions("other", "s", 1, 4)
    with pytest.raises(SessionNotFoundError):
        store.pin_revisions("owner", "missing", 0, 1)


def test_a_bounded_collection_resumes_from_its_durable_cursor(tmp_path: Path) -> None:
    """A stopped run leaves a cursor; a new process finishes the same sweep."""
    store = _history(tmp_path)
    first = store.collect_garbage(batch_size=10, maximum_batches=2)
    assert not first.complete
    with sqlite3.connect(tmp_path / "s.sqlite") as raw:
        assert raw.execute("SELECT cursor FROM state_collection").fetchone()[0] > 0
    rest = SessionStore(tmp_path / "s.sqlite").collect_garbage(batch_size=10)
    assert rest.complete
    assert first.reclaimed_nodes + rest.reclaimed_nodes == 5 * 2
    assert store.read_session("s").state == _world(5)


def test_a_half_reclaimed_revision_cannot_be_pinned_and_collection_continues(
    tmp_path: Path,
) -> None:
    """A bounded sweep that stopped between a revision's child and its root leaves it unpinnable.

    Genesis stores its children before its root, so a one-row batch sweep reaches the point
    where the old rows subtree is gone and the genesis root row is still present.
    """
    store = _history(tmp_path)
    old = store.ref_at("s", 0)
    root = bytes.fromhex(old.root_sha256)
    for _ in range(_node_count(tmp_path / "s.sqlite")):
        store.collect_garbage(batch_size=1, maximum_batches=1)
        with sqlite3.connect(tmp_path / "s.sqlite") as raw:
            remaining = raw.execute(
                "SELECT COUNT(*) FROM state_nodes WHERE digest = ?", (root,)
            ).fetchone()[0]
        try:
            store.read_state(old)
        except StateUnavailableError:
            break
    assert remaining == 1, "the sweep must stop after a child of genesis and before its root"
    with pytest.raises(StateUnavailableError, match="partly reclaimed"):
        store.pin_revisions("late", "s", 0, 1)
    assert not store.release_pin("late", "s")
    assert store.collect_garbage().complete
    assert store.read_session("s").state == _world(5)


def test_a_commit_between_batches_that_reuses_a_candidate_keeps_it_alive(
    tmp_path: Path,
) -> None:
    """A node unreachable at the mark but reused by a later commit is never deleted."""
    store = _history(tmp_path)
    store.collect_garbage(batch_size=1, maximum_batches=1)
    _commit(store, "back", (SetValue(path=("rows",), value=_world(1)["rows"]),))
    store.collect_garbage(batch_size=3)
    assert store.read_session("s").state == _world(1)
    head = store.read_head("s")
    assert store.read_state(head.state) == _world(1)


def test_a_reader_holding_a_snapshot_sees_a_complete_revision_during_collection(
    tmp_path: Path,
) -> None:
    """Each read runs in one snapshot, so a concurrent sweep cannot tear it."""
    store = _history(tmp_path)
    old = store.ref_at("s", 1)
    reader = connect_database(tmp_path / "s.sqlite")
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0] > 0
        store.collect_garbage()
        nodes = fetch_subtree(reader, bytes.fromhex(old.root_sha256))
        assert len(nodes) == 3
        reader.execute("COMMIT")
    finally:
        reader.close()
    with pytest.raises(StateUnavailableError):
        store.read_state(old)
    assert _node_count(tmp_path / "s.sqlite") == 3


def test_a_commit_inside_one_sweep_that_reuses_candidates_keeps_them_alive(
    tmp_path: Path,
) -> None:
    """Between two batches of one run, a commit revives unmarked nodes; the sweep spares them."""
    store = _history(tmp_path)
    database = tmp_path / "s.sqlite"
    batches = [0]

    @contextmanager
    def transaction() -> Iterator[sqlite3.Connection]:
        """Open one write transaction, committing a reviving step before the second batch."""
        batches[0] += 1
        if batches[0] == 3:
            _commit(store, "revive", (SetValue(path=("rows",), value=_world(1)["rows"]),))
            _commit(store, "nest", (SetValue(path=("copy",), value=_world(2)["rows"]),))
        connection = connect_database(database)
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        finally:
            connection.close()

    reader = connect_database(database)
    try:
        report = collect(reader, transaction, batch_size=1, maximum_batches=None)
    finally:
        reader.close()
    assert report.complete and batches[0] > 3
    expected = {**_world(1), "copy": _world(2)["rows"]}
    assert store.read_session("s").state == expected
    later = store.collect_garbage()
    assert later.complete and later.reclaimed_nodes > 0
    assert store.read_session("s").state == expected


def test_a_preview_reports_what_collection_reclaims_and_what_each_pin_holds(
    tmp_path: Path,
) -> None:
    """The preview writes nothing; its counts equal the collection that follows a release."""
    store = _history(tmp_path)
    store.pin_revisions("checkpoint-a", "s", 0, 3)
    store.pin_revisions("fork-b", "s", 2, 5)
    before = _node_count(tmp_path / "s.sqlite")
    preview = store.preview_collection()
    assert _node_count(tmp_path / "s.sqlite") == before
    assert preview.reclaimable_nodes == 0 and preview.retained_nodes == before
    holdings = {row.pin.owner_id: row for row in preview.pins}
    assert (holdings["checkpoint-a"].exclusive_nodes, holdings["fork-b"].exclusive_nodes) == (
        4,
        4,
    )
    assert store.read_state(store.ref_at("s", 1)) == _world(1)
    assert store.release_pin("checkpoint-a", "s")
    released = store.preview_collection()
    assert released.reclaimable_nodes == holdings["checkpoint-a"].exclusive_nodes
    assert released.reclaimable_bytes == holdings["checkpoint-a"].exclusive_bytes
    assert [row.pin.owner_id for row in released.pins] == ["fork-b"]
    report = store.collect_garbage()
    assert (report.reclaimed_nodes, report.reclaimed_bytes) == (
        released.reclaimable_nodes,
        released.reclaimable_bytes,
    )
    assert store.preview_collection().reclaimable_nodes == 0


def test_a_vacuum_after_collection_returns_freed_pages_and_keeps_every_revision(
    tmp_path: Path,
) -> None:
    """The file shrinks; heads read the same and the event log still replays every revision."""
    store = _history(tmp_path, revisions=40)
    store.collect_garbage()
    assert store.vacuum() > 0
    assert store.read_session("s").state == _world(40)
    assert len(store.verify_history("s")) == 41
