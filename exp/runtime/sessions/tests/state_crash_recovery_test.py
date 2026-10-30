"""Crash and interrupt tests for incremental session state at every step of a commit.

Each test fails a commit at every possible point (every SQLite statement, every SQLite virtual
machine step, or a real ``SIGKILL`` of the committing process) and then checks the store's
durable contract: either nothing of the commit is visible and the step stays open, or the whole
commit is visible and the step replays its response. No partial head, event or node set may
ever be observable, and every surviving revision must replay from its event log.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.sqlite.connection import persistent_connection
from exp.runtime.sessions import store as store_module
from exp.runtime.sessions.edits import (
    AppendValue,
    CopyValue,
    DeleteValue,
    SetValue,
    StateEdit,
    StateEditAdapter,
    apply_json_edit,
)
from exp.runtime.sessions.errors import ClaimLostError, StepAmbiguousError
from exp.runtime.sessions.records import SessionBinding, StepClaim
from exp.runtime.sessions.store import SessionStore

BINDING = SessionBinding(
    deadline=datetime(2099, 1, 1, tzinfo=UTC),
    target_id="crash@v1",
    initialization_receipt_sha256=sha256_json({"receipt": 1}),
    knowledge_sha256=sha256_json({"knowledge": 1}),
    corpus_sha256=sha256_json({"corpus": 1}),
    representation_sha256=sha256_json({"representation": 1}),
    execution_policy_sha256=sha256_json({"policy": "crash-test"}),
)

INITIAL: JsonObject = {
    "business_state": {"orders": {"o1": {"status": "open", "lines": [1, 2]}}, "log": []},
    "history": [],
    "transitions": [],
}

EDITS: tuple[StateEdit, ...] = (
    AppendValue(path=("transitions",), value={"operation_id": "op", "before": None, "after": None}),
    CopyValue(path=("transitions", 0, "before"), source=("business_state",)),
    SetValue(path=("business_state", "orders", "o1", "status"), value="cancelled"),
    DeleteValue(path=("business_state", "orders", "o1", "lines", 0)),
    SetValue(path=("business_state", "orders", "o2"), value={"status": "new", "lines": [3]}),
    CopyValue(path=("transitions", 0, "after"), source=("business_state",)),
    AppendValue(path=("history",), value={"actions": ["cancel"], "observations": ["ok"]}),
)

RESPONSE: JsonObject = {"observations": [{"content": "ok"}]}

type Parameter = str | bytes | int | float | None
"""One SQLite statement parameter the store binds."""


def _expected_after() -> JsonObject:
    """Return the state a successful commit of ``EDITS`` publishes."""
    document = json.loads(json.dumps(INITIAL))
    for edit in EDITS:
        document = apply_json_edit(document, edit)
    return document


def _prepared(path: Path) -> tuple[SessionStore, StepClaim]:
    """Create a session with one dispatched step whose proposal is already durable.

    Args:
        path: Database file.

    Returns:
        The store and the open provider-started claim.
    """
    store = SessionStore(path)
    store.create_session("s", BINDING, INITIAL)
    claim = store.begin_step("s", "op", sha256_json("op"), 0, BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 10)
    store.record_proposal(claim, {"proposal": "paid output"})
    return store, claim


def _counts(path: Path) -> tuple[int, int, int, str]:
    """Return node, event and head counts plus the step status, read outside the store."""
    with sqlite3.connect(path) as raw:
        nodes = raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]
        events = raw.execute("SELECT COUNT(*) FROM state_events").fetchone()[0]
        version = raw.execute("SELECT version FROM sessions").fetchone()[0]
        status = raw.execute("SELECT status FROM steps WHERE step_id = 'op'").fetchone()[0]
    return nodes, events, version, status


def _assert_atomic(store: SessionStore, path: Path, before: tuple[int, int, int, str]) -> bool:
    """Check that a failed commit left either nothing or everything.

    Args:
        store: Store to read through.
        path: Database file.
        before: Counts before the attempt.

    Returns:
        Whether the commit became durable.
    """
    nodes, events, version, status = _counts(path)
    with sqlite3.connect(path) as raw:
        assert raw.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    store.verify_history("s")
    if version == 0:
        assert (nodes, events, version, status) == before
        assert store.read_session("s").state == INITIAL
        return False
    assert (events, version, status) == (2, 1, "completed")
    assert store.read_session("s").state == _expected_after()
    replay = store.begin_step("s", "op", sha256_json("op"), 0, BINDING)
    assert replay.completed is not None and replay.completed.response == RESPONSE
    return True


class _FailingConnection:
    """Forwards to a real connection but raises on one chosen statement.

    Attributes:
        connection: Wrapped SQLite connection.
        countdown: Shared remaining statements before the failure; negative when disarmed.
    """

    def __init__(self, connection: sqlite3.Connection, countdown: list[int]) -> None:
        """Wrap ``connection`` with a shared statement countdown."""
        self.connection = connection
        self.countdown = countdown

    def _tick(self) -> None:
        """Raise when the countdown reaches the chosen statement."""
        if self.countdown[0] == 0:
            self.countdown[0] = -1
            raise sqlite3.OperationalError("injected failure before this statement")
        if self.countdown[0] > 0:
            self.countdown[0] -= 1

    def execute(self, sql: str, parameters: Sequence[Parameter] = ()) -> sqlite3.Cursor:
        """Run one statement unless it is the chosen failure."""
        self._tick()
        return self.connection.execute(sql, parameters)

    def executemany(self, sql: str, parameters: Iterable[Sequence[Parameter]]) -> sqlite3.Cursor:
        """Run one batched statement unless it is the chosen failure."""
        self._tick()
        return self.connection.executemany(sql, parameters)

    @property
    def in_transaction(self) -> bool:
        """Whether the wrapped connection has an open transaction."""
        return self.connection.in_transaction


def _patched(monkeypatch: pytest.MonkeyPatch, wrap: Callable[[sqlite3.Connection], object]) -> None:
    """Make the store check out wrapped connections.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        wrap: Wraps or arms each checked-out connection.
    """

    @contextmanager
    def checkout(path: Path) -> Iterator[object]:
        """Yield the wrapped form of the store's real reusable connection."""
        with persistent_connection(path) as connection:
            try:
                yield wrap(connection)
            finally:
                connection.set_progress_handler(None, 0)

    monkeypatch.setattr(store_module, "persistent_connection", checkout)


def test_a_failure_before_any_statement_of_a_commit_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing each statement in turn never publishes part of a commit."""
    path = tmp_path / "s.sqlite"
    store, claim = _prepared(path)
    before = _counts(path)
    countdown = [-1]
    _patched(monkeypatch, lambda connection: _FailingConnection(connection, countdown))
    statement = 0
    while True:
        countdown[0] = statement
        try:
            store.complete_step(claim, EDITS, RESPONSE)
        except sqlite3.OperationalError:
            committed = _assert_atomic(store, path, before)
        else:
            break
        if committed:
            break
        statement += 1
    assert statement > 8
    assert _assert_atomic(store, path, before)


def test_an_interrupt_at_every_sqlite_step_of_a_commit_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupting the SQLite machine anywhere inside the commit publishes all or nothing."""
    path = tmp_path / "s.sqlite"
    store, claim = _prepared(path)
    before = _counts(path)
    remaining = [-1]

    def handler() -> int:
        """Interrupt exactly once, at the chosen virtual-machine step."""
        if remaining[0] == 0:
            remaining[0] = -1
            return 1
        if remaining[0] > 0:
            remaining[0] -= 1
        return 0

    def arm(connection: sqlite3.Connection) -> sqlite3.Connection:
        """Install the interrupting progress handler on one checkout."""
        connection.set_progress_handler(handler, 1)
        return connection

    _patched(monkeypatch, arm)
    attempts = committed = 0
    step = 0
    while not committed:
        remaining[0] = step
        try:
            store.complete_step(claim, EDITS, RESPONSE)
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            attempts += 1
            committed = _assert_atomic(store, path, before)
        except ClaimLostError:
            committed = _assert_atomic(store, path, before)
        else:
            committed = True
        step += 1
    assert attempts > 200, attempts
    assert _assert_atomic(store, path, before)


_CHILD = """
import json, os, signal, sys
from contextlib import contextmanager
from pathlib import Path
from exp.common.sqlite.connection import persistent_connection
from exp.runtime.sessions import store as store_module
from exp.runtime.sessions.edits import StateEditAdapter
from exp.runtime.sessions.records import StepClaim

path, claim, edits, response, kill_at = sys.argv[1:6]
remaining = [int(kill_at)]

def handler():
    remaining[0] -= 1
    if remaining[0] == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    return 0

@contextmanager
def checkout(database):
    with persistent_connection(database) as connection:
        connection.set_progress_handler(handler, 1)
        yield connection

store = store_module.SessionStore(Path(path))
store_module.persistent_connection = checkout
store.complete_step(
    StepClaim.model_validate_json(claim),
    [StateEditAdapter.validate_python(edit) for edit in json.loads(edits)],
    json.loads(response),
)
print(json.dumps({"steps": int(kill_at) - remaining[0]}))
"""


def _child(path: Path, claim: StepClaim, kill_at: int) -> subprocess.CompletedProcess[str]:
    """Run one commit in a fresh process that kills itself at a chosen SQLite step.

    Args:
        path: Database file.
        claim: Open claim to complete.
        kill_at: Virtual-machine step at which the process sends itself ``SIGKILL``.

    Returns:
        The finished process.
    """
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD,
            str(path),
            claim.model_dump_json(),
            json.dumps([edit.model_dump(mode="json") for edit in EDITS]),
            json.dumps(RESPONSE),
            str(kill_at),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="requires POSIX SIGKILL")
def test_a_killed_committing_process_leaves_all_or_nothing(tmp_path: Path) -> None:
    """SIGKILL anywhere in a commit, including after COMMIT, never tears durable state."""
    probe_path = tmp_path / "probe.sqlite"
    _, probe_claim = _prepared(probe_path)
    finished = _child(probe_path, probe_claim, 10**9)
    assert finished.returncode == 0, finished.stderr
    total = json.loads(finished.stdout)["steps"]
    points = sorted({1, 2, total - 1, *range(3, total, max(1, total // 12))})
    outcomes = []
    for index, kill_at in enumerate(points):
        path = tmp_path / f"kill-{index}.sqlite"
        _, claim = _prepared(path)
        before = _counts(path)
        killed = _child(path, claim, kill_at)
        assert killed.returncode == -signal.SIGKILL, killed.stderr
        store = SessionStore(path)
        committed = _assert_atomic(store, path, before)
        if not committed:
            store.recover()
            with pytest.raises(StepAmbiguousError):
                store.begin_step("s", "op", sha256_json("op"), 0, BINDING)
            assert store.proposal("s", "op") == {"proposal": "paid output"}
        outcomes.append(committed)
    assert outcomes[0] is False and outcomes[-1] is True


def test_two_workers_completing_one_claim_publish_exactly_once(tmp_path: Path) -> None:
    """Racing completions of the same claim commit one revision; the other loses its claim."""
    path = tmp_path / "s.sqlite"
    _, claim = _prepared(path)
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        """Complete the shared claim from a separate store instance."""
        own = SessionStore(path)
        barrier.wait()
        try:
            own.complete_step(claim, EDITS, RESPONSE)
            outcomes.append("committed")
        except ClaimLostError:
            outcomes.append("lost")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["committed", "lost"]
    store = SessionStore(path)
    assert store.read_head("s").version == 1
    assert store.read_session("s").state == _expected_after()
    assert [edit.model_dump(mode="json") for edit in EDITS] == [
        StateEditAdapter.validate_python(edit.model_dump(mode="json")).model_dump(mode="json")
        for edit in EDITS
    ]
