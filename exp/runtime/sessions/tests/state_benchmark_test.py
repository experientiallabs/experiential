"""Offline storage benchmark: incremental session state against full-state replacement.

The fixtures are deterministic local worlds; no provider, model or network is involved. Each
workload commits through the real store and measures the bytes it actually inserts (node rows
plus event rows) against the bytes a full-state store rewrites on every commit (the complete
declared envelope text, which is what replacing one ``state_json`` column writes). Assertions use
these deterministic byte and row counts; wall-clock latencies and file sizes are logged for
inspection only, because they depend on the disk.

Run with ``-o log_cli=true --log-cli-level=INFO`` to see the measurements.
"""

from __future__ import annotations

import copy
import logging
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.runtime.sessions.edits import (
    AppendValue,
    CopyValue,
    DeleteValue,
    SetValue,
    StateEdit,
    apply_json_edit,
    prefixed,
)
from exp.runtime.sessions.nodes import declared_text
from exp.runtime.sessions.records import SessionBinding
from exp.runtime.sessions.store import SessionStore

logger = logging.getLogger(__name__)

BINDING = SessionBinding(
    deadline=datetime(2099, 1, 1, tzinfo=UTC),
    target_id="benchmark@v1",
    initialization_receipt_sha256=sha256_json({"receipt": 1}),
    knowledge_sha256=sha256_json({"knowledge": 1}),
    corpus_sha256=sha256_json({"corpus": 1}),
    representation_sha256=sha256_json({"representation": 1}),
    execution_policy_sha256=sha256_json({"policy": "benchmark"}),
)

WORLDS = {"small": (8, 10), "medium": (40, 25), "large": (120, 40)}
"""Tables and rows per table of each deterministic world."""


def world(tables: int, rows: int) -> JsonObject:
    """Build one deterministic retail-like world.

    Args:
        tables: Number of tables.
        rows: Rows per table.

    Returns:
        The world.
    """
    return {
        f"table_{t}": {
            f"row_{r}": {"id": f"{t}-{r}", "status": "open", "qty": r, "tags": ["a", "b"]}
            for r in range(rows)
        }
        for t in range(tables)
    }


@dataclass
class Measurement:
    """Bytes and latencies of one workload.

    Attributes:
        incremental_bytes: Node and event bytes the store inserted, per commit.
        full_state_bytes: Envelope bytes a full-state store would rewrite, per commit.
        latencies: Wall-clock seconds per commit.
    """

    incremental_bytes: list[int]
    full_state_bytes: list[int]
    latencies: list[float]


def _inserted(path: Path, marks: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    """Return bytes inserted since ``marks`` and the new marks.

    Args:
        path: Database file.
        marks: Largest node row and event row seen at the previous measurement.

    Returns:
        Inserted bytes and the updated marks.
    """
    with sqlite3.connect(path) as raw:
        nodes = raw.execute(
            "SELECT COALESCE(SUM(length(template) + length(children) + 32), 0),"
            " COALESCE(MAX(node_id), 0) FROM state_nodes WHERE node_id > ?",
            (marks[0],),
        ).fetchone()
        events = raw.execute(
            "SELECT COALESCE(SUM(length(effect_json) + 32), 0), COALESCE(MAX(rowid), 0)"
            " FROM state_events WHERE rowid > ?",
            (marks[1],),
        ).fetchone()
    return nodes[0] + events[0], (max(marks[0], nodes[1]), max(marks[1], events[1]))


class Session:
    """A durable-envelope session driven like the world-model coordinator drives it.

    Attributes:
        store: Store under measurement.
        session_id: Session identity.
        envelope: In-memory full-state oracle of the envelope.
        path: Database file.
        marks: Inserted-row marks for byte accounting.
    """

    def __init__(
        self, store: SessionStore, path: Path, session_id: str, initial: JsonObject
    ) -> None:
        """Create the session with an empty history."""
        self.store, self.path, self.session_id = store, path, session_id
        self.envelope: JsonObject = {
            "business_state": copy.deepcopy(initial),
            "history": [],
            "transitions": [],
        }
        store.create_session(session_id, BINDING, copy.deepcopy(self.envelope))
        _, self.marks = _inserted(path, (0, 0))

    def commit(self, step: str, business: tuple[StateEdit, ...], measurement: Measurement) -> None:
        """Commit one transition exactly as the coordinator encodes it, and measure it.

        Args:
            step: Step identity.
            business: Edits relative to the business state.
            measurement: Accumulator.
        """
        version = self.store.read_head(self.session_id).version
        record: JsonObject = {"operation_id": step, "before": None, "after": None, "reply": "ok"}
        edits = (
            AppendValue(path=("transitions",), value=record),
            CopyValue(path=("transitions", version, "before"), source=("business_state",)),
            *prefixed(("business_state",), business),
            CopyValue(path=("transitions", version, "after"), source=("business_state",)),
            AppendValue(path=("history",), value={"actions": [step], "observations": ["ok"]}),
        )
        for edit in edits:
            self.envelope = apply_json_edit(self.envelope, edit)
        started = time.perf_counter()
        claim = self.store.begin_step(
            self.session_id, step, sha256_json(step), version, BINDING
        ).claim
        assert claim is not None
        self.store.mark_provider_started(claim, 10_000)
        self.store.record_proposal(claim, {"step": step})
        self.store.complete_step(claim, edits, {"step": step})
        measurement.latencies.append(time.perf_counter() - started)
        written, self.marks = _inserted(self.path, self.marks)
        measurement.incremental_bytes.append(written)
        measurement.full_state_bytes.append(len(declared_text(self.envelope).encode("utf-8")))


def _node_bytes(path: Path) -> int:
    """Return the stored bytes of every node row."""
    with sqlite3.connect(path) as raw:
        return raw.execute(
            "SELECT COALESCE(SUM(length(template) + length(children) + 32), 0) FROM state_nodes"
        ).fetchone()[0]


def _report(name: str, measurement: Measurement, path: Path) -> None:
    """Log one workload's measurements."""
    latencies = sorted(measurement.latencies)
    logger.info(
        "%s: commits=%d incremental_bytes=%d full_state_bytes=%d ratio=%.1fx "
        "p50=%.1fms p95=%.1fms db_bytes=%d",
        name,
        len(latencies),
        sum(measurement.incremental_bytes),
        sum(measurement.full_state_bytes),
        sum(measurement.full_state_bytes) / max(1, sum(measurement.incremental_bytes)),
        statistics.median(latencies) * 1000,
        latencies[int(0.95 * (len(latencies) - 1))] * 1000,
        sum(p.stat().st_size for p in path.parent.glob(path.name + "*")),
    )


@pytest.mark.parametrize("size", list(WORLDS))
def test_sparse_leaf_commits_cost_the_change_not_the_world_or_history(
    tmp_path: Path, size: str
) -> None:
    """A leaf update writes a bounded amount however large the world and history grow."""
    tables, rows = WORLDS[size]
    path = tmp_path / "sparse.sqlite"
    session = Session(SessionStore(path), path, "s", world(tables, rows))
    measurement = Measurement([], [], [])
    rng = random.Random(7)
    for step in range(60):
        table, row = f"table_{rng.randrange(tables)}", f"row_{rng.randrange(rows)}"
        session.commit(
            f"op-{step}",
            (SetValue(path=(table, row, "status"), value=f"status-{step}"),),
            measurement,
        )
    _report(f"sparse/{size}", measurement, path)
    early = statistics.mean(measurement.incremental_bytes[:10])
    late = statistics.mean(measurement.incremental_bytes[-10:])
    assert (late - early) / 50 <= 128, "only the two append-only history arrays may grow"
    assert measurement.full_state_bytes[-1] > 20 * measurement.full_state_bytes[0] / 2
    assert sum(measurement.full_state_bytes) > 25 * sum(measurement.incremental_bytes)
    assert session.store.read_session("s").state == session.envelope


def test_dense_replacements_write_only_changed_subtrees(tmp_path: Path) -> None:
    """A model-authored full replacement still stores only the containers that changed."""
    tables, rows = WORLDS["medium"]
    path = tmp_path / "dense.sqlite"
    current = world(tables, rows)
    session = Session(SessionStore(path), path, "s", current)
    measurement = Measurement([], [], [])
    rng = random.Random(11)
    for step in range(20):
        current = copy.deepcopy(current)
        for _ in range(tables * rows // 10):
            table, row = f"table_{rng.randrange(tables)}", f"row_{rng.randrange(rows)}"
            table_rows = current[table]
            assert isinstance(table_rows, dict)
            table_rows[row] = {"id": row, "status": f"s{step}", "qty": step, "tags": []}
        session.commit(f"op-{step}", (SetValue(path=(), value=current),), measurement)
    _report("dense/medium", measurement, path)
    world_bytes = len(declared_text(world(tables, rows)).encode("utf-8"))
    assert max(measurement.incremental_bytes) < 2 * world_bytes
    assert sum(measurement.full_state_bytes) > 6 * sum(measurement.incremental_bytes)
    assert session.store.read_session("s").state == session.envelope


def test_array_deletes_rewrite_only_the_affected_array(tmp_path: Path) -> None:
    """Deleting from a large array rewrites that array's entries and its ancestors only."""
    path = tmp_path / "array.sqlite"
    initial: JsonObject = {"items": [{"n": i} for i in range(2000)], "other": world(10, 10)}
    session = Session(SessionStore(path), path, "s", initial)
    measurement = Measurement([], [], [])
    rng = random.Random(3)
    for step in range(30):
        items = session.envelope["business_state"]
        assert isinstance(items, dict) and isinstance(items["items"], list)
        index = rng.randrange(len(items["items"]))
        session.commit(f"op-{step}", (DeleteValue(path=("items", index)),), measurement)
    _report("array-deletes", measurement, path)
    array_bytes = 2000 * 33
    assert max(measurement.incremental_bytes) < 2 * array_bytes
    assert session.store.read_session("s").state == session.envelope


def test_forks_from_one_prefix_share_every_unchanged_node(tmp_path: Path) -> None:
    """Sibling sessions created from one prefix store only their own changes."""
    tables, rows = WORLDS["large"]
    path = tmp_path / "forks.sqlite"
    store = SessionStore(path)
    prefix = world(tables, rows)
    measurement = Measurement([], [], [])
    first = Session(store, path, "fork-0", prefix)
    one_world = _node_bytes(path)
    siblings = [first, *(Session(store, path, f"fork-{index}", prefix) for index in range(1, 5))]
    assert _node_bytes(path) < one_world + 5 * 1024, "siblings share the prefix's nodes"
    _, marks = _inserted(path, (0, 0))
    for index, sibling in enumerate(siblings):
        sibling.marks = marks
        for step in range(10):
            sibling.commit(
                f"op-{step}",
                (SetValue(path=("table_0", "row_0", "status"), value=f"fork-{index}-{step}"),),
                measurement,
            )
        marks = sibling.marks
    _report("forks/large", measurement, path)
    logger.info(
        "forks/large: one_world_node_bytes=%d all_node_bytes=%d", one_world, _node_bytes(path)
    )
    for index, sibling in enumerate(siblings):
        state = store.read_state(store.read_head(f"fork-{index}").state, ("business_state",))
        assert state == sibling.envelope["business_state"]
