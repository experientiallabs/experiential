"""Tests for the incremental state projection: differential commits, sparse writes and replay."""

from __future__ import annotations

import copy
import random
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue, sha256_json
from exp.runtime.sessions.edits import (
    AppendValue,
    CopyValue,
    DeleteValue,
    SetValue,
    StateEdit,
    StateEditError,
    apply_json_edit,
)
from exp.runtime.sessions.errors import StateReferenceError, StepConflictError
from exp.runtime.sessions.nodes import declared_text
from exp.runtime.sessions.records import SessionBinding
from exp.runtime.sessions.store import SessionStore

BINDING = SessionBinding(
    deadline=datetime(2099, 1, 1, tzinfo=UTC),
    target_id="projection@v1",
    initialization_receipt_sha256=sha256_json({"receipt": 1}),
    knowledge_sha256=sha256_json({"knowledge": 1}),
    corpus_sha256=sha256_json({"corpus": 1}),
    representation_sha256=sha256_json({"representation": 1}),
    execution_policy_sha256=sha256_json({"policy": "projection-test"}),
)

INITIAL: JsonObject = {
    "orders": {"o1": {"status": "open", "lines": [{"sku": "a", "qty": 1}]}},
    "0": ["string key, not an index"],
    "empty": {"object": {}, "array": []},
    "nullable": None,
}


def _commit(store: SessionStore, session_id: str, step: str, edits: tuple[StateEdit, ...]) -> None:
    """Claim, dispatch, record and commit one step with the given edits.

    Args:
        store: Store under test.
        session_id: Session to advance.
        step: Distinct step identifier.
        edits: Structural edits to publish.
    """
    version = store.read_head(session_id).version
    claim = store.begin_step(session_id, step, sha256_json(step), version, BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 10_000)
    store.record_proposal(claim, {"step": step})
    store.complete_step(claim, edits, {"step": step})


def _paths(value: JsonValue, prefix: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """List every location in a document, the root included.

    Args:
        value: Document or subtree.
        prefix: Location of ``value``.

    Returns:
        Every path, parents before children.
    """
    found = [prefix]
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_paths(item, (*prefix, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_paths(item, (*prefix, index)))
    return found


def _value_at(document: JsonValue, path: tuple[str | int, ...]) -> JsonValue:
    """Return the value at an existing path of a plain document."""
    for part in path:
        if isinstance(document, dict) and isinstance(part, str):
            document = document[part]
        elif isinstance(document, list) and isinstance(part, int):
            document = document[part]
        else:
            raise AssertionError(part)
    return document


def _random_edit(rng: random.Random, document: JsonObject) -> StateEdit:
    """Draw one edit that applies to ``document``.

    Args:
        rng: Seeded generator.
        document: Current oracle document.

    Returns:
        An applicable primitive.
    """
    paths = _paths(document)
    containers = [p for p in paths if isinstance(_value_at(document, p), (dict, list))]
    arrays = [p for p in containers if isinstance(_value_at(document, p), list)]
    values: list[JsonValue] = [
        None,
        0,
        0.0,
        True,
        "",
        "0",
        {"k": [1, {"deep": "x"}]},
        [],
        {},
        [None, False],
    ]
    choice = rng.randrange(5)
    if choice == 0 and arrays:
        return AppendValue(path=rng.choice(arrays), value=copy.deepcopy(rng.choice(values)))
    deletable = [p for p in paths if p]
    if choice == 1 and deletable:
        return DeleteValue(path=rng.choice(deletable))
    if choice == 2 and deletable:
        source = rng.choice(paths)
        target = rng.choice(containers)
        parent = _value_at(document, target)
        if isinstance(parent, list) and parent:
            return CopyValue(path=(*target, rng.randrange(len(parent))), source=source)
        if isinstance(parent, dict):
            return CopyValue(path=(*target, rng.choice(["copy", "0", "orders"])), source=source)
    target = rng.choice(containers)
    parent = _value_at(document, target)
    if isinstance(parent, list) and parent:
        return SetValue(path=(*target, rng.randrange(len(parent))), value=rng.choice(values))
    if isinstance(parent, dict):
        key = rng.choice([*parent, "new", "0", ""]) if parent else "new"
        return SetValue(path=(*target, key), value=copy.deepcopy(rng.choice(values)))
    return SetValue(path=("fresh",), value=copy.deepcopy(rng.choice(values)))


@pytest.mark.parametrize("seed", range(6))
def test_stored_revisions_equal_the_full_state_reducer_on_random_edit_sequences(
    tmp_path: Path, seed: int
) -> None:
    """Every revision reads back byte-identical to an in-memory full-state reducer."""
    rng = random.Random(seed)
    store = SessionStore(tmp_path / "s.sqlite")
    store.create_session("s", BINDING, INITIAL)
    oracle = copy.deepcopy(INITIAL)
    expected = [declared_text(oracle)]
    for step in range(40):
        edits = tuple(_random_edit(rng, oracle) for _ in range(rng.randint(1, 4)))
        candidate = copy.deepcopy(oracle)
        try:
            for edit in edits:
                candidate = apply_json_edit(candidate, edit)
        except StateEditError:
            continue
        _commit(store, "s", f"step-{step}", edits)
        oracle = candidate
        expected.append(declared_text(oracle))
    store.pin_revisions("audit", "s", 0, store.read_head("s").version + 1)
    assert store.collect_garbage().complete
    assert declared_text(store.read_session("s").state) == expected[-1]
    for revision, text in enumerate(expected):
        assert declared_text(store.read_state(store.ref_at("s", revision))) == text
    assert len(store.verify_history("s")) == len(expected)


def test_a_leaf_edit_writes_only_the_changed_path(tmp_path: Path) -> None:
    """A sparse edit inserts one node per changed container, not the whole world."""
    world: JsonObject = {f"table-{i}": {f"row-{j}": {"v": j} for j in range(50)} for i in range(20)}
    store = SessionStore(tmp_path / "s.sqlite")
    store.create_session("s", BINDING, world)
    with sqlite3.connect(tmp_path / "s.sqlite") as raw:
        before = raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]
    _commit(store, "s", "leaf", (SetValue(path=("table-3", "row-7", "v"), value="changed"),))
    with sqlite3.connect(tmp_path / "s.sqlite") as raw:
        after = raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]
    assert after - before == 2
    state = store.read_session("s").state
    table = world["table-3"]
    assert isinstance(table, dict)
    assert state["table-3"] == {**table, "row-7": {"v": "changed"}}


def test_a_failed_edit_publishes_nothing_and_leaves_the_step_open(tmp_path: Path) -> None:
    """Invalid structure rolls back the event, the nodes and the head together."""
    path = tmp_path / "s.sqlite"
    store = SessionStore(path)
    store.create_session("s", BINDING, INITIAL)
    claim = store.begin_step("s", "bad", sha256_json("bad"), 0, BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 10)
    store.record_proposal(claim, {"bad": True})
    with sqlite3.connect(path) as raw:
        nodes = raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]
    edits = (SetValue(path=("orders", "o2"), value={"new": 1}), DeleteValue(path=("absent",)))
    with pytest.raises(StateEditError):
        store.complete_step(claim, edits, {})
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0] == nodes
        assert raw.execute("SELECT COUNT(*) FROM state_events").fetchone()[0] == 1
    assert store.read_session("s").state == INITIAL
    assert store.open_steps("s") == ("bad",)
    store.complete_step(claim, edits[:1], {})
    assert store.read_head("s").version == 1


def test_events_are_immutable_and_one_operation_commits_once(tmp_path: Path) -> None:
    """The event log refuses edits, deletes and a second event for one operation."""
    path = tmp_path / "s.sqlite"
    store = SessionStore(path)
    store.create_session("s", BINDING, INITIAL)
    _commit(store, "s", "once", (SetValue(path=("nullable",), value=1),))
    with sqlite3.connect(path) as raw:
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("UPDATE state_events SET effect_json = '{}'")
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute("DELETE FROM state_events")
        with pytest.raises(sqlite3.IntegrityError):
            raw.execute(
                "INSERT INTO state_events (session_id, revision, step_id, effect_json, root)"
                " SELECT session_id, 2, step_id, effect_json, root FROM state_events"
                " WHERE revision = 1"
            )
    with pytest.raises(StepConflictError):
        store.begin_step("s", "once", sha256_json("other"), 1, BINDING)


def test_references_are_bound_to_their_store_revision_and_root(tmp_path: Path) -> None:
    """A reference from another store or with another root is refused, not reinterpreted."""
    store = SessionStore(tmp_path / "a.sqlite")
    other = SessionStore(tmp_path / "b.sqlite")
    for each in (store, other):
        each.create_session("s", BINDING, INITIAL)
    ref = store.read_head("s").state
    assert store.read_state(ref, ("orders", "o1", "lines", 0, "sku")) == "a"
    assert store.read_state(ref, ("0",)) == ["string key, not an index"]
    with pytest.raises(StateReferenceError):
        other.read_state(ref)
    with pytest.raises(StateReferenceError):
        store.read_state(ref.model_copy(update={"root_sha256": "0" * 64}))
    with pytest.raises(StateReferenceError):
        store.read_state(ref.model_copy(update={"revision": 5}))
    with pytest.raises(KeyError):
        store.read_state(ref, ("orders", "missing"))


def test_returned_values_are_detached_from_stored_history(tmp_path: Path) -> None:
    """Mutating a read value never changes what a later read returns."""
    store = SessionStore(tmp_path / "s.sqlite")
    store.create_session("s", BINDING, INITIAL)
    first = store.read_session("s").state
    orders = first["orders"]
    assert isinstance(orders, dict)
    orders["o1"] = "mutated"
    assert store.read_session("s").state == INITIAL


def test_transition_copies_share_storage_with_the_states_they_copy(tmp_path: Path) -> None:
    """Copying the current state into history costs no new nodes for its unchanged subtrees."""
    world: JsonObject = {"rows": [{"id": i, "payload": "x" * 50} for i in range(200)]}
    store = SessionStore(tmp_path / "s.sqlite")
    store.create_session("s", BINDING, {"business_state": world, "transitions": []})
    with sqlite3.connect(tmp_path / "s.sqlite") as raw:
        before = raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0]
    _commit(
        store,
        "s",
        "copy",
        (
            AppendValue(path=("transitions",), value={"before": None}),
            CopyValue(path=("transitions", 0, "before"), source=("business_state",)),
        ),
    )
    with sqlite3.connect(tmp_path / "s.sqlite") as raw:
        assert raw.execute("SELECT COUNT(*) FROM state_nodes").fetchone()[0] - before == 3
    state = store.read_session("s").state
    assert state["transitions"] == [{"before": world}]
