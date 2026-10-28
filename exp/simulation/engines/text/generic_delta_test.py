"""Regression checks for the declared structural edit semantics."""

import copy
import json
import random

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.sessions.edits import apply_json_edit
from exp.runtime.sessions.nodes import declared_text
from exp.simulation.engines.text.generic_delta import (
    DELTA_OPS,
    DeltaError,
    apply_delta,
    delta_edits,
)


def test_sequential_edits_keep_literal_map_keys_and_missing_null_empty_distinct() -> None:
    """Object-key paths are literal; later operations observe earlier changes."""
    state: JsonObject = {"a/b.c": {"nullable": None, "empty": [], "keep": {}}}
    original = copy.deepcopy(state)
    post = apply_delta(
        state,
        [
            {"op": "set", "path": ["a/b.c", "new", "list"], "value": []},
            {"op": "append", "path": ["a/b.c", "new", "list"], "value": None},
            {"op": "delete", "path": ["a/b.c", "nullable"]},
        ],
    )
    assert post == {"a/b.c": {"empty": [], "keep": {}, "new": {"list": [None]}}}
    assert state == original


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "set", "path": ["list", True], "value": 1},
        {"op": "set", "path": ["list", 2], "value": 1},
        {"op": "replace", "path": ["list", 0], "value": 1},
        {"op": "append", "path": ["missing"], "value": 1},
        {"op": "set", "path": ["list"], "value": 1, "extra": True},
        {"op": "set", "path": ["list"]},
        {"op": "set", "path": []},
    ],
)
def test_bad_late_operation_rejects_atomically(operation: JsonObject) -> None:
    """A late invalid operation never changes the caller's state."""
    state: JsonObject = {"list": [1], "untouched": None}
    with pytest.raises(DeltaError):
        apply_delta(state, [{"op": "append", "path": ["list"], "value": 2}, operation])
    assert state == {"list": [1], "untouched": None}


def test_remove_value_is_type_aware_and_removes_only_first_match() -> None:
    """The existing JSON equality semantics do not confuse boolean and numeric values."""
    post = apply_delta(
        {"list": [True, 1, 1.0]},
        [{"op": "remove_value", "path": ["list"], "value": 1}],
    )
    assert post == {"list": [True, 1.0]}


def test_copy_then_edit_preserves_typed_source_without_aliasing() -> None:
    """A row deletion copies exact metadata, rather than regenerating unchanged cells."""
    state: JsonObject = {
        "input": [{"value": "key", "data_type": "s"}, {"value": 1}],
        "output": None,
    }
    post = apply_delta(
        state,
        [
            {"op": "copy", "from": ["input"], "path": ["output"]},
            {"op": "delete", "path": ["output", 1]},
        ],
    )
    assert post["output"] == [{"value": "key", "data_type": "s"}]
    assert post["input"] == state["input"]
    assert state["output"] is None


@pytest.mark.parametrize("source", [[], ["missing"], ["input", True]])
def test_copy_refuses_missing_or_ambiguous_sources(source: list[str | int]) -> None:
    """Copy never turns absent evidence into null or treats booleans as indices."""
    with pytest.raises(DeltaError):
        apply_delta({"input": [1]}, [{"op": "copy", "from": source, "path": ["output"]}])


# The full-state reducer as it was before structural edits, kept verbatim as the differential
# oracle: the structural interpreter must reproduce it exactly, errors included.


def _reference_resolve(container: JsonValue, parts: list[str | int]) -> JsonValue:
    """Return the value at ``parts`` inside ``container``.

    Args:
        container: Document to walk.
        parts: Object keys and list indices.

    Returns:
        The value found.

    Raises:
        DeltaError: If any segment is missing or cannot be followed.
    """
    node = container
    for part in parts:
        if isinstance(node, dict) and isinstance(part, str) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            raise DeltaError("missing path segment: " + repr(part))
    return node


def _reference_json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare JSON values recursively, keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _reference_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _reference_json_equal(value, right[key]) for key, value in left.items()
        )
    return type(left) is type(right) and left == right


def reference_apply_delta(state: JsonObject, delta: JsonValue) -> JsonObject:
    """Apply generic delta ops exactly as the full-state reducer did before structural edits.

    Ops: ``set`` creates or replaces an object key or an in-range list index, ``delete`` removes
    an existing key or index, ``append`` adds to an existing list, ``remove_value`` removes the
    first JSON-equal element of an existing list. Booleans remain distinct from numbers,
    while numeric equality admits equivalent integer and floating-point values.
    ``copy`` deep-copies an existing ``from`` path to a set-compatible destination. Sources
    resolve against the working state, so later operations observe earlier edits.
    Any other op or shape is refused.

    Args:
        state: The current business state. Not mutated.
        delta: The ``delta`` list from the model output.

    Returns:
        The next business state.

    Raises:
        DeltaError: On any undeclared op, bad path, or missing target.
    """
    if not isinstance(delta, list):
        raise DeltaError("delta must be a list of ops")
    new = copy.deepcopy(state)
    for op in delta:
        if not isinstance(op, dict) or op.get("op") not in DELTA_OPS:
            raise DeltaError("undeclared op")
        if op["op"] == "copy":
            source = op.get("from")
            if (
                set(op) != {"op", "path", "from"}
                or not isinstance(source, list)
                or not source
                or not all(type(part) in (str, int) for part in source)
            ):
                raise DeltaError("copy requires an exact nonempty source path")
            op = {
                "op": "set",
                "path": op["path"],
                "value": copy.deepcopy(_reference_resolve(new, source)),
            }
        if set(op) - {"op", "path", "value"}:
            raise DeltaError("op carries undeclared fields")
        path = op.get("path")
        if (
            not isinstance(path, list)
            or not path
            or not all(isinstance(p, (str, int)) and not isinstance(p, bool) for p in path)
        ):
            raise DeltaError("bad path")
        parent = (
            _reference_resolve(new, path[:-1])
            if op["op"] != "set"
            else _reference_parent_for_set(new, path)
        )
        key = path[-1]
        kind = op["op"]
        assert isinstance(kind, str)
        if kind == "set":
            if "value" not in op:
                raise DeltaError("set requires value")
            if isinstance(parent, dict) and isinstance(key, str):
                parent[key] = copy.deepcopy(op["value"])
            elif isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
                parent[key] = copy.deepcopy(op["value"])
            else:
                raise DeltaError("set target invalid")
        elif kind == "delete":
            if isinstance(parent, dict) and key in parent:
                del parent[key]
            elif isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent):
                del parent[key]
            else:
                raise DeltaError("delete target absent")
        else:
            if "value" not in op:
                raise DeltaError(kind + " requires value")
            target = _reference_resolve(new, path)
            if not isinstance(target, list):
                raise DeltaError(kind + " target is not a list")
            if kind == "append":
                target.append(copy.deepcopy(op["value"]))
            else:
                for index, item in enumerate(target):
                    if _reference_json_equal(item, op["value"]):
                        del target[index]
                        break
                else:
                    raise DeltaError("remove_value value absent")
    return new


def _reference_parent_for_set(state: JsonObject, path: list[str | int]) -> JsonValue:
    """Return the container for a ``set`` op, creating only missing object keys on the way.

    Args:
        state: Working copy of the business state.
        path: Full path of the ``set`` op.

    Returns:
        The container that will receive the value.

    Raises:
        DeltaError: If an intermediate list index is out of range.
    """
    node: JsonValue = state
    for part in path[:-1]:
        if isinstance(node, dict) and isinstance(part, str):
            if part not in node:
                node[part] = {}
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            raise DeltaError("cannot descend into: " + repr(part))
    return node


def _paths(value: JsonValue, prefix: list[str | int]) -> list[list[str | int]]:
    """List every location of a document, the root included."""
    found = [prefix]
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(_paths(item, [*prefix, key]))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_paths(item, [*prefix, index]))
    return found


def _random_op(rng: random.Random, state: JsonObject) -> JsonObject:
    """Draw one generic op, valid or not, aimed at existing and missing locations."""
    paths = [path for path in _paths(state, []) if path]
    targets = [*paths, ["missing"], ["missing", "deeper", "leaf"], [], ["list", True]]
    path = rng.choice(targets)
    extended = [*path, rng.choice(["new", "0", 0, 1, 5])] if rng.random() < 0.4 else path
    values: list[JsonValue] = [None, 0, 1, 1.0, True, False, "1", [], {}, [1, 1.0], {"k": [None]}]
    kind = rng.choice([*DELTA_OPS, "replace"])
    if kind == "copy":
        source = rng.choice([*paths, ["missing"], []])
        return {"op": "copy", "from": source, "path": extended}
    if kind == "delete":
        return {"op": "delete", "path": extended}
    return {"op": kind, "path": extended, "value": copy.deepcopy(rng.choice(values))}


@pytest.mark.parametrize("seed", range(40))
def test_structural_edits_reproduce_the_full_state_reducer(seed: int) -> None:
    """Random op sequences give byte-identical states, or the same refusal, as the old reducer."""
    rng = random.Random(seed)
    state: JsonObject = {
        "list": [True, 1, 1.0, {"a": [1]}, [None]],
        "map": {"0": "key", "nested": {"x": {"y": None}}},
        "empty": {},
        "nullable": None,
    }
    for _ in range(30):
        delta: JsonValue = [_random_op(rng, state) for _ in range(rng.randint(1, 4))]
        before = json.loads(declared_text(state))
        try:
            expected = reference_apply_delta(state, delta)
        except DeltaError:
            with pytest.raises(DeltaError):
                delta_edits(state, delta)
            continue
        actual, edits = delta_edits(state, delta)
        assert declared_text(actual) == declared_text(expected)
        assert declared_text(state) == declared_text(before)
        replayed = json.loads(declared_text(state))
        for edit in edits:
            replayed = apply_json_edit(replayed, edit)
        assert declared_text(replayed) == declared_text(expected)
        state = actual


def test_copy_into_a_created_child_of_its_source_keeps_the_original_value() -> None:
    """A copy whose new parents sit under its source captures the source before creation."""
    state: JsonObject = {"a": {"x": 1}}
    delta: JsonValue = [{"op": "copy", "from": ["a"], "path": ["a", "b", "c"]}]
    actual, edits = delta_edits(state, delta)
    assert actual == reference_apply_delta(state, delta) == {"a": {"x": 1, "b": {"c": {"x": 1}}}}
    replayed = copy.deepcopy(state)
    for edit in edits:
        replayed = apply_json_edit(replayed, edit)
    assert replayed == actual
