"""Sequential generic delta ops on a copy, resolved into runtime structural edits."""

from __future__ import annotations

import copy

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.sessions.edits import (
    AppendValue,
    CopyValue,
    DeleteValue,
    SetValue,
    StateEdit,
    apply_json_edit,
)

DELTA_OPS: tuple[str, ...] = ("set", "delete", "append", "remove_value", "copy")


class DeltaError(ValueError):
    """The proposal violates the declared generic delta contract."""


def _resolve(container: JsonValue, parts: list[str | int]) -> JsonValue:
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


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    """Compare JSON values recursively, keeping booleans distinct from numbers."""
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    return type(left) is type(right) and left == right


def apply_delta(state: JsonObject, delta: JsonValue) -> JsonObject:
    """Apply declared generic delta ops to a deep copy of ``state``.

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
    return delta_edits(state, delta)[0]


def delta_edits(state: JsonObject, delta: JsonValue) -> tuple[JsonObject, tuple[StateEdit, ...]]:
    """Apply generic delta ops and return the next state with the structural edits it took.

    Each op is checked against the working state, then resolved into runtime structural
    primitives that are applied with the runtime's own interpreter: missing ``set`` parents
    become explicit empty-object sets, ``remove_value`` becomes a delete at the matched index,
    and ``copy`` stays a copy so a store can share the copied subtree. Replaying the returned
    edits on ``state`` reproduces the returned state exactly.

    Args:
        state: The current business state. Not mutated.
        delta: The ``delta`` list from the model output.

    Returns:
        The next business state and its ordered structural edits.

    Raises:
        DeltaError: On any undeclared op, bad path, or missing target.
    """
    if not isinstance(delta, list):
        raise DeltaError("delta must be a list of ops")
    new: JsonObject = copy.deepcopy(state)
    edits: list[StateEdit] = []

    def emit(edit: StateEdit) -> None:
        """Apply one resolved primitive to the working state and record it."""
        apply_json_edit(new, edit)
        edits.append(edit)

    for op in delta:
        if not isinstance(op, dict) or op.get("op") not in DELTA_OPS:
            raise DeltaError("undeclared op")
        source: list[str | int] | None = None
        if op["op"] == "copy":
            raw_source = op.get("from")
            if (
                set(op) != {"op", "path", "from"}
                or not isinstance(raw_source, list)
                or not raw_source
                or not all(type(part) in (str, int) for part in raw_source)
            ):
                raise DeltaError("copy requires an exact nonempty source path")
            source = [part for part in raw_source if isinstance(part, (str, int))]
            op = {"op": "set", "path": op["path"], "value": copy.deepcopy(_resolve(new, source))}
        if set(op) - {"op", "path", "value"}:
            raise DeltaError("op carries undeclared fields")
        raw_path = op.get("path")
        if (
            not isinstance(raw_path, list)
            or not raw_path
            or not all(isinstance(p, (str, int)) and not isinstance(p, bool) for p in raw_path)
        ):
            raise DeltaError("bad path")
        path = [part for part in raw_path if isinstance(part, (str, int))]
        kind = op["op"]
        assert isinstance(kind, str)
        if kind == "set":
            if "value" not in op:
                raise DeltaError("set requires value")
            parent, created = _parent_for_set(new, path)
            key = path[-1]
            if not (
                (isinstance(parent, dict) and isinstance(key, str))
                or (isinstance(parent, list) and _index(parent, key) is not None)
            ):
                raise DeltaError("set target invalid")
            for missing in created:
                emit(SetValue(path=tuple(missing), value={}))
            if source is not None and not any(
                missing[: len(source)] == source for missing in created
            ):
                emit(CopyValue(path=tuple(path), source=tuple(source)))
            else:
                emit(SetValue(path=tuple(path), value=op["value"]))
            continue
        parent = _resolve(new, path[:-1])
        key = path[-1]
        if kind == "delete":
            if (isinstance(parent, dict) and key in parent) or (
                isinstance(parent, list) and _index(parent, key) is not None
            ):
                emit(DeleteValue(path=tuple(path)))
            else:
                raise DeltaError("delete target absent")
            continue
        if "value" not in op:
            raise DeltaError(kind + " requires value")
        target = _resolve(new, path)
        if not isinstance(target, list):
            raise DeltaError(kind + " target is not a list")
        if kind == "append":
            emit(AppendValue(path=tuple(path), value=op["value"]))
            continue
        for index, item in enumerate(target):
            if _json_equal(item, op["value"]):
                emit(DeleteValue(path=(*path, index)))
                break
        else:
            raise DeltaError("remove_value value absent")
    return new, tuple(edits)


def _index(node: list[JsonValue], part: str | int) -> int | None:
    """Return ``part`` when it is an existing index of ``node``; booleans are never indexes."""
    if isinstance(part, int) and not isinstance(part, bool) and 0 <= part < len(node):
        return part
    return None


def _parent_for_set(
    state: JsonObject, path: list[str | int]
) -> tuple[JsonValue, list[list[str | int]]]:
    """Return the container for a ``set`` op and the missing object keys it would create.

    Only object keys are created on the way; nothing is mutated here.

    Args:
        state: Working copy of the business state.
        path: Full path of the ``set`` op.

    Returns:
        The receiving container (a fresh empty object when created) and the paths of the
        objects that must be created first, outermost first.

    Raises:
        DeltaError: If an intermediate list index is out of range or a segment cannot be
            followed.
    """
    node: JsonValue = state
    created: list[list[str | int]] = []
    for depth, part in enumerate(path[:-1]):
        if isinstance(node, dict) and isinstance(part, str):
            if part not in node:
                node = {}
                created.append(path[: depth + 1])
            else:
                node = node[part]
        elif isinstance(node, list) and (index := _index(node, part)) is not None:
            node = node[index]
        else:
            raise DeltaError("cannot descend into: " + repr(part))
    return node, created
