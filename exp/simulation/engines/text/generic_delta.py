"""Sequential structural edits on a copy, without target business logic."""

from __future__ import annotations

import copy

from exp.common.core.artifacts import JsonObject, JsonValue

DELTA_OPS: tuple[str, ...] = ("set", "delete", "append", "remove_value")


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
            _resolve(new, path[:-1]) if op["op"] != "set" else _resolve_parent_for_set(new, path)
        )
        key = path[-1]
        kind = op["op"]
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
            target = _resolve(new, path)
            if not isinstance(target, list):
                raise DeltaError(kind + " target is not a list")
            if kind == "append":
                target.append(copy.deepcopy(op["value"]))
            else:
                for index, item in enumerate(target):
                    if _json_equal(item, op["value"]):
                        del target[index]
                        break
                else:
                    raise DeltaError("remove_value value absent")
    return new


def _resolve_parent_for_set(state: JsonObject, path: list[str | int]) -> JsonValue:
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
