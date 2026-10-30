"""Runtime-owned structural state edits and their single interpreter.

A commit describes its state change as an ordered sequence of four structural primitives. They
carry no business semantics: the coordinator that validated a proposal translates it into these
primitives, and the store only checks that each one applies. The same interpreter edits a plain
in-memory document and a stored draft whose untouched containers stay unloaded references, so
a committed state equals the state the coordinator validated.

Semantics, applied in order, each against the result of the previous edit:

- ``set`` replaces the whole document (empty path), creates or replaces an object key (keeping
  an existing key's position), or replaces an in-range array element. Parents must exist.
- ``delete`` removes an existing object key or in-range array element.
- ``append`` appends to an existing array.
- ``copy`` sets ``path`` to the value found at ``source`` before this edit, without aliasing.

Values are deep-copied into the document, so later edits never reach a caller's objects.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, StrictInt, StrictStr, TypeAdapter

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.sessions.nodes import Draft, DraftValue, NodeRef

type PathPart = StrictStr | StrictInt
"""An object key or an array index; booleans are neither."""


class StateEditError(ValueError):
    """A structural edit does not apply to the document it was given."""


class SetValue(ContractModel):
    """Create or replace one value; an empty path replaces the whole document.

    Attributes:
        op: Discriminator.
        path: Target location; its parent must exist.
        value: New value, deep-copied into the document.
    """

    op: Literal["set"] = "set"
    path: tuple[PathPart, ...]
    value: JsonValue


class DeleteValue(ContractModel):
    """Remove one existing object key or array element.

    Attributes:
        op: Discriminator.
        path: Existing location; never the document itself.
    """

    op: Literal["delete"] = "delete"
    path: tuple[PathPart, ...] = Field(min_length=1)


class AppendValue(ContractModel):
    """Append one value to an existing array.

    Attributes:
        op: Discriminator.
        path: Location of the array.
        value: Appended value, deep-copied into the document.
    """

    op: Literal["append"] = "append"
    path: tuple[PathPart, ...]
    value: JsonValue


class CopyValue(ContractModel):
    """Set one location to the value at another, as it was before this edit.

    Stored subtrees are shared by reference, so copying an unchanged container costs nothing.

    Attributes:
        op: Discriminator.
        path: Target location with ``set`` semantics; never the document itself.
        source: Existing location read before the edit applies.
    """

    op: Literal["copy"] = "copy"
    path: tuple[PathPart, ...] = Field(min_length=1)
    source: tuple[PathPart, ...]


type StateEdit = Annotated[
    SetValue | DeleteValue | AppendValue | CopyValue, Field(discriminator="op")
]
"""One structural primitive."""

StateEditAdapter: TypeAdapter[StateEdit] = TypeAdapter(StateEdit)
"""Validates one recorded primitive from its JSON form."""

type Loader = Callable[[NodeRef], Draft]
"""Loads an unloaded stored container as a fresh mutable draft."""


def prefixed(prefix: tuple[str | int, ...], edits: Iterable[StateEdit]) -> tuple[StateEdit, ...]:
    """Re-root edits written against a subdocument at ``prefix`` of an enclosing document.

    Args:
        prefix: Location of the subdocument.
        edits: Edits relative to the subdocument.

    Returns:
        The same edits relative to the enclosing document.
    """
    rooted: list[StateEdit] = []
    for edit in edits:
        update: dict[str, tuple[str | int, ...]] = {"path": (*prefix, *edit.path)}
        if isinstance(edit, CopyValue):
            update["source"] = (*prefix, *edit.source)
        rooted.append(edit.model_copy(update=update))
    return tuple(rooted)


def apply_edits(document: Draft, edits: Iterable[StateEdit], load: Loader | None = None) -> Draft:
    """Apply edits in order and return the resulting document.

    The document is edited in place except when a ``set`` replaces it whole, so callers keep
    the returned value.

    Args:
        document: Root object, plain JSON or a stored draft.
        edits: Ordered primitives.
        load: Loads unloaded containers; required only for stored drafts.

    Returns:
        The edited root object.

    Raises:
        StateEditError: If an edit does not apply; the document may then be partially edited.
    """
    for edit in edits:
        document = apply_edit(document, edit, load)
    return document


def apply_json_edit(document: JsonObject, edit: StateEdit) -> JsonObject:
    """Apply one primitive to a plain JSON document, which holds no stored references.

    Args:
        document: Root object; edited in place unless the edit replaces it whole.
        edit: Primitive to apply.

    Returns:
        The edited root object, still plain JSON.

    Raises:
        StateEditError: If the edit does not apply.
    """
    return cast("JsonObject", apply_edit(cast("Draft", document), edit))


def apply_edit(
    document: Draft,
    edit: StateEdit,
    load: Loader | None = None,
    *,
    encoded: NodeRef | None = None,
) -> Draft:
    """Apply one primitive.

    Args:
        document: Root object, plain JSON or a stored draft.
        edit: Primitive to apply.
        load: Loads unloaded containers; required only for stored drafts.
        encoded: The already stored form of a ``set`` or ``append`` container value, placed
            instead of a copy of ``edit.value``.

    Returns:
        The edited root object.

    Raises:
        StateEditError: If the edit does not apply.
    """
    if isinstance(edit, CopyValue):
        value = _detached(_lookup(document, edit.source, load))
        return _set(document, edit.path, value, load)
    if isinstance(edit, SetValue):
        value = copy.deepcopy(edit.value) if encoded is None else encoded
        return _set(document, edit.path, value, load)
    if isinstance(edit, DeleteValue):
        parent, key = _container(document, edit.path[:-1], load), edit.path[-1]
        if isinstance(parent, dict) and isinstance(key, str) and key in parent:
            del parent[key]
        elif isinstance(parent, list) and (index := _index(parent, key)) is not None:
            del parent[index]
        else:
            raise StateEditError(f"delete target is absent: {list(edit.path)!r}")
        return document
    target = _container(document, edit.path, load)
    if not isinstance(target, list):
        raise StateEditError(f"append target is not an array: {list(edit.path)!r}")
    target.append(copy.deepcopy(edit.value) if encoded is None else encoded)
    return document


def _set(
    document: Draft, path: tuple[str | int, ...], value: DraftValue, load: Loader | None
) -> Draft:
    """Place ``value`` at ``path`` with ``set`` semantics.

    Args:
        document: Root object.
        path: Target location.
        value: Detached value to place.
        load: Loads unloaded containers.

    Returns:
        The edited root object.

    Raises:
        StateEditError: If the parent is missing or the key does not fit it.
    """
    if not path:
        if isinstance(value, NodeRef) and load is not None:
            value = load(value)
        if not isinstance(value, dict):
            raise StateEditError("the document root must remain an object")
        return value
    parent, key = _container(document, path[:-1], load), path[-1]
    if isinstance(parent, dict) and isinstance(key, str):
        parent[key] = value
    elif isinstance(parent, list) and (index := _index(parent, key)) is not None:
        parent[index] = value
    else:
        raise StateEditError(f"set target is invalid: {list(path)!r}")
    return document


def _container(document: Draft, path: tuple[str | int, ...], load: Loader | None) -> Draft:
    """Return the loaded container at ``path``, loading references on the way.

    Args:
        document: Root object.
        path: Location of an existing container.
        load: Loads unloaded containers.

    Returns:
        The mutable container, now loaded in place within its parent.

    Raises:
        StateEditError: If a segment is missing or the target is a scalar.
    """
    node: Draft = document
    for part in path:
        child = _child(node, part, path)
        if isinstance(child, NodeRef):
            if load is None:
                raise StateEditError("a stored reference requires a loader")
            child = load(child)
            if isinstance(node, dict) and isinstance(part, str):
                node[part] = child
            elif isinstance(node, list) and (index := _index(node, part)) is not None:
                node[index] = child
        if not isinstance(child, (dict, list)):
            raise StateEditError(f"path does not lead to a container: {list(path)!r}")
        node = child
    return node


def _lookup(document: Draft, path: tuple[str | int, ...], load: Loader | None) -> DraftValue:
    """Return the value at ``path`` without loading its own subtree.

    Args:
        document: Root object.
        path: Existing location.
        load: Loads unloaded containers on the way.

    Returns:
        A scalar, a loaded container or an unloaded reference.
    """
    if not path:
        return document
    return _child(_container(document, path[:-1], load), path[-1], path)


def _child(node: Draft, part: str | int, path: tuple[str | int, ...]) -> DraftValue:
    """Return one existing entry of a container.

    Args:
        node: Loaded container.
        part: Key or index.
        path: Full path, for the error message.

    Returns:
        The entry.

    Raises:
        StateEditError: If the entry does not exist.
    """
    if isinstance(node, dict) and isinstance(part, str) and part in node:
        return node[part]
    if isinstance(node, list) and (index := _index(node, part)) is not None:
        return node[index]
    raise StateEditError(f"missing path segment {part!r} in {list(path)!r}")


def _index(node: list[DraftValue], part: str | int) -> int | None:
    """Return ``part`` when it is an existing index of ``node``; booleans are never indexes."""
    if isinstance(part, int) and not isinstance(part, bool) and 0 <= part < len(node):
        return part
    return None


def _detached(value: DraftValue) -> DraftValue:
    """Copy a value so later edits cannot reach its source; references are immutable."""
    if isinstance(value, NodeRef):
        return value
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detached(item) for item in value]
    return value
