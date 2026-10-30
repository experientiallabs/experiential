"""Tests for the runtime structural edit primitives and their interpreter."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.sessions.edits import (
    AppendValue,
    CopyValue,
    DeleteValue,
    SetValue,
    StateEdit,
    StateEditAdapter,
    StateEditError,
    apply_json_edit,
    prefixed,
)


def _apply(document: JsonObject, *edits: StateEdit) -> JsonObject:
    """Apply edits in order to a plain document.

    Args:
        document: Root object, edited in place.
        *edits: Primitives to apply.

    Returns:
        The resulting root.
    """
    for edit in edits:
        document = apply_json_edit(document, edit)
    return document


def test_set_creates_or_replaces_in_place_and_keeps_existing_key_positions() -> None:
    """Replacing a key keeps its position; a new key goes last; indexes must exist."""
    document = _apply(
        {"a": 1, "b": [0, 1], "c": 3},
        SetValue(path=("a",), value={"x": None}),
        SetValue(path=("d",), value=[]),
        SetValue(path=("b", 1), value="one"),
    )
    assert list(document) == ["a", "b", "c", "d"]
    assert document == {"a": {"x": None}, "b": [0, "one"], "c": 3, "d": []}
    with pytest.raises(StateEditError):
        _apply(document, SetValue(path=("b", 2), value=1))
    with pytest.raises(StateEditError):
        _apply(document, SetValue(path=("missing", "child"), value=1))
    with pytest.raises(StateEditError):
        _apply(document, SetValue(path=("b", "0"), value=1))


def test_root_set_replaces_the_document_but_never_with_a_scalar() -> None:
    """An empty path replaces the whole document, which must stay an object."""
    assert _apply({"a": 1}, SetValue(path=(), value={"b": 2})) == {"b": 2}
    with pytest.raises(StateEditError):
        _apply({"a": 1}, SetValue(path=(), value=[1]))


def test_delete_and_append_require_existing_targets() -> None:
    """Deletes reindex arrays; appends need an array; absent targets refuse."""
    document = _apply(
        {"rows": [1, 2, 3], "map": {"k": "v"}},
        DeleteValue(path=("rows", 0)),
        DeleteValue(path=("map", "k")),
        AppendValue(path=("rows",), value=2),
    )
    assert document == {"rows": [2, 3, 2], "map": {}}
    for edit in (
        DeleteValue(path=("rows", 3)),
        DeleteValue(path=("map", "k")),
        AppendValue(path=("map",), value=1),
    ):
        with pytest.raises(StateEditError):
            _apply(document, edit)


def test_values_and_copies_never_alias_the_caller_or_their_source() -> None:
    """Later edits cannot reach a caller's value or a copied source."""
    value: JsonObject = {"inner": [1]}
    edit = SetValue(path=("a",), value=value)
    document = _apply({}, edit, CopyValue(path=("b",), source=("a",)))
    _apply(document, AppendValue(path=("a", "inner"), value=2))
    assert value == {"inner": [1]}
    assert edit.value == {"inner": [1]}
    assert document == {"a": {"inner": [1, 2]}, "b": {"inner": [1]}}


def test_copy_reads_its_source_before_writing_inside_it() -> None:
    """Copying a container into itself captures the value before the edit."""
    document = _apply({"a": {"x": 1}}, CopyValue(path=("a", "self"), source=("a",)))
    assert document == {"a": {"x": 1, "self": {"x": 1}}}


def test_paths_refuse_booleans_and_prefixing_reroots_paths_and_sources() -> None:
    """Booleans are neither keys nor indexes; prefixing rewrites both copy paths."""
    with pytest.raises(ValueError):
        SetValue.model_validate({"path": [True], "value": 1})
    rooted = prefixed(("state",), (CopyValue(path=("b",), source=("a",)),))
    assert rooted == (CopyValue(path=("state", "b"), source=("state", "a")),)
    assert StateEditAdapter.validate_python(rooted[0].model_dump(mode="json")) == rooted[0]


def test_set_value_keeps_exact_scalar_types() -> None:
    """Validation never turns 1.0 into 1 or true into 1."""
    edit = SetValue(path=("v",), value=[1, 1.0, True, None, "1"])
    values = _apply({}, edit)["v"]
    assert isinstance(values, list)
    assert [type(item) for item in values] == [int, float, bool, type(None), str]
