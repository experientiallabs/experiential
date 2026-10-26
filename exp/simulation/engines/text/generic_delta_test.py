"""Regression checks for the declared structural edit semantics."""

import copy

import pytest

from exp.common.core.artifacts import JsonObject
from exp.simulation.engines.text.generic_delta import DeltaError, apply_delta


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
