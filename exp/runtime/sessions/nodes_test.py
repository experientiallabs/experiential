"""Tests for the content-addressed JSON node codec: exact text, sharing and decoding."""

from __future__ import annotations

import json

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.sessions.nodes import (
    INLINE_CHARACTERS,
    EncodedNode,
    NodeRef,
    assemble,
    declared_text,
    decode,
    encode,
    resolve_path,
)

ADVERSARIAL: JsonObject = {
    "zeta": 1,
    "alpha": [1, 1.0, -0.0, 1e16, 12345678901234567890, True, False, None],
    "strings": ["", "\u0000", "NaN", "\x00NaN\x00", "é ✓ 𝄞", '"quoted"\\', "line\nbreak"],
    "\u0000key": {"": {}, "nested": [[], [{}], [[None]]]},
    "null_vs_missing": {"present": None},
    "mixed": [{"0": "string key"}, ["index 0"], {"a": [1, {"b": [2, {"c": 3}]}]}],
    "duplicates": [{"same": 1}, {"same": 1}, {"same": 1.0}, {"same": True}],
}


def _round_trip(value: JsonObject) -> tuple[bytes, dict[bytes, EncodedNode], str]:
    """Encode a value and reassemble its text.

    Args:
        value: Object to encode.

    Returns:
        Root digest, node rows and reassembled text.
    """
    nodes: dict[bytes, EncodedNode] = {}
    root = encode(value, nodes)
    return root, nodes, assemble(root, nodes)


def test_reassembled_text_is_byte_identical_to_declared_json() -> None:
    """Key order, scalar spellings, escapes and empty containers survive exactly."""
    _, _, text = _round_trip(ADVERSARIAL)
    assert text == declared_text(ADVERSARIAL)
    decoded = json.loads(text)
    assert decoded == ADVERSARIAL
    assert list(decoded) == list(ADVERSARIAL)
    alpha = ADVERSARIAL["alpha"]
    assert isinstance(alpha, list)
    assert [type(item) for item in decoded["alpha"]] == [type(item) for item in alpha]


def test_equal_subtrees_share_one_node_and_distinct_scalars_do_not() -> None:
    """Sharing follows exact content: 1, 1.0 and true stay distinct nodes."""
    padding = "x" * INLINE_CHARACTERS
    value: JsonObject = {
        "duplicates": [
            {"same": 1, "pad": padding},
            {"same": 1, "pad": padding},
            {"same": 1.0, "pad": padding},
            {"same": True, "pad": padding},
        ]
    }
    root, nodes, _ = _round_trip(value)
    outer = decode(nodes[root])
    assert isinstance(outer, dict)
    array = outer["duplicates"]
    assert isinstance(array, NodeRef)
    items = decode(nodes[array.digest])
    assert isinstance(items, list)
    assert items[0] == items[1]
    assert len({item for item in items if isinstance(item, NodeRef)}) == 3
    assert encode({"a": 2, "b": 1}, {}) != encode({"b": 1, "a": 2}, {})


def test_small_nested_containers_stay_inline_and_encode_canonically() -> None:
    """Only the root and long containers are nodes, however a state was assembled."""
    padding = "y" * INLINE_CHARACTERS
    nodes: dict[bytes, EncodedNode] = {}
    root = encode({"small": {"b": [1]}, "large": {"pad": padding}}, nodes)
    assert len(nodes) == 2
    container = decode(nodes[root])
    assert isinstance(container, dict)
    assert container["small"] == {"b": [1]}
    assert isinstance(container["large"], NodeRef)
    shrunk = encode({"small": {"b": [1]}, "large": {"pad": padding, "gone": padding}}, {})
    rebuilt = {"small": {"b": [1]}, "large": {"pad": padding}}
    assert shrunk != root == encode(rebuilt, {})


def test_decode_keeps_real_nulls_distinct_from_child_placeholders() -> None:
    """A stored null is a scalar; only node positions become references."""
    padding = "z" * INLINE_CHARACTERS
    value: JsonObject = {"a": None, "b": {"pad": padding}, "c": [None, [padding]]}
    nodes: dict[bytes, EncodedNode] = {}
    root = encode(value, nodes)
    container = decode(nodes[root])
    assert isinstance(container, dict)
    assert container["a"] is None
    assert isinstance(container["b"], NodeRef)
    inner = container["c"]
    assert isinstance(inner, NodeRef)
    items = decode(nodes[inner.digest])
    assert isinstance(items, list)
    assert items[0] is None and isinstance(items[1], NodeRef)


def test_encoding_refuses_non_finite_numbers_and_scalars_at_the_root() -> None:
    """Storage never normalizes: non-finite numbers and scalar documents are refused."""
    with pytest.raises(ValueError):
        encode({"bad": float("nan")}, {})
    with pytest.raises(TypeError):
        encode("scalar", {})


def test_unloaded_references_are_kept_by_digest() -> None:
    """Encoding a draft with references never needs the referenced subtree."""
    nodes: dict[bytes, EncodedNode] = {}
    child = encode({"deep": [1, 2]}, nodes)
    draft: dict[str, JsonValue | NodeRef] = {"kept": NodeRef(child), "new": 1}
    fresh: dict[bytes, EncodedNode] = {}
    root = encode(draft, fresh)
    assert child not in fresh
    assert assemble(root, {**nodes, **fresh}) == '{"kept":{"deep":[1,2]},"new":1}'


def test_resolve_path_loads_only_the_path_and_rejects_boolean_indexes() -> None:
    """Path reads follow keys and indexes exactly; a boolean is never an index."""
    nodes: dict[bytes, EncodedNode] = {}
    root = encode({"rows": [{"id": "x"}], "0": "key"}, nodes)
    loaded: list[bytes] = []

    def load(digest: bytes) -> EncodedNode:
        """Record and return one node."""
        loaded.append(digest)
        return nodes[digest]

    assert resolve_path(root, ("rows", 0, "id"), load) == "x"
    assert len(loaded) == 1
    assert resolve_path(root, ("0",), load) == "key"
    with pytest.raises(KeyError):
        resolve_path(root, ("rows", True), load)
    with pytest.raises(KeyError):
        resolve_path(root, ("rows", "0"), load)
    with pytest.raises(KeyError):
        resolve_path(root, ("missing",), load)
