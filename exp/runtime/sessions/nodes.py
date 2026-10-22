"""Content-addressed JSON nodes: the codec behind incremental session state.

A stored state is a tree of immutable container nodes. A node's ``template`` is its exact
declared JSON text with every nested node replaced by one NUL placeholder, and its ``children``
are the 32-byte digests of those nested nodes in textual order. Scalars, and nested containers
whose whole text is short, stay inline in their parent's template, so the template keeps the
original string, number and boolean spellings and the declared key order byte for byte.

A node's digest covers its template and its children's digests, so equal digests mean equal
subtrees, and an unchanged subtree is shared by every revision, session and fork that contains
it. JSON text never contains a raw NUL (the encoder escapes it inside strings), so the
placeholder is unambiguous. Decoding substitutes ``NaN`` for each placeholder and resolves it
through ``parse_constant``; stored values never contain ``NaN`` because the encoder refuses
non-finite numbers.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

from exp.common.core.artifacts import JsonValue

PLACEHOLDER = "\x00"
"""Marker standing for one nested container inside a node template."""

DIGEST_BYTES = 32
"""Width of one SHA-256 child digest inside a node's ``children`` blob."""

INLINE_CHARACTERS = 128
"""Nested containers whose whole text is at most this long stay inline in their parent.

Whether a nested container is a node depends only on its content, so equal states always
encode to equal digests however they were built.
"""

_dumps = partial(json.dumps, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class NodeRef:
    """An unloaded stored container, standing in a draft for its whole immutable subtree.

    Attributes:
        digest: SHA-256 node digest of the referenced container.
    """

    digest: bytes


type Draft = dict[str, DraftValue] | list[DraftValue]
"""A loaded container whose nested containers may still be unloaded references."""

type DraftValue = JsonValue | NodeRef | Draft
"""One value in a draft: a scalar, an unloaded reference or a loaded container."""


class EncodedNode(NamedTuple):
    """One immutable container row.

    Attributes:
        template: Declared JSON text with a placeholder for each nested container.
        children: Concatenated digests of the nested containers, in textual order.
    """

    template: str
    children: bytes


def declared_text(value: JsonValue) -> str:
    """Render JSON as compact UTF-8 text that keeps declared key order and exact scalars.

    Args:
        value: JSON value to render.

    Returns:
        The text every stored state, proposal and response uses.
    """
    return _dumps(value)


def node_digest(template: str, children: bytes) -> bytes:
    """Return the identity of one node from its template and its children's identities.

    Args:
        template: Node template text.
        children: Concatenated child digests.

    Returns:
        A 32-byte SHA-256 digest.
    """
    encoded = template.encode("utf-8")
    digest = hashlib.sha256(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(children)
    return digest.digest()


def encode(value: DraftValue, nodes: dict[bytes, EncodedNode]) -> bytes:
    """Encode one container as a node, collecting the rows of its node descendants.

    The value itself always becomes a node, whatever its size, because it is a document root
    or an explicitly placed subtree. Unloaded references are kept by digest without visiting
    their subtrees, so encoding a draft costs only its loaded containers.

    Args:
        value: Container or reference to encode.
        nodes: Output map from digest to node row; the caller persists it.

    Returns:
        The container's digest.

    Raises:
        TypeError: If ``value`` is a scalar or an object has a non-string key.
        ValueError: If the value contains a non-finite number.
    """
    if isinstance(value, NodeRef):
        return value.digest
    if not isinstance(value, (dict, list)):
        raise TypeError("only objects and arrays are stored as nodes")
    template, children = _parts(value, nodes)
    digest = node_digest(template, children)
    nodes.setdefault(digest, EncodedNode(template, children))
    return digest


def encode_nested(value: DraftValue, nodes: dict[bytes, EncodedNode]) -> bytes | None:
    """Encode a container placed inside another, or report that it stays inline there.

    Args:
        value: Container or reference.
        nodes: Output map from digest to node row.

    Returns:
        The node digest, or None when the container is small enough to stay inline text.
    """
    if isinstance(value, NodeRef):
        return value.digest
    if not isinstance(value, (dict, list)):
        return None
    template, children = _parts(value, nodes)
    if not children and len(template) <= INLINE_CHARACTERS:
        return None
    digest = node_digest(template, children)
    nodes.setdefault(digest, EncodedNode(template, children))
    return digest


def _parts(value: Draft, nodes: dict[bytes, EncodedNode]) -> tuple[str, bytes]:
    """Return a container's template and child digests, encoding node children first.

    Args:
        value: Loaded container.
        nodes: Output node map.

    Returns:
        The template and the concatenated child digests.

    Raises:
        TypeError: If an object has a non-string key.
    """
    entries = value.values() if isinstance(value, dict) else value
    if not any(isinstance(item, (dict, list, NodeRef)) for item in entries):
        if isinstance(value, dict) and not all(isinstance(key, str) for key in value):
            raise TypeError("stored objects require string keys")
        return _dumps(value), b""
    digests: list[bytes] = []
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("stored objects require string keys")
            parts.append(_dumps(key) + ":" + _slot(item, nodes, digests))
        return "{" + ",".join(parts) + "}", b"".join(digests)
    template = "[" + ",".join(_slot(item, nodes, digests) for item in value) + "]"
    return template, b"".join(digests)


def _slot(item: DraftValue, nodes: dict[bytes, EncodedNode], digests: list[bytes]) -> str:
    """Render one entry: a placeholder for a node child, the exact text for anything else.

    Args:
        item: Entry value.
        nodes: Output node map shared with the enclosing encoding.
        digests: Child digests of the enclosing container, appended in textual order.

    Returns:
        The entry's template text.
    """
    if isinstance(item, (dict, list, NodeRef)):
        digest = encode_nested(item, nodes)
        if digest is None:
            return _dumps(item)
        digests.append(digest)
        return PLACEHOLDER
    return _dumps(item)


def child_digests(children: bytes) -> list[bytes]:
    """Split a ``children`` blob into its ordered child digests.

    Args:
        children: Concatenated 32-byte digests.

    Returns:
        The digests in textual order.
    """
    return [children[i : i + DIGEST_BYTES] for i in range(0, len(children), DIGEST_BYTES)]


def decode(node: EncodedNode) -> Draft:
    """Decode one node into a container whose nested containers are unloaded references.

    Args:
        node: Stored node row.

    Returns:
        A fresh mutable container.

    Raises:
        ValueError: If the template and children disagree.
    """
    references = iter([NodeRef(digest) for digest in child_digests(node.children)])

    def reference(_: str) -> NodeRef:
        """Resolve the next placeholder to the next child reference."""
        return next(references)

    value = json.loads(node.template.replace(PLACEHOLDER, "NaN"), parse_constant=reference)
    if next(references, None) is not None or not isinstance(value, (dict, list)):
        raise ValueError("stored node template and children disagree")
    return value


def assemble(root: bytes, nodes: Mapping[bytes, EncodedNode]) -> str:
    """Reassemble the exact declared JSON text of a stored subtree.

    Each distinct node is rendered once, so subtrees shared between revisions cost one
    rendering however often they occur.

    Args:
        root: Digest of the subtree's root container.
        nodes: Every node reachable from ``root``.

    Returns:
        The subtree's declared JSON text.
    """
    texts: dict[bytes, str] = {}

    def render(digest: bytes) -> str:
        """Render one node, reusing renderings of shared subtrees."""
        text = texts.get(digest)
        if text is None:
            node = nodes[digest]
            if not node.children:
                text = node.template
            else:
                pieces = node.template.split(PLACEHOLDER)
                rendered = [render(child) for child in child_digests(node.children)]
                text = "".join(
                    piece + child for piece, child in zip(pieces, [*rendered, ""], strict=True)
                )
            texts[digest] = text
        return text

    return render(root)


def resolve_path(
    root: bytes, path: tuple[str | int, ...], load: Callable[[bytes], EncodedNode]
) -> DraftValue:
    """Follow a path from a stored root, loading only the nodes on that path.

    Args:
        root: Digest of the root container.
        path: Object keys and array indexes; booleans are never indexes.
        load: Reads one node by digest.

    Returns:
        The scalar, inline container or unloaded reference at ``path``.

    Raises:
        KeyError: If a segment is missing or cannot be followed.
    """
    value: DraftValue = NodeRef(root)
    for part in path:
        container = decode(load(value.digest)) if isinstance(value, NodeRef) else value
        if isinstance(container, dict) and isinstance(part, str) and part in container:
            value = container[part]
        elif (
            isinstance(container, list)
            and isinstance(part, int)
            and not isinstance(part, bool)
            and 0 <= part < len(container)
        ):
            value = container[part]
        else:
            raise KeyError(part)
    return value
