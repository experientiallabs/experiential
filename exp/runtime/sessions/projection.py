"""Incremental SQLite projection of session state: genesis, sparse commits and revision reads.

Every session revision has one immutable event row: genesis records the initial state, and each
commit records its ordered structural edits with the operation that produced them. The event
names the resulting root node, and the session head names the current one. A commit loads only
the containers its edits touch, re-encodes those containers and their ancestors, and inserts
only nodes the store does not already hold. Unchanged subtrees, including the state copies that
transitions keep, stay shared by digest.

Every function here runs inside a transaction its caller owns, so the event, the new nodes and
the head publish together or not at all.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, Sha256
from exp.runtime.sessions.edits import (
    AppendValue,
    SetValue,
    StateEdit,
    StateEditAdapter,
    apply_edit,
)
from exp.runtime.sessions.errors import StateUnavailableError
from exp.runtime.sessions.nodes import (
    Draft,
    EncodedNode,
    NodeRef,
    assemble,
    child_digests,
    declared_text,
    decode,
    encode,
    encode_nested,
    node_digest,
    resolve_path,
)

_CHUNK = 500
"""Digests fetched per query while loading a subtree."""


class StateRef(ContractModel):
    """A revision-bound reference to one immutable session state; it carries no JSON.

    Reading it is an explicit store call. Holding it does not retain the revision: only a
    retention pin or the session head keeps a revision's nodes from collection.

    Attributes:
        store_id: Identity of the store that issued the reference.
        session_id: Session whose revision this is.
        revision: Committed revision; 0 is genesis.
        root_sha256: Digest of the revision's root node.
    """

    store_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    root_sha256: Sha256


class PreparedEdits(NamedTuple):
    """Edits whose container values were encoded before the commit transaction.

    Attributes:
        edits: The ordered primitives, exactly as recorded in the event log.
        values: Digest of each edit's encoded container value, or None.
    """

    edits: tuple[StateEdit, ...]
    values: tuple[bytes | None, ...]


def prepare_edits(edits: Sequence[StateEdit], nodes: dict[bytes, EncodedNode]) -> PreparedEdits:
    """Encode every container value outside the commit transaction.

    Args:
        edits: Ordered primitives.
        nodes: Output node rows of the encoded values.

    Returns:
        The edits with the digest of each container value.
    """
    values: list[bytes | None] = []
    for edit in edits:
        if isinstance(edit, SetValue) and not edit.path:
            values.append(encode(edit.value, nodes) if isinstance(edit.value, dict) else None)
        elif isinstance(edit, (SetValue, AppendValue)):
            values.append(encode_nested(edit.value, nodes))
        else:
            values.append(None)
    return PreparedEdits(edits=tuple(edits), values=tuple(values))


def insert_nodes(connection: sqlite3.Connection, nodes: Mapping[bytes, EncodedNode]) -> None:
    """Insert node rows the store does not hold yet; existing digests are left untouched.

    Args:
        connection: Connection inside a write transaction.
        nodes: Rows keyed by digest.
    """
    connection.executemany(
        "INSERT OR IGNORE INTO state_nodes (digest, template, children) VALUES (?, ?, ?)",
        ((digest, node.template, node.children) for digest, node in nodes.items()),
    )


def load_node(connection: sqlite3.Connection, digest: bytes) -> EncodedNode:
    """Read and verify one node.

    Args:
        connection: Open connection.
        digest: Node digest.

    Returns:
        The verified node row.

    Raises:
        StateUnavailableError: If the node is missing or its content differs from its digest.
    """
    row = connection.execute(
        "SELECT template, children FROM state_nodes WHERE digest = ?", (digest,)
    ).fetchone()
    if row is None:
        raise StateUnavailableError("state node is missing; its revision was reclaimed")
    return _verified(digest, row["template"], row["children"])


def fetch_subtree(connection: sqlite3.Connection, root: bytes) -> dict[bytes, EncodedNode]:
    """Read and verify every distinct node reachable from ``root``.

    Args:
        connection: Connection inside a read or write transaction.
        root: Root digest.

    Returns:
        The reachable nodes keyed by digest.

    Raises:
        StateUnavailableError: If any reachable node is missing or corrupt.
    """
    nodes: dict[bytes, EncodedNode] = {}
    frontier = {root}
    while frontier:
        wanted = sorted(frontier)
        found: dict[bytes, EncodedNode] = {}
        for start in range(0, len(wanted), _CHUNK):
            chunk = wanted[start : start + _CHUNK]
            rows = connection.execute(
                "SELECT digest, template, children FROM state_nodes WHERE digest IN"
                f" ({', '.join('?' for _ in chunk)})",
                chunk,
            ).fetchall()
            for row in rows:
                found[row["digest"]] = _verified(row["digest"], row["template"], row["children"])
        if len(found) != len(wanted):
            raise StateUnavailableError("state node is missing; its revision was reclaimed")
        nodes.update(found)
        frontier = {
            child
            for node in found.values()
            for child in child_digests(node.children)
            if child not in nodes
        }
    return nodes


def read_value(
    connection: sqlite3.Connection, root: bytes, path: tuple[str | int, ...]
) -> JsonValue:
    """Materialize the value at ``path`` of one stored state as fresh JSON.

    Args:
        connection: Connection inside a read transaction, so the revision stays consistent.
        root: Root digest of the revision.
        path: Location inside the state; empty for the whole state.

    Returns:
        A detached JSON value with its declared key order and exact scalars.

    Raises:
        KeyError: If the path does not exist.
        StateUnavailableError: If the revision's nodes were reclaimed.
    """
    value = resolve_path(root, path, lambda digest: load_node(connection, digest))
    if not isinstance(value, NodeRef):
        return value
    return json.loads(assemble(value.digest, fetch_subtree(connection, value.digest)))


def commit_edits(
    connection: sqlite3.Connection,
    root: bytes,
    prepared: PreparedEdits,
    values: Mapping[bytes, EncodedNode],
) -> tuple[bytes, dict[bytes, EncodedNode]]:
    """Apply prepared edits to a stored root, loading only the containers they touch.

    Args:
        connection: Connection inside the commit's write transaction.
        root: Root digest of the base revision.
        prepared: Edits with their pre-encoded container values.
        values: Node rows of the pre-encoded values, which may not be stored yet.

    Returns:
        The resulting root digest and every node row the commit must insert.

    Raises:
        StateEditError: If an edit does not apply; the caller's transaction then rolls back.
    """
    stored: set[bytes] = set()

    def load(reference: NodeRef) -> Draft:
        """Load one container into the draft, remembering which ones the store holds."""
        pending = values.get(reference.digest)
        if pending is not None:
            return decode(pending)
        stored.add(reference.digest)
        return decode(load_node(connection, reference.digest))

    document: Draft = load(NodeRef(root))
    for edit, value in zip(prepared.edits, prepared.values, strict=True):
        document = apply_edit(
            document, edit, load, encoded=None if value is None else NodeRef(value)
        )
    nodes = dict(values)
    result = encode(document, nodes)
    return result, _unstored_reachable(result, nodes, stored)


def _unstored_reachable(
    root: bytes, nodes: Mapping[bytes, EncodedNode], stored: set[bytes]
) -> dict[bytes, EncodedNode]:
    """Return the encoded nodes the new root reaches that the store may not hold yet.

    A value an edit placed and a later edit replaced is never inserted, and a subtree loaded
    from the base revision already exists, so its descendants are not visited.

    Args:
        root: Resulting root digest.
        nodes: Every node encoded during the commit.
        stored: Digests loaded from the store.

    Returns:
        The rows to insert.
    """
    reachable: dict[bytes, EncodedNode] = {}
    frontier = [root]
    while frontier:
        digest = frontier.pop()
        node = nodes.get(digest)
        if digest in reachable or digest in stored or node is None:
            continue
        reachable[digest] = node
        frontier.extend(child_digests(node.children))
    return reachable


def record_event(
    connection: sqlite3.Connection,
    session_id: str,
    revision: int,
    step_id: str | None,
    effect: JsonObject,
    root: bytes,
) -> None:
    """Append one immutable event naming the revision's root.

    Args:
        connection: Connection inside the publishing write transaction.
        session_id: Session the event belongs to.
        revision: Resulting revision.
        step_id: Operation that produced it; None for genesis.
        effect: Replayable effect: the initial state or the ordered edits.
        root: Resulting root digest.
    """
    connection.execute(
        "INSERT INTO state_events (session_id, revision, step_id, effect_json, root)"
        " VALUES (?, ?, ?, ?, ?)",
        (session_id, revision, step_id, declared_text(effect), root),
    )


def edits_effect(edits: Iterable[StateEdit]) -> JsonObject:
    """Return the replayable event effect of ordered edits."""
    return {"kind": "edits", "edits": [edit.model_dump(mode="json") for edit in edits]}


def genesis_effect(state: JsonObject) -> JsonObject:
    """Return the replayable event effect of a session's initial state."""
    return {"kind": "genesis", "state": state}


def event_root(connection: sqlite3.Connection, session_id: str, revision: int) -> bytes | None:
    """Return the recorded root of one revision, or None when the revision does not exist."""
    row = connection.execute(
        "SELECT root FROM state_events WHERE session_id = ? AND revision = ?",
        (session_id, revision),
    ).fetchone()
    return None if row is None else bytes(row["root"])


def replay_roots(connection: sqlite3.Connection, session_id: str) -> list[bytes]:
    """Recompute every revision's root from the event log alone, without stored nodes.

    Args:
        connection: Connection inside a read transaction.
        session_id: Session to replay.

    Returns:
        The recomputed root of each revision, from genesis.
    """
    scratch: dict[bytes, EncodedNode] = {}

    def load(reference: NodeRef) -> Draft:
        """Load one container produced earlier in this replay."""
        return decode(scratch[reference.digest])

    roots: list[bytes] = []
    for row in connection.execute(
        "SELECT effect_json FROM state_events WHERE session_id = ? ORDER BY revision",
        (session_id,),
    ):
        effect = json.loads(row["effect_json"])
        if effect["kind"] == "genesis":
            roots.append(encode(effect["state"], scratch))
            continue
        document = load(NodeRef(roots[-1]))
        for raw in effect["edits"]:
            document = apply_edit(document, StateEditAdapter.validate_python(raw), load)
        roots.append(encode(document, scratch))
    return roots


def _verified(digest: bytes, template: str, children: bytes) -> EncodedNode:
    """Return a node row after checking it against its digest.

    Raises:
        StateUnavailableError: If the stored content does not match its digest.
    """
    if node_digest(template, children) != digest:
        raise StateUnavailableError("stored state node does not match its digest")
    return EncodedNode(template, children)
