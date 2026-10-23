"""Durable retention owners and resumable reclamation of unowned historical state nodes.

Session heads always keep their current state. A retention pin is a durable, named owner of a
half-open revision range of one session, registered before a historical revision is exposed or
read; it lasts until its owner releases it explicitly, never because a process ended. Collection
deletes only nodes that no head and no pinned revision reaches. The immutable event log is never
reclaimed, so every revision stays replayable even after its nodes are collected.

Collection marks reachability once, then sweeps candidates in bounded write transactions with a
durable cursor, so an interrupted collection resumes where it stopped. A commit that lands between
two batches may reuse a candidate node by digest; every batch therefore re-checks the current
roots and the children of nodes created since the mark, and keeps any candidate they reach.
Readers see a complete revision because each read runs in one SQLite snapshot.

A preview reads one snapshot and writes nothing: it reports what a collection would reclaim now
and, for each pin, the nodes that only that pin keeps, which its release would make collectable.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.runtime.sessions.errors import (
    RetentionConflictError,
    SessionNotFoundError,
    StateUnavailableError,
)
from exp.runtime.sessions.nodes import DIGEST_BYTES, child_digests

type Transaction = Callable[[], AbstractContextManager[sqlite3.Connection]]
"""Opens one write transaction; its context exit commits or rolls back."""


class RetentionPin(ContractModel):
    """A durable owner of one session's revisions ``[start_revision, end_revision)``.

    Attributes:
        owner_id: Stable owner name, such as a checkpoint, fork or frozen evaluation identity.
        session_id: Session whose history is retained.
        start_revision: First retained revision.
        end_revision: First revision after the retained range.
    """

    owner_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    start_revision: int = Field(ge=0)
    end_revision: int = Field(gt=0)


class CollectionReport(ContractModel):
    """What one collection run reclaimed and kept.

    Deleting rows frees pages for reuse inside the database; the file does not shrink on its
    own, and the operating system gets space back only after an explicit vacuum.

    Attributes:
        reclaimed_nodes: Node rows deleted by this run.
        reclaimed_bytes: Template and children bytes of the deleted rows.
        retained_nodes: Nodes reachable from heads and pins at the mark.
        complete: Whether the sweep finished; False after a bounded stop, resumable later.
    """

    reclaimed_nodes: int = Field(ge=0)
    reclaimed_bytes: int = Field(ge=0)
    retained_nodes: int = Field(ge=0)
    complete: bool


class PinHolding(ContractModel):
    """One pin and what it alone keeps alive.

    Attributes:
        pin: The registered owner and revision range.
        exclusive_nodes: Nodes no head and no other pin reaches; releasing the pin frees them.
        exclusive_bytes: Template, children and digest bytes of those nodes.
    """

    pin: RetentionPin
    exclusive_nodes: int = Field(ge=0)
    exclusive_bytes: int = Field(ge=0)


class CollectionPreview(ContractModel):
    """What a collection would reclaim now, and what each pin holds, from one snapshot.

    Attributes:
        reclaimable_nodes: Stored nodes no head and no pin reaches.
        reclaimable_bytes: Template, children and digest bytes of those nodes.
        retained_nodes: Nodes reachable from heads and pins.
        pins: Every registered pin, ordered by owner and session.
    """

    reclaimable_nodes: int = Field(ge=0)
    reclaimable_bytes: int = Field(ge=0)
    retained_nodes: int = Field(ge=0)
    pins: tuple[PinHolding, ...]


def pin_revisions(connection: sqlite3.Connection, pin: RetentionPin) -> RetentionPin:
    """Register a pin after checking that every revision in its range is still materialized.

    A bounded or interrupted collection can delete an unowned revision's shared child and stop
    before its root, so the check walks every node each revision reaches, not only its root.
    Pinning a half-reclaimed revision would expose unreadable state and make every later
    collection refuse to mark.

    Args:
        connection: Connection inside a write transaction.
        pin: Requested owner and range.

    Returns:
        The registered pin; an identical existing pin is returned unchanged.

    Raises:
        SessionNotFoundError: If the session does not exist.
        RetentionConflictError: If the owner already pins another range of the session, or
            the range extends past the session head.
        StateUnavailableError: If a node of a revision in the range was already reclaimed.
    """
    head = connection.execute(
        "SELECT version FROM sessions WHERE session_id = ?", (pin.session_id,)
    ).fetchone()
    if head is None:
        raise SessionNotFoundError(f"session {pin.session_id!r} does not exist")
    if pin.start_revision >= pin.end_revision or pin.end_revision > head["version"] + 1:
        raise RetentionConflictError(
            f"pin range [{pin.start_revision}, {pin.end_revision}) is outside the committed "
            f"revisions 0..{head['version']} of session {pin.session_id!r}"
        )
    existing = connection.execute(
        "SELECT start_revision, end_revision FROM state_pins WHERE owner_id = ? AND session_id = ?",
        (pin.owner_id, pin.session_id),
    ).fetchone()
    if existing is not None:
        if (existing["start_revision"], existing["end_revision"]) != (
            pin.start_revision,
            pin.end_revision,
        ):
            raise RetentionConflictError(
                f"owner {pin.owner_id!r} already pins another range of {pin.session_id!r}; "
                "release it first"
            )
        return pin
    roots = {
        bytes(row[0])
        for row in connection.execute(
            "SELECT root FROM state_events WHERE session_id = ? AND revision >= ? AND revision < ?",
            (pin.session_id, pin.start_revision, pin.end_revision),
        )
    }
    try:
        _reachable(connection, roots)
    except StateUnavailableError as error:
        raise StateUnavailableError(
            f"revisions [{pin.start_revision}, {pin.end_revision}) of {pin.session_id!r} were"
            " partly reclaimed; they cannot be pinned"
        ) from error
    connection.execute(
        "INSERT INTO state_pins (owner_id, session_id, start_revision, end_revision)"
        " VALUES (?, ?, ?, ?)",
        (pin.owner_id, pin.session_id, pin.start_revision, pin.end_revision),
    )
    return pin


def release_pin(connection: sqlite3.Connection, owner_id: str, session_id: str) -> bool:
    """Release one owner's pin; its revisions become collectable unless another owner holds them.

    Args:
        connection: Connection inside a write transaction.
        owner_id: Owner that registered the pin.
        session_id: Pinned session.

    Returns:
        Whether a pin was released.
    """
    cursor = connection.execute(
        "DELETE FROM state_pins WHERE owner_id = ? AND session_id = ?", (owner_id, session_id)
    )
    return cursor.rowcount > 0


def live_roots(connection: sqlite3.Connection) -> set[bytes]:
    """Return the roots of every session head and every pinned revision.

    Args:
        connection: Open connection inside a transaction.

    Returns:
        Root digests that collection must keep.
    """
    roots = {bytes(row[0]) for row in connection.execute("SELECT root FROM sessions")}
    roots.update(
        bytes(row[0])
        for row in connection.execute(
            "SELECT e.root FROM state_events e JOIN state_pins p ON p.session_id = e.session_id"
            " AND e.revision >= p.start_revision AND e.revision < p.end_revision"
        )
    )
    return roots


def registered_pins(connection: sqlite3.Connection) -> tuple[RetentionPin, ...]:
    """Return every registered pin, ordered by owner and session.

    Args:
        connection: Open connection.

    Returns:
        The pins.
    """
    return tuple(
        RetentionPin(
            owner_id=row["owner_id"],
            session_id=row["session_id"],
            start_revision=row["start_revision"],
            end_revision=row["end_revision"],
        )
        for row in connection.execute(
            "SELECT owner_id, session_id, start_revision, end_revision FROM state_pins"
            " ORDER BY owner_id, session_id"
        )
    )


def preview_collection(connection: sqlite3.Connection) -> CollectionPreview:
    """Report reclaimable nodes and each pin's exclusive holding without writing anything.

    Args:
        connection: Connection inside a read transaction, so every query sees one snapshot.

    Returns:
        The preview.
    """
    sizes = {
        bytes(row[0]): row[1] + DIGEST_BYTES
        for row in connection.execute(
            "SELECT digest, length(template) + length(children) FROM state_nodes"
        )
    }
    heads = {bytes(row[0]) for row in connection.execute("SELECT root FROM sessions")}
    pins = registered_pins(connection)
    pin_roots = {
        (pin.owner_id, pin.session_id): {
            bytes(row[0])
            for row in connection.execute(
                "SELECT root FROM state_events WHERE session_id = ? AND revision >= ?"
                " AND revision < ?",
                (pin.session_id, pin.start_revision, pin.end_revision),
            )
        }
        for pin in pins
    }
    reachable = _reachable(connection, heads.union(*pin_roots.values()))
    holdings = []
    for pin in pins:
        key = (pin.owner_id, pin.session_id)
        others = heads.union(*(roots for owner, roots in pin_roots.items() if owner != key))
        exclusive = reachable - _reachable(connection, others)
        holdings.append(
            PinHolding(
                pin=pin,
                exclusive_nodes=len(exclusive),
                exclusive_bytes=sum(sizes[digest] for digest in exclusive),
            )
        )
    unreachable = sizes.keys() - reachable
    return CollectionPreview(
        reclaimable_nodes=len(unreachable),
        reclaimable_bytes=sum(sizes[digest] for digest in unreachable),
        retained_nodes=len(reachable),
        pins=tuple(holdings),
    )


def collect(
    connection: sqlite3.Connection,
    transaction: Transaction,
    *,
    batch_size: int,
    maximum_batches: int | None,
) -> CollectionReport:
    """Reclaim nodes unreachable from heads and pins, in bounded resumable batches.

    The mark reads one snapshot and blocks no writer. Node rows are numbered monotonically and
    never reuse a number, so every row above the recorded high-water mark was created after it
    and is never a candidate.

    Args:
        connection: Connection outside any transaction, used for the reachability mark.
        transaction: Opens each sweep batch's write transaction.
        batch_size: Candidate rows examined per batch.
        maximum_batches: Stop after this many batches; None sweeps to completion.

    Returns:
        What this run reclaimed; ``complete`` is False when it stopped early.
    """
    connection.execute("BEGIN")
    try:
        state = connection.execute(
            "SELECT high_water, cursor FROM state_collection WHERE singleton = 1"
        ).fetchone()
        if state is None:
            high_water = connection.execute(
                "SELECT seq FROM sqlite_sequence WHERE name = 'state_nodes'"
            ).fetchone()
            high_water, cursor = (0 if high_water is None else high_water[0]), 0
        else:
            high_water, cursor = state["high_water"], state["cursor"]
        reachable = _reachable(connection, live_roots(connection))
    finally:
        connection.execute("COMMIT")
    if state is None:
        with transaction() as start:
            start.execute(
                "INSERT OR IGNORE INTO state_collection (singleton, high_water, cursor)"
                " VALUES (1, ?, 0)",
                (high_water,),
            )
    reclaimed_nodes = reclaimed_bytes = batches = 0
    while maximum_batches is None or batches < maximum_batches:
        with transaction() as batch:
            rows = batch.execute(
                "SELECT node_id, digest, length(template) + length(children) AS size"
                " FROM state_nodes WHERE node_id > ? AND node_id <= ? ORDER BY node_id LIMIT ?",
                (cursor, high_water, batch_size),
            ).fetchall()
            if not rows:
                batch.execute("DELETE FROM state_collection WHERE singleton = 1")
                return CollectionReport(
                    reclaimed_nodes=reclaimed_nodes,
                    reclaimed_bytes=reclaimed_bytes,
                    retained_nodes=len(reachable),
                    complete=True,
                )
            candidates = [row for row in rows if bytes(row["digest"]) not in reachable]
            protected = _reused_since_mark(batch, reachable, high_water) if candidates else set()
            doomed = [row for row in candidates if bytes(row["digest"]) not in protected]
            batch.executemany(
                "DELETE FROM state_nodes WHERE node_id = ?", ((row["node_id"],) for row in doomed)
            )
            cursor = rows[-1]["node_id"]
            batch.execute("UPDATE state_collection SET cursor = ? WHERE singleton = 1", (cursor,))
            reclaimed_nodes += len(doomed)
            reclaimed_bytes += sum(row["size"] + DIGEST_BYTES for row in doomed)
        batches += 1
    return CollectionReport(
        reclaimed_nodes=reclaimed_nodes,
        reclaimed_bytes=reclaimed_bytes,
        retained_nodes=len(reachable),
        complete=False,
    )


def _reachable(connection: sqlite3.Connection, roots: Iterable[bytes]) -> set[bytes]:
    """Return every stored node reachable from ``roots``.

    Args:
        connection: Connection inside a transaction.
        roots: Starting digests.

    Returns:
        Reachable digests.

    Raises:
        StateUnavailableError: If a live root reaches a missing node.
    """
    seen: set[bytes] = set()
    frontier = set(roots)
    while frontier:
        seen.update(frontier)
        found = _children(connection, frontier)
        if len(found) != len(frontier):
            raise StateUnavailableError("a live session state reaches a missing node")
        frontier = {child for children in found.values() for child in children} - seen
    return seen


def _reused_since_mark(
    connection: sqlite3.Connection, reachable: set[bytes], high_water: int
) -> set[bytes]:
    """Return unmarked nodes that commits after the mark made reachable again.

    Args:
        connection: Connection inside the batch's write transaction.
        reachable: Nodes reachable at the mark; their subtrees are closed under reachability.
        high_water: Largest node row that existed at the mark.

    Returns:
        Candidate digests that must survive this batch.
    """
    starts = live_roots(connection)
    for row in connection.execute(
        "SELECT children FROM state_nodes WHERE node_id > ?", (high_water,)
    ):
        starts.update(child_digests(row[0]))
    protected: set[bytes] = set()
    frontier = starts - reachable
    while frontier:
        old = {
            bytes(row["digest"]): child_digests(row["children"])
            for row in _rows(connection, frontier)
            if row["node_id"] <= high_water
        }
        protected.update(old)
        frontier = {child for children in old.values() for child in children}
        frontier -= reachable | protected
    return protected


def _children(connection: sqlite3.Connection, digests: set[bytes]) -> dict[bytes, list[bytes]]:
    """Return the child digests of each stored node among ``digests``."""
    return {
        bytes(row["digest"]): child_digests(row["children"]) for row in _rows(connection, digests)
    }


def _rows(connection: sqlite3.Connection, digests: set[bytes]) -> list[sqlite3.Row]:
    """Return the stored rows for ``digests``, querying in bounded chunks."""
    wanted = sorted(digests)
    rows: list[sqlite3.Row] = []
    for start in range(0, len(wanted), 500):
        chunk = wanted[start : start + 500]
        rows.extend(
            connection.execute(
                "SELECT node_id, digest, children FROM state_nodes WHERE digest IN"
                f" ({', '.join('?' for _ in chunk)})",
                chunk,
            ).fetchall()
        )
    return rows
