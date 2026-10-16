"""Root exposure ledger: which recorded trace content a person has seen, and what it grouped with.

Held-out evidence is independent only while nobody has read its per-trace results. Project-local
leakage lineage IDs cannot carry that fact across builds: they change on reimport and when
grouping changes. The ledger therefore lives at the ``.exp`` root, above any one project, in one
SQLite file, and keys everything on content:

- Nodes are fingerprints of normalized traces (task text, recorded actions, recorded replies)
  under a versioned scheme, with the source trace ID and any declared task identity
  (``task_lineage``) kept as attributes.
- Edges are grouping relationships recorded for every build, whether or not anything is exposed:
  two traces that declare the same task identity, or the build's own near-duplicate union with
  its similarity. Each edge names the project and build that produced it. Nothing weaker (a
  shared customer, conversation or time bucket) is an edge.
- Exposures mark nodes a person has seen, with the panel, receipt and report that showed them.
- Reservations claim content for one audit receipt before any of its runs is registered; a
  release by that receipt's owner frees an unused reservation.

Content is exposed when it shares a connected component with an exposed node, and taken when it
shares one with content another receipt reserved. Both relations are transitive and ignore time:
an edge recorded before or after the exposure propagates it, and it keeps propagating after the
exposed trace leaves every later import. Every table is append-only (SQL triggers refuse edits
and deletions), so a snapshot identity names exactly the rows a decision used, and independence
is only relative to that ledger.

Atomicity holds for processes that share this one ledger file on one host; SQLite and OS file
locks over network filesystems are unreliable. A panel lease is an exclusive OS file lock, so it
dies with its process and a crash leaves no stale lease.
"""

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal, Self

from filelock import FileLock, Timeout
from pydantic import AwareDatetime, BaseModel, Field, JsonValue, model_validator

from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.core.text import normalize_durable_text
from exp.common.core.union_find import UnionFind
from exp.common.sqlite.connection import connect_database
from exp.common.traces import Trace
from exp.simulation.mining.deduplicate import DuplicateEdge
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_real_transitions

FINGERPRINT_SCHEME = "trace-content-v1"
LEDGER_DIRECTORY = "exposure-ledger"
LEDGER_FILE = "ledger.sqlite"
SCHEMA_VERSION: Final = 1
TASK_LINEAGE_KEY = "task_lineage"
SAME_HOST_LIMIT = (
    "Reservations and panel leases are atomic only for runs sharing one ledger file on one host; "
    "SQLite and file locks over network filesystems are unreliable."
)
_FINGERPRINT_LINEAGE = "lineage-fingerprint"  # placeholder binding; never part of the content
Fingerprint = str
EdgeReason = Literal["declared-task-identity", "near-duplicate"]
RecordKind = Literal["build", "node", "edge", "exposure"]


class PanelInUseError(RuntimeError):
    """Another process holds the panel's lease."""


class LedgerSchemaError(RuntimeError):
    """The file is not an exposure ledger of this schema; it is refused, never migrated."""


class LedgerNode(ContractModel):
    """One observation of trace content in one build.

    Attributes:
        fingerprint: Content fingerprint under ``scheme``.
        scheme: Fingerprint scheme and version.
        trace_id: Source trace ID of this observation.
        task_lineage: Declared task identity from the trace's request context, if any.
        project: Project whose build observed the trace.
        build: Build (task-set artifact) that observed the trace.
    """

    fingerprint: Fingerprint = Field(pattern=r"^[0-9a-f]{64}$")
    scheme: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    task_lineage: str | None = None
    project: str = Field(min_length=1)
    build: str = Field(min_length=1)


class LedgerEdge(ContractModel):
    """One grouping relationship between two fingerprints, never deleted.

    Attributes:
        left: Lexically smaller fingerprint.
        right: Lexically larger fingerprint.
        reason: ``declared-task-identity`` or the build's ``near-duplicate`` union.
        task_lineage: The shared declared task identity, for ``declared-task-identity``.
        duplicate_kind: Exact or semantic evidence, for ``near-duplicate``.
        similarity: Cosine similarity of the union, for ``near-duplicate``; exact is 1.0.
        project: Project whose build produced the edge.
        build: Build (task-set artifact) that produced the edge.
    """

    left: Fingerprint = Field(pattern=r"^[0-9a-f]{64}$")
    right: Fingerprint = Field(pattern=r"^[0-9a-f]{64}$")
    reason: EdgeReason
    task_lineage: str | None = None
    duplicate_kind: Literal["exact", "semantic"] | None = None
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    project: str = Field(min_length=1)
    build: str = Field(min_length=1)

    @model_validator(mode="after")
    def _require_reason_evidence(self) -> Self:
        """Require sorted distinct ends and exactly the evidence the reason names."""
        if self.left >= self.right:
            raise ValueError("ledger edge ends must be distinct and sorted")
        declared = self.task_lineage is not None
        duplicate = self.duplicate_kind is not None and self.similarity is not None
        if self.reason == "declared-task-identity" and (not declared or duplicate):
            raise ValueError("a declared-task-identity edge carries only the task lineage")
        if self.reason == "near-duplicate" and (not duplicate or declared):
            raise ValueError("a near-duplicate edge carries its kind and similarity only")
        return self


class LedgerBuild(ContractModel):
    """Marker that one build recorded its complete grouping in the ledger.

    Attributes:
        project: Project that built.
        build: Build (task-set artifact).
        scheme: Fingerprint scheme its nodes and edges use.
    """

    project: str = Field(min_length=1)
    build: str = Field(min_length=1)
    scheme: str = Field(min_length=1)


class LedgerExposure(ContractModel):
    """Fingerprints a person saw, and how.

    Attributes:
        fingerprints: Exposed content, sorted.
        how: ``inspected`` for an independent audit, ``re-audited`` for a development re-audit.
        panel: Identity of the panel shown.
        receipt: Audit receipt recording the report.
        report: Report that showed per-trace results.
        project: Project the report ran in.
        at: When the report was recorded.
    """

    fingerprints: tuple[Fingerprint, ...] = Field(min_length=1)
    how: Literal["inspected", "re-audited"]
    panel: str = Field(min_length=1)
    receipt: str = Field(min_length=1)
    report: str = Field(min_length=1)
    project: str = Field(min_length=1)
    at: AwareDatetime

    @property
    def record_id(self) -> str:
        """Content address of this exposure record."""
        return _record_id(self)


class LedgerReservation(ContractModel):
    """Content one audit receipt claimed before registering any run.

    Attributes:
        receipt: Owning audit receipt; only its owner may evaluate the panel independently or
            release the reservation.
        panel: Panel identity the receipt audits.
        project: Project of the receipt.
        lineages: The build's held-out lineages the panel covers.
        fingerprints: Member content of those lineages, sorted.
        snapshot: Ledger snapshot identity the reservation was decided against.
        at: When it was committed.
    """

    receipt: str = Field(min_length=1)
    panel: str = Field(min_length=1)
    project: str = Field(min_length=1)
    lineages: tuple[str, ...] = Field(min_length=1)
    fingerprints: tuple[Fingerprint, ...] = Field(min_length=1)
    snapshot: str = Field(min_length=1)
    at: AwareDatetime


class LedgerRelease(ContractModel):
    """The owner released an unused reservation; its content is free again.

    Attributes:
        receipt: Receipt whose reservation was released.
        at: When the release committed.
    """

    receipt: str = Field(min_length=1)
    at: AwareDatetime


class ExposurePath(ContractModel):
    """Why some content counts as exposed or taken under one ledger snapshot.

    Attributes:
        exposed_node: Nearest marked fingerprint in the component.
        component_size: Fingerprints in the connected component, so over-long chains show.
        path: Edges from the queried content to ``exposed_node``; empty when it is marked itself.
        by: Exposure records, or reserving receipts, that mark ``exposed_node``.
    """

    exposed_node: Fingerprint
    component_size: int = Field(ge=1)
    path: tuple[LedgerEdge, ...] = ()
    by: tuple[str, ...] = Field(min_length=1)


class LedgerExport(ContractModel):
    """Every row of a ledger, for sharing or audit.

    Attributes:
        identity: Snapshot identity of exactly these rows.
        scheme: Fingerprint scheme this code writes.
        builds: Builds that recorded their grouping.
        nodes: Content observations.
        edges: Grouping relationships.
        exposures: Exposure records.
        reservations: Content claimed by audit receipts.
        releases: Reservations their owners released.
    """

    identity: str
    scheme: str
    builds: tuple[LedgerBuild, ...] = ()
    nodes: tuple[LedgerNode, ...] = ()
    edges: tuple[LedgerEdge, ...] = ()
    exposures: tuple[LedgerExposure, ...] = ()
    reservations: tuple[LedgerReservation, ...] = ()
    releases: tuple[LedgerRelease, ...] = ()


class LedgerSnapshot:
    """An immutable view of the ledger used for one decision."""

    def __init__(self, export: LedgerExport) -> None:
        """Index the rows for component and path queries.

        Args:
            export: Every row of the snapshot.
        """
        self.records = export
        self.identity = export.identity
        self._builds = {(build.project, build.build) for build in export.builds}
        self.reservations = {item.receipt: item for item in export.reservations}
        self.released = frozenset(item.receipt for item in export.releases)
        self._union = UnionFind(node.fingerprint for node in export.nodes)
        self._adjacent: dict[Fingerprint, list[tuple[Fingerprint, LedgerEdge]]] = defaultdict(list)
        for edge in export.edges:
            self._union.union(edge.left, edge.right)
            self._adjacent[edge.left].append((edge.right, edge))
            self._adjacent[edge.right].append((edge.left, edge))
        self._exposed: dict[Fingerprint, list[str]] = defaultdict(list)
        for exposure in export.exposures:
            for fingerprint in exposure.fingerprints:
                self._exposed[fingerprint].append(exposure.record_id)
        self._reserved: dict[Fingerprint, list[str]] = defaultdict(list)
        for reservation in export.reservations:
            if reservation.receipt not in self.released:
                for fingerprint in reservation.fingerprints:
                    self._reserved[fingerprint].append(reservation.receipt)
        self._sizes: dict[str, int] = defaultdict(int)
        for fingerprint in {
            *(node.fingerprint for node in export.nodes),
            *self._adjacent,
            *self._exposed,
            *self._reserved,
        }:
            self._sizes[self._union.find(fingerprint)] += 1

    def has_build(self, project: str, build: str) -> bool:
        """Whether this build recorded its grouping in the ledger."""
        return (project, build) in self._builds

    def active(self, receipt: str) -> bool:
        """Whether ``receipt`` holds a reservation that was not released."""
        return receipt in self.reservations and receipt not in self.released

    def exposure(self, fingerprints: Iterable[Fingerprint]) -> ExposurePath | None:
        """The shortest path from any of ``fingerprints`` to exposed content, if one exists.

        Args:
            fingerprints: Content of one lineage's member traces.

        Returns:
            The path, or None when no member shares a component with exposed content.
        """
        return self._nearest(fingerprints, self._exposed)

    def taken(self, fingerprints: Iterable[Fingerprint], owner: str | None) -> ExposurePath | None:
        """The shortest path to content another receipt reserved and did not release.

        Args:
            fingerprints: Content of one lineage's member traces.
            owner: Receipt whose own reservation does not count; None counts every one.

        Returns:
            The path, or None when no member shares a component with another reservation.
        """
        others = {
            fingerprint: [receipt for receipt in receipts if receipt != owner]
            for fingerprint, receipts in self._reserved.items()
        }
        return self._nearest(fingerprints, {key: value for key, value in others.items() if value})

    def _nearest(
        self, fingerprints: Iterable[Fingerprint], marked: Mapping[Fingerprint, list[str]]
    ) -> ExposurePath | None:
        """Breadth-first search from ``fingerprints`` to the nearest marked fingerprint."""
        starts = sorted(set(fingerprints))
        previous: dict[Fingerprint, tuple[Fingerprint, LedgerEdge] | None] = dict.fromkeys(starts)
        queue = deque(starts)
        while queue:
            current = queue.popleft()
            if current in marked:
                path: list[LedgerEdge] = []
                step = previous[current]
                while step is not None:
                    path.append(step[1])
                    step = previous[step[0]]
                return ExposurePath(
                    exposed_node=current,
                    component_size=max(1, self._sizes[self._union.find(current)]),
                    path=tuple(reversed(path)),
                    by=tuple(sorted(set(marked[current]))),
                )
            for neighbour, edge in sorted(
                self._adjacent.get(current, ()), key=lambda item: _record_id(item[1])
            ):
                if neighbour not in previous:
                    previous[neighbour] = (current, edge)
                    queue.append(neighbour)
        return None


class LedgerTransaction:
    """One ``BEGIN IMMEDIATE`` transaction: reads see a stable ledger, writes commit together."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Bind an open connection inside its transaction."""
        self._connection = connection

    def snapshot(self) -> LedgerSnapshot:
        """Every row as of now, inside this transaction."""
        return LedgerSnapshot(_export(self._connection))

    def record(self, kind: RecordKind, record: BaseModel) -> None:
        """Add one content-addressed record unless it is already present."""
        self._connection.execute(
            "INSERT OR IGNORE INTO records (kind, record_id, body) VALUES (?, ?, ?)",
            (kind, _record_id(record), _body(record)),
        )

    def reserve(self, reservation: LedgerReservation) -> None:
        """Claim content for one receipt; a receipt reserves at most once."""
        self._connection.execute(
            "INSERT INTO reservations (receipt, body) VALUES (?, ?)",
            (reservation.receipt, _body(reservation)),
        )

    def release(self, release: LedgerRelease) -> None:
        """Free a reservation; its owner must have checked that no slot was claimed."""
        self._connection.execute(
            "INSERT INTO releases (receipt, body) VALUES (?, ?)",
            (release.receipt, _body(release)),
        )


class ExposureLedger:
    """The append-only exposure ledger of one ``.exp`` root."""

    def __init__(self, root: Path) -> None:
        """Bind the ledger of one ``.exp`` root without touching the file.

        Args:
            root: The ``.exp`` root shared by every project.
        """
        self.directory = Path(root) / LEDGER_DIRECTORY
        self.path = self.directory / LEDGER_FILE

    @contextmanager
    def transaction(self) -> Iterator[LedgerTransaction]:
        """Hold the ledger's write lock: read, decide and write, then commit or roll back."""
        self.directory.mkdir(parents=True, exist_ok=True)
        connection = connect_database(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            _require_schema(connection)
            yield LedgerTransaction(connection)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def lease(self, panel: str) -> Iterator[FileLock]:
        """Hold the exclusive cross-process lease of one panel, or fail at once.

        Yields:
            The held lock; ``is_locked`` tells a guard the lease is still held.

        Raises:
            PanelInUseError: Another process holds it.
        """
        directory = self.directory / "leases"
        directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(directory / f"panel-{panel}.lock", timeout=0, thread_local=False)
        try:
            lock.acquire()
        except Timeout:
            raise PanelInUseError(
                f"panel {panel} in use: another process is executing, resuming or releasing it"
            ) from None
        try:
            yield lock
        finally:
            lock.release()

    def record_build(
        self,
        *,
        project: str,
        build: str,
        traces: Sequence[Trace],
        near_duplicates: Sequence[DuplicateEdge],
    ) -> LedgerBuild:
        """Record one build's content nodes and grouping edges, idempotently.

        Args:
            project: Project that built.
            build: Build (task-set artifact) identity.
            traces: Every trace the build mined.
            near_duplicates: The build's exact and semantic duplicate unions.

        Returns:
            The build marker.
        """
        fingerprints = {trace.trace_id: trace_fingerprint(trace) for trace in traces}
        nodes = tuple(
            LedgerNode(
                fingerprint=fingerprints[trace.trace_id],
                scheme=FINGERPRINT_SCHEME,
                trace_id=trace.trace_id,
                task_lineage=declared_task_identity(trace),
                project=project,
                build=build,
            )
            for trace in traces
        )
        edges: list[LedgerEdge] = []
        declared: dict[str, set[Fingerprint]] = defaultdict(set)
        for item in nodes:
            if item.task_lineage is not None:
                declared[item.task_lineage].add(item.fingerprint)
        for task_lineage, members in sorted(declared.items()):
            chain = sorted(members)
            edges.extend(
                LedgerEdge(
                    left=left,
                    right=right,
                    reason="declared-task-identity",
                    task_lineage=task_lineage,
                    project=project,
                    build=build,
                )
                for left, right in zip(chain, chain[1:], strict=False)
            )
        for duplicate in near_duplicates:
            ends = sorted(
                {fingerprints[duplicate.left_trace_id], fingerprints[duplicate.right_trace_id]}
            )
            if len(ends) == 2:
                edges.append(
                    LedgerEdge(
                        left=ends[0],
                        right=ends[1],
                        reason="near-duplicate",
                        duplicate_kind=duplicate.kind,
                        similarity=duplicate.similarity,
                        project=project,
                        build=build,
                    )
                )
        marker = LedgerBuild(project=project, build=build, scheme=FINGERPRINT_SCHEME)
        self.record_grouping(marker, nodes=nodes, edges=edges)
        return marker

    def record_grouping(
        self, marker: LedgerBuild, *, nodes: Sequence[LedgerNode], edges: Sequence[LedgerEdge]
    ) -> None:
        """Write a build's nodes, edges and marker in one transaction.

        Args:
            marker: The build that produced them.
            nodes: Content observations.
            edges: Grouping relationships.
        """
        with self.transaction() as transaction:
            for item in nodes:
                transaction.record("node", item)
            for edge in edges:
                transaction.record("edge", edge)
            transaction.record("build", marker)

    def record_exposure(self, exposure: LedgerExposure) -> None:
        """Mark content exposed; the row is permanent.

        Args:
            exposure: What was shown, and how.
        """
        with self.transaction() as transaction:
            transaction.record("exposure", exposure)

    def snapshot(self) -> LedgerSnapshot:
        """Every row now in the ledger, as one queryable snapshot."""
        return LedgerSnapshot(self.export())

    def export(self) -> LedgerExport:
        """Every row now in the ledger, with its snapshot identity."""
        if not self.path.exists():
            return _export(None)
        with self.transaction() as transaction:
            return transaction.snapshot().records

    def merge(self, export: LedgerExport) -> None:
        """Add every row of another ledger's export; nothing is replaced or removed.

        Args:
            export: Rows from another ledger.
        """
        with self.transaction() as transaction:
            snapshot = transaction.snapshot()
            for kind, records in (
                ("node", export.nodes),
                ("edge", export.edges),
                ("exposure", export.exposures),
                ("build", export.builds),
            ):
                for record in records:
                    transaction.record(kind, record)
            for reservation in export.reservations:
                if reservation.receipt not in snapshot.reservations:
                    transaction.reserve(reservation)
            for release in export.releases:
                if release.receipt not in snapshot.released:
                    transaction.release(release)


def trace_fingerprint(trace: Trace) -> Fingerprint:
    """Fingerprint a trace's content, independent of IDs, timestamps and source records.

    The content is the normalized task text and every recorded action with the observation that
    followed it, in recorded order. Trace and span IDs, times, source identity and request
    context are excluded, so a reimport of the same content has the same fingerprint.

    Args:
        trace: One normalized trace.

    Returns:
        A SHA-256 hex digest under :data:`FINGERPRINT_SCHEME`.
    """
    transitions = extract_real_transitions(
        (trace,),
        (
            RAGLineageBinding(
                trace_id=trace.trace_id, lineage_id=_FINGERPRINT_LINEAGE, partition="fit"
            ),
        ),
        included_partitions=frozenset({"fit"}),
    )
    started = {span.span_id: span.started_at for span in trace.spans}
    steps = [
        {
            "action": transition.action.model_dump(mode="json"),
            "observation": {
                "kind": transition.observation.kind,
                "content": _decoded(transition.observation.content),
            },
        }
        for transition in transitions
    ]
    order = [started[transition.action_span_id] for transition in transitions]
    ordered = [
        step
        for _, step in sorted(
            zip(order, steps, strict=True),
            key=lambda item: (item[0], json.dumps(item[1], sort_keys=True)),
        )
    ]
    return sha256_json(
        {
            "scheme": FINGERPRINT_SCHEME,
            "task": normalize_durable_text(trace.task.strip()),
            "steps": ordered,
        }
    )


def declared_task_identity(trace: Trace) -> str | None:
    """The task identity a trace declares in its request context, if any."""
    value = trace.initial_context.get(TASK_LINEAGE_KEY)
    return value if isinstance(value, str) and value else None


def _require_schema(connection: sqlite3.Connection) -> None:
    """Create the append-only schema in an empty file, or refuse any other schema."""
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version == SCHEMA_VERSION:
        return
    if (
        version != 0
        or connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
    ):
        raise LedgerSchemaError(
            f"exposure ledger schema {version} is not {SCHEMA_VERSION}; it is never migrated"
        )
    connection.execute(
        """CREATE TABLE records (
            kind TEXT NOT NULL CHECK (kind IN ('build', 'node', 'edge', 'exposure')),
            record_id TEXT NOT NULL,
            body TEXT NOT NULL,
            PRIMARY KEY (kind, record_id)
        ) STRICT"""
    )
    connection.execute(
        "CREATE TABLE reservations (receipt TEXT PRIMARY KEY, body TEXT NOT NULL) STRICT"
    )
    connection.execute(
        """CREATE TABLE releases (
            receipt TEXT PRIMARY KEY REFERENCES reservations (receipt),
            body TEXT NOT NULL
        ) STRICT"""
    )
    for table in ("records", "reservations", "releases"):
        for action in ("UPDATE", "DELETE"):
            connection.execute(
                f"""CREATE TRIGGER {table}_no_{action.lower()} BEFORE {action} ON {table}
                BEGIN SELECT RAISE(ABORT, 'exposure ledger rows are append-only'); END"""
            )
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _export(connection: sqlite3.Connection | None) -> LedgerExport:
    """Every row, parsed, with the snapshot identity over their keys."""
    rows: dict[str, list[str]] = defaultdict(list)
    keys: list[str] = []
    if connection is not None:
        for kind, record_id, body in connection.execute(
            "SELECT kind, record_id, body FROM records ORDER BY kind, record_id"
        ):
            rows[kind].append(body)
            keys.append(f"{kind}/{record_id}")
        for table in ("reservations", "releases"):
            for receipt, body in connection.execute(
                f"SELECT receipt, body FROM {table} ORDER BY receipt"
            ):
                rows[table].append(body)
                keys.append(f"{table}/{receipt}/{sha256_json(body)[:16]}")
    return LedgerExport(
        identity="ledger-" + sha256_json(sorted(keys))[:20],
        scheme=FINGERPRINT_SCHEME,
        builds=_parse(rows["build"], LedgerBuild),
        nodes=_parse(rows["node"], LedgerNode),
        edges=_parse(rows["edge"], LedgerEdge),
        exposures=_parse(rows["exposure"], LedgerExposure),
        reservations=_parse(rows["reservations"], LedgerReservation),
        releases=_parse(rows["releases"], LedgerRelease),
    )


def _parse[Row: BaseModel](bodies: list[str], model: type[Row]) -> tuple[Row, ...]:
    """Validate stored JSON rows."""
    return tuple(model.model_validate_json(body) for body in bodies)


def _body(record: BaseModel) -> str:
    """Canonical JSON text of one row."""
    return json.dumps(record.model_dump(mode="json"), sort_keys=True)


def _decoded(content: str) -> JsonValue:
    """Recorded text as JSON when it is JSON, so encoding whitespace never changes content."""
    try:
        return json.loads(content)
    except ValueError:
        return content


def _record_id(record: BaseModel) -> str:
    """Content address of one ledger record."""
    return sha256_json(record)[:32]
