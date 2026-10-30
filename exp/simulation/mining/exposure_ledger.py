"""Root exposure ledger: which recorded trace content a person has seen, and what it grouped with.

Held-out evidence is independent only while nobody has read its per-trace results. Project-local
leakage lineage IDs cannot carry that fact across builds: they change on reimport and when
grouping changes. The ledger therefore lives at the ``.exp`` root, above any one project, in one
SQLite file, and keys everything on content (:mod:`exp.simulation.mining.fingerprint`):

- Nodes are traces fingerprinted under a scheme, each with a pointer to its immutable verified
  source record (project, trace dataset, trace ID and the record's SHA-256), the source trace ID
  and any declared task identity (``task_lineage``). A trace a scheme cannot fingerprint is an
  unresolved node, still linked by grouping edges.
- Edges are grouping relationships recorded for every build, whether or not anything is exposed:
  two traces that declare the same task identity within one source and customer, exactly as the
  build's own partition scopes declarations (:mod:`exp.simulation.mining.lineage`), or the
  build's near-duplicate union with its similarity. Each build also links its declaring traces
  to a vertex for the scoped identity itself, so one task joins its traces across builds even
  when a later import holds only new content of it. Each edge names the project and build that
  produced it. Nothing weaker (a shared customer, conversation or time bucket, or a declared
  name reused by another source or customer) is an edge.
- Equivalence links join the nodes of one verified source under every scheme that fingerprinted
  it, for unexposed traces too, so edges recorded under any scheme stay valid.
- Exposures mark nodes a person has seen, with the panel, receipt and report that showed them.
  The ledger also notes which exposure rows it wrote itself; a merge never imports that note.
- Reservations claim content for one audit receipt before any of its runs is registered; a
  release by that receipt's owner frees an unused reservation.

Content is exposed when it shares a connected component with an exposed node, and taken when it
shares one with content another receipt reserved. Both relations are transitive and ignore time:
an edge recorded before or after the exposure propagates it, and it keeps propagating after the
exposed trace leaves every later import. Every table is append-only (SQL triggers refuse edits
and deletions), so a snapshot identity names exactly the rows a decision used, and independence
is only relative to that ledger.

A scheme recorded in the ledger with the same name and version as a scheme this code implements,
but a different definition or vector set, is a specification changed without a version bump:
recording under it is refused. On import, such a scheme is refused and its nodes stay as
unknown-scheme nodes, which block independence wherever no verified equivalence resolves them.

Atomicity holds for processes that share this one ledger file on one host; SQLite and OS file
locks over network filesystems are unreliable, so the ledger's connection and panel leases refuse
another machine or an undeclared network filesystem (`exp.common.core.host_scope`). A panel
lease is an exclusive OS file lock, so it dies with its process and a crash leaves no stale lease.
"""

import json
import sqlite3
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal, Self

from filelock import FileLock, Timeout
from pydantic import AwareDatetime, BaseModel, Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.core.locks import CoordinationLock
from exp.common.core.union_find import UnionFind
from exp.common.sqlite.connection import connect_database
from exp.common.traces import Trace
from exp.simulation.mining.deduplicate import DuplicateEdge
from exp.simulation.mining.fingerprint import SCHEMES, Scheme, SchemeIdentity
from exp.simulation.mining.lineage import LineageAssignment

LEDGER_DIRECTORY = "exposure-ledger"
LEDGER_FILE = "ledger.sqlite"
SCHEMA_VERSION: Final = 2
TASK_LINEAGE_KEY = "task_lineage"
DECLARED_PREFIX = "declared:"
SAME_HOST_LIMIT = (
    "Reservations and panel leases are atomic only for runs sharing one ledger file on one host; "
    "SQLite and file locks over network filesystems are unreliable."
)
EdgeReason = Literal["declared-task-identity", "near-duplicate"]
RecordKind = Literal["build", "node", "edge", "exposure", "equivalence", "scheme"]
Vertex = str


class PanelInUseError(RuntimeError):
    """Another process holds the panel's lease."""


class LedgerSchemaError(RuntimeError):
    """The file is not an exposure ledger of this schema; it is refused, never migrated."""


class SchemeConflictError(ValueError):
    """A scheme's name and version are already recorded with a different definition."""


class SourceRecord(ContractModel):
    """Pointer to the immutable verified record a node was fingerprinted from.

    Attributes:
        project: Project holding the trace dataset.
        dataset: Trace dataset artifact.
        trace_id: Trace ID inside it.
        record_sha256: SHA-256 of the trace record's canonical JSON.
    """

    project: str = Field(min_length=1)
    dataset: ArtifactInput
    trace_id: str = Field(min_length=1)
    record_sha256: Sha256


class LedgerNode(ContractModel):
    """One trace fingerprinted under one scheme in one build.

    Attributes:
        scheme: Full scheme identity.
        fingerprint: Content fingerprint, or None when the scheme cannot pair its calls.
        source: Verified source record.
        task_lineage: Declared task identity from the trace's request context, if any.
        build: Build (task-set artifact) that observed the trace.
    """

    scheme: SchemeIdentity
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source: SourceRecord
    task_lineage: str | None = None
    build: str = Field(min_length=1)

    @property
    def vertex(self) -> Vertex:
        """Graph vertex: the scheme-qualified fingerprint, or the unresolved source."""
        return node_vertex(self.scheme, self.fingerprint, self.source)


class LedgerEdge(ContractModel):
    """One grouping relationship between two vertices, never deleted.

    Attributes:
        left: Lexically smaller vertex.
        right: Lexically larger vertex.
        reason: ``declared-task-identity`` or the build's ``near-duplicate`` union.
        task_lineage: The shared declared task identity, for ``declared-task-identity``.
        duplicate_kind: Exact or semantic evidence, for ``near-duplicate``.
        similarity: Cosine similarity of the union, for ``near-duplicate``; exact is 1.0.
        project: Project whose build produced the edge.
        build: Build (task-set artifact) that produced the edge.
    """

    left: Vertex = Field(min_length=1)
    right: Vertex = Field(min_length=1)
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


class LedgerEquivalence(ContractModel):
    """One verified source fingerprinted under several schemes.

    Attributes:
        source_sha256: The source record's SHA-256.
        vertices: Its vertex under each scheme, sorted.
    """

    source_sha256: Sha256
    vertices: tuple[Vertex, ...] = Field(min_length=2)


class LedgerBuild(ContractModel):
    """Marker that one build recorded its complete grouping in the ledger.

    Attributes:
        project: Project that built.
        build: Build (task-set artifact).
        schemes: Schemes its nodes were recorded under; edges use the last.
    """

    project: str = Field(min_length=1)
    build: str = Field(min_length=1)
    schemes: tuple[SchemeIdentity, ...] = Field(min_length=1)


class LedgerExposure(ContractModel):
    """Vertices a person saw, and how.

    Attributes:
        vertices: Exposed content, sorted.
        how: ``inspected`` for an independent audit, ``re-audited`` for a development re-audit.
        panel: Identity of the panel shown.
        receipt: Audit receipt recording the report.
        report: Report that showed per-trace results.
        project: Project the report ran in.
        at: When the report was recorded, by the recording host's clock; informational only.
    """

    vertices: tuple[Vertex, ...] = Field(min_length=1)
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
        vertices: Member content of those lineages, sorted.
        snapshot: Ledger snapshot identity the reservation was decided against.
        at: When it was committed.
    """

    receipt: str = Field(min_length=1)
    panel: str = Field(min_length=1)
    project: str = Field(min_length=1)
    lineages: tuple[str, ...] = Field(min_length=1)
    vertices: tuple[Vertex, ...] = Field(min_length=1)
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
        exposed_node: Nearest marked vertex in the component.
        component_size: Vertices in the connected component, so over-long chains show.
        path: Edges from the queried content to ``exposed_node``; empty when it is marked
            itself or reached only through equivalence links.
        by: Exposure records, or reserving receipts, that mark ``exposed_node``.
    """

    exposed_node: Vertex
    component_size: int = Field(ge=1)
    path: tuple[LedgerEdge, ...] = ()
    by: tuple[str, ...] = Field(min_length=1)


class UnknownScheme(ContractModel):
    """Nodes recorded under a scheme this machine cannot compute, with no verified equivalence.

    Attributes:
        scheme: The scheme's identity as recorded.
        nodes: Unresolved node count.
    """

    scheme: SchemeIdentity
    nodes: int = Field(ge=1)


class LedgerExport(ContractModel):
    """Every row of a ledger, for sharing or audit.

    Attributes:
        identity: Snapshot identity of exactly these rows.
        schemes: Scheme identities recorded.
        builds: Builds that recorded their grouping.
        nodes: Fingerprinted traces.
        edges: Grouping relationships.
        equivalences: Cross-scheme links of one source.
        exposures: Exposure records.
        reservations: Content claimed by audit receipts.
        releases: Reservations their owners released.
        recorded_here: Exposure record IDs and reserving receipts this ledger wrote itself, never
            merged ones; local
            provenance, so it is outside the identity and a merge never imports it.
    """

    identity: str
    schemes: tuple[SchemeIdentity, ...] = ()
    builds: tuple[LedgerBuild, ...] = ()
    nodes: tuple[LedgerNode, ...] = ()
    edges: tuple[LedgerEdge, ...] = ()
    equivalences: tuple[LedgerEquivalence, ...] = ()
    exposures: tuple[LedgerExposure, ...] = ()
    reservations: tuple[LedgerReservation, ...] = ()
    releases: tuple[LedgerRelease, ...] = ()
    recorded_here: tuple[str, ...] = ()


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
        self._union = UnionFind(node.vertex for node in export.nodes)
        self._adjacent: dict[Vertex, list[tuple[Vertex, LedgerEdge | None]]] = defaultdict(list)
        for edge in export.edges:
            self._link(edge.left, edge.right, edge)
        for equivalence in export.equivalences:
            first, *rest = equivalence.vertices
            for vertex in rest:
                self._link(first, vertex, None)
        self._reserved: dict[Vertex, list[str]] = defaultdict(list)
        for reservation in export.reservations:
            if reservation.receipt not in self.released:
                for vertex in reservation.vertices:
                    self._reserved[vertex].append(reservation.receipt)
        self._sizes: dict[str, int] = defaultdict(int)
        for vertex in {
            *(node.vertex for node in export.nodes),
            *self._adjacent,
            *(vertex for item in export.exposures for vertex in item.vertices),
            *self._reserved,
        }:
            if not vertex.startswith(DECLARED_PREFIX):
                self._sizes[self._union.find(vertex)] += 1

    def _link(self, left: Vertex, right: Vertex, edge: LedgerEdge | None) -> None:
        """Join two vertices by a grouping edge or an equivalence link (``None``)."""
        self._union.union(left, right)
        self._adjacent[left].append((right, edge))
        self._adjacent[right].append((left, edge))

    def has_build(self, project: str, build: str) -> bool:
        """Whether this build recorded its grouping in the ledger."""
        return (project, build) in self._builds

    def active(self, receipt: str) -> bool:
        """Whether ``receipt`` holds a reservation that was not released."""
        return receipt in self.reservations and receipt not in self.released

    def exposure(
        self,
        vertices: Iterable[Vertex],
        *,
        counts: Callable[[LedgerExposure], bool] | None = None,
    ) -> ExposurePath | None:
        """The shortest path from any of ``vertices`` to exposed content, if one exists.

        Args:
            vertices: Content of one lineage's member traces, under every computable scheme.
            counts: Which exposure records count, such as only those not provably recorded
                after an audit receipt (late-link reassessment); every record when omitted.
                Exposure times (``at``) are informational and never decide this.

        Returns:
            The path, or None when no member shares a component with exposed content.
        """
        marked: dict[Vertex, list[str]] = defaultdict(list)
        for item in self.records.exposures:
            if counts is None or counts(item):
                for vertex in item.vertices:
                    marked[vertex].append(item.record_id)
        return self._nearest(vertices, marked)

    def taken(self, vertices: Iterable[Vertex], owner: str | None) -> ExposurePath | None:
        """The shortest path to content another receipt reserved and did not release.

        Args:
            vertices: Content of one lineage's member traces.
            owner: Receipt whose own reservation does not count; None counts every one.

        Returns:
            The path, or None when no member shares a component with another reservation.
        """
        others = {
            vertex: [receipt for receipt in receipts if receipt != owner]
            for vertex, receipts in self._reserved.items()
        }
        return self._nearest(vertices, {key: value for key, value in others.items() if value})

    def unknown_schemes(self, computable: Sequence[SchemeIdentity]) -> tuple[UnknownScheme, ...]:
        """Nodes under schemes not computable here, with no equivalence to a computable one.

        Args:
            computable: Identities of the schemes that pass their vectors on this machine.

        Returns:
            Unresolved node counts per unknown scheme.
        """
        known = set(computable)
        resolved = self.resolved_vertices(computable)
        counts: dict[SchemeIdentity, int] = defaultdict(int)
        for node in self.records.nodes:
            if node.scheme not in known and node.vertex not in resolved:
                counts[node.scheme] += 1
        return tuple(
            UnknownScheme(scheme=scheme, nodes=count)
            for scheme, count in sorted(counts.items(), key=lambda item: item[0].key)
        )

    def resolved_vertices(self, computable: Sequence[SchemeIdentity]) -> frozenset[Vertex]:
        """Vertices tied to a content fingerprint computable here.

        A node under a computable scheme resolves only when it has a fingerprint: an unresolved
        vertex names its source record, not its content, so it can never match the same task
        imported again with call IDs. Every other vertex resolves only through verified
        equivalences, followed transitively; a shared grouping edge never resolves one.

        Args:
            computable: Identities of the schemes that pass their vectors on this machine.

        Returns:
            The resolved vertices.
        """
        known = set(computable)
        anchors = {
            node.vertex
            for node in self.records.nodes
            if node.scheme in known and node.fingerprint is not None
        }
        union = UnionFind(anchors)
        for equivalence in self.records.equivalences:
            first, *rest = equivalence.vertices
            for vertex in rest:
                union.union(first, vertex)
        roots = {union.find(vertex) for vertex in anchors}
        linked = {
            vertex
            for equivalence in self.records.equivalences
            for vertex in equivalence.vertices
            if union.find(vertex) in roots
        }
        return frozenset(anchors | linked)

    def _nearest(
        self, vertices: Iterable[Vertex], marked: Mapping[Vertex, list[str]]
    ) -> ExposurePath | None:
        """Breadth-first search from ``vertices`` to the nearest marked vertex."""
        starts = sorted(set(vertices))
        previous: dict[Vertex, tuple[Vertex, LedgerEdge | None] | None] = dict.fromkeys(starts)
        queue = deque(starts)
        while queue:
            current = queue.popleft()
            if current in marked:
                path: list[LedgerEdge] = []
                step = previous[current]
                while step is not None:
                    if step[1] is not None:
                        path.append(step[1])
                    step = previous[step[0]]
                return ExposurePath(
                    exposed_node=current,
                    component_size=max(1, self._sizes[self._union.find(current)]),
                    path=tuple(reversed(path)),
                    by=tuple(sorted(set(marked[current]))),
                )
            for neighbour, edge in sorted(
                self._adjacent.get(current, ()),
                key=lambda item: (item[1] is not None, _record_id(item[1]) if item[1] else ""),
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

    def record(self, kind: RecordKind, record: BaseModel) -> bool:
        """Add one content-addressed record unless it is already present; True when added."""
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO records (kind, record_id, body) VALUES (?, ?, ?)",
            (kind, _record_id(record), _body(record)),
        )
        return cursor.rowcount == 1

    def record_scheme(self, scheme: SchemeIdentity) -> None:
        """Record a scheme this code computes under, refusing a changed spec without a bump.

        Raises:
            SchemeConflictError: The ledger records this name and version differently.
        """
        for recorded in self.snapshot().records.schemes:
            if recorded.conflicts(scheme):
                raise SchemeConflictError(
                    f"scheme {scheme.name} v{scheme.version} is recorded with definition "
                    f"{recorded.definition_sha256[:12]} and vectors "
                    f"{recorded.vectors_sha256[:12]}, but this code defines "
                    f"{scheme.definition_sha256[:12]} and {scheme.vectors_sha256[:12]}; a "
                    "changed specification needs a new scheme version"
                )
        self.record("scheme", scheme)

    def record_here(self, exposure: LedgerExposure) -> None:
        """Note that this ledger, not a merge, recorded the exposure."""
        self._connection.execute(
            "INSERT OR IGNORE INTO recorded_here (record_id) VALUES (?)", (exposure.record_id,)
        )

    def reserve(self, reservation: LedgerReservation, *, here: bool = True) -> None:
        """Claim content for one receipt, at most once; ``here`` is False when merged."""
        self._connection.execute(
            "INSERT INTO reservations (receipt, body) VALUES (?, ?)",
            (reservation.receipt, _body(reservation)),
        )
        if here:
            self._connection.execute(
                "INSERT INTO recorded_here (record_id) VALUES (?)", (reservation.receipt,)
            )

    def release(self, release: LedgerRelease) -> None:
        """Free a reservation; its owner must have checked that no slot was claimed."""
        self._connection.execute(
            "INSERT INTO releases (receipt, body) VALUES (?, ?)",
            (release.receipt, _body(release)),
        )


class ExposureLedger:
    """The append-only exposure ledger of one ``.exp`` root."""

    def __init__(self, root: Path, schemes: Sequence[Scheme] = SCHEMES) -> None:
        """Bind the ledger of one ``.exp`` root without touching the file.

        Args:
            root: The ``.exp`` root shared by every project.
            schemes: Scheme implementations, newest last.
        """
        self.directory = Path(root) / LEDGER_DIRECTORY
        self.path = self.directory / LEDGER_FILE
        self.schemes = tuple(schemes)

    @property
    def computable(self) -> tuple[Scheme, ...]:
        """Schemes whose golden vectors reproduce on this machine, newest last."""
        return tuple(scheme for scheme in self.schemes if scheme.computable)

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
        lock = CoordinationLock(directory / f"panel-{panel}.lock", timeout=0, thread_local=False)
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

    def vertices(self, trace: Trace) -> dict[SchemeIdentity, str | None]:
        """A trace's fingerprint under every computable scheme (None when unresolved)."""
        return {scheme.identity: scheme.compute(trace) for scheme in self.computable}

    def record_build(
        self,
        *,
        project: str,
        build: str,
        dataset: ArtifactInput,
        traces: Sequence[Trace],
        lineages: Sequence[LineageAssignment],
        near_duplicates: Sequence[DuplicateEdge],
    ) -> LedgerBuild:
        """Record one build's nodes, equivalences and grouping edges, idempotently.

        Every trace is fingerprinted under every computable scheme; edges use the newest. A
        declared-task-identity edge joins traces that declare one task identity within one
        source and customer, the scope the build's own partition gives declarations, so a
        task name reused by an unrelated source or customer never links their content. One
        more such edge links them to the scoped identity's own vertex, shared by every build.

        Args:
            project: Project that built.
            build: Build (task-set artifact) identity.
            dataset: Trace dataset the traces come from.
            traces: Every trace the build mined.
            lineages: The build's source lineage assignments, one per trace, which carry the
                customer scope and the declared task identity the build partitioned by.
            near_duplicates: The build's exact and semantic duplicate unions.

        Returns:
            The build marker.

        Raises:
            SchemeConflictError: A scheme's specification changed without a version bump.
            ValueError: No scheme is computable on this machine, or a trace has no lineage
                assignment.
        """
        if not self.computable:
            raise ValueError(
                "no fingerprint scheme reproduces its golden vectors on this machine, so the "
                "build cannot be recorded in the exposure ledger"
            )
        primary = self.computable[-1].identity
        nodes = _nodes(self, project, build, dataset, traces)
        by_trace = {node.source.trace_id: node for node in nodes if node.scheme == primary}
        edges: list[LedgerEdge] = []
        for scope, members in sorted(_declared_partitions(traces, lineages, by_trace).items()):
            chain = sorted(members)
            declared = declared_vertex(*scope)
            edges.extend(
                LedgerEdge(
                    left=min(left, right),
                    right=max(left, right),
                    reason="declared-task-identity",
                    task_lineage=scope[2],
                    project=project,
                    build=build,
                )
                for left, right in (*zip(chain, chain[1:], strict=False), (chain[0], declared))
            )
        for duplicate in near_duplicates:
            ends = sorted(
                {
                    by_trace[duplicate.left_trace_id].vertex,
                    by_trace[duplicate.right_trace_id].vertex,
                }
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
        marker = LedgerBuild(
            project=project,
            build=build,
            schemes=tuple(scheme.identity for scheme in self.computable),
        )
        with self.transaction() as transaction:
            for scheme in self.computable:
                transaction.record_scheme(scheme.identity)
            _write_nodes(transaction, nodes)
            for edge in edges:
                transaction.record("edge", edge)
            transaction.record("build", marker)
        return marker

    def record_nodes(
        self, *, project: str, build: str, dataset: ArtifactInput, traces: Sequence[Trace]
    ) -> None:
        """Fingerprint traces under every computable scheme and link their equivalences.

        A scheme added after a build only adds nodes and links; earlier edges stay valid.

        Raises:
            SchemeConflictError: A scheme's specification changed without a version bump.
        """
        nodes = _nodes(self, project, build, dataset, traces)
        with self.transaction() as transaction:
            for scheme in self.computable:
                transaction.record_scheme(scheme.identity)
            _write_nodes(transaction, nodes)

    def resolve_sources(self, load: Callable[[SourceRecord], Trace | None]) -> int:
        """Re-fingerprint unknown-scheme nodes from their verified sources where available.

        Args:
            load: The trace a source record points to, or None when it is unavailable. The
                record is used only when its canonical JSON still has the recorded SHA-256.

        Returns:
            Nodes resolved by a new equivalence link.
        """
        computable = [scheme.identity for scheme in self.computable]
        snapshot = self.snapshot()
        if not snapshot.unknown_schemes(computable):
            return 0
        resolved = 0
        linked = snapshot.resolved_vertices(computable)
        for node in snapshot.records.nodes:
            if node.scheme in computable or node.vertex in linked:
                continue
            trace = load(node.source)
            if trace is None or record_sha256(trace) != node.source.record_sha256:
                continue
            fresh = _nodes(self, node.source.project, node.build, node.source.dataset, (trace,))
            with self.transaction() as transaction:
                _write_nodes(transaction, fresh, extra={node.source.record_sha256: node.vertex})
            # A link to an unresolved vertex groups the node but leaves its content unknown.
            resolved += any(item.fingerprint is not None for item in fresh)
        return resolved

    def record_exposure(self, exposure: LedgerExposure) -> None:
        """Mark content exposed, and note that this ledger recorded it; both rows are permanent.

        A row already present, for example from a merge, keeps its provenance.

        Args:
            exposure: What was shown, and how.
        """
        with self.transaction() as transaction:
            if transaction.record("exposure", exposure):
                transaction.record_here(exposure)

    def snapshot(self) -> LedgerSnapshot:
        """Every row now in the ledger, as one queryable snapshot."""
        return LedgerSnapshot(self.export())

    def export(self) -> LedgerExport:
        """Every row now in the ledger, with its snapshot identity."""
        if not self.path.exists():
            return _export(None)
        with self.transaction() as transaction:
            return transaction.snapshot().records

    def merge(self, export: LedgerExport) -> tuple[SchemeIdentity, ...]:
        """Add every row of another ledger's export; nothing is replaced or removed.

        A scheme with the same name and version as one recorded here or implemented by this
        code, but a different definition or vector set, is refused: its scheme row is not
        imported, and its nodes stay as unknown-scheme nodes until an equivalence resolves them.

        Args:
            export: Rows from another ledger.

        Returns:
            The refused scheme identities.
        """
        refused: list[SchemeIdentity] = []
        with self.transaction() as transaction:
            snapshot = transaction.snapshot()
            local = [*snapshot.records.schemes, *(scheme.identity for scheme in self.schemes)]
            for scheme in export.schemes:
                if any(scheme.conflicts(known) for known in local):
                    refused.append(scheme)
                else:
                    transaction.record("scheme", scheme)
            for kind, records in (
                ("node", export.nodes),
                ("edge", export.edges),
                ("equivalence", export.equivalences),
                ("exposure", export.exposures),
                ("build", export.builds),
            ):
                for record in records:
                    transaction.record(kind, record)
            for reservation in export.reservations:
                if reservation.receipt not in snapshot.reservations:
                    transaction.reserve(reservation, here=False)
            for release in export.releases:
                if release.receipt not in snapshot.released:
                    transaction.release(release)
        return tuple(refused)


def node_vertex(scheme: SchemeIdentity, fingerprint: str | None, source: SourceRecord) -> Vertex:
    """Graph vertex of one fingerprinted trace, or of its unresolved source."""
    if fingerprint is None:
        return f"unresolved:{scheme.key}:{source.record_sha256}"
    return f"{scheme.key}:{fingerprint}"


def declared_vertex(source_id: str, customer_id: str, task_lineage: str) -> Vertex:
    """Graph vertex of one task identity declared within one source and customer."""
    return DECLARED_PREFIX + sha256_json([source_id, customer_id, task_lineage])


def record_sha256(trace: Trace) -> str:
    """SHA-256 of a trace record's canonical JSON."""
    return sha256_json(trace.model_dump(mode="json"))


def declared_task_identity(trace: Trace) -> str | None:
    """The task identity a trace declares in its request context, if any."""
    value = trace.initial_context.get(TASK_LINEAGE_KEY)
    return value if isinstance(value, str) and value else None


def _declared_partitions(
    traces: Sequence[Trace],
    lineages: Sequence[LineageAssignment],
    by_trace: Mapping[str, LedgerNode],
) -> dict[tuple[str, str, str], set[Vertex]]:
    """Group declaring traces' vertices by the build's ``(source, customer, task)`` scope.

    Raises:
        ValueError: A trace has no lineage assignment, so its declaration scope is unknown.
    """
    assigned = {assignment.trace_id: assignment for assignment in lineages}
    declared: dict[tuple[str, str, str], set[Vertex]] = defaultdict(set)
    for trace in traces:
        assignment = assigned.get(trace.trace_id)
        if assignment is None:
            raise ValueError(
                f"trace {trace.trace_id} has no source lineage assignment, so its declared "
                "task identity cannot be scoped in the exposure ledger"
            )
        if assignment.task_lineage is not None:
            key = (trace.source.identity.source_id, assignment.customer_id, assignment.task_lineage)
            declared[key].add(by_trace[trace.trace_id].vertex)
    return declared


def _nodes(
    ledger: ExposureLedger,
    project: str,
    build: str,
    dataset: ArtifactInput,
    traces: Sequence[Trace],
) -> tuple[LedgerNode, ...]:
    """Every trace under every computable scheme."""
    return tuple(
        LedgerNode(
            scheme=scheme.identity,
            fingerprint=scheme.compute(trace),
            source=SourceRecord(
                project=project,
                dataset=dataset,
                trace_id=trace.trace_id,
                record_sha256=record_sha256(trace),
            ),
            task_lineage=declared_task_identity(trace),
            build=build,
        )
        for trace in traces
        for scheme in ledger.computable
    )


def _write_nodes(
    transaction: LedgerTransaction,
    nodes: Sequence[LedgerNode],
    *,
    extra: Mapping[str, Vertex] | None = None,
) -> None:
    """Write nodes and link every source fingerprinted under more than one scheme."""
    by_source: dict[str, set[Vertex]] = defaultdict(set)
    for node in nodes:
        transaction.record("node", node)
        by_source[node.source.record_sha256].add(node.vertex)
    for source, vertex in (extra or {}).items():
        by_source[source].add(vertex)
    for source, vertices in sorted(by_source.items()):
        if len(vertices) > 1:
            transaction.record(
                "equivalence",
                LedgerEquivalence(source_sha256=source, vertices=tuple(sorted(vertices))),
            )


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
            kind TEXT NOT NULL CHECK (
                kind IN ('build', 'node', 'edge', 'exposure', 'equivalence', 'scheme')
            ),
            record_id TEXT NOT NULL,
            body TEXT NOT NULL,
            PRIMARY KEY (kind, record_id)
        ) STRICT"""
    )
    connection.execute(
        "CREATE TABLE reservations (receipt TEXT PRIMARY KEY, body TEXT NOT NULL) STRICT"
    )
    connection.execute("CREATE TABLE recorded_here (record_id TEXT PRIMARY KEY) STRICT")
    connection.execute(
        """CREATE TABLE releases (
            receipt TEXT PRIMARY KEY REFERENCES reservations (receipt),
            body TEXT NOT NULL
        ) STRICT"""
    )
    for table in ("records", "reservations", "releases", "recorded_here"):
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
        for (record_id,) in connection.execute(
            "SELECT record_id FROM recorded_here ORDER BY record_id"
        ):
            rows["recorded_here"].append(record_id)
    return LedgerExport(
        identity="ledger-" + sha256_json(sorted(keys))[:20],
        schemes=_parse(rows["scheme"], SchemeIdentity),
        builds=_parse(rows["build"], LedgerBuild),
        nodes=_parse(rows["node"], LedgerNode),
        edges=_parse(rows["edge"], LedgerEdge),
        equivalences=_parse(rows["equivalence"], LedgerEquivalence),
        exposures=_parse(rows["exposure"], LedgerExposure),
        reservations=_parse(rows["reservations"], LedgerReservation),
        releases=_parse(rows["releases"], LedgerRelease),
        recorded_here=tuple(rows["recorded_here"]),
    )


def _parse[Row: BaseModel](bodies: list[str], model: type[Row]) -> tuple[Row, ...]:
    """Validate stored JSON rows."""
    return tuple(model.model_validate_json(body) for body in bodies)


def _body(record: BaseModel) -> str:
    """Canonical JSON text of one row."""
    return json.dumps(record.model_dump(mode="json"), sort_keys=True)


def _record_id(record: BaseModel) -> str:
    """Content address of one ledger record."""
    return sha256_json(record)[:32]
