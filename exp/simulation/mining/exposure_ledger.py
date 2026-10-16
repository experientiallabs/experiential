"""Root exposure ledger: which recorded trace content a person has seen, and what it grouped with.

Held-out evidence is independent only while nobody has read its per-trace results. Project-local
leakage lineage IDs cannot carry that fact across builds: they change on reimport and when
grouping changes. The ledger therefore lives at the ``.exp`` root, above any one project, and
keys everything on content:

- Nodes are fingerprints of normalized traces (task text, recorded actions, recorded replies)
  under a versioned scheme, with the source trace ID and any declared task identity
  (``task_lineage``) kept as attributes.
- Edges are grouping relationships recorded for every build, whether or not anything is exposed:
  two traces that declare the same task identity, or the build's own near-duplicate union with
  its similarity. Each edge names the project and build that produced it. Nothing weaker (a
  shared customer, conversation or time bucket) is an edge.
- Exposures mark nodes a person has seen, with the panel, receipt and report that showed them.

Content is exposed when it shares a connected component with an exposed node. The relation is
transitive and ignores time: an edge recorded before or after the exposure propagates it, and it
keeps propagating after the exposed trace leaves every later import. Records are content-addressed
files that are written once and never removed, so a ledger only grows; a snapshot identity names
exactly the records a decision used, and independence is only relative to that ledger.
"""

import json
import os
import tempfile
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Self, TypeVar

from pydantic import AwareDatetime, BaseModel, Field, JsonValue, model_validator

from exp.common.core.artifacts import ContractModel, sha256_json
from exp.common.core.text import normalize_durable_text
from exp.common.core.union_find import UnionFind
from exp.common.traces import Trace
from exp.simulation.mining.deduplicate import DuplicateEdge
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_real_transitions

FINGERPRINT_SCHEME = "trace-content-v1"
LEDGER_DIRECTORY = "exposure-ledger"
TASK_LINEAGE_KEY = "task_lineage"
_FINGERPRINT_LINEAGE = "lineage-fingerprint"  # placeholder binding; never part of the content
Fingerprint = str
EdgeReason = Literal["declared-task-identity", "near-duplicate"]
_Record = TypeVar("_Record", bound=BaseModel)


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
        how: ``inspected`` for an audit's own panel, ``re-audited`` for a development re-audit.
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


class ExposurePath(ContractModel):
    """Why some content counts as exposed under one ledger snapshot.

    Attributes:
        exposed_node: Nearest exposed fingerprint in the component.
        component_size: Fingerprints in the connected component, so over-long chains show.
        path: Edges from the queried content to ``exposed_node``; empty when it is exposed itself.
        exposures: Exposure records that marked ``exposed_node``.
    """

    exposed_node: Fingerprint
    component_size: int = Field(ge=1)
    path: tuple[LedgerEdge, ...] = ()
    exposures: tuple[str, ...] = Field(min_length=1)


class LedgerExport(ContractModel):
    """Every record of a ledger, for sharing or audit.

    Attributes:
        identity: Snapshot identity of exactly these records.
        scheme: Fingerprint scheme this code writes.
        builds: Builds that recorded their grouping.
        nodes: Content observations.
        edges: Grouping relationships.
        exposures: Exposure records.
    """

    identity: str
    scheme: str
    builds: tuple[LedgerBuild, ...] = ()
    nodes: tuple[LedgerNode, ...] = ()
    edges: tuple[LedgerEdge, ...] = ()
    exposures: tuple[LedgerExposure, ...] = ()


class LedgerSnapshot:
    """An immutable view of the ledger used for one decision."""

    def __init__(self, export: LedgerExport) -> None:
        """Index the records for component and path queries.

        Args:
            export: Every record of the snapshot.
        """
        self.records = export
        self.identity = export.identity
        self._builds = {(build.project, build.build) for build in export.builds}
        self._union = UnionFind(node.fingerprint for node in export.nodes)
        self._adjacent: dict[Fingerprint, list[tuple[Fingerprint, LedgerEdge]]] = defaultdict(list)
        for edge in export.edges:
            self._union.union(edge.left, edge.right)
            self._adjacent[edge.left].append((edge.right, edge))
            self._adjacent[edge.right].append((edge.left, edge))
        self._exposed: dict[Fingerprint, list[str]] = defaultdict(list)
        for exposure in export.exposures:
            for fingerprint in exposure.fingerprints:
                self._union.add(fingerprint)
                self._exposed[fingerprint].append(exposure.record_id)
        self._sizes: dict[str, int] = defaultdict(int)
        for fingerprint in {
            *(node.fingerprint for node in export.nodes),
            *self._adjacent,
            *self._exposed,
        }:
            self._sizes[self._union.find(fingerprint)] += 1

    def has_build(self, project: str, build: str) -> bool:
        """Whether this build recorded its grouping in the ledger."""
        return (project, build) in self._builds

    def exposure(self, fingerprints: Iterable[Fingerprint]) -> ExposurePath | None:
        """The shortest path from any of ``fingerprints`` to exposed content, if one exists.

        Args:
            fingerprints: Content of one lineage's member traces.

        Returns:
            The path, or None when no member shares a component with exposed content.
        """
        starts = sorted(set(fingerprints))
        previous: dict[Fingerprint, tuple[Fingerprint, LedgerEdge] | None] = dict.fromkeys(starts)
        queue = deque(starts)
        while queue:
            current = queue.popleft()
            if current in self._exposed:
                path: list[LedgerEdge] = []
                step = previous[current]
                while step is not None:
                    path.append(step[1])
                    step = previous[step[0]]
                return ExposurePath(
                    exposed_node=current,
                    component_size=self._sizes[self._union.find(current)],
                    path=tuple(reversed(path)),
                    exposures=tuple(sorted(self._exposed[current])),
                )
            for neighbour, edge in sorted(
                self._adjacent.get(current, ()), key=lambda item: _record_id(item[1])
            ):
                if neighbour not in previous:
                    previous[neighbour] = (current, edge)
                    queue.append(neighbour)
        return None


class ExposureLedger:
    """Append-only, content-addressed ledger files under ``<root>/exposure-ledger``."""

    def __init__(self, root: Path) -> None:
        """Bind the ledger of one ``.exp`` root without writing anything.

        Args:
            root: The ``.exp`` root shared by every project.
        """
        self.directory = Path(root) / LEDGER_DIRECTORY

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
        """Write a build's nodes and edges, then its marker, skipping records already present.

        Args:
            marker: The build that produced them.
            nodes: Content observations.
            edges: Grouping relationships.
        """
        for item in nodes:
            self._write("nodes", item)
        for edge in edges:
            self._write("edges", edge)
        self._write("builds", marker)

    def record_exposure(self, exposure: LedgerExposure) -> None:
        """Mark content exposed; the record is permanent.

        Args:
            exposure: What was shown, and how.
        """
        self._write("exposures", exposure)

    def snapshot(self) -> LedgerSnapshot:
        """Every record now in the ledger, as one queryable snapshot."""
        return LedgerSnapshot(self.export())

    def export(self) -> LedgerExport:
        """Every record now in the ledger, with its snapshot identity."""
        builds = self._read("builds", LedgerBuild)
        nodes = self._read("nodes", LedgerNode)
        edges = self._read("edges", LedgerEdge)
        exposures = self._read("exposures", LedgerExposure)
        identity = (
            "ledger-"
            + sha256_json(
                sorted(
                    f"{kind}/{_record_id(record)}"
                    for kind, records in (
                        ("builds", builds),
                        ("nodes", nodes),
                        ("edges", edges),
                        ("exposures", exposures),
                    )
                    for record in records
                )
            )[:20]
        )
        return LedgerExport(
            identity=identity,
            scheme=FINGERPRINT_SCHEME,
            builds=builds,
            nodes=nodes,
            edges=edges,
            exposures=exposures,
        )

    def merge(self, export: LedgerExport) -> None:
        """Add every record of another ledger's export; nothing is replaced or removed.

        Args:
            export: Records from another ledger.
        """
        for kind, records in (
            ("nodes", export.nodes),
            ("edges", export.edges),
            ("exposures", export.exposures),
            ("builds", export.builds),
        ):
            for record in records:
                self._write(kind, record)

    def _write(self, kind: str, record: BaseModel) -> None:
        """Write one content-addressed record once, atomically."""
        directory = self.directory / kind
        target = directory / f"{_record_id(record)}.json"
        if target.exists():
            return
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(record.model_dump(mode="json"), sort_keys=True)
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, suffix=".tmp") as file:
            file.write(payload)
            temporary = Path(file.name)
        os.replace(temporary, target)

    def _read(self, kind: str, model: type[_Record]) -> tuple[_Record, ...]:
        """Every record of one kind, in content-address order."""
        directory = self.directory / kind
        if not directory.is_dir():
            return ()
        return tuple(
            model.model_validate_json(path.read_text()) for path in sorted(directory.glob("*.json"))
        )


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


def _decoded(content: str) -> JsonValue:
    """Recorded text as JSON when it is JSON, so encoding whitespace never changes content."""
    try:
        return json.loads(content)
    except ValueError:
        return content


def _record_id(record: BaseModel) -> str:
    """Content address of one ledger record."""
    return sha256_json(record)[:32]
