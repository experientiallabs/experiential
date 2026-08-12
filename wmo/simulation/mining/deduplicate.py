"""Duplicate union and frozen leakage-safe fit versus held-out partitioning."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from wmo.common.core.artifacts import stable_id
from wmo.common.traces import Trace
from wmo.simulation.mining.descriptors import CoverageDescriptor, RoutingDescriptor
from wmo.simulation.mining.lineage import LineageAssignment

DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD = 0.95
DEFAULT_FIT_LINEAGE_TARGET = 50
DEFAULT_HELD_OUT_LINEAGE_TARGET = 20


@dataclass(frozen=True)
class DuplicateEdge:
    """One exact or semantic duplicate relationship found from routing descriptors.

    Args:
        left_trace_id: First canonical trace identity, sorted before the right identity.
        right_trace_id: Second canonical trace identity.
        left_lineage_group_id: Initial lineage for the first trace.
        right_lineage_group_id: Initial lineage for the second trace.
        kind: Exact fingerprint or semantic-vector duplicate evidence.
        similarity: Cosine similarity. Exact duplicates use exactly ``1.0``.
    """

    left_trace_id: str
    right_trace_id: str
    left_lineage_group_id: str
    right_lineage_group_id: str
    kind: Literal["exact", "semantic"]
    similarity: float


@dataclass(frozen=True)
class LeakageGroup:
    """Connected initial lineages that must remain entirely fit or held out.

    Args:
        lineage_group_id: Stable connected-component identity.
        initial_lineage_group_ids: Source lineages joined by duplicate evidence.
        source_trace_ids: Every trace protected by this split boundary.
        workload_mass: Uncollapsed source trace count represented by this component.
    """

    lineage_group_id: str
    initial_lineage_group_ids: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    workload_mass: int


@dataclass(frozen=True)
class DeduplicatedTrace:
    """One real source trace representing a duplicate component during task selection.

    Args:
        representative_trace_id: Real trace selected as the duplicate-component medoid.
        lineage_group_id: Connected leakage group that owns every represented source trace.
        source_trace_ids: Exact and semantic duplicate source traces represented by this exemplar.
        workload_mass: Number of source traces collapsed into this candidate.
        routing_descriptor: Request-visible descriptor used for clustering and router features.
        coverage_descriptor: Mining-only coverage facts aggregated across represented traces.
        source_coverage_descriptors: Mining-only facts retained per source trace for exact coverage
            mass accounting after duplicate collapse.
        vector: Unit routing-descriptor embedding used for duplicate and core-set geometry.
    """

    representative_trace_id: str
    lineage_group_id: str
    source_trace_ids: tuple[str, ...]
    workload_mass: int
    routing_descriptor: RoutingDescriptor
    coverage_descriptor: CoverageDescriptor
    source_coverage_descriptors: tuple[CoverageDescriptor, ...]
    vector: tuple[float, ...]


@dataclass(frozen=True)
class DuplicateAnalysis:
    """All duplicate evidence, sealed leakage groups, and selection candidates.

    Args:
        edges: Auditable exact and semantic duplicate evidence.
        leakage_groups: Connected source lineages that cannot cross the split.
        candidates: Deduplicated real-source candidates for representative task selection.
    """

    edges: tuple[DuplicateEdge, ...]
    leakage_groups: tuple[LeakageGroup, ...]
    candidates: tuple[DeduplicatedTrace, ...]


@dataclass(frozen=True)
class FrozenTaskPartition:
    """One deterministic lineage partition frozen before task selection.

    Args:
        fit_lineage_group_ids: Connected lineages available to fit-only work.
        held_out_lineage_group_ids: Connected lineages sealed for final reporting.
        fit_workload_mass: Source workload represented by fit groups.
        held_out_workload_mass: Source workload represented by held-out groups.
        underfilled_reason: Explicit reason when eligible lineages cannot fill the requested split.
    """

    fit_lineage_group_ids: tuple[str, ...]
    held_out_lineage_group_ids: tuple[str, ...]
    fit_workload_mass: int
    held_out_workload_mass: int
    underfilled_reason: str | None = None

    def partition_for(self, lineage_group_id: str) -> Literal["fit", "held_out"]:
        """Return the frozen partition for one connected lineage.

        Args:
            lineage_group_id: Connected lineage identity to look up.

        Returns:
            The fit or held-out partition.

        Raises:
            KeyError: The lineage was not part of this frozen task partition.
        """
        if lineage_group_id in self.fit_lineage_group_ids:
            return "fit"
        if lineage_group_id in self.held_out_lineage_group_ids:
            return "held_out"
        raise KeyError(f"lineage group {lineage_group_id!r} is not in the frozen task partition")


def analyze_duplicates(
    traces: Sequence[Trace],
    assignments: Sequence[LineageAssignment],
    routing_descriptors: Sequence[RoutingDescriptor],
    coverage_descriptors: Sequence[CoverageDescriptor],
    vectors: Sequence[tuple[float, ...]],
    *,
    semantic_duplicate_threshold: float = DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
) -> DuplicateAnalysis:
    """Find duplicate edges, union leakage groups, and retain real representative candidates.

    Args:
        traces: Canonical traces in the same order as all companion sequences.
        assignments: Initial source lineage assignments.
        routing_descriptors: Request-visible descriptors only.
        coverage_descriptors: Mining-only descriptors, never used to form duplicate edges.
        vectors: Unit routing-descriptor vectors in input order.
        semantic_duplicate_threshold: Inclusive cosine threshold for semantic duplicate edges.

    Returns:
        Duplicate evidence, connected leakage groups, and collapsed real-source candidates.

    Raises:
        ValueError: Inputs are misaligned, repeated, or use an invalid semantic threshold.
    """
    _validate_inputs(traces, assignments, routing_descriptors, coverage_descriptors, vectors)
    if not 0.0 <= semantic_duplicate_threshold <= 1.0:
        raise ValueError("semantic duplicate threshold must be between 0 and 1")

    assignment_by_trace = {assignment.trace_id: assignment for assignment in assignments}
    descriptor_by_trace = {descriptor.trace_id: descriptor for descriptor in routing_descriptors}
    coverage_by_trace = {descriptor.trace_id: descriptor for descriptor in coverage_descriptors}
    vector_by_trace = {
        trace.trace_id: vector for trace, vector in zip(traces, vectors, strict=True)
    }

    trace_union = _UnionFind(trace.trace_id for trace in traces)
    lineage_union = _UnionFind(assignment.lineage_group_id for assignment in assignments)
    edges = _duplicate_edges(
        traces,
        assignments,
        routing_descriptors,
        vectors,
        semantic_duplicate_threshold=semantic_duplicate_threshold,
    )
    for edge in edges:
        trace_union.union(edge.left_trace_id, edge.right_trace_id)
        lineage_union.union(edge.left_lineage_group_id, edge.right_lineage_group_id)
    for assignment in assignments:
        lineage_union.add(assignment.lineage_group_id)

    leakage_by_initial = _leakage_ids(assignments, lineage_union)
    leakage_groups = _build_leakage_groups(assignments, leakage_by_initial)
    candidate_groups = _groups_by_root((trace.trace_id for trace in traces), trace_union)
    candidates = []
    for trace_ids in candidate_groups.values():
        candidates.append(
            _candidate_from_duplicate_group(
                trace_ids=tuple(sorted(trace_ids)),
                assignment_by_trace=assignment_by_trace,
                descriptor_by_trace=descriptor_by_trace,
                coverage_by_trace=coverage_by_trace,
                vector_by_trace=vector_by_trace,
                leakage_by_initial=leakage_by_initial,
            )
        )
    candidates.sort(key=lambda candidate: candidate.representative_trace_id)
    return DuplicateAnalysis(
        edges=tuple(edges),
        leakage_groups=tuple(leakage_groups),
        candidates=tuple(candidates),
    )


def freeze_task_partition(
    groups: Sequence[LeakageGroup],
    *,
    fit_lineage_target: int = DEFAULT_FIT_LINEAGE_TARGET,
    held_out_lineage_target: int = DEFAULT_HELD_OUT_LINEAGE_TARGET,
) -> FrozenTaskPartition:
    """Freeze a deterministic partition over connected source lineages.

    Args:
        groups: Connected leakage groups output by ``analyze_duplicates``.
        fit_lineage_target: Desired fit lineage count before representative selection.
        held_out_lineage_target: Desired held-out lineage count before representative selection.

    Returns:
        A partition that never splits a connected lineage or duplicate component.

    Raises:
        ValueError: Groups repeat, have invalid workload mass, or targets are invalid.
    """
    if fit_lineage_target < 0 or held_out_lineage_target < 0:
        raise ValueError("lineage targets cannot be negative")
    if fit_lineage_target + held_out_lineage_target == 0:
        raise ValueError("at least one lineage target must be positive")
    group_ids = tuple(group.lineage_group_id for group in groups)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("frozen task partition needs unique leakage group IDs")
    if any(group.workload_mass <= 0 for group in groups):
        raise ValueError("leakage groups need positive workload mass")
    if not groups:
        return FrozenTaskPartition((), (), 0, 0, "no eligible lineage groups")
    target_count = fit_lineage_target + held_out_lineage_target
    underfilled_reason = (
        None
        if len(groups) >= target_count
        else (
            f"{len(groups)} eligible lineage groups are below the requested {target_count} "
            "representative capacity; every lineage remains in the deterministic split"
        )
    )
    if len(groups) == 1:
        group = groups[0]
        return FrozenTaskPartition(
            fit_lineage_group_ids=(group.lineage_group_id,),
            held_out_lineage_group_ids=(),
            fit_workload_mass=group.workload_mass,
            held_out_workload_mass=0,
            underfilled_reason=underfilled_reason,
        )

    by_id = {group.lineage_group_id: group for group in groups}
    ranked = sorted(
        by_id,
        key=lambda group_id: (
            hashlib.sha256(f"task-split-v1\0{group_id}".encode()).digest(),
            group_id,
        ),
    )
    if held_out_lineage_target == 0:
        fit_count = len(ranked)
    elif fit_lineage_target == 0:
        fit_count = 0
    else:
        fit_count = len(ranked) * fit_lineage_target // target_count
        fit_count = min(max(fit_count, 1), len(ranked) - 1)
    fit_ids = ranked[:fit_count]
    held_out_ids = ranked[fit_count:]
    return FrozenTaskPartition(
        fit_lineage_group_ids=tuple(sorted(fit_ids)),
        held_out_lineage_group_ids=tuple(sorted(held_out_ids)),
        fit_workload_mass=sum(by_id[group_id].workload_mass for group_id in fit_ids),
        held_out_workload_mass=sum(by_id[group_id].workload_mass for group_id in held_out_ids),
        underfilled_reason=underfilled_reason,
    )


def _validate_inputs(
    traces: Sequence[Trace],
    assignments: Sequence[LineageAssignment],
    routing_descriptors: Sequence[RoutingDescriptor],
    coverage_descriptors: Sequence[CoverageDescriptor],
    vectors: Sequence[tuple[float, ...]],
) -> None:
    """Reject sequence misalignment before combining source evidence."""
    sizes = {
        len(traces),
        len(assignments),
        len(routing_descriptors),
        len(coverage_descriptors),
        len(vectors),
    }
    if len(sizes) != 1:
        raise ValueError("duplicate analysis inputs must have matching lengths")
    trace_ids = tuple(trace.trace_id for trace in traces)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("duplicate analysis needs unique trace IDs")
    expected = set(trace_ids)
    for label, ids in (
        ("lineage assignments", {item.trace_id for item in assignments}),
        ("routing descriptors", {item.trace_id for item in routing_descriptors}),
        ("coverage descriptors", {item.trace_id for item in coverage_descriptors}),
    ):
        if ids != expected:
            raise ValueError(f"{label} do not match trace identities")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("duplicate analysis vectors need one non-empty shared dimension")


def _duplicate_edges(
    traces: Sequence[Trace],
    assignments: Sequence[LineageAssignment],
    descriptors: Sequence[RoutingDescriptor],
    vectors: Sequence[tuple[float, ...]],
    *,
    semantic_duplicate_threshold: float,
) -> list[DuplicateEdge]:
    """Build exact and semantic duplicate edges using routing data only."""
    assignment_by_trace = {assignment.trace_id: assignment for assignment in assignments}
    descriptor_by_trace = {descriptor.trace_id: descriptor for descriptor in descriptors}
    vector_by_trace = {
        trace.trace_id: vector for trace, vector in zip(traces, vectors, strict=True)
    }
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for trace in traces:
        by_fingerprint[descriptor_by_trace[trace.trace_id].fingerprint()].append(trace.trace_id)
    edge_keys: set[tuple[str, str]] = set()
    edges: list[DuplicateEdge] = []
    for trace_ids in by_fingerprint.values():
        ordered = sorted(trace_ids)
        for left_index, left_trace_id in enumerate(ordered):
            for right_trace_id in ordered[left_index + 1 :]:
                edge_keys.add((left_trace_id, right_trace_id))
                edges.append(
                    _edge(
                        left_trace_id,
                        right_trace_id,
                        assignment_by_trace,
                        kind="exact",
                        similarity=1.0,
                    )
                )
    ordered_trace_ids = sorted(trace.trace_id for trace in traces)
    for left_index, left_trace_id in enumerate(ordered_trace_ids):
        for right_trace_id in ordered_trace_ids[left_index + 1 :]:
            pair = (left_trace_id, right_trace_id)
            if pair in edge_keys:
                continue
            similarity = _cosine(vector_by_trace[left_trace_id], vector_by_trace[right_trace_id])
            if similarity >= semantic_duplicate_threshold:
                edges.append(
                    _edge(
                        left_trace_id,
                        right_trace_id,
                        assignment_by_trace,
                        kind="semantic",
                        similarity=similarity,
                    )
                )
    return edges


def _edge(
    left_trace_id: str,
    right_trace_id: str,
    assignments: dict[str, LineageAssignment],
    *,
    kind: Literal["exact", "semantic"],
    similarity: float,
) -> DuplicateEdge:
    """Build an auditable edge in a stable trace-identity order."""
    return DuplicateEdge(
        left_trace_id=left_trace_id,
        right_trace_id=right_trace_id,
        left_lineage_group_id=assignments[left_trace_id].lineage_group_id,
        right_lineage_group_id=assignments[right_trace_id].lineage_group_id,
        kind=kind,
        similarity=similarity,
    )


def _leakage_ids(assignments: Sequence[LineageAssignment], _union: _UnionFind) -> dict[str, str]:
    """Create stable IDs for connected initial-lineage components."""
    by_root = _groups_by_root((assignment.lineage_group_id for assignment in assignments), _union)
    by_initial: dict[str, str] = {}
    for initial_ids in by_root.values():
        stable_group_id = stable_id(
            "leakage",
            {"version": "leakage-group-v1", "initial_lineage_group_ids": sorted(initial_ids)},
        )
        for initial_id in initial_ids:
            by_initial[initial_id] = stable_group_id
    return by_initial


def _build_leakage_groups(
    assignments: Sequence[LineageAssignment], leakage_by_initial: dict[str, str]
) -> list[LeakageGroup]:
    """Materialize connected lineage groups with workload mass for splitting."""
    initial_by_leakage: dict[str, set[str]] = defaultdict(set)
    traces_by_leakage: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        leakage_id = leakage_by_initial[assignment.lineage_group_id]
        initial_by_leakage[leakage_id].add(assignment.lineage_group_id)
        traces_by_leakage[leakage_id].append(assignment.trace_id)
    return [
        LeakageGroup(
            lineage_group_id=leakage_id,
            initial_lineage_group_ids=tuple(sorted(initial_by_leakage[leakage_id])),
            source_trace_ids=tuple(sorted(traces_by_leakage[leakage_id])),
            workload_mass=len(traces_by_leakage[leakage_id]),
        )
        for leakage_id in sorted(traces_by_leakage)
    ]


def _candidate_from_duplicate_group(
    *,
    trace_ids: tuple[str, ...],
    assignment_by_trace: dict[str, LineageAssignment],
    descriptor_by_trace: dict[str, RoutingDescriptor],
    coverage_by_trace: dict[str, CoverageDescriptor],
    vector_by_trace: dict[str, tuple[float, ...]],
    leakage_by_initial: dict[str, str],
) -> DeduplicatedTrace:
    """Collapse one duplicate component to a real central source trace."""
    representative_trace_id = min(
        trace_ids,
        key=lambda trace_id: (
            -sum(_cosine(vector_by_trace[trace_id], vector_by_trace[other]) for other in trace_ids),
            trace_id,
        ),
    )
    lineage_ids = {
        leakage_by_initial[assignment_by_trace[trace_id].lineage_group_id] for trace_id in trace_ids
    }
    if len(lineage_ids) != 1:
        raise ValueError("one duplicate component must belong to one leakage group")
    coverage = _aggregate_coverage(
        representative_trace_id,
        tuple(coverage_by_trace[trace_id] for trace_id in trace_ids),
    )
    return DeduplicatedTrace(
        representative_trace_id=representative_trace_id,
        lineage_group_id=next(iter(lineage_ids)),
        source_trace_ids=trace_ids,
        workload_mass=len(trace_ids),
        routing_descriptor=descriptor_by_trace[representative_trace_id],
        coverage_descriptor=coverage,
        source_coverage_descriptors=tuple(coverage_by_trace[trace_id] for trace_id in trace_ids),
        vector=vector_by_trace[representative_trace_id],
    )


def _aggregate_coverage(
    representative_trace_id: str, descriptors: Sequence[CoverageDescriptor]
) -> CoverageDescriptor:
    """Keep tail coverage facts when duplicate source traces collapse to one exemplar."""
    transitions = sorted(
        {tool for descriptor in descriptors for tool in descriptor.tool_transitions}
    )
    return CoverageDescriptor(
        trace_id=representative_trace_id,
        tool_transitions=tuple(transitions),
        outcome_statuses=tuple(
            sorted(status for descriptor in descriptors for status in descriptor.outcome_statuses)
        ),
        is_escalation=any(descriptor.is_escalation for descriptor in descriptors),
        span_counts=tuple(
            sorted(
                span_count for descriptor in descriptors for span_count in descriptor.span_counts
            )
        ),
        domains=tuple(
            sorted(domain for descriptor in descriptors for domain in descriptor.domains)
        ),
    )


def _groups_by_root(values: Iterable[str], union: _UnionFind) -> dict[str, list[str]]:
    """Group stable string identities by a union-find root."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in values:
        grouped[union.find(value)].append(value)
    return grouped


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a bounded dot product for pre-normalized vectors."""
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


class _UnionFind:
    """Small deterministic union-find implementation for lineage and duplicate components."""

    def __init__(self, values: Iterable[str]) -> None:
        """Initialize each source identity as its own component."""
        self._parent: dict[str, str] = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        """Add one identity when it has not been seen before."""
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        """Return one component root with path compression."""
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        """Join two components using the lexically smallest root for deterministic ownership."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root
