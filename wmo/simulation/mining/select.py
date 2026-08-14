"""Deterministic real-trace medoid, tail-slot, and farthest-first selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from wmo.simulation.mining.deduplicate import DeduplicatedTrace

_RESERVED_SLOT_NAMES = ("rare_tool", "failure", "escalation", "long", "boundary")


@dataclass(frozen=True)
class SelectedRepresentative:
    """One real selected trace and the workload mass it represents.

    Args:
        representative_trace_id: Source trace retained as the actual task.
        lineage_group_id: Connected lineage that fixes the task partition.
        partition: Frozen fit or held-out partition.
        cluster_id: Deterministic request-descriptor cluster used for selection audit.
        selection_reasons: Medoid, tail-slot, and farthest-first selection reasons.
        source_trace_ids: Direct duplicate-source provenance for the retained real task.
        workload_mass: Unnormalized source workload represented by this task.
    """

    representative_trace_id: str
    lineage_group_id: str
    partition: Literal["fit", "held_out"]
    cluster_id: int
    selection_reasons: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    workload_mass: int


@dataclass(frozen=True)
class ClusterSelection:
    """One deterministic cluster's source mass and medoid selection state.

    Args:
        cluster_id: Stable local cluster index within one partition.
        candidate_count: Deduplicated candidate count assigned to the cluster.
        workload_mass: Uncollapsed trace workload represented by the cluster.
        medoid_trace_id: Real central source trace for the cluster.
        selected_medoid: Whether available budget retained that real medoid.
    """

    cluster_id: int
    candidate_count: int
    workload_mass: int
    medoid_trace_id: str
    selected_medoid: bool


@dataclass(frozen=True)
class PartitionSelection:
    """Selection result and explicit missing-tail coverage for one frozen partition.

    Args:
        partition: Fit or held-out partition selected independently.
        requested_budget: Requested task count before underfill.
        selected: Real selected source traces in deterministic selection order.
        missing_reserved_slots: Important tail slots that had no eligible source or no capacity.
        clusters: Source cluster sizes and their real medoid retention state.
    """

    partition: Literal["fit", "held_out"]
    requested_budget: int
    selected: tuple[SelectedRepresentative, ...]
    missing_reserved_slots: tuple[str, ...]
    clusters: tuple[ClusterSelection, ...]

    @property
    def underfilled(self) -> bool:
        """Return whether source eligibility could not fill the requested task budget."""
        return len(self.selected) < self.requested_budget


def select_partition_representatives(
    candidates: Sequence[DeduplicatedTrace],
    *,
    partition: Literal["fit", "held_out"],
    budget: int,
) -> PartitionSelection:
    """Select a deterministic weighted core set within one frozen partition.

    The function first reserves available rare-tool, failure, escalation, long-episode, and
    cluster-boundary slots. It then adds real cluster medoids and farthest-first coverage points.
    Every unselected candidate is assigned to its nearest retained representative so workload mass
    remains visible rather than disappearing with duplicate removal.

    Args:
        candidates: Deduplicated real-source candidates from exactly one partition.
        partition: Partition named in the resulting audit records.
        budget: Maximum task count for this partition.

    Returns:
        Selected real traces, workload assignment, and unavailable tail-slot reasons.

    Raises:
        ValueError: The budget is negative or candidates repeat source identities.
    """
    if budget < 0:
        raise ValueError("representative task budget cannot be negative")
    candidate_ids = tuple(candidate.representative_trace_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("representative selection needs unique source trace IDs")
    if budget == 0 or not candidates:
        missing = tuple(f"{name}: no selection capacity" for name in _RESERVED_SLOT_NAMES)
        return PartitionSelection(partition, budget, (), missing, ())

    target = min(budget, len(candidates))
    clusters = _cluster_candidates(candidates, target)
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    missing_slots: list[str] = []
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}

    for slot_name in _RESERVED_SLOT_NAMES:
        if len(selected_ids) >= target:
            missing_slots.append(f"{slot_name}: budget exhausted")
            continue
        candidate_id = _reserved_candidate(slot_name, candidates, clusters, selected_ids)
        if candidate_id is None:
            missing_slots.append(f"{slot_name}: no eligible source trace")
            continue
        _select(selected_ids, reasons, candidate_id, f"reserved:{slot_name}")

    for lineage_group_id in sorted({candidate.lineage_group_id for candidate in candidates}):
        if any(
            by_id[selected_id].lineage_group_id == lineage_group_id for selected_id in selected_ids
        ):
            continue
        if len(selected_ids) >= target:
            break
        lineage_member_ids = [
            candidate.representative_trace_id
            for candidate in candidates
            if candidate.lineage_group_id == lineage_group_id
        ]
        _select(
            selected_ids,
            reasons,
            _medoid(lineage_member_ids, by_id),
            "lineage-anchor",
        )

    for cluster_id in sorted(set(clusters.values())):
        member_ids = [
            candidate.representative_trace_id
            for candidate in candidates
            if clusters[candidate.representative_trace_id] == cluster_id
        ]
        medoid_id = _medoid(member_ids, by_id)
        if medoid_id in selected_ids:
            reasons[medoid_id].append(f"cluster-medoid:{cluster_id}")
            continue
        if len(selected_ids) >= target:
            break
        _select(selected_ids, reasons, medoid_id, f"cluster-medoid:{cluster_id}")

    while len(selected_ids) < target:
        next_id = _farthest_first_candidate(candidates, selected_ids)
        _select(selected_ids, reasons, next_id, "farthest-first")

    selected = _assign_workload(candidates, selected_ids, reasons, clusters, partition)
    cluster_summaries = _cluster_summaries(candidates, clusters, selected_ids, by_id)
    return PartitionSelection(
        partition=partition,
        requested_budget=budget,
        selected=tuple(selected),
        missing_reserved_slots=tuple(missing_slots),
        clusters=cluster_summaries,
    )


def _cluster_candidates(candidates: Sequence[DeduplicatedTrace], target: int) -> dict[str, int]:
    """Build deterministic request-vector clusters and return candidate-to-cluster assignments."""
    cluster_count = min(len(candidates), max(1, round(math.sqrt(len(candidates)))), target)
    if cluster_count == 1:
        return {candidate.representative_trace_id: 0 for candidate in candidates}
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
    centers = _initial_centers(candidates, cluster_count)
    assignments: dict[str, int] = {}
    for _ in range(20):
        updated = {
            candidate.representative_trace_id: _nearest_center(candidate, centers, by_id)
            for candidate in candidates
        }
        if updated == assignments:
            break
        assignments = updated
        centers = _reseed_centers(candidates, assignments, cluster_count, by_id)
    return assignments


def _initial_centers(candidates: Sequence[DeduplicatedTrace], count: int) -> list[str]:
    """Seed clusters with a lexical anchor followed by deterministic farthest-first points."""
    selected = [min(candidate.representative_trace_id for candidate in candidates)]
    while len(selected) < count:
        selected.append(_farthest_first_candidate(candidates, selected))
    return selected


def _nearest_center(
    candidate: DeduplicatedTrace,
    centers: Sequence[str],
    by_id: dict[str, DeduplicatedTrace],
) -> int:
    """Return the best center index, breaking vector ties by center identity."""
    ranked = sorted(
        (
            (-_cosine(candidate.vector, by_id[center_id].vector), center_id, index)
            for index, center_id in enumerate(centers)
        )
    )
    return ranked[0][2]


def _reseed_centers(
    candidates: Sequence[DeduplicatedTrace],
    assignments: dict[str, int],
    cluster_count: int,
    by_id: dict[str, DeduplicatedTrace],
) -> list[str]:
    """Choose a real weighted medoid for each non-empty cluster, retaining stable empty centers."""
    centers: list[str] = []
    for cluster_id in range(cluster_count):
        member_ids = [
            candidate.representative_trace_id
            for candidate in candidates
            if assignments[candidate.representative_trace_id] == cluster_id
        ]
        if member_ids:
            medoid = _medoid(member_ids, by_id)
            centers.append(medoid)
    while len(centers) < cluster_count:
        centers.append(_farthest_first_candidate(candidates, centers))
    return centers


def _medoid(member_ids: Sequence[str], by_id: dict[str, DeduplicatedTrace]) -> str:
    """Return a real weighted central member rather than a synthetic centroid."""
    return min(
        member_ids,
        key=lambda candidate_id: (
            -sum(
                _cosine(by_id[candidate_id].vector, by_id[other_id].vector)
                * by_id[other_id].workload_mass
                for other_id in member_ids
            ),
            candidate_id,
        ),
    )


def _reserved_candidate(
    slot_name: str,
    candidates: Sequence[DeduplicatedTrace],
    clusters: dict[str, int],
    selected_ids: Sequence[str],
) -> str | None:
    """Return one unselected trace for a required mining-only coverage tail slot."""
    unavailable = set(selected_ids)
    eligible: list[DeduplicatedTrace]
    if slot_name == "rare_tool":
        tool_mass: dict[str, int] = defaultdict(int)
        total_mass = sum(candidate.workload_mass for candidate in candidates)
        for candidate in candidates:
            for tool in set(candidate.coverage_descriptor.tool_transitions):
                tool_mass[tool] += candidate.workload_mass
        rarity_limit = max(1, math.ceil(total_mass * 0.10))
        eligible = [
            candidate
            for candidate in candidates
            if any(
                tool_mass[tool] <= rarity_limit
                for tool in candidate.coverage_descriptor.tool_transitions
            )
        ]
        available = [
            candidate
            for candidate in eligible
            if candidate.representative_trace_id not in unavailable
        ]
        if not available:
            return None
        return min(
            available,
            key=lambda candidate: (
                min(
                    tool_mass[tool]
                    for tool in candidate.coverage_descriptor.tool_transitions
                    if tool_mass[tool] <= rarity_limit
                ),
                -candidate.workload_mass,
                candidate.representative_trace_id,
            ),
        ).representative_trace_id
    if slot_name == "failure":
        eligible = [
            candidate
            for candidate in candidates
            if candidate.coverage_descriptor.has_failure
            and candidate.representative_trace_id not in unavailable
        ]
    elif slot_name == "escalation":
        eligible = [
            candidate
            for candidate in candidates
            if candidate.coverage_descriptor.is_escalation
            and candidate.representative_trace_id not in unavailable
        ]
    elif slot_name == "long":
        threshold = _long_span_threshold(candidates)
        eligible = [
            candidate
            for candidate in candidates
            if threshold is not None
            and any(
                span_count >= threshold for span_count in candidate.coverage_descriptor.span_counts
            )
            and candidate.representative_trace_id not in unavailable
        ]
    elif slot_name == "boundary":
        eligible = [
            candidate
            for candidate in candidates
            if candidate.representative_trace_id not in unavailable
        ]
        if len(set(clusters.values())) < 2:
            return None
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda candidate: (
                _boundary_gap(candidate, candidates, clusters),
                candidate.representative_trace_id,
            ),
        ).representative_trace_id
    else:  # pragma: no cover - fixed local slot tuple guards this branch
        raise ValueError(f"unknown reserved selection slot {slot_name!r}")
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (-candidate.workload_mass, candidate.representative_trace_id),
    ).representative_trace_id


def _long_span_threshold(candidates: Sequence[DeduplicatedTrace]) -> int | None:
    """Return the deterministic upper-decile span-count threshold for long episode coverage."""
    counts = sorted(
        span_count
        for candidate in candidates
        for span_count in candidate.coverage_descriptor.span_counts
    )
    if counts[0] == counts[-1]:
        return None
    index = min(len(counts) - 1, math.ceil(len(counts) * 0.9) - 1)
    return counts[index]


def _boundary_gap(
    candidate: DeduplicatedTrace,
    candidates: Sequence[DeduplicatedTrace],
    clusters: dict[str, int],
) -> float:
    """Measure closeness to the two nearest cluster medoids as a boundary proxy."""
    by_id = {item.representative_trace_id: item for item in candidates}
    medoids = {
        cluster_id: _medoid(
            [
                item.representative_trace_id
                for item in candidates
                if clusters[item.representative_trace_id] == cluster_id
            ],
            by_id,
        )
        for cluster_id in set(clusters.values())
    }
    similarities = sorted(
        (_cosine(candidate.vector, by_id[medoid_id].vector) for medoid_id in medoids.values()),
        reverse=True,
    )
    return similarities[0] - similarities[1]


def _farthest_first_candidate(
    candidates: Sequence[DeduplicatedTrace], selected_ids: Sequence[str]
) -> str:
    """Choose the real candidate farthest from every already selected representative."""
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
    remaining = [
        candidate
        for candidate in candidates
        if candidate.representative_trace_id not in set(selected_ids)
    ]
    if not remaining:
        raise ValueError("farthest-first selection has no remaining candidate")
    if not selected_ids:
        return _medoid(tuple(by_id), by_id)
    return min(
        remaining,
        key=lambda candidate: (
            max(
                _cosine(candidate.vector, by_id[selected_id].vector) for selected_id in selected_ids
            ),
            candidate.representative_trace_id,
        ),
    ).representative_trace_id


def _select(
    selected_ids: list[str], reasons: dict[str, list[str]], candidate_id: str, reason: str
) -> None:
    """Add one selected source trace or append another auditable reason to it."""
    if candidate_id not in selected_ids:
        selected_ids.append(candidate_id)
    reasons[candidate_id].append(reason)


def _assign_workload(
    candidates: Sequence[DeduplicatedTrace],
    selected_ids: Sequence[str],
    reasons: dict[str, list[str]],
    clusters: dict[str, int],
    partition: Literal["fit", "held_out"],
) -> list[SelectedRepresentative]:
    """Assign every candidate's full duplicate workload to its nearest selected task."""
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
    mass: dict[str, int] = dict.fromkeys(selected_ids, 0)
    for candidate in candidates:
        selected_id = min(
            selected_ids,
            key=lambda selected: (
                -_cosine(candidate.vector, by_id[selected].vector),
                selected,
            ),
        )
        mass[selected_id] += candidate.workload_mass
    return [
        SelectedRepresentative(
            representative_trace_id=selected_id,
            lineage_group_id=by_id[selected_id].lineage_group_id,
            partition=partition,
            cluster_id=clusters[selected_id],
            selection_reasons=tuple(reasons[selected_id]),
            source_trace_ids=by_id[selected_id].source_trace_ids,
            workload_mass=mass[selected_id],
        )
        for selected_id in selected_ids
    ]


def _cluster_summaries(
    candidates: Sequence[DeduplicatedTrace],
    clusters: dict[str, int],
    selected_ids: Sequence[str],
    by_id: dict[str, DeduplicatedTrace],
) -> tuple[ClusterSelection, ...]:
    """Describe full source cluster mass and whether its real medoid survived selection."""
    summaries: list[ClusterSelection] = []
    selected = set(selected_ids)
    for cluster_id in sorted(set(clusters.values())):
        member_ids = [
            candidate.representative_trace_id
            for candidate in candidates
            if clusters[candidate.representative_trace_id] == cluster_id
        ]
        medoid_id = _medoid(member_ids, by_id)
        summaries.append(
            ClusterSelection(
                cluster_id=cluster_id,
                candidate_count=len(member_ids),
                workload_mass=sum(by_id[member_id].workload_mass for member_id in member_ids),
                medoid_trace_id=medoid_id,
                selected_medoid=medoid_id in selected,
            )
        )
    return tuple(summaries)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a bounded dot product for pre-normalized vectors."""
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
