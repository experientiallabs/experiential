"""Deterministic real-trace medoid, tail-slot, and farthest-first selection."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from exp.simulation.mining.clustering import (
    ClusterSpec,
    DescriptorClusterSpec,
    cluster_candidates,
)
from exp.simulation.mining.deduplicate import DeduplicatedTrace, validate_selection_geometry
from exp.simulation.mining.geometry import CoreSetGeometry, cosine
from exp.simulation.mining.sampling import (
    MedoidSamplingSpec,
    SamplingSpec,
    fill_budget,
    members_by_group,
    record_selection,
)

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
    cluster: ClusterSpec | None = None,
    sampling: SamplingSpec | None = None,
    probabilities: Mapping[str, tuple[float, ...]] | None = None,
) -> PartitionSelection:
    """Select a deterministic weighted core set within one frozen partition.

    The phase order is the contract, because each phase pays for a guarantee the next one assumes.
    Reserved rare-tool, failure, escalation, long-episode and cluster-boundary slots come first,
    while capacity is protected for lineages nothing covers yet. Then one anchor per lineage, so no
    lineage goes unrepresented. Only then is the remaining budget discretionary, and that is the
    part a sampling spec decides. Every unselected candidate is finally assigned to its nearest
    retained representative, so workload mass stays visible rather than disappearing with
    duplicate removal.

    Args:
        candidates: Deduplicated real-source candidates from exactly one partition.
        partition: Partition named in the resulting audit records.
        budget: Maximum task count for this partition.
        cluster: Which clustering backend groups the candidates. None uses the provider-free
            descriptor clustering, which is seedless and needs no model.
        sampling: How the remaining budget is spent. None fills with one real cluster medoid each
            and then farthest-first coverage.
        probabilities: Untransformed decision-model answers per candidate, required only by a
            clustering backend that reads tag probabilities directly, which is the Bernoulli
            mixture and the pattern hierarchy.

    Returns:
        Selected real traces, workload assignment, and unavailable tail-slot reasons.

    Raises:
        ValueError: The budget is negative, candidates repeat source identities, or their
            selection vectors do not share one width.
        ClusteringError: The clustering backend cannot read the supplied evidence.
    """
    if budget < 0:
        raise ValueError("representative task budget cannot be negative")
    candidate_ids = tuple(candidate.representative_trace_id for candidate in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("representative selection needs unique source trace IDs")
    if budget == 0 or not candidates:
        missing = tuple(f"{name}: no selection capacity" for name in _RESERVED_SLOT_NAMES)
        return PartitionSelection(partition, budget, (), missing, ())

    validate_selection_geometry(candidates)
    target = min(budget, len(candidates))
    geometry = CoreSetGeometry.of(
        {candidate.representative_trace_id: candidate.selection_vector for candidate in candidates},
        {candidate.representative_trace_id: candidate.workload_mass for candidate in candidates},
    )
    assignment = cluster_candidates(
        cluster or DescriptorClusterSpec(),
        geometry=geometry,
        target=target,
        probabilities=probabilities,
    )
    clusters = dict(assignment.labels)
    selected_ids: list[str] = []
    reasons: dict[str, list[str]] = defaultdict(list)
    missing_slots: list[str] = []
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
    lineage_ids = {candidate.lineage_group_id for candidate in candidates}

    for slot_name in _RESERVED_SLOT_NAMES:
        if len(selected_ids) >= target:
            missing_slots.append(f"{slot_name}: budget exhausted")
            continue
        covered_lineages = {by_id[selected_id].lineage_group_id for selected_id in selected_ids}
        uncovered_count = len(lineage_ids - covered_lineages)
        protected_ids = (
            tuple(
                candidate.representative_trace_id
                for candidate in candidates
                if candidate.lineage_group_id in covered_lineages
            )
            if target - len(selected_ids) <= uncovered_count
            else ()
        )
        candidate_id = _reserved_candidate(
            slot_name, candidates, clusters, (*selected_ids, *protected_ids), geometry
        )
        if candidate_id is None:
            reason = (
                "capacity reserved for uncovered lineages"
                if protected_ids
                and _reserved_candidate(slot_name, candidates, clusters, selected_ids, geometry)
                is not None
                else "no eligible source trace"
            )
            missing_slots.append(f"{slot_name}: {reason}")
            continue
        record_selection(selected_ids, reasons, candidate_id, f"reserved:{slot_name}")

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
        record_selection(
            selected_ids,
            reasons,
            geometry.medoid(lineage_member_ids),
            "lineage-anchor",
        )

    fill_budget(
        sampling or MedoidSamplingSpec(),
        geometry=geometry,
        labels=clusters,
        selected_ids=selected_ids,
        reasons=reasons,
        target=target,
        strata=assignment.strata,
    )

    selected = _assign_workload(candidates, selected_ids, reasons, clusters, partition)
    cluster_summaries = _cluster_summaries(clusters, selected_ids, geometry)
    return PartitionSelection(
        partition=partition,
        requested_budget=budget,
        selected=tuple(selected),
        missing_reserved_slots=tuple(missing_slots),
        clusters=cluster_summaries,
    )


def _reserved_candidate(
    slot_name: str,
    candidates: Sequence[DeduplicatedTrace],
    clusters: dict[str, int],
    selected_ids: Sequence[str],
    geometry: CoreSetGeometry,
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
                _boundary_gap(candidate.representative_trace_id, clusters, geometry),
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


def _boundary_gap(candidate_id: str, clusters: dict[str, int], geometry: CoreSetGeometry) -> float:
    """Measure closeness to the two nearest cluster medoids as a boundary proxy.

    A case sitting between two groups is where a rubric is most likely to be wrong, so one slot is
    reserved for the smallest gap between the nearest and second-nearest cluster centre.
    """
    medoids = {
        cluster_id: geometry.medoid(
            [member_id for member_id in geometry.ids if clusters[member_id] == cluster_id]
        )
        for cluster_id in set(clusters.values())
    }
    similarities = sorted(
        (geometry.similarity(candidate_id, medoid_id) for medoid_id in medoids.values()),
        reverse=True,
    )
    return similarities[0] - similarities[1]


def _assign_workload(
    candidates: Sequence[DeduplicatedTrace],
    selected_ids: Sequence[str],
    reasons: dict[str, list[str]],
    clusters: dict[str, int],
    partition: Literal["fit", "held_out"],
) -> list[SelectedRepresentative]:
    """Keep each selected case's own mass and assign unselected cases by distance."""
    by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
    mass: dict[str, int] = dict.fromkeys(selected_ids, 0)
    for candidate in candidates:
        selected_id = nearest_representative(candidate, selected_ids, by_id)
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


def nearest_representative(
    candidate: DeduplicatedTrace,
    selected_ids: Sequence[str],
    by_id: dict[str, DeduplicatedTrace],
) -> str:
    """Keep selected cases with themselves and assign other cases by distance and identity.

    Args:
        candidate: Source case whose workload and coverage need one consistent assignment.
        selected_ids: Nonempty set of selected representative identities.
        by_id: Source candidates keyed by their representative trace identity.

    Returns:
        The selected identity receiving this case's workload and coverage records.
    """
    return min(
        selected_ids,
        key=lambda selected: (
            selected != candidate.representative_trace_id,
            -cosine(candidate.selection_vector, by_id[selected].selection_vector),
            selected,
        ),
    )


def _cluster_summaries(
    clusters: dict[str, int], selected_ids: Sequence[str], geometry: CoreSetGeometry
) -> tuple[ClusterSelection, ...]:
    """Describe full source cluster mass and whether its real medoid survived selection.

    Only clusters that kept a member are reported, so a backend whose group lost every candidate
    never publishes an empty cluster into coverage evidence.
    """
    summaries: list[ClusterSelection] = []
    selected = set(selected_ids)
    for cluster_id, member_ids in sorted(members_by_group(clusters).items()):
        medoid_id = geometry.medoid(member_ids)
        summaries.append(
            ClusterSelection(
                cluster_id=cluster_id,
                candidate_count=len(member_ids),
                workload_mass=sum(geometry.masses[member_id] for member_id in member_ids),
                medoid_trace_id=medoid_id,
                selected_medoid=medoid_id in selected,
            )
        )
    return tuple(summaries)
