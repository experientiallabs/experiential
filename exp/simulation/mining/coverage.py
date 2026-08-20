"""Inspectable coverage reporting for canonical representative task selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.deduplicate import (
    DeduplicatedTrace,
    DuplicateAnalysis,
    FrozenTaskPartition,
)
from exp.simulation.mining.descriptors import CoverageDescriptor
from exp.simulation.mining.select import (
    ClusterSelection,
    PartitionSelection,
    SelectedRepresentative,
)

DEFAULT_COVERAGE_SIMILARITY_THRESHOLD = 0.70
PartitionName = Literal["fit", "held_out"]
FacetDimension = Literal["tool", "domain", "outcome", "complexity"]
_PARTITIONS: tuple[PartitionName, PartitionName] = ("fit", "held_out")


class CoverageFacet(ContractModel):
    """Workload coverage for one tool, domain, outcome, or complexity stratum."""

    dimension: FacetDimension
    value: str = Field(min_length=1)
    input_workload_mass: int = Field(ge=0)
    directly_selected_workload_mass: int = Field(ge=0)


class TraceCoverageDistance(ContractModel):
    """Distance from one input trace to the real representative task covering it."""

    trace_id: str = Field(min_length=1)
    partition: PartitionName
    nearest_task_id: str = Field(min_length=1)
    similarity: float = Field(ge=-1.0, le=1.0)
    low_similarity: bool


class SelectedTaskCoverage(ContractModel):
    """Selection provenance, assigned workload, and normalized task weight."""

    task_id: str = Field(min_length=1)
    representative_trace_id: str = Field(min_length=1)
    partition: PartitionName
    lineage_group_id: str = Field(min_length=1)
    cluster_id: int = Field(ge=0)
    selection_reasons: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    workload_mass: int = Field(gt=0)
    workload_weight: float = Field(gt=0.0, le=1.0)


class ClusterCoverage(ContractModel):
    """Cluster size, workload mass, and real-medoid retention state."""

    partition: PartitionName
    cluster_id: int = Field(ge=0)
    candidate_count: int = Field(gt=0)
    workload_mass: int = Field(gt=0)
    medoid_trace_id: str = Field(min_length=1)
    selected_medoid: bool


class PartitionCoverage(ContractModel):
    """Coverage summary for one frozen fit or held-out task partition."""

    partition: PartitionName
    requested_task_budget: int = Field(ge=0)
    eligible_trace_count: int = Field(ge=0)
    deduplicated_candidate_count: int = Field(ge=0)
    selected_task_count: int = Field(ge=0)
    selected_workload_mass: int = Field(ge=0)
    low_similarity_workload_mass: int = Field(ge=0)
    underfilled: bool
    missing_reserved_slots: tuple[str, ...]
    clusters: tuple[ClusterCoverage, ...]


class CoverageReport(ContractModel):
    """Versioned task-mining coverage evidence stored beside a canonical task set."""

    schema_version: int = Field(default=1, ge=1)
    input_trace_count: int = Field(ge=0)
    invalid_trace_count: int = Field(ge=0)
    eligible_trace_count: int = Field(ge=0)
    duplicate_trace_count: int = Field(ge=0)
    selected_task_count: int = Field(ge=0)
    operating_range: Literal["below", "within", "above"]
    split_separation_verified: bool
    fit_held_out_lineage_overlap: tuple[str, ...]
    partition_underfilled_reason: str | None = None
    fit: PartitionCoverage
    held_out: PartitionCoverage
    selections: tuple[SelectedTaskCoverage, ...]
    facets: tuple[CoverageFacet, ...]
    distances: tuple[TraceCoverageDistance, ...]


def build_coverage_report(
    *,
    input_trace_count: int,
    invalid_trace_count: int,
    analysis: DuplicateAnalysis,
    partition: FrozenTaskPartition,
    fit_selection: PartitionSelection,
    held_out_selection: PartitionSelection,
    task_ids_by_representative: dict[str, str],
    weights_by_representative: dict[str, float],
    similarity_threshold: float = DEFAULT_COVERAGE_SIMILARITY_THRESHOLD,
) -> CoverageReport:
    """Build the complete coverage report consumed by task review and router reporting.

    Args:
        input_trace_count: All source records before invalid evidence exclusion.
        invalid_trace_count: Source records excluded by canonical normalization.
        analysis: Duplicate and leakage-group analysis over eligible traces.
        partition: Frozen connected-lineage split.
        fit_selection: Real representative selection for fit lineages.
        held_out_selection: Real representative selection for held-out lineages.
        task_ids_by_representative: Canonical task IDs keyed by selected source trace IDs.
        weights_by_representative: Normalized workload weights keyed by selected source trace IDs.
        similarity_threshold: Maximum nearest distance threshold used to flag weak coverage.

    Returns:
        A deterministic report that keeps every invalid, duplicate, and low-coverage fact visible.

    Raises:
        ValueError: Task mappings or similarity threshold are inconsistent with selected evidence.
    """
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("coverage similarity threshold must be between -1 and 1")
    selections = (*fit_selection.selected, *held_out_selection.selected)
    selected_ids = {selection.representative_trace_id for selection in selections}
    if set(task_ids_by_representative) != selected_ids:
        raise ValueError("coverage task IDs must name exactly the selected source traces")
    if set(weights_by_representative) != selected_ids:
        raise ValueError("coverage weights must name exactly the selected source traces")

    candidates_by_partition: dict[PartitionName, tuple[DeduplicatedTrace, ...]] = {
        "fit": tuple(
            candidate
            for candidate in analysis.candidates
            if partition.partition_for(candidate.lineage_group_id) == "fit"
        ),
        "held_out": tuple(
            candidate
            for candidate in analysis.candidates
            if partition.partition_for(candidate.lineage_group_id) == "held_out"
        ),
    }
    selection_by_partition: dict[PartitionName, PartitionSelection] = {
        "fit": fit_selection,
        "held_out": held_out_selection,
    }
    partition_reports: dict[PartitionName, PartitionCoverage] = {
        name: _partition_coverage(
            partition=name,
            candidates=candidates_by_partition[name],
            selection=selection_by_partition[name],
            task_ids_by_representative=task_ids_by_representative,
            similarity_threshold=similarity_threshold,
        )
        for name in _PARTITIONS
    }
    distances = _distances(
        candidates_by_partition,
        selection_by_partition,
        task_ids_by_representative,
        similarity_threshold,
    )
    selected_lookup = {selection.representative_trace_id: selection for selection in selections}
    selection_records = tuple(
        SelectedTaskCoverage(
            task_id=task_ids_by_representative[selection.representative_trace_id],
            representative_trace_id=selection.representative_trace_id,
            partition=selection.partition,
            lineage_group_id=selection.lineage_group_id,
            cluster_id=selection.cluster_id,
            selection_reasons=selection.selection_reasons,
            source_trace_ids=selection.source_trace_ids,
            workload_mass=selection.workload_mass,
            workload_weight=weights_by_representative[selection.representative_trace_id],
        )
        for selection in selections
    )
    overlap = tuple(
        sorted(
            set(partition.fit_lineage_group_ids).intersection(partition.held_out_lineage_group_ids)
        )
    )
    eligible_trace_count = sum(candidate.workload_mass for candidate in analysis.candidates)
    return CoverageReport(
        input_trace_count=input_trace_count,
        invalid_trace_count=invalid_trace_count,
        eligible_trace_count=eligible_trace_count,
        duplicate_trace_count=eligible_trace_count - len(analysis.candidates),
        selected_task_count=len(selections),
        operating_range=_operating_range(eligible_trace_count),
        split_separation_verified=not overlap,
        fit_held_out_lineage_overlap=overlap,
        partition_underfilled_reason=partition.underfilled_reason,
        fit=partition_reports["fit"],
        held_out=partition_reports["held_out"],
        selections=selection_records,
        facets=_facet_coverage(analysis.candidates, selected_lookup),
        distances=distances,
    )


def _partition_coverage(
    *,
    partition: PartitionName,
    candidates: Sequence[DeduplicatedTrace],
    selection: PartitionSelection,
    task_ids_by_representative: dict[str, str],
    similarity_threshold: float,
) -> PartitionCoverage:
    """Create one partition report including all nearest-representative coverage distances."""
    low_similarity_mass = 0
    if selection.selected:
        by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
        for candidate in candidates:
            best_similarity = max(
                _cosine(candidate.vector, by_id[selected.representative_trace_id].vector)
                for selected in selection.selected
            )
            if best_similarity < similarity_threshold:
                low_similarity_mass += candidate.workload_mass
    return PartitionCoverage(
        partition=partition,
        requested_task_budget=selection.requested_budget,
        eligible_trace_count=sum(candidate.workload_mass for candidate in candidates),
        deduplicated_candidate_count=len(candidates),
        selected_task_count=len(selection.selected),
        selected_workload_mass=sum(selected.workload_mass for selected in selection.selected),
        low_similarity_workload_mass=low_similarity_mass,
        underfilled=selection.underfilled,
        missing_reserved_slots=selection.missing_reserved_slots,
        clusters=tuple(_cluster_coverage(partition, cluster) for cluster in selection.clusters),
    )


def _cluster_coverage(partition: PartitionName, cluster: ClusterSelection) -> ClusterCoverage:
    """Convert local selection geometry to persisted coverage evidence."""
    return ClusterCoverage(
        partition=partition,
        cluster_id=cluster.cluster_id,
        candidate_count=cluster.candidate_count,
        workload_mass=cluster.workload_mass,
        medoid_trace_id=cluster.medoid_trace_id,
        selected_medoid=cluster.selected_medoid,
    )


def _distances(
    candidates_by_partition: dict[PartitionName, tuple[DeduplicatedTrace, ...]],
    selection_by_partition: dict[PartitionName, PartitionSelection],
    task_ids_by_representative: dict[str, str],
    similarity_threshold: float,
) -> tuple[TraceCoverageDistance, ...]:
    """Expand representative distances back to every original source trace."""
    distances: list[TraceCoverageDistance] = []
    for partition_name in _PARTITIONS:
        candidates = candidates_by_partition[partition_name]
        selected = selection_by_partition[partition_name].selected
        if not selected:
            continue
        by_id = {candidate.representative_trace_id: candidate for candidate in candidates}
        for candidate in candidates:
            nearest_id = min(
                (selection.representative_trace_id for selection in selected),
                key=lambda selected_id: (
                    -_cosine(candidate.vector, by_id[selected_id].vector),
                    selected_id,
                ),
            )
            similarity = _cosine(candidate.vector, by_id[nearest_id].vector)
            for trace_id in candidate.source_trace_ids:
                distances.append(
                    TraceCoverageDistance(
                        trace_id=trace_id,
                        partition=partition_name,
                        nearest_task_id=task_ids_by_representative[nearest_id],
                        similarity=similarity,
                        low_similarity=similarity < similarity_threshold,
                    )
                )
    return tuple(sorted(distances, key=lambda distance: distance.trace_id))


def _facet_coverage(
    candidates: Sequence[DeduplicatedTrace], selected: dict[str, SelectedRepresentative]
) -> tuple[CoverageFacet, ...]:
    """Report source-level input and direct-selection mass for every coverage stratum."""
    input_mass: dict[tuple[FacetDimension, str], int] = defaultdict(int)
    selected_mass: dict[tuple[FacetDimension, str], int] = defaultdict(int)
    for candidate in candidates:
        for descriptor in candidate.source_coverage_descriptors:
            for facet in _source_facets(descriptor):
                input_mass[facet] += 1
                if candidate.representative_trace_id in selected:
                    selected_mass[facet] += 1
    return tuple(
        CoverageFacet(
            dimension=dimension,
            value=value,
            input_workload_mass=mass,
            directly_selected_workload_mass=selected_mass[(dimension, value)],
        )
        for (dimension, value), mass in sorted(input_mass.items())
    )


def _source_facets(
    descriptor: CoverageDescriptor,
) -> tuple[tuple[FacetDimension, str], ...]:
    """Return source-level tool, domain, outcome, and complexity strata without collapse."""
    facets: set[tuple[FacetDimension, str]] = {
        ("tool", tool) for tool in descriptor.tool_transitions
    }
    facets.update(("domain", domain) for domain in descriptor.domains)
    facets.update(("outcome", status) for status in descriptor.outcome_statuses)
    facets.update(
        ("complexity", _complexity_bucket(span_count)) for span_count in descriptor.span_counts
    )
    return tuple(sorted(facets))


def _complexity_bucket(span_count: int) -> str:
    """Bucket episode length for coverage reports without exposing it to routing features."""
    if span_count <= 2:
        return "short"
    if span_count <= 6:
        return "medium"
    return "long"


def _operating_range(trace_count: int) -> Literal["below", "within", "above"]:
    """Classify the fixed 100 to 1,000 trace operating range without rejecting underfill."""
    if trace_count < 100:
        return "below"
    if trace_count > 1_000:
        return "above"
    return "within"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a bounded dot product for pre-normalized vectors."""
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
