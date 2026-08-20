"""Canonical trace-to-representative-task mining service and immutable task-set materialization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    SourceIdentity,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import TaskCase, TaskSet
from exp.common.traces import Trace, load_trace_dataset
from exp.simulation.mining.bindings import (
    TaskSetLineageBindings,
    build_task_set_lineage_bindings,
)
from exp.simulation.mining.cleanup import (
    InstructionCleanupModel,
    InstructionCleanupResult,
    clean_instruction,
)
from exp.simulation.mining.coverage import (
    DEFAULT_COVERAGE_SIMILARITY_THRESHOLD,
    CoverageReport,
    build_coverage_report,
)
from exp.simulation.mining.deduplicate import (
    DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
    DeduplicatedTrace,
    DuplicateAnalysis,
    FrozenTaskPartition,
    analyze_duplicates,
    freeze_task_partition,
)
from exp.simulation.mining.descriptors import (
    DescriptorEmbedder,
    coverage_descriptor,
    normalized_vectors,
    routing_descriptor,
)
from exp.simulation.mining.lineage import (
    DEFAULT_LINEAGE_TIME_BUCKET_SECONDS,
    LineageAssignment,
    assign_source_lineages,
)
from exp.simulation.mining.select import PartitionSelection, select_partition_representatives


class MiningSpec(ContractModel):
    """Versioned representative-task mining controls with fixed 50-fit and 20-held-out defaults."""

    fit_task_budget: int = Field(default=50, ge=0)
    held_out_task_budget: int = Field(default=20, ge=0)
    semantic_duplicate_threshold: float = Field(
        default=DEFAULT_SEMANTIC_DUPLICATE_THRESHOLD,
        ge=0.0,
        le=1.0,
    )
    coverage_similarity_threshold: float = Field(
        default=DEFAULT_COVERAGE_SIMILARITY_THRESHOLD,
        ge=-1.0,
        le=1.0,
    )
    lineage_time_bucket_seconds: int = Field(default=DEFAULT_LINEAGE_TIME_BUCKET_SECONDS, gt=0)


@dataclass(frozen=True)
class TaskMiningResult:
    """One complete no-network mining result ready for immutable task-set persistence.

    Args:
        tasks: Canonical real-source representative tasks with globally normalized workload weights.
        coverage: Complete coverage and split report stored beside the task set.
        analysis: Duplicate edges, connected leakage groups, and deduplicated candidates.
        lineage_assignments: Initial source lineage evidence before duplicate unions.
        partition: Frozen connected-lineage partition used by all downstream work.
        cleanup_results: Source back-check evidence keyed by selected representative trace ID.
    """

    tasks: tuple[TaskCase, ...]
    coverage: CoverageReport
    analysis: DuplicateAnalysis
    lineage_assignments: tuple[LineageAssignment, ...]
    partition: FrozenTaskPartition
    cleanup_results: tuple[tuple[str, InstructionCleanupResult], ...]


def mine_tasks(
    traces: Sequence[Trace],
    mining_spec: MiningSpec | None = None,
    *,
    embedder: DescriptorEmbedder | None = None,
    instruction_cleanup_model: InstructionCleanupModel | None = None,
    input_trace_count: int | None = None,
    invalid_trace_count: int = 0,
) -> TaskMiningResult:
    """Mine a small weighted, leakage-safe task set from canonical production traces.

    Production composition supplies the configured W3 ``EmbeddingClient`` through the narrow
    ``DescriptorEmbedder`` protocol. Tests and explicit offline callers can pass
    ``HashingDescriptorEmbedder``. Optional instruction cleanup is likewise an injected model
    interface and is source-back-checked before any task instruction changes.

    Args:
        traces: Valid canonical traces from OTLP or PostHog conversion.
        mining_spec: Task budgets and leakage-safe mining controls. Defaults to 50 fit and 20
            held out.
        embedder: Explicit request-descriptor embedder. ``None`` is rejected to avoid a hidden
            model or low-fidelity fallback.
        instruction_cleanup_model: Optional injected proposal model. ``None`` makes no model call.
        input_trace_count: Source record count before invalid exclusions. Defaults to valid count.
        invalid_trace_count: Source record count excluded by canonical normalization.

    Returns:
        Canonical tasks, coverage evidence, duplicate lineage evidence, and frozen split membership.

    Raises:
        ValueError: Inputs are empty, invalid counts are inconsistent, or selection cannot be
            audited.
    """
    if not traces:
        raise ValueError("task mining needs at least one eligible canonical trace")
    if embedder is None:
        raise ValueError(
            "task mining needs an explicit DescriptorEmbedder; pass a configured embedding client "
            "or an explicit deterministic offline fake"
        )
    if invalid_trace_count < 0:
        raise ValueError("invalid trace count cannot be negative")
    if input_trace_count is None:
        input_trace_count = len(traces) + invalid_trace_count
    if input_trace_count < len(traces) + invalid_trace_count:
        raise ValueError("input trace count cannot be smaller than valid plus invalid trace counts")
    spec = mining_spec or MiningSpec()
    assignments = assign_source_lineages(
        traces,
        time_bucket_seconds=spec.lineage_time_bucket_seconds,
    )
    routing_descriptors = tuple(routing_descriptor(trace) for trace in traces)
    coverage_descriptors = tuple(coverage_descriptor(trace) for trace in traces)
    vectors = normalized_vectors(embedder, routing_descriptors)
    analysis = analyze_duplicates(
        traces,
        assignments,
        routing_descriptors,
        coverage_descriptors,
        vectors,
        semantic_duplicate_threshold=spec.semantic_duplicate_threshold,
    )
    partition = freeze_task_partition(
        analysis.leakage_groups,
        fit_lineage_target=spec.fit_task_budget,
        held_out_lineage_target=spec.held_out_task_budget,
    )
    fit_candidates = tuple(
        candidate
        for candidate in analysis.candidates
        if partition.partition_for(candidate.lineage_group_id) == "fit"
    )
    held_out_candidates = tuple(
        candidate
        for candidate in analysis.candidates
        if partition.partition_for(candidate.lineage_group_id) == "held_out"
    )
    fit_budget, held_out_budget = _selection_budgets(spec, fit_candidates, held_out_candidates)
    fit_selection = select_partition_representatives(
        fit_candidates,
        partition="fit",
        budget=fit_budget,
    )
    held_out_selection = select_partition_representatives(
        held_out_candidates,
        partition="held_out",
        budget=held_out_budget,
    )
    tasks, task_ids, weights, cleanup_results = _materialize_tasks(
        traces,
        fit_selection,
        held_out_selection,
        instruction_cleanup_model,
    )
    coverage = build_coverage_report(
        input_trace_count=input_trace_count,
        invalid_trace_count=invalid_trace_count,
        analysis=analysis,
        partition=partition,
        fit_selection=fit_selection,
        held_out_selection=held_out_selection,
        task_ids_by_representative=task_ids,
        weights_by_representative=weights,
        similarity_threshold=spec.coverage_similarity_threshold,
    )
    if not coverage.split_separation_verified:
        raise ValueError("task mining produced overlapping fit and held-out lineage groups")
    return TaskMiningResult(
        tasks=tasks,
        coverage=coverage,
        analysis=analysis,
        lineage_assignments=assignments,
        partition=partition,
        cleanup_results=cleanup_results,
    )


def persist_task_set(
    result: TaskMiningResult,
    store: ArtifactStore,
    *,
    task_set_id: str,
    created_at: datetime,
    code_revision: str,
    inputs: tuple[ArtifactInput, ...] = (),
    source: SourceIdentity | None = None,
) -> TaskSet:
    """Write canonical tasks and coverage as one immutable W2 task-set artifact.

    Args:
        result: Completed trace-mining result to materialize without reinterpretation.
        store: Project-local immutable artifact store.
        task_set_id: New stable artifact identity.
        created_at: Time the task-set artifact was completed.
        code_revision: Exact EXP code revision that produced the artifact.
        inputs: Sorted immutable input artifact identities and digests.
        source: Optional direct normalized trace source identity.

    Returns:
        The canonical ``TaskSet`` manifest stored with ``tasks.jsonl`` and ``coverage.json``.

    Raises:
        ValueError: An existing artifact differs from the deterministic replay.
    """
    task_payload = canonical_jsonl_bytes(result.tasks)
    coverage_payload = canonical_json_bytes(result.coverage)
    lineage_payload = None
    if inputs:
        if len(inputs) != 1:
            raise ValueError("built task sets require exactly one source trace-dataset input")
        lineage_bindings = build_task_set_lineage_bindings(task_set_id, inputs[0], result)
        _require_complete_source_bindings(store, inputs[0], lineage_bindings)
        lineage_payload = canonical_json_bytes(lineage_bindings)
    task_set = TaskSet(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        source=source,
        task_set_id=task_set_id,
        task_ids=tuple(task.task_id for task in result.tasks),
        tasks_path="tasks.jsonl",
        tasks_sha256=sha256_bytes(task_payload),
        coverage_path="coverage.json",
        coverage_sha256=sha256_bytes(coverage_payload),
    )
    files = {
        "tasks.jsonl": task_payload,
        "coverage.json": coverage_payload,
        "task-set.json": canonical_json_bytes(task_set),
    }
    if lineage_payload is not None:
        files["lineage-bindings.json"] = lineage_payload
    try:
        stored, _ = store.write_or_replay(
            artifact_id=task_set_id,
            artifact_type="task-set",
            envelope=task_set,
            envelope_path="task-set.json",
            envelope_type=TaskSet,
            files=files,
        )
    except ValueError as exc:
        raise ValueError("existing task set differs from replayed mining evidence") from exc
    return stored


def _require_complete_source_bindings(
    store: ArtifactStore,
    trace_dataset_input: ArtifactInput,
    payload: TaskSetLineageBindings,
) -> None:
    """Reject incomplete mining lineage evidence before task-set publication.

    Args:
        store: Project artifact store owning the exact source trace dataset.
        trace_dataset_input: Claimed immutable source dataset manifest input.
        payload: Typed complete lineage payload produced from mining.

    Raises:
        ValueError: Bindings omit or add traces relative to the exact source dataset.
        ArtifactCorruptionError: The source artifact or manifest input cannot be verified.
    """
    stored = store.read(trace_dataset_input.artifact_id)
    if artifact_input(stored.manifest) != trace_dataset_input:
        raise ValueError("task-set source trace-dataset manifest digest differs")
    source = load_trace_dataset(store, trace_dataset_input.artifact_id)
    source_trace_ids = tuple(sorted(trace.trace_id for trace in source.traces))
    binding_trace_ids = tuple(binding.trace_id for binding in payload.bindings)
    if binding_trace_ids != source_trace_ids:
        raise ValueError(
            "task-set lineage bindings must cover every exact source trace once and only once"
        )


def _materialize_tasks(
    traces: Sequence[Trace],
    fit_selection: PartitionSelection,
    held_out_selection: PartitionSelection,
    cleanup_model: InstructionCleanupModel | None,
) -> tuple[
    tuple[TaskCase, ...],
    dict[str, str],
    dict[str, float],
    tuple[tuple[str, InstructionCleanupResult], ...],
]:
    """Convert selected real traces to the one canonical task contract and global weights."""
    by_trace_id = {trace.trace_id: trace for trace in traces}
    selections = (*fit_selection.selected, *held_out_selection.selected)
    if not selections:
        raise ValueError("task mining selected no representative source trace")
    total_mass = sum(selection.workload_mass for selection in selections)
    if total_mass <= 0:
        raise ValueError("representative task workload mass must be positive")
    tasks: list[TaskCase] = []
    task_ids: dict[str, str] = {}
    weights: dict[str, float] = {}
    cleanup_results: list[tuple[str, InstructionCleanupResult]] = []
    for selection in selections:
        trace = by_trace_id[selection.representative_trace_id]
        cleanup = clean_instruction(trace, cleanup_model)
        task_id = stable_id(
            "task",
            {
                "version": "task-case-v1",
                "lineage_group_id": selection.lineage_group_id,
                "partition": selection.partition,
                "representative_trace_id": selection.representative_trace_id,
                "source_trace_ids": list(selection.source_trace_ids),
            },
        )
        weight = selection.workload_mass / total_mass
        tasks.append(
            TaskCase(
                task_id=task_id,
                lineage_group_id=selection.lineage_group_id,
                partition=selection.partition,
                instruction=cleanup.instruction,
                initial_context=trace.initial_context,
                tools=trace.tools,
                workload_weight=weight,
                source_trace_ids=selection.source_trace_ids,
            )
        )
        task_ids[selection.representative_trace_id] = task_id
        weights[selection.representative_trace_id] = weight
        cleanup_results.append((selection.representative_trace_id, cleanup))
    return tuple(tasks), task_ids, weights, tuple(cleanup_results)


def _selection_budgets(
    spec: MiningSpec,
    fit_candidates: Sequence[DeduplicatedTrace],
    held_out_candidates: Sequence[DeduplicatedTrace],
) -> tuple[int, int]:
    """Reallocate the fixed total budget only when needed to retain every small-set lineage."""
    fit_lineages = {candidate.lineage_group_id for candidate in fit_candidates}
    held_out_lineages = {candidate.lineage_group_id for candidate in held_out_candidates}
    total_budget = spec.fit_task_budget + spec.held_out_task_budget
    if len(fit_lineages) + len(held_out_lineages) > total_budget:
        return spec.fit_task_budget, spec.held_out_task_budget
    fit_budget = max(spec.fit_task_budget, len(fit_lineages))
    held_out_budget = max(spec.held_out_task_budget, len(held_out_lineages))
    overflow = fit_budget + held_out_budget - total_budget
    fit_budget -= min(overflow, fit_budget - len(fit_lineages))
    overflow = fit_budget + held_out_budget - total_budget
    held_out_budget -= min(overflow, held_out_budget - len(held_out_lineages))
    return fit_budget, held_out_budget
