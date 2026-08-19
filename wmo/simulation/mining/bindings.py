"""Complete immutable trace lineage assignments owned by a built task set."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    ContractModel,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    envelope_matches_manifest,
    stable_id,
)
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.common.tasks import TaskCase, TaskSet, load_task_set
from wmo.common.traces import load_trace_dataset
from wmo.simulation.ingest.dataset import verify_current_trace_dataset
from wmo.simulation.mining.coverage import CoverageReport

if TYPE_CHECKING:
    from wmo.simulation.mining.service import TaskMiningResult

LINEAGE_BINDINGS_PATH = "lineage-bindings.json"
_TASK_SET_FILES = frozenset(
    {"coverage.json", LINEAGE_BINDINGS_PATH, "task-set.json", "tasks.jsonl"}
)


class TaskTraceLineageBinding(ContractModel):
    """Frozen leakage lineage and partition for one imported trace."""

    trace_id: str = Field(min_length=1, max_length=512)
    lineage_id: ArtifactId
    partition: Literal["fit", "held_out"]


class TaskSetLineageBindings(ContractModel):
    """Complete build-owned lineage assignments for a task set's source dataset."""

    schema_version: Literal[1] = 1
    binding_set_id: ArtifactId
    task_set_id: ArtifactId
    trace_dataset: ArtifactInput
    bindings: tuple[TaskTraceLineageBinding, ...] = Field(min_length=1)

    @field_validator("bindings")
    @classmethod
    def _require_sorted_unique_bindings(
        cls, value: tuple[TaskTraceLineageBinding, ...]
    ) -> tuple[TaskTraceLineageBinding, ...]:
        """Require exactly one deterministically ordered assignment per trace.

        Args:
            value: Complete trace lineage assignments.

        Returns:
            The validated assignments without modification.

        Raises:
            ValueError: Trace IDs repeat or are not sorted.
        """
        trace_ids = tuple(binding.trace_id for binding in value)
        if len(set(trace_ids)) != len(trace_ids):
            raise ValueError("task-set lineage bindings must not repeat trace IDs")
        if trace_ids != tuple(sorted(trace_ids)):
            raise ValueError("task-set lineage bindings must be sorted by trace ID")
        return value

    @model_validator(mode="after")
    def _require_content_identity(self) -> TaskSetLineageBindings:
        """Verify that the binding-set identity covers every semantic field.

        Returns:
            The validated content-addressed payload.

        Raises:
            ValueError: The stored identity differs from the canonical binding content.
        """
        expected = task_set_lineage_binding_id(
            self.task_set_id,
            self.trace_dataset,
            self.bindings,
        )
        if self.binding_set_id != expected:
            raise ValueError("task-set lineage binding ID differs from its complete content")
        return self


def bindings_for_mining(result: TaskMiningResult) -> tuple[TaskTraceLineageBinding, ...]:
    """Derive complete assignments for every trace in frozen leakage groups.

    Args:
        result: Completed mining evidence with leakage groups and a frozen partition.

    Returns:
        One sorted binding for every imported trace.

    Raises:
        ValueError: Mining evidence repeats a trace or omits a partitioned lineage.
    """
    bindings: list[TaskTraceLineageBinding] = []
    seen: set[str] = set()
    for group in result.analysis.leakage_groups:
        partition = result.partition.partition_for(group.lineage_group_id)
        for trace_id in group.source_trace_ids:
            if trace_id in seen:
                raise ValueError(
                    f"mining evidence repeats trace {trace_id!r} across leakage groups"
                )
            seen.add(trace_id)
            bindings.append(
                TaskTraceLineageBinding(
                    trace_id=trace_id,
                    lineage_id=group.lineage_group_id,
                    partition=partition,
                )
            )
    return tuple(sorted(bindings, key=lambda item: item.trace_id))


def task_set_lineage_binding_material(
    bindings: tuple[TaskTraceLineageBinding, ...],
) -> dict[str, object]:
    """Return canonical versioned material used by the parent task-set identity.

    Args:
        bindings: Complete sorted assignments derived from mining.

    Returns:
        JSON-compatible versioned identity material.
    """
    return {
        "version": "task-set-lineage-bindings-v1",
        "bindings": [binding.model_dump(mode="json") for binding in bindings],
    }


def task_set_content_id(
    trace_dataset: ArtifactInput,
    tasks: tuple[TaskCase, ...],
    coverage: CoverageReport,
    bindings: tuple[TaskTraceLineageBinding, ...],
) -> str:
    """Return the content identity for a build-owned task set and complete bindings.

    Args:
        trace_dataset: Exact source trace-dataset manifest input.
        tasks: Ordered representative tasks.
        coverage: Complete mining coverage report.
        bindings: Complete sorted imported-trace assignments.

    Returns:
        Stable task-set artifact identity.
    """
    return stable_id(
        "task-set",
        {
            "trace_dataset": trace_dataset.model_dump(mode="json"),
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "coverage": coverage.model_dump(mode="json"),
            "lineage_bindings": task_set_lineage_binding_material(bindings),
        },
    )


def task_set_lineage_binding_id(
    task_set_id: str,
    trace_dataset: ArtifactInput,
    bindings: tuple[TaskTraceLineageBinding, ...],
) -> str:
    """Return the content identity for one complete lineage-binding payload.

    Args:
        task_set_id: Parent task-set artifact identity.
        trace_dataset: Exact source trace-dataset manifest input.
        bindings: Complete sorted trace assignments.

    Returns:
        Stable lineage-binding payload identity.
    """
    return stable_id(
        "task-set-lineage-bindings",
        {
            "task_set_id": task_set_id,
            "trace_dataset": trace_dataset.model_dump(mode="json"),
            **task_set_lineage_binding_material(bindings),
        },
    )


def build_task_set_lineage_bindings(
    task_set_id: str,
    trace_dataset: ArtifactInput,
    result: TaskMiningResult,
) -> TaskSetLineageBindings:
    """Materialize a complete typed binding payload from mining evidence.

    Args:
        task_set_id: Parent task-set artifact identity.
        trace_dataset: Exact source trace-dataset manifest input.
        result: Completed mining evidence.

    Returns:
        Content-addressed lineage-binding payload.
    """
    bindings = bindings_for_mining(result)
    return TaskSetLineageBindings(
        binding_set_id=task_set_lineage_binding_id(task_set_id, trace_dataset, bindings),
        task_set_id=task_set_id,
        trace_dataset=trace_dataset,
        bindings=bindings,
    )


def load_task_set_lineage_bindings(
    store: ArtifactStore,
    task_set_id: str,
) -> TaskSetLineageBindings:
    """Load and recursively verify complete lineage assignments for a built task set.

    Args:
        store: Project-local immutable artifact store.
        task_set_id: Task-set artifact whose build-owned bindings are required.

    Returns:
        Verified complete lineage assignments.

    Raises:
        ArtifactCorruptionError: The payload is missing, corrupt, incomplete, or disagrees with
            its task set or source trace dataset.
    """
    stored = store.read(task_set_id)
    if stored.manifest.artifact_type != "task-set":
        raise ArtifactCorruptionError(f"artifact {task_set_id} is not a task-set")
    if stored.manifest.schema_version != 1:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} uses unsupported schema version "
            f"{stored.manifest.schema_version}"
        )
    paths = {entry.path for entry in stored.manifest.files}
    if LINEAGE_BINDINGS_PATH not in paths:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} has no complete lineage bindings; rebuild the project"
        )
    loaded_tasks = load_task_set(store, task_set_id)
    _require_task_set_manifest_matches_envelope(stored.manifest, loaded_tasks.task_set)
    if loaded_tasks.task_set.tasks_path != "tasks.jsonl":
        raise ArtifactCorruptionError(f"task set {task_set_id} uses a noncanonical task path")
    if loaded_tasks.task_set.coverage_path != "coverage.json":
        raise ArtifactCorruptionError(f"task set {task_set_id} uses a noncanonical coverage path")
    if paths != _TASK_SET_FILES:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} does not have the exact complete-lineage file set"
        )
    if store.read_bytes(task_set_id, "task-set.json") != canonical_json_bytes(
        loaded_tasks.task_set
    ):
        raise ArtifactCorruptionError(
            f"task set {task_set_id} envelope is not canonical current-build JSON"
        )
    task_bytes = store.read_bytes(task_set_id, "tasks.jsonl")
    if task_bytes != canonical_jsonl_bytes(loaded_tasks.tasks):
        raise ArtifactCorruptionError(
            f"task set {task_set_id} records are not canonical current-build JSONL"
        )
    lineage_bytes = store.read_bytes(task_set_id, LINEAGE_BINDINGS_PATH)
    try:
        payload = TaskSetLineageBindings.model_validate_json(lineage_bytes)
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} has invalid complete lineage bindings"
        ) from exc
    if lineage_bytes != canonical_json_bytes(payload):
        raise ArtifactCorruptionError(
            f"task set {task_set_id} lineage bindings are not canonical current-build JSON"
        )
    if payload.task_set_id != task_set_id:
        raise ArtifactCorruptionError(
            f"task-set lineage bindings name {payload.task_set_id}, expected {task_set_id}"
        )
    if len(stored.manifest.inputs) != 1:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} must have exactly one source trace-dataset input"
        )
    trace_input = stored.manifest.inputs[0]
    if payload.trace_dataset != trace_input:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} lineage bindings differ from its source dataset input"
        )
    trace_stored = store.read(trace_input.artifact_id)
    if trace_stored.manifest.schema_version not in {1, 2}:
        raise ArtifactCorruptionError(
            f"trace dataset {trace_input.artifact_id} uses unsupported schema version "
            f"{trace_stored.manifest.schema_version}"
        )
    source = load_trace_dataset(store, trace_input.artifact_id)
    verify_current_trace_dataset(store, source)
    if artifact_input(trace_stored.manifest) != trace_input:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} source trace-dataset manifest digest differs"
        )
    if loaded_tasks.task_set.code_revision != source.dataset.code_revision:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} producer revision differs from its source trace dataset"
        )
    if loaded_tasks.task_set.created_at != source.dataset.created_at:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} build timestamp differs from its source trace dataset"
        )
    if loaded_tasks.task_set.source is not None:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} must derive source provenance from its trace-dataset input"
        )
    expected_trace_ids = tuple(sorted(trace.trace_id for trace in source.traces))
    actual_trace_ids = tuple(binding.trace_id for binding in payload.bindings)
    if actual_trace_ids != expected_trace_ids:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} lineage bindings do not cover its exact source traces"
        )
    if loaded_tasks.task_set.coverage_path is None:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} has no coverage evidence for lineage verification"
        )
    coverage_bytes = store.read_bytes(task_set_id, loaded_tasks.task_set.coverage_path)
    if hashlib.sha256(coverage_bytes).hexdigest() != loaded_tasks.task_set.coverage_sha256:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} coverage digest does not match its envelope"
        )
    try:
        coverage = CoverageReport.model_validate_json(coverage_bytes)
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} has invalid coverage evidence"
        ) from exc
    if coverage_bytes != canonical_json_bytes(coverage):
        raise ArtifactCorruptionError(
            f"task set {task_set_id} coverage is not canonical current-build JSON"
        )
    expected_task_set_id = task_set_content_id(
        trace_input,
        loaded_tasks.tasks,
        coverage,
        payload.bindings,
    )
    if task_set_id != expected_task_set_id:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} identity does not bind its complete lineage evidence"
        )
    _require_selected_tasks_match_bindings(loaded_tasks.task_set, loaded_tasks.tasks, payload)
    return payload


def _require_task_set_manifest_matches_envelope(
    manifest: ArtifactManifest,
    task_set: TaskSet,
) -> None:
    """Verify task-set envelope provenance against its artifact manifest.

    Args:
        manifest: Immutable artifact manifest surrounding the task-set envelope.
        task_set: Parsed task-set envelope.

    Raises:
        ArtifactCorruptionError: The envelope inputs differ from the manifest.
    """
    if not envelope_matches_manifest(task_set, manifest):
        raise ArtifactCorruptionError(
            f"task set {task_set.task_set_id} envelope differs from its artifact manifest"
        )


def _require_selected_tasks_match_bindings(
    task_set: TaskSet,
    tasks: tuple[TaskCase, ...],
    payload: TaskSetLineageBindings,
) -> None:
    """Verify representative tasks agree with complete source assignments.

    Args:
        task_set: Verified task-set envelope used for actionable error context.
        tasks: Verified ordered task records.
        payload: Complete source-trace assignments.

    Raises:
        ArtifactCorruptionError: A task source is absent or crosses lineage or partition boundaries.
    """
    by_trace = {binding.trace_id: binding for binding in payload.bindings}
    for task in tasks:
        for trace_id in task.source_trace_ids:
            binding = by_trace.get(trace_id)
            if binding is None:
                raise ArtifactCorruptionError(
                    f"task set {task_set.task_set_id} task source {trace_id!r} has no lineage "
                    "binding"
                )
            if binding.lineage_id != task.lineage_group_id or binding.partition != task.partition:
                raise ArtifactCorruptionError(
                    f"task set {task_set.task_set_id} task source {trace_id!r} disagrees with its "
                    "lineage binding"
                )
