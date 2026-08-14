"""Selection and loading of real trace evidence for manual judge review."""

from __future__ import annotations

from collections.abc import Sequence

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.project import ProjectStore
from wmo.common.rollouts import RolloutArtifact
from wmo.common.tasks import TaskCase
from wmo.common.traces import Trace
from wmo.optimize.router.judging.artifacts import rollout_id
from wmo.optimize.router.judging.contracts import (
    JudgeTracePreview,
    ManualJudgeError,
)


def representative_pairs(
    tasks: Sequence[TaskCase], traces: Sequence[Trace], limit: int
) -> tuple[tuple[TaskCase, Trace], ...]:
    """Match one real trace per distinct fit lineage in deterministic task order.

    Args:
        tasks: Verified representative tasks from the completed build.
        traces: Verified normalized real traces from the completed build.
        limit: Maximum number of fit lineages to retain.

    Returns:
        Ordered task and real-trace pairs.

    Raises:
        ManualJudgeError: No fit task can be matched to real trace evidence.
    """
    by_id = {trace.trace_id: trace for trace in traces}
    selected: list[tuple[TaskCase, Trace]] = []
    seen: set[str] = set()
    for task in tasks:
        if task.partition != "fit" or task.lineage_group_id in seen:
            continue
        trace = next((by_id[item] for item in task.source_trace_ids if item in by_id), None)
        if trace is None:
            continue
        selected.append((task, trace))
        seen.add(task.lineage_group_id)
        if len(selected) == limit:
            break
    if not selected:
        raise ManualJudgeError("completed build has no real fit-lineage trace available for review")
    return tuple(selected)


def pairwise_references(
    selected: Sequence[tuple[TaskCase, Trace]],
    traces: Sequence[Trace],
    response_shape: str,
) -> tuple[Trace | None, ...]:
    """Resolve a distinct real output for each selected canonical pairwise task.

    Args:
        selected: Target task and trace pairs in calibration order.
        traces: Complete verified build trace dataset.
        response_shape: Finalized structured-feedback shape.

    Returns:
        Same-task comparison traces, or aligned ``None`` values for non-pairwise feedback.

    Raises:
        ManualJudgeError: A pairwise task lacks a second real output for the same task.
    """
    if response_shape != "pairwise":
        return tuple(None for _item in selected)
    by_id = {trace.trace_id: trace for trace in traces}
    references: list[Trace] = []
    for task, target in selected:
        reference = next(
            (
                by_id[trace_id]
                for trace_id in task.source_trace_ids
                if trace_id != target.trace_id and trace_id in by_id
            ),
            None,
        )
        if reference is None:
            raise ManualJudgeError(
                "pairwise judge calibration needs two real outputs for each canonical fit task; "
                "collect another trace for the same request or choose a non-pairwise schema"
            )
        references.append(reference)
    return tuple(references)


def representative_pairwise_pairs(
    tasks: Sequence[TaskCase], traces: Sequence[Trace], limit: int
) -> tuple[tuple[TaskCase, Trace], ...]:
    """Select fit tasks backed by at least two real same-task outputs.

    Args:
        tasks: Verified representative tasks from the completed build.
        traces: Verified normalized real traces from the completed build.
        limit: Maximum number of pairable fit lineages to retain.

    Returns:
        Target task and first real output pairs in deterministic task order.

    Raises:
        ManualJudgeError: No fit task has two real outputs for comparison.
    """
    by_id = {trace.trace_id: trace for trace in traces}
    selected = tuple(
        (task, by_id[available[0]])
        for task in tasks
        if task.partition == "fit"
        for available in (
            tuple(trace_id for trace_id in task.source_trace_ids if trace_id in by_id),
        )
        if len(available) >= 2
    )[:limit]
    if not selected:
        raise ManualJudgeError(
            "pairwise judge calibration needs two real outputs for the same canonical fit task; "
            "collect another trace for the same request or choose a non-pairwise schema"
        )
    return selected


def trace_preview(
    task: TaskCase, trace: Trace, reference: Trace | None = None
) -> JudgeTracePreview:
    """Render one concise local trace preview without persisting a rollout.

    Args:
        task: Representative fit task linked to the trace.
        trace: Normalized real trace selected for preview.
        reference: Optional same-task real comparison trace.

    Returns:
        Stable human-readable preview metadata.
    """
    outcome = "unknown" if trace.outcome is None else trace.outcome.status
    return JudgeTracePreview(
        trace_id=trace.trace_id,
        rollout_id=rollout_id(task, trace),
        task_id=task.task_id,
        lineage_id=task.lineage_group_id,
        task=trace.task,
        outcome=outcome,
        span_names=tuple(span.name for span in trace.spans),
        reference_trace_id=reference.trace_id if reference is not None else None,
        reference_rollout_id=(rollout_id(task, reference) if reference is not None else None),
    )


def read_rollout(store: ProjectStore, expected: ArtifactInput) -> RolloutArtifact:
    """Read one production rollout through its exact manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        expected: Rollout pointer created or resumed for calibration.

    Returns:
        Verified immutable production rollout.

    Raises:
        ManualJudgeError: Rollout content or manifest identity changed.
    """
    try:
        rollout, rollout_input = read_artifact_json(
            store,
            artifact_id=expected.artifact_id,
            expected_artifact_type="rollout",
            relative_path="rollout.json",
            model_type=RolloutArtifact,
        )
    except JudgingProvenanceError as exc:
        raise ManualJudgeError("manual judge production rollout is unavailable") from exc
    if rollout_input != expected:
        raise ManualJudgeError("manual judge production rollout manifest changed")
    return rollout
