"""Direct local composition from normalized trace evidence to immutable representative tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import ArtifactInput, stable_id
from wmo.common.project import ArtifactStore, artifact_input
from wmo.common.tasks import TaskSet
from wmo.simulation.ingest.dataset import PersistedTraceDataset, persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.descriptors import DescriptorEmbedder, HashingDescriptorEmbedder
from wmo.simulation.mining.service import MiningSpec, TaskMiningResult, mine_tasks, persist_task_set


@dataclass(frozen=True)
class TaskSetBuild:
    """Completed canonical trace-dataset and task-set artifacts from one normalized input.

    Args:
        trace_dataset: Immutable raw-source-normalized evidence and manifest.
        mining: Deterministic representative-task selection evidence.
        task_set: Immutable task-set artifact derived only from the trace-dataset manifest.
    """

    trace_dataset: PersistedTraceDataset
    mining: TaskMiningResult
    task_set: TaskSet


def build_task_set(
    normalized: TraceNormalizationResult,
    store: ArtifactStore,
    *,
    created_at: datetime,
    code_revision: str,
    mining_spec: MiningSpec | None = None,
    embedder: DescriptorEmbedder | None = None,
) -> TaskSetBuild:
    """Persist normalized evidence, mine representatives, and persist a dependent task set.

    This is the supported no-network Python composition path. It intentionally accepts a
    pre-normalized result rather than a file path, so the only raw-source read remains in one
    selected canonical OTLP or PostHog loader.

    Args:
        normalized: Canonical traces and explicit validation exclusions from one source loader.
        store: Project-local immutable artifact store for both completed artifacts.
        created_at: Shared completion timestamp for this local build.
        code_revision: Exact WMO revision producing the artifacts.
        mining_spec: Optional representative selection controls, defaulting to 50 fit and 20 held
            out tasks.
        embedder: Explicit descriptor embedder. The deterministic local hashing embedder is used
            when omitted.

    Returns:
        Immutable trace-dataset and task-set artifacts plus the mining evidence between them.
    """
    trace_dataset = persist_trace_dataset(
        normalized,
        store,
        created_at=created_at,
        code_revision=code_revision,
    )
    mining = mine_tasks(
        trace_dataset.traces,
        mining_spec,
        embedder=embedder or HashingDescriptorEmbedder(),
        input_trace_count=len(trace_dataset.traces) + trace_dataset.dataset.invalid_trace_count,
        invalid_trace_count=trace_dataset.dataset.invalid_trace_count,
    )
    dataset_input = artifact_input(trace_dataset.manifest)
    task_set = persist_task_set(
        mining,
        store,
        task_set_id=_task_set_id(dataset_input, mining),
        created_at=created_at,
        code_revision=code_revision,
        inputs=(dataset_input,),
    )
    return TaskSetBuild(trace_dataset=trace_dataset, mining=mining, task_set=task_set)


def _task_set_id(dataset_input: ArtifactInput, mining: TaskMiningResult) -> str:
    """Return the content-addressed task-set ID for one immutable dataset and selection result."""
    return stable_id(
        "task-set",
        {
            "trace_dataset": dataset_input.model_dump(mode="json"),
            "tasks": [task.model_dump(mode="json") for task in mining.tasks],
            "coverage": mining.coverage.model_dump(mode="json"),
        },
    )
