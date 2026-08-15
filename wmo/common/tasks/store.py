"""Verified reads of immutable representative task-set artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import ValidationError

from wmo.common.project import ArtifactCorruptionError, ArtifactStore
from wmo.common.tasks.task import TaskCase, TaskSet


@dataclass(frozen=True)
class LoadedTaskSet:
    """One verified task-set envelope and its ordered canonical task cases.

    Args:
        task_set: Immutable task-set envelope that names the selected task IDs and data files.
        tasks: Exact ordered canonical task cases from the digest-verified task JSONL.
    """

    task_set: TaskSet
    tasks: tuple[TaskCase, ...]


def load_task_set(store: ArtifactStore, task_set_id: str) -> LoadedTaskSet:
    """Load one immutable task-set artifact without reopening its source trace export.

    Args:
        store: Project-local store that owns the immutable task-set artifact.
        task_set_id: Stable task-set artifact ID to verify and load.

    Returns:
        The typed task-set envelope and its ordered task cases.

    Raises:
        ArtifactCorruptionError: The artifact is not a valid task set or its task records disagree
            with its envelope.
    """
    stored = store.read(task_set_id)
    if stored.manifest.artifact_type != "task-set":
        raise ArtifactCorruptionError(f"artifact {task_set_id} is not a task-set")
    try:
        task_set = TaskSet.model_validate_json(store.read_bytes(task_set_id, "task-set.json"))
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"task set {task_set_id} has an invalid envelope") from exc
    if task_set.task_set_id != task_set_id:
        raise ArtifactCorruptionError(
            f"task-set envelope ID {task_set.task_set_id} does not match artifact {task_set_id}"
        )
    task_payload = store.read_bytes(task_set_id, task_set.tasks_path)
    if hashlib.sha256(task_payload).hexdigest() != task_set.tasks_sha256:
        raise ArtifactCorruptionError(f"task set {task_set_id} task digest does not match envelope")
    try:
        tasks = tuple(
            TaskCase.model_validate_json(line)
            for line in task_payload.decode("utf-8").splitlines()
            if line
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(f"task set {task_set_id} has invalid task records") from exc
    task_ids = tuple(task.task_id for task in tasks)
    if task_ids != task_set.task_ids:
        raise ArtifactCorruptionError(
            f"task set {task_set_id} task records do not match its ordered task IDs"
        )
    return LoadedTaskSet(task_set=task_set, tasks=tasks)


def resolve_task_set(store: ArtifactStore, task_set_id: str | None = None) -> LoadedTaskSet:
    """Load one named task set, or the only task set owned by a project.

    Args:
        store: Project-local immutable artifact store to inspect.
        task_set_id: Optional exact immutable task-set identity.

    Returns:
        The requested task set, or the project's only completed task set.

    Raises:
        ArtifactCorruptionError: There is no unambiguous task set to consume, or the selected
            artifact fails immutable verification.
    """
    if task_set_id is not None:
        return load_task_set(store, task_set_id)
    candidates = tuple(
        artifact_id
        for artifact_id in store.list_ids()
        if store.read(artifact_id).manifest.artifact_type == "task-set"
    )
    if not candidates:
        raise ArtifactCorruptionError(
            "project has no immutable task set; run wmo build on a declared trace export first"
        )
    if len(candidates) > 1:
        raise ArtifactCorruptionError(
            "project has multiple task sets; pass --task-set with one of: " + ", ".join(candidates)
        )
    return load_task_set(store, candidates[0])
