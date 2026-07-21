"""Exact task selection for the harbor scorer.

Harbor's own `DatasetConfig.task_names` filter uses fnmatch semantics, so a task id containing a
glob character would silently over-match — and an optimizer's train/heldout split firewall relies
on exact selection. This module resolves a dataset once, post-filters by exact id, and returns
the pinned `TaskConfig` list a candidate job runs directly (`tasks=[...]`, no dataset filters).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig


async def resolve_harbor_tasks(
    dataset: DatasetConfig | Path,
    task_ids: Sequence[str],
) -> list[TaskConfig]:
    """Resolve `task_ids` from a harbor dataset (or local task dir) by exact identity.

    Args:
        dataset: A harbor `DatasetConfig`, or a local directory of task dirs (shorthand for
            `DatasetConfig(path=...)`).
        task_ids: Exact task names to select, in the order the caller wants them evaluated.

    Returns:
        One pinned `TaskConfig` per requested id, in request order. Git-sourced tasks are marked
        `overwrite=True` so harbor re-downloads them: its cache does not verify that an existing
        checkout still matches the requested commit.

    Raises:
        ValueError: On empty/duplicate ids, a dataset that resolves duplicate task names, or ids
            the dataset does not contain.
    """
    requested = list(task_ids)
    if not requested or any(not task_id for task_id in requested):
        raise ValueError("task_ids must be nonempty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("task_ids must be unique")

    if isinstance(dataset, Path):
        dataset = DatasetConfig(path=dataset)
    # Resolve the dataset WITHOUT harbor's filters, then select by exact id ourselves:
    # task_names is an fnmatch pattern list, not an exact-selection API.
    resolved = DatasetConfig.model_validate(dataset.model_dump(mode="python"))
    resolved.task_names = None
    resolved.exclude_task_names = None
    resolved.n_tasks = None
    configs = await resolved.get_task_configs()

    by_id: dict[str, TaskConfig] = {}
    for config in configs:
        task_id = config.get_task_id().get_name()
        if task_id in by_id:
            raise ValueError(f"harbor dataset resolved duplicate task {task_id!r}")
        by_id[task_id] = config
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(
            f"harbor task selection was not exact: missing={missing}; "
            "check the ids against the dataset's task names"
        )

    selected: list[TaskConfig] = []
    for task_id in requested:
        config = by_id[task_id].model_copy(deep=True)
        if config.is_git_task():
            config = config.model_copy(update={"overwrite": True})
        selected.append(config)
    return selected
