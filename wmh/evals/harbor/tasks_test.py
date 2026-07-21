"""Tests for exact harbor task selection."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig

from wmh.evals.harbor.tasks import resolve_harbor_tasks


def _make_task_dir(dataset: Path, task_id: str) -> None:
    task_dir = dataset / task_id
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.19\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    (task_dir / "instruction.md").write_text(f"Complete {task_id}.\n", encoding="utf-8")
    (task_dir / "task.toml").write_text('version = "1.0"\n\n[environment]\n', encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    for task_id in ("task-a", "task-b", "task-abc"):
        _make_task_dir(root, task_id)
    return root


def test_resolves_exact_ids_in_request_order(dataset: Path) -> None:
    selected = asyncio.run(resolve_harbor_tasks(dataset, ["task-b", "task-a"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-b", "task-a"]
    assert all(isinstance(task, TaskConfig) for task in selected)


def test_glob_shaped_ids_never_over_match(dataset: Path) -> None:
    """harbor's task_names filter is fnmatch; exact post-filtering must not expand globs."""
    with pytest.raises(ValueError, match=r"missing=\['task-\*'\]"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-*"]))
    # A glob that WOULD match several tasks under fnmatch selects nothing here, while the
    # literal ids it would have matched remain individually selectable.
    selected = asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-abc"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-a", "task-abc"]


def test_rejects_missing_duplicate_and_empty_ids(dataset: Path) -> None:
    with pytest.raises(ValueError, match="missing=\\['task-z'\\]"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-z"]))
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(resolve_harbor_tasks(dataset, ["task-a", "task-a"]))
    with pytest.raises(ValueError, match="nonempty"):
        asyncio.run(resolve_harbor_tasks(dataset, []))


def test_dataset_filters_are_ignored_and_git_tasks_get_overwrite(dataset: Path) -> None:
    # Preconfigured fnmatch filters on the dataset must not shadow exact selection.
    config = DatasetConfig(path=dataset, task_names=["task-b"])
    selected = asyncio.run(resolve_harbor_tasks(config, ["task-a"]))
    assert [task.get_task_id().get_name() for task in selected] == ["task-a"]

    git_task = TaskConfig(
        path=Path("tasks/task-g"),
        git_url="https://example.com/tasks.git",
        git_commit_id="a" * 40,
    )

    async def fake_get_task_configs(
        self: DatasetConfig,
        disable_verification: bool = False,
    ) -> list[TaskConfig]:
        del self, disable_verification
        return [git_task]

    original = DatasetConfig.get_task_configs
    DatasetConfig.get_task_configs = fake_get_task_configs
    try:
        [pinned] = asyncio.run(resolve_harbor_tasks(config, ["task-g"]))
    finally:
        DatasetConfig.get_task_configs = original
    # harbor's git cache does not verify commit provenance; force a fresh download.
    assert pinned.overwrite is True
    assert git_task.overwrite is False  # the caller's config is never mutated
