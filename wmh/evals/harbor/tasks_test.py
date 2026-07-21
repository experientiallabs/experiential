"""Tests for exact, provenance-bound Harbor task resolution."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from harbor.publisher.packager import Packager
from harbor.tasks.client import BatchDownloadResult, TaskDownloadResult, TaskIdType

from wmh.evals.harbor.tasks import (
    _pin_task_config,
    resolve_harbor_task_set,
)


def _task(parent: Path, name: str) -> Path:
    task_dir = parent / name
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM alpine:3.19\n",
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.sh").write_text(
        "#!/usr/bin/env sh\nexit 0\n",
        encoding="utf-8",
    )
    (task_dir / "instruction.md").write_text(
        f"Complete {name}.\n",
        encoding="utf-8",
    )
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\n',
        encoding="utf-8",
    )
    return task_dir


def _download(
    path: Path,
    *,
    content_hash: str | None = None,
    resolved_git_commit_id: str | None = None,
) -> TaskDownloadResult:
    return TaskDownloadResult(
        path=path,
        download_time_sec=0.0,
        cached=False,
        content_hash=content_hash,
        resolved_git_commit_id=resolved_git_commit_id,
    )


def test_resolver_selects_literal_task_ids_without_fnmatch_semantics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks"
    _task(dataset_path, "task[1]")
    _task(dataset_path, "task1")

    task_set = asyncio.run(
        resolve_harbor_task_set(
            DatasetConfig(path=dataset_path),
            ("task[1]",),
        )
    )

    assert task_set.task_ids == ("task[1]",)
    assert [task.get_task_id().get_name() for task in task_set.task_configs()] == ["task[1]"]
    definition, definition_path = task_set.task_inputs()[0]
    assert definition.schema_version == "1.0"
    assert definition_path == (dataset_path / "task[1]").resolve()
    assert task_set.requested_dataset_config().path == dataset_path
    assert task_set.resolved_dataset_config().task_names is None


def test_resolver_rejects_duplicate_requested_resolved_and_missing_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "tasks"
    task_path = _task(dataset_path, "task-a")
    dataset = DatasetConfig(path=dataset_path)

    with pytest.raises(ValueError, match="must be unique"):
        asyncio.run(resolve_harbor_task_set(dataset, ("task-a", "task-a")))
    with pytest.raises(ValueError, match="missing=.*task-b"):
        asyncio.run(resolve_harbor_task_set(dataset, ("task-b",)))

    async def duplicate_configs(
        _self: DatasetConfig,
        disable_verification: bool = False,
    ) -> list[TaskConfig]:
        del disable_verification
        config = TaskConfig(path=task_path, source="tasks")
        return [config, config.model_copy(deep=True)]

    monkeypatch.setattr(DatasetConfig, "get_task_configs", duplicate_configs)
    with pytest.raises(ValueError, match="duplicate task"):
        asyncio.run(resolve_harbor_task_set(dataset, ("task-a",)))


def test_resolved_local_task_is_absolute_and_rehashed_before_reuse(tmp_path: Path) -> None:
    dataset_path = tmp_path / "tasks"
    task_path = _task(dataset_path, "task-a")
    task_set = asyncio.run(resolve_harbor_task_set(DatasetConfig(path=dataset_path), ("task-a",)))

    assert task_set.task_configs()[0].path == task_path.resolve()
    task_set.verify()
    (task_path / "instruction.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed on disk"):
        task_set.verify()


def test_local_task_rejects_a_download_from_a_different_path(tmp_path: Path) -> None:
    configured = _task(tmp_path / "configured", "task-a")
    downloaded = _task(tmp_path / "downloaded", "task-a")

    with pytest.raises(ValueError, match="different download provenance"):
        _pin_task_config(
            TaskConfig(path=configured, source="configured"),
            _download(downloaded),
        )


def test_package_task_binds_config_download_and_cached_bytes(tmp_path: Path) -> None:
    task_path = _task(tmp_path, "task-a")
    content_hash, _ = Packager.compute_content_hash(task_path)
    digest = f"sha256:{content_hash}"
    download = _download(task_path, content_hash=content_hash)

    pinned = _pin_task_config(
        TaskConfig(name="org/task-a", ref=digest),
        download,
    )
    assert pinned.ref == digest

    with pytest.raises(ValueError, match="different package"):
        _pin_task_config(
            TaskConfig(name="org/task-a", ref="sha256:" + "0" * 64),
            download,
        )

    (task_path / "instruction.md").write_text("corrupt cache\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cache.*corrupt"):
        _pin_task_config(
            TaskConfig(name="org/task-a", ref=digest),
            download,
        )


@pytest.mark.parametrize("selector", [None, "main", "release-tag"])
def test_git_task_resolves_symbolic_selector_to_full_commit(
    tmp_path: Path,
    selector: str | None,
) -> None:
    task_path = _task(tmp_path, "task-a")
    commit = "A" * 40

    pinned = _pin_task_config(
        TaskConfig(
            path=Path("suite/task-a"),
            git_url="https://example.invalid/repo.git",
            git_commit_id=selector,
        ),
        _download(task_path, resolved_git_commit_id=commit),
    )

    assert pinned.git_commit_id == commit.lower()


def test_git_task_rejects_mismatched_or_unresolved_commit(tmp_path: Path) -> None:
    task_path = _task(tmp_path, "task-a")
    config = TaskConfig(
        path=Path("suite/task-a"),
        git_url="https://example.invalid/repo.git",
        git_commit_id="b" * 40,
    )

    with pytest.raises(ValueError, match="different git commit"):
        _pin_task_config(
            config,
            _download(task_path, resolved_git_commit_id="a" * 40),
        )
    with pytest.raises(ValueError, match="did not resolve"):
        _pin_task_config(
            config.model_copy(update={"git_commit_id": "main"}),
            _download(task_path, resolved_git_commit_id="short"),
        )


def test_remote_git_resolution_forces_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial_task_path = _task(tmp_path / "initial", "task-a")
    pinned_task_path = _task(tmp_path / "pinned", "task-a")
    dataset = DatasetConfig(name="dataset", overwrite=False)
    config = TaskConfig(
        path=Path("suite/task-a"),
        git_url="https://example.invalid/repo.git",
        git_commit_id="main",
        source="dataset",
    )

    async def task_configs(
        _self: DatasetConfig,
        disable_verification: bool = False,
    ) -> list[TaskConfig]:
        del disable_verification
        return [config]

    class _Client:
        overwrites: list[bool]

        def __init__(self) -> None:
            self.overwrites = []

        async def download_tasks(
            self,
            task_ids: list[TaskIdType],
            overwrite: bool = False,
            output_dir: Path | None = None,
        ) -> BatchDownloadResult:
            del task_ids, output_dir
            self.overwrites.append(overwrite)
            path = initial_task_path if len(self.overwrites) == 1 else pinned_task_path
            return BatchDownloadResult(
                results=[_download(path, resolved_git_commit_id="a" * 40)],
                total_time_sec=0.0,
            )

    monkeypatch.setattr(DatasetConfig, "get_task_configs", task_configs)
    client = _Client()
    task_set = asyncio.run(resolve_harbor_task_set(dataset, ("task-a",), task_client=client))

    assert client.overwrites == [True, True]
    assert task_set.task_configs()[0].git_commit_id == "a" * 40
    assert task_set.task_configs()[0].overwrite is False
    task_set.verify()
    (initial_task_path / "instruction.md").write_text(
        "old symbolic cache changed\n",
        encoding="utf-8",
    )
    task_set.verify()
    (pinned_task_path / "instruction.md").write_text(
        "pinned cache changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed on disk"):
        task_set.verify()
