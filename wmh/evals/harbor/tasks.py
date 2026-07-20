"""Resolve and freeze exact Harbor task inputs before evaluator spend."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from harbor.models.job.config import DatasetConfig
from harbor.models.job.lock import build_trial_lock
from harbor.models.task.id import GitTaskId, PackageTaskId
from harbor.models.task.task import Task
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.publisher.packager import Packager
from harbor.tasks.client import (
    BatchDownloadResult,
    TaskClient,
    TaskDownloadResult,
    TaskIdType,
)
from pydantic import BaseModel, ConfigDict, Field

_CHECKSUM_PATTERN = r"^[0-9a-f]{64}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$")


class HarborTaskIdentity(BaseModel):
    """The two task-content identities emitted by Harbor 0.20."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_checksum: str = Field(strict=True, pattern=_CHECKSUM_PATTERN)
    lock_digest: str = Field(strict=True, pattern=_DIGEST_PATTERN)


class HarborTaskDownloader(Protocol):
    """Minimal task-download surface used by the exact resolver."""

    async def download_tasks(
        self,
        task_ids: list[TaskIdType],
        overwrite: bool = False,
        output_dir: Path | None = None,
    ) -> BatchDownloadResult: ...


@dataclass(frozen=True)
class _ResolvedTask:
    task_id: str
    identity: HarborTaskIdentity
    config_json: str
    download_json: str

    @classmethod
    def create(
        cls,
        config: TaskConfig,
        download: TaskDownloadResult,
        identity: HarborTaskIdentity,
    ) -> _ResolvedTask:
        snapshot = TaskConfig.model_validate(config.model_dump(mode="python"))
        download_snapshot = TaskDownloadResult.model_validate(
            download.model_dump(mode="python")
        ).model_copy(
            update={"path": download.path.resolve(strict=True)},
            deep=True,
        )
        return cls(
            task_id=snapshot.get_task_id().get_name(),
            identity=HarborTaskIdentity.model_validate(identity),
            config_json=snapshot.model_dump_json(),
            download_json=download_snapshot.model_dump_json(),
        )

    def config(self) -> TaskConfig:
        return TaskConfig.model_validate_json(self.config_json)

    def download(self) -> TaskDownloadResult:
        return TaskDownloadResult.model_validate_json(self.download_json)


@dataclass(frozen=True)
class ResolvedHarborTaskSet:
    """Immutable selected task configs and content identities for repeated jobs."""

    _requested_dataset_json: str
    _resolved_dataset_json: str
    _tasks: tuple[_ResolvedTask, ...]

    @classmethod
    def from_tasks(
        cls,
        *,
        requested_dataset: DatasetConfig,
        resolved_dataset: DatasetConfig,
        tasks: Sequence[tuple[TaskConfig, TaskDownloadResult, HarborTaskIdentity]],
    ) -> ResolvedHarborTaskSet:
        if not tasks:
            raise ValueError("resolved Harbor task set must be nonempty")
        records = tuple(
            _ResolvedTask.create(config, download, identity) for config, download, identity in tasks
        )
        ids = [record.task_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("resolved Harbor task ids must be unique")
        requested_snapshot = DatasetConfig.model_validate(
            requested_dataset.model_dump(mode="python")
        )
        resolved_snapshot = DatasetConfig.model_validate(resolved_dataset.model_dump(mode="python"))
        resolved_snapshot.task_names = None
        resolved_snapshot.exclude_task_names = None
        resolved_snapshot.n_tasks = None
        return cls(
            _requested_dataset_json=requested_snapshot.model_dump_json(),
            _resolved_dataset_json=resolved_snapshot.model_dump_json(),
            _tasks=records,
        )

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(record.task_id for record in self._tasks)

    @property
    def identities(self) -> dict[str, HarborTaskIdentity]:
        return {record.task_id: record.identity for record in self._tasks}

    def requested_dataset_config(self) -> DatasetConfig:
        """Return the exact selector supplied before Harbor resolves remote provenance."""
        return DatasetConfig.model_validate_json(self._requested_dataset_json)

    def resolved_dataset_config(self) -> DatasetConfig:
        """Return Harbor's resolved dataset provenance without task filters."""
        return DatasetConfig.model_validate_json(self._resolved_dataset_json)

    def task_configs(self) -> list[TaskConfig]:
        return [record.config() for record in self._tasks]

    def verify(self) -> None:
        """Re-hash every resolved task before a candidate job can incur spend."""
        for record in self._tasks:
            observed = _task_identity(record.config(), record.download())
            if observed != record.identity:
                raise ValueError(f"resolved Harbor task {record.task_id!r} changed on disk")


async def resolve_harbor_task_set(
    dataset: DatasetConfig,
    task_ids: Sequence[str],
    *,
    task_client: HarborTaskDownloader | None = None,
) -> ResolvedHarborTaskSet:
    """Resolve exact selected tasks, pin remote sources, and hash their downloaded bytes."""
    requested = tuple(task_ids)
    if not requested or any(not task_id for task_id in requested):
        raise ValueError("task_ids must be nonempty strings")
    if len(requested) != len(set(requested)):
        raise ValueError("task_ids must be unique")

    requested_dataset = DatasetConfig.model_validate(dataset.model_dump(mode="python"))
    resolved_dataset = DatasetConfig.model_validate(dataset.model_dump(mode="python"))
    # Resolve the dataset once, then select by exact identity ourselves. Harbor's
    # DatasetConfig.task_names uses fnmatch semantics and is not an exact-selection API.
    resolved_dataset.task_names = None
    resolved_dataset.exclude_task_names = None
    resolved_dataset.n_tasks = None
    configs = await resolved_dataset.get_task_configs(disable_verification=False)
    configs_by_id: dict[str, TaskConfig] = {}
    for config in configs:
        task_id = config.get_task_id().get_name()
        if task_id in configs_by_id:
            raise ValueError(f"Harbor dataset resolved duplicate task {task_id!r}")
        configs_by_id[task_id] = config
    missing = sorted(set(requested) - set(configs_by_id))
    if missing:
        raise ValueError(f"Harbor task selection was not exact: missing={missing}")

    ordered_configs = [configs_by_id[task_id] for task_id in requested]
    client = task_client or TaskClient()
    downloads = await client.download_tasks(
        [config.get_task_id() for config in ordered_configs],
        # Harbor's git cache does not prove an existing checkout still came from the
        # requested commit. Refresh it once during the no-model preflight, then freeze
        # and re-hash those bytes before every candidate job.
        overwrite=resolved_dataset.overwrite
        or any(isinstance(config.get_task_id(), GitTaskId) for config in ordered_configs),
        output_dir=resolved_dataset.download_dir,
    )
    if len(downloads.results) != len(ordered_configs):
        raise ValueError("Harbor returned an incomplete task download result")

    pinned_downloads: list[tuple[TaskConfig, TaskDownloadResult]] = []
    for config, download in zip(ordered_configs, downloads.results, strict=True):
        download = _absolute_download(download)
        pinned = _pin_task_config(config, download)
        pinned_downloads.append((pinned, download))

    git_indexes = [
        index
        for index, (config, _) in enumerate(pinned_downloads)
        if isinstance(config.get_task_id(), GitTaskId)
    ]
    if git_indexes:
        final_downloads = await client.download_tasks(
            [pinned_downloads[index][0].get_task_id() for index in git_indexes],
            overwrite=True,
            output_dir=resolved_dataset.download_dir,
        )
        if len(final_downloads.results) != len(git_indexes):
            raise ValueError("Harbor returned an incomplete pinned git download result")
        for index, download in zip(git_indexes, final_downloads.results, strict=True):
            download = _absolute_download(download)
            pinned, _ = pinned_downloads[index]
            pinned_downloads[index] = (_pin_task_config(pinned, download), download)

    resolved = [
        (config, download, _task_identity(config, download))
        for config, download in pinned_downloads
    ]
    return ResolvedHarborTaskSet.from_tasks(
        requested_dataset=requested_dataset,
        resolved_dataset=resolved_dataset,
        tasks=resolved,
    )


def _absolute_download(download: TaskDownloadResult) -> TaskDownloadResult:
    return download.model_copy(
        update={"path": download.path.resolve(strict=True)},
        deep=True,
    )


def _task_identity(
    config: TaskConfig,
    download: TaskDownloadResult,
) -> HarborTaskIdentity:
    _validate_download_provenance(config, download)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        trial_checksum = Task(download.path).checksum
    trial_lock = build_trial_lock(
        trial_config=TrialConfig(task=config),
        task_download_result=download,
    )
    if trial_lock.task.name != config.get_task_id().get_name():
        raise ValueError("Harbor task lock resolved a different task identity")
    return HarborTaskIdentity(
        trial_checksum=trial_checksum,
        lock_digest=trial_lock.task.digest,
    )


def _pin_task_config(config: TaskConfig, download: TaskDownloadResult) -> TaskConfig:
    task_id = config.get_task_id()
    if isinstance(task_id, GitTaskId):
        commit = download.resolved_git_commit_id
        if commit is None or _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"Harbor did not resolve a git commit for task {task_id.get_name()!r}")
        requested_commit = config.git_commit_id
        if (
            requested_commit is not None
            and _GIT_COMMIT_PATTERN.fullmatch(requested_commit) is not None
            and requested_commit.lower() != commit.lower()
        ):
            raise ValueError(
                f"Harbor resolved a different git commit for task {task_id.get_name()!r}"
            )
        pinned = config.model_copy(
            update={"git_commit_id": commit.lower(), "overwrite": False},
            deep=True,
        )
        _validate_download_provenance(pinned, download)
        return pinned
    if isinstance(task_id, PackageTaskId):
        downloaded_ref = (
            f"sha256:{download.content_hash}" if download.content_hash is not None else None
        )
        if (
            config.ref is not None
            and config.ref.startswith("sha256:")
            and downloaded_ref is not None
            and config.ref != downloaded_ref
        ):
            raise ValueError(
                f"Harbor downloaded a different package for task {task_id.get_name()!r}"
            )
        digest_ref = (
            config.ref if config.ref and config.ref.startswith("sha256:") else downloaded_ref
        )
        if digest_ref is None or not digest_ref.startswith("sha256:"):
            raise ValueError(
                f"Harbor did not resolve a package digest for task {task_id.get_name()!r}"
            )
        pinned = config.model_copy(
            update={"ref": digest_ref, "overwrite": False},
            deep=True,
        )
        _validate_download_provenance(pinned, download)
        return pinned
    if config.path is None:
        raise ValueError(f"Harbor local task {task_id.get_name()!r} has no configured path")
    configured_path = config.path.resolve(strict=True)
    downloaded_path = download.path.resolve(strict=True)
    if downloaded_path != configured_path:
        raise ValueError(
            f"Harbor local task {task_id.get_name()!r} has different download provenance"
        )
    return config.model_copy(
        update={"path": downloaded_path, "overwrite": False},
        deep=True,
    )


def _validate_download_provenance(
    config: TaskConfig,
    download: TaskDownloadResult,
) -> None:
    task_id = config.get_task_id()
    if isinstance(task_id, GitTaskId):
        commit = config.git_commit_id
        if commit is None or _GIT_COMMIT_PATTERN.fullmatch(commit) is None:
            raise ValueError(f"Harbor git task {task_id.get_name()!r} is not commit-pinned")
        resolved_commit = download.resolved_git_commit_id
        if resolved_commit is None or resolved_commit.lower() != commit.lower():
            raise ValueError(
                f"Harbor git task {task_id.get_name()!r} has different download provenance"
            )
    elif isinstance(task_id, PackageTaskId):
        expected = config.ref
        if expected is None or re.fullmatch(_DIGEST_PATTERN, expected) is None:
            raise ValueError(f"Harbor package task {task_id.get_name()!r} is not digest-pinned")
        if download.content_hash is not None and f"sha256:{download.content_hash}" != expected:
            raise ValueError(
                f"Harbor package task {task_id.get_name()!r} has different download provenance"
            )
        content_hash, _ = Packager.compute_content_hash(download.path)
        if f"sha256:{content_hash}" != expected:
            raise ValueError(f"Harbor package cache for task {task_id.get_name()!r} is corrupt")
