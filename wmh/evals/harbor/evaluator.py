"""Reusable Harbor 0.18 evaluator for ground-truth WMH benchmark runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol, override
from uuid import UUID

from harbor.environments.factory import EnvironmentFactory
from harbor.job import Job
from harbor.metrics.uv_script import UvScript
from harbor.models.dataset.paths import DatasetPaths
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.job.lock import JobLock, TrialLock, build_job_lock
from harbor.models.job.result import JobResult
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.registry import DatasetMetadata
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import task_has_any_separate_verifier
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.models.trial.result import TrialResult
from harbor.registry.client.factory import RegistryClientFactory
from harbor.registry.client.package import PackageDatasetClient
from harbor.tasks.client import TaskIdType
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wmh.core.text import normalize_durable_text
from wmh.evals.benchmark import (
    MAX_BENCHMARK_TASK_INSTRUCTION_CHARS,
    BenchmarkCell,
    BenchmarkRunIdentity,
    BenchmarkTaskEnvironment,
)
from wmh.evals.harbor._file_lease import exclusive_posix_file_lease
from wmh.evals.harbor.agent import WMH_PI_AGENT_VERSION, WmhPiAgent
from wmh.evals.harbor.config import (
    SUPPORTED_HARBOR_VERSION,
    HarborEnvironmentBackend,
    HarborJobSpec,
    build_harbor_job_config,
    validate_controlled_harbor_environment,
)
from wmh.evals.harbor.results import (
    HarborTrialManifest,
    HarborTrialManifestEntry,
    LoadedHarborJobResult,
    harbor_agent_config_digest,
    harbor_trial_lock_digest,
    load_harbor_job_result,
)
from wmh.evals.harbor.task_security import (
    PrebuiltImageTaskBoundaryError,
    TaskCredentialBoundaryError,
    validate_prebuilt_image_task_tree,
    validate_task_credential_boundary,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_local import (
    PI_CONTAINER_IMAGE,
    validate_pi_container_image,
    verify_container_pi_runner_ready,
)
from wmh.providers.base import ProviderConfig

_MANIFEST_FILENAME = "wmh-manifest.json"
_MAX_TASK_HOST_TEXT_BYTES = 1024 * 1024
_MAX_EXTRA_INSTRUCTION_FILES = 16
_TASK_SNAPSHOT_ROOT = ".wmh-task-snapshots"
_JOB_ROOT_FILES = frozenset(
    {_MANIFEST_FILENAME, "config.json", "job.log", "lock.json", "result.json"}
)
HARBOR_EVALUATOR_VERSION = "1"


class StaleHarborJobError(RuntimeError):
    """A pre-existing Harbor job directory cannot be proved to match this evaluation."""


class ConcurrentHarborJobError(RuntimeError):
    """Another process already holds the exclusive execution lease for this Harbor job."""


class UnsupportedHarborTaskError(ValueError):
    """A task uses Harbor behavior whose failures cannot yet be observed safely."""


class UnsupportedHarborMetricError(ValueError):
    """A Harbor metric could execute untrusted code on the credential-bearing host."""


class HarborRunExpectation(BaseModel):
    """Canonical identity one Harbor evaluator invocation must publish.

    This public contract lets higher-level experiment protocols freeze the exact evaluator
    identity without copying Harbor adapter internals. It intentionally contains no credentials or
    host paths.
    """

    model_config = ConfigDict(frozen=True)

    job_name: str = Field(min_length=1)
    harness_execution_hash: str = Field(min_length=1)
    harness_execution_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    identity: BenchmarkRunIdentity


def harbor_run_expectation(
    *,
    candidate: HarnessDoc,
    spec: HarborJobSpec,
    provider_config: ProviderConfig,
    runner_image: str,
    turn_timeout_s: float,
    require_provider_receipts: bool = True,
) -> HarborRunExpectation:
    """Build the path-independent identity expected from one exact Harbor run."""
    validate_pi_container_image(runner_image)
    if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
        raise ValueError("turn_timeout_s must be finite and positive")
    frozen_spec = HarborJobSpec.model_validate(spec.model_dump())
    frozen_provider = ProviderConfig.model_validate(provider_config.model_dump())
    agent = _build_harbor_agent_config(
        candidate=candidate,
        provider_config=frozen_provider,
        runner_image=runner_image,
        turn_timeout_s=turn_timeout_s,
        agent_n_concurrent=frozen_spec.agent_n_concurrent,
        require_provider_receipts=require_provider_receipts,
    )
    run_config_digest = harbor_run_config_digest(
        frozen_spec,
        harbor_agent_config_digest(agent),
    )
    environment = {
        HarborEnvironmentBackend.LOCAL: BenchmarkTaskEnvironment.DOCKER,
        HarborEnvironmentBackend.E2B: BenchmarkTaskEnvironment.E2B,
    }[frozen_spec.environment_backend]
    return HarborRunExpectation(
        job_name=frozen_spec.job_name,
        harness_execution_hash=candidate.execution_hash,
        harness_execution_digest=candidate.execution_digest,
        identity=BenchmarkRunIdentity(
            candidate_hash=candidate.execution_hash,
            agent_name=WmhPiAgent.name(),
            agent_version=WMH_PI_AGENT_VERSION,
            provider=frozen_provider.kind.value,
            model_name=frozen_provider.model,
            task_environment=environment,
            runner_image=runner_image,
            run_config_digest=run_config_digest,
        ),
    )


def _build_harbor_agent_config(
    *,
    candidate: HarnessDoc,
    provider_config: ProviderConfig,
    runner_image: str,
    turn_timeout_s: float,
    agent_n_concurrent: int | None,
    require_provider_receipts: bool,
) -> AgentConfig:
    model_name = f"{provider_config.kind.value}/{provider_config.model}"
    return AgentConfig(
        import_path=WmhPiAgent.import_path(),
        model_name=model_name,
        n_concurrent=agent_n_concurrent,
        kwargs={
            "harness": candidate.model_dump(mode="json"),
            "provider_config": provider_config.model_dump(mode="json"),
            "runner_image": runner_image,
            "turn_timeout_s": turn_timeout_s,
            "require_provider_receipts": require_provider_receipts,
        },
    )


class HarborEvaluatorSession:
    """Share one trusted runner-readiness probe across concurrent exact Harbor jobs."""

    def __init__(self, *, runner_image: str = PI_CONTAINER_IMAGE) -> None:
        validate_pi_container_image(runner_image)
        self._runner_image = runner_image
        self._ready = False
        self._lock = asyncio.Lock()

    async def ensure_runner_ready(self, *, runner_image: str) -> None:
        """Probe the immutable runner image once for this event-loop-scoped session."""
        if runner_image != self._runner_image:
            raise ValueError("Harbor evaluator session runner image differs from evaluator")
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await asyncio.to_thread(
                verify_container_pi_runner_ready,
                image=self._runner_image,
            )
            self._ready = True


class _DigestWriter(Protocol):
    def update(self, value: bytes, /) -> None: ...


class _AtomicHarborJob(Job):
    """Pinned Harbor job with job-root-safe replacement of its live root result."""

    @override
    def _write_job_result(self, *, exclude_trial_results: bool = False) -> None:
        if exclude_trial_results:
            payload = self._job_result.model_dump_json(
                indent=4,
                exclude={"trial_results"},
            )
        else:
            payload = self._job_result.model_dump_json(indent=4)
        _atomic_replace_job_result(self._job_result_path, payload)


@dataclass(frozen=True)
class _PreparedTrial:
    config: TrialConfig
    task_key: str
    task_name: str
    task_identity: str
    task_checksum: str
    task_source: str | None
    task_instruction: str
    trial_lock_digest: str

    @property
    def immutable_key(self) -> tuple[str, str, str, str, str | None, str, str]:
        return (
            self.task_key,
            self.task_name,
            self.task_identity,
            self.task_checksum,
            self.task_source,
            self.task_instruction,
            self.trial_lock_digest,
        )


@dataclass
class _OpenedDatasetSource:
    """One source path identity held open from task discovery through snapshot copy."""

    path: Path
    fd: int
    metadata: os.stat_result
    task_names: tuple[str, ...]


class HarborEvaluator:
    """Evaluate one immutable harness candidate over a fixed Harbor task matrix."""

    def __init__(
        self,
        spec: HarborJobSpec,
        provider_config: ProviderConfig,
        *,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
        require_provider_receipts: bool = True,
        session: HarborEvaluatorSession | None = None,
    ) -> None:
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        if not isinstance(require_provider_receipts, bool):
            raise ValueError("require_provider_receipts must be a boolean")
        validated_spec = HarborJobSpec.model_validate(spec.model_dump())
        _validate_job_name(validated_spec.job_name)
        self._spec = validated_spec.model_copy(
            update={"jobs_dir": validated_spec.jobs_dir.expanduser().resolve()}, deep=True
        )
        self._provider_config = provider_config.model_copy(deep=True)
        self._runner_image = runner_image
        self._turn_timeout_s = turn_timeout_s
        self._require_provider_receipts = require_provider_receipts
        self._session = session
        self._runner_ready = False

    async def evaluate(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
        """Run or safely reuse one exact Harbor job, then strictly ingest its evidence."""
        if candidate.runtime_kind() != "pi-node":
            raise ValueError(
                "Harbor WMH evaluation requires a pi-node candidate, got "
                f"{candidate.runtime_kind()!r}"
            )
        with _exclusive_job_run_lock(self._spec.jobs_dir, self._spec.job_name):
            return await self._evaluate_locked(candidate)

    async def _evaluate_locked(self, candidate: HarnessDoc) -> LoadedHarborJobResult:
        """Evaluate while holding this job name's interprocess execution lease."""
        agent = self._build_agent(candidate)
        agent_digest = harbor_agent_config_digest(agent)
        run_config_digest = harbor_run_config_digest(self._spec, agent_digest)
        identity = self._run_identity(
            candidate,
            run_config_digest=run_config_digest,
        )
        job_config = build_harbor_job_config(self._spec, agent=agent)
        job_config = _snapshot_local_datasets(job_config)
        job_dir = job_config.jobs_dir / job_config.job_name
        await _reject_executable_harbor_metrics(job_config)
        existing_manifest = _inspect_existing_job(
            job_dir,
            expected_identity=identity,
            expected_agent_digest=agent_digest,
            expected_job_config=job_config,
        )
        await asyncio.to_thread(_preflight_task_environment, job_config)
        await self._ensure_runner_ready()

        try:
            job = await _AtomicHarborJob.create(job_config)
        except FileExistsError as exc:
            if existing_manifest is not None:
                raise StaleHarborJobError(
                    f"Harbor job directory {job_dir} is stale: {exc}"
                ) from exc
            raise

        try:
            _reject_resolved_executable_harbor_metrics(job)
            if not job._trial_configs:
                raise ValueError(
                    "Harbor resolved zero trials; refusing to publish an empty evaluation"
                )
            expected_environment_type = job_config.environment.type
            validate_controlled_harbor_environment(
                job.config.environment,
                expected_type=expected_environment_type,
            )
            for trial_config in job._trial_configs:
                validate_controlled_harbor_environment(
                    trial_config.environment,
                    expected_type=expected_environment_type,
                )
            tasks = _load_and_validate_tasks(
                job,
                environment_backend=self._spec.environment_backend,
            )
            job_lock = _build_prepared_job_lock(job)
            prepared_trials = _prepare_trials(
                job,
                job_lock,
                tasks,
                agent_config_digest=agent_digest,
            )
            if existing_manifest is not None:
                _validate_existing_job_lock(job_dir, job_lock)
                _restore_manifest_trial_names(job, prepared_trials, existing_manifest)
            manifest = _build_manifest(
                job.config.job_name,
                prepared_trials,
                identity=identity,
                agent_config_digest=agent_digest,
            )
            _persist_or_compare_manifest(job_dir, manifest, existing_manifest)
            await job.run()
        finally:
            # Job.run() closes these itself. This idempotent call also covers pre-run rejection.
            job._close_logger_handlers()

        return load_harbor_job_result(job_dir, manifest)

    async def _ensure_runner_ready(self) -> None:
        """Probe the trusted local runner once before Harbor can create a job."""
        if self._session is not None:
            await self._session.ensure_runner_ready(runner_image=self._runner_image)
            return
        if self._runner_ready:
            return
        await asyncio.to_thread(
            verify_container_pi_runner_ready,
            image=self._runner_image,
        )
        self._runner_ready = True

    def _build_agent(self, candidate: HarnessDoc) -> AgentConfig:
        return _build_harbor_agent_config(
            candidate=candidate,
            provider_config=self._provider_config,
            runner_image=self._runner_image,
            turn_timeout_s=self._turn_timeout_s,
            agent_n_concurrent=self._spec.agent_n_concurrent,
            require_provider_receipts=self._require_provider_receipts,
        )

    def _run_identity(
        self,
        candidate: HarnessDoc,
        *,
        run_config_digest: str,
    ) -> BenchmarkRunIdentity:
        environment = {
            HarborEnvironmentBackend.LOCAL: BenchmarkTaskEnvironment.DOCKER,
            HarborEnvironmentBackend.E2B: BenchmarkTaskEnvironment.E2B,
        }[self._spec.environment_backend]
        return BenchmarkRunIdentity(
            candidate_hash=candidate.execution_hash,
            agent_name=WmhPiAgent.name(),
            agent_version=WMH_PI_AGENT_VERSION,
            provider=self._provider_config.kind.value,
            model_name=self._provider_config.model,
            reasoning_effort=self._provider_config.reasoning_effort,
            task_environment=environment,
            runner_image=self._runner_image,
            run_config_digest=run_config_digest,
        )


async def _reject_executable_harbor_metrics(config: JobConfig) -> None:
    """Reject every Harbor metric source that can run code in the host process."""
    _reject_uv_script_metrics(config.metrics, source="the Harbor job configuration")
    for dataset in config.datasets:
        metadata = await _load_dataset_metadata_for_metric_audit(dataset)
        if metadata is None:
            continue
        source = f"Harbor dataset {metadata.name!r}"
        _reject_uv_script_metrics(metadata.metrics, source=source)
        if dataset.is_package() and any(
            file.path == DatasetPaths.METRIC_FILENAME for file in metadata.files
        ):
            raise UnsupportedHarborMetricError(
                f"{source} includes executable {DatasetPaths.METRIC_FILENAME!r}; "
                "WMH ground-truth evaluation accepts only non-executable built-in aggregate "
                "metrics because Harbor dataset scripts run on the credential-bearing host. "
                "Remove the executable metric and compute trusted analysis from canonical "
                "per-trial rewards."
            )


def _reject_uv_script_metrics(metrics: list[MetricConfig], *, source: str) -> None:
    """Reject Harbor's configured host-side script metric without constructing it."""
    if any(metric.type is MetricType.UV_SCRIPT for metric in metrics):
        raise UnsupportedHarborMetricError(
            f"{source} declares an executable {MetricType.UV_SCRIPT.value!r} metric; "
            "WMH ground-truth evaluation accepts only non-executable built-in aggregate metrics "
            "because Harbor metric scripts run on the credential-bearing host. Remove the "
            "executable metric and compute trusted analysis from canonical per-trial rewards."
        )


def _reject_resolved_executable_harbor_metrics(job: Job) -> None:
    """Reject executable metrics from Harbor's actual post-resolution metric set."""
    executable_sources = sorted(
        source
        for source, metrics in job._metrics.items()
        if any(isinstance(metric, UvScript) for metric in metrics)
    )
    if executable_sources:
        raise UnsupportedHarborMetricError(
            "Harbor resolved executable host-side metrics for "
            f"{executable_sources}; WMH ground-truth evaluation accepts only non-executable "
            "built-in aggregate metrics because Harbor metric scripts run on the "
            "credential-bearing host. Pin a dataset without executable metrics and compute "
            "trusted analysis from canonical per-trial rewards."
        )


async def _load_dataset_metadata_for_metric_audit(
    dataset: DatasetConfig,
) -> DatasetMetadata | None:
    """Load remote metadata without downloading tasks or dataset-level executable files."""
    if dataset.is_local():
        return None
    if dataset.is_repo():
        client = RegistryClientFactory.create(
            repo=dataset.repo,
            path=dataset.path,
            registry_path=dataset.registry_path,
        )
        if dataset.name is None:
            name = ""
        else:
            name = f"{dataset.name}@{dataset.version}" if dataset.version else dataset.name
        return await client.get_dataset_metadata(name)
    if dataset.is_package():
        if dataset.name is None:
            raise RuntimeError("Package dataset config is missing name")
        name = f"{dataset.name}@{dataset.ref or 'latest'}"
        return await PackageDatasetClient().get_dataset_metadata(name)
    if dataset.is_registry():
        if dataset.name is None:
            raise RuntimeError("Registry dataset config is missing name")
        client = RegistryClientFactory.create(
            registry_url=dataset.registry_url,
            registry_path=dataset.registry_path,
        )
        name = f"{dataset.name}@{dataset.version}" if dataset.version else dataset.name
        return await client.get_dataset_metadata(name)
    raise RuntimeError("Harbor dataset config has no supported source")


def _preflight_task_environment(config: JobConfig) -> None:
    """Fail before runner or job creation when Harbor's task backend is unavailable."""
    environment = config.environment
    missing_e2b_modules = [
        module
        for module in ("e2b", "dockerfile_parse")
        if environment.type is EnvironmentType.E2B and find_spec(module) is None
    ]
    if missing_e2b_modules:
        raise RuntimeError(
            "Harbor E2B task environments require the WMH e2b extra; install "
            "world-model-harness[e2b] or run `uv sync --extra e2b` "
            f"(missing modules: {', '.join(missing_e2b_modules)})"
        )
    try:
        EnvironmentFactory.run_preflight(
            type=environment.type,
            import_path=environment.import_path,
        )
    except SystemExit as exc:
        detail = str(exc).strip() or "task environment is unavailable"
        raise RuntimeError(f"Harbor task-environment preflight failed: {detail}") from None


def _build_prepared_job_lock(job: Job) -> JobLock:
    lock = build_job_lock(
        config=job.config,
        trial_configs=job._trial_configs,
        task_download_results=job._task_download_results,
    )
    if len(lock.trials) != len(job._trial_configs):
        raise RuntimeError(
            "Harbor prepared a different number of trial locks and trial configs: "
            f"{len(lock.trials)} != {len(job._trial_configs)}"
        )
    return lock


def _preflight_local_task_trees(config: JobConfig) -> None:
    """Reject unsafe local task paths before Harbor can inspect task configuration."""
    for dataset in config.datasets:
        if not dataset.is_local():
            continue
        if dataset.path is None:
            raise RuntimeError("local Harbor dataset is missing its path")
        root = dataset.path.expanduser()
        if root.is_symlink():
            raise UnsupportedHarborTaskError(
                f"local dataset root cannot be a symbolic link: {root}"
            )
        if not root.is_dir():
            continue
        for task_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if task_dir.is_symlink():
                raise UnsupportedHarborTaskError(
                    f"local task root cannot be a symbolic link: {task_dir}"
                )
            if not task_dir.is_dir():
                continue
            config_path = task_dir / "task.toml"
            if not (config_path.exists() or config_path.is_symlink()):
                continue
            _validate_task_tree_for_host_reads(task_dir, extra_instruction_paths=())
            if config.environment.type is EnvironmentType.DOCKER:
                _validate_prebuilt_local_task_tree(task_dir)


def _snapshot_local_datasets(config: JobConfig) -> JobConfig:
    """Run every local dataset from a content-addressed WMH-controlled task snapshot.

    Harbor 0.18 resolves a local task path again when each trial starts. Pointing it at the source
    checkout would leave a validation-to-execution race in which a Compose or dotenv file could be
    added after preflight. Copying through held directory descriptors without following links,
    validating the copy, and then atomically publishing a read-only content-addressed tree makes the
    source path irrelevant to execution for both Docker and E2B. Remote/repository/package
    acquisition remains disabled because Harbor 0.18 may dereference task symlinks before WMH
    receives the downloaded tree to validate.
    """
    local_dataset_indexes = {
        index for index, dataset in enumerate(config.datasets) if dataset.is_local()
    }
    if len(local_dataset_indexes) != len(config.datasets):
        raise UnsupportedHarborTaskError(
            "ground-truth evaluation requires preflightable local dataset paths for both "
            "Docker and E2B"
        )
    if not local_dataset_indexes:
        return config
    snapshots_root = config.jobs_dir / _TASK_SNAPSHOT_ROOT
    if snapshots_root.is_symlink():
        raise UnsupportedHarborTaskError("Harbor task snapshot root cannot be a symbolic link")
    snapshots_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not snapshots_root.is_dir():
        raise UnsupportedHarborTaskError("Harbor task snapshot root must be a directory")
    try:
        jobs_root = config.jobs_dir.resolve(strict=True)
        snapshots_root_resolved = snapshots_root.resolve(strict=True)
    except OSError:
        raise UnsupportedHarborTaskError(
            "Harbor task snapshot namespace cannot be resolved"
        ) from None
    _require_private_owned_directory(jobs_root, label="Harbor jobs directory")
    _require_private_owned_directory(
        snapshots_root_resolved,
        label="Harbor task snapshot root",
    )

    opened_datasets: dict[int, _OpenedDatasetSource] = {}
    source_names: set[str] = set()
    try:
        for index in sorted(local_dataset_indexes):
            dataset = config.datasets[index]
            if dataset.path is None:
                raise RuntimeError("local Harbor dataset is missing its path")
            requested_root = dataset.path.expanduser().absolute()
            source_fd, source_metadata, task_names = _open_local_dataset_snapshot_source(
                requested_root
            )
            try:
                source_root = requested_root.resolve(strict=True)
                resolved_metadata = source_root.stat(follow_symlinks=False)
            except OSError:
                os.close(source_fd)
                raise UnsupportedHarborTaskError(
                    "local dataset root cannot be resolved while its identity is held open"
                ) from None
            if (resolved_metadata.st_dev, resolved_metadata.st_ino) != (
                source_metadata.st_dev,
                source_metadata.st_ino,
            ):
                os.close(source_fd)
                raise UnsupportedHarborTaskError(
                    "local dataset path changed identity while it was resolved"
                )
            if source_root.is_relative_to(jobs_root) or jobs_root.is_relative_to(source_root):
                os.close(source_fd)
                raise UnsupportedHarborTaskError(
                    "local dataset and Harbor jobs directory must not contain one another"
                )
            if source_root.name in source_names:
                os.close(source_fd)
                raise UnsupportedHarborTaskError(
                    "local datasets must have unique directory names because Harbor uses the "
                    "directory name as the dataset source identity"
                )
            source_names.add(source_root.name)
            opened_datasets[index] = _OpenedDatasetSource(
                path=source_root,
                fd=source_fd,
                metadata=source_metadata,
                task_names=task_names,
            )

        frozen_datasets: list[DatasetConfig] = []
        for index, dataset in enumerate(config.datasets):
            source = opened_datasets[index]
            source_key = hashlib.sha256(str(source.path).encode()).hexdigest()
            source_snapshot_root = snapshots_root_resolved / source_key
            source_snapshot_root.mkdir(mode=0o700, exist_ok=True)
            if source_snapshot_root.is_symlink() or not source_snapshot_root.is_dir():
                raise UnsupportedHarborTaskError(
                    "Harbor dataset snapshot namespace must be a regular directory"
                )
            _require_private_owned_directory(
                source_snapshot_root,
                label="Harbor dataset snapshot namespace",
            )
            snapshot = _build_or_reuse_dataset_snapshot(
                config,
                dataset,
                source=source,
                source_snapshot_root=source_snapshot_root,
            )
            os.close(source.fd)
            source.fd = -1
            frozen_datasets.append(dataset.model_copy(update={"path": snapshot}, deep=True))
    finally:
        for source in opened_datasets.values():
            if source.fd >= 0:
                os.close(source.fd)

    frozen = config.model_copy(update={"datasets": frozen_datasets}, deep=True)
    _preflight_local_task_trees(frozen)
    return frozen


def _build_or_reuse_dataset_snapshot(
    config: JobConfig,
    dataset: DatasetConfig,
    *,
    source: _OpenedDatasetSource,
    source_snapshot_root: Path,
) -> Path:
    temporary_root = Path(tempfile.mkdtemp(prefix=".pending-", dir=source_snapshot_root))
    # Harbor uses the local dataset directory basename as its source identity. Keep that stable
    # across snapshots; the source path namespace above guarantees different roots cannot share
    # this leaf, while the preflight rejects ambiguous basenames within one job.
    temporary_dataset = temporary_root / source.path.name

    try:
        # Do not use shutil.copytree here. Its lstat/open sequence can follow a source entry that
        # is swapped to a symlink between those operations. The fd-relative copier below holds
        # every parent directory open and uses O_NOFOLLOW for each child.
        _copy_local_tasks_from_open_dataset(
            source.fd,
            source.metadata,
            temporary_dataset,
            task_names=source.task_names,
        )
        temporary_config = config.model_copy(
            update={
                "datasets": [dataset.model_copy(update={"path": temporary_dataset}, deep=True)]
            },
            deep=True,
        )
        _preflight_local_task_trees(temporary_config)
        digest = _task_snapshot_digest(temporary_dataset)
        snapshot_version_root = source_snapshot_root / digest.removeprefix("sha256:")
        try:
            snapshot_version_root.mkdir(mode=0o700, exist_ok=True)
        except OSError:
            raise UnsupportedHarborTaskError(
                "Harbor task snapshot version directory could not be created"
            ) from None
        if snapshot_version_root.is_symlink() or not snapshot_version_root.is_dir():
            raise UnsupportedHarborTaskError(
                "Harbor task snapshot version path must be a regular directory"
            )
        _require_private_owned_directory(
            snapshot_version_root,
            label="Harbor task snapshot version directory",
        )
        final = snapshot_version_root / source.path.name
        if final.exists() or final.is_symlink():
            _validate_existing_dataset_snapshot(final, expected_digest=digest)
            return final

        _make_task_snapshot_read_only(temporary_dataset)
        try:
            temporary_dataset.rename(final)
        except FileExistsError:
            _validate_existing_dataset_snapshot(final, expected_digest=digest)
        except OSError:
            if not final.is_dir():
                raise UnsupportedHarborTaskError(
                    "Harbor task snapshot could not be published atomically"
                ) from None
            _validate_existing_dataset_snapshot(final, expected_digest=digest)
        else:
            try:
                final.chmod(0o555)
            except OSError:
                raise UnsupportedHarborTaskError(
                    "Harbor task snapshot could not be made read-only"
                ) from None
        _validate_existing_dataset_snapshot(final, expected_digest=digest)
        return final
    finally:
        _remove_temporary_snapshot(temporary_root)


def _open_local_dataset_snapshot_source(
    source_root: Path,
) -> tuple[int, os.stat_result, tuple[str, ...]]:
    """Open one exact dataset root and discover its task names through the held fd."""
    _require_secure_copy_primitives()
    source_fd: int | None = None
    try:
        before = source_root.stat(follow_symlinks=False)
        source_fd = os.open(source_root, _source_directory_open_flags())
        opened = os.fstat(source_fd)
    except OSError:
        if source_fd is not None:
            os.close(source_fd)
        raise UnsupportedHarborTaskError(
            "local Harbor dataset root could not be opened without following links"
        ) from None
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or not stat.S_ISDIR(
        opened.st_mode
    ):
        os.close(source_fd)
        raise UnsupportedHarborTaskError("local Harbor dataset root changed identity")
    try:
        _require_owned_nonwritable_by_others(opened, label="local Harbor dataset root")
        task_names = _discover_local_task_names(source_fd, opened)
    except BaseException:
        os.close(source_fd)
        raise
    if not task_names:
        os.close(source_fd)
        raise UnsupportedHarborTaskError("local Harbor dataset contains no task directories")
    return source_fd, opened, task_names


def _copy_local_tasks_from_open_dataset(
    source_fd: int,
    source_metadata: os.stat_result,
    destination_root: Path,
    *,
    task_names: tuple[str, ...],
) -> None:
    """Copy selected task trees through held directory fds without following links."""
    source_flags = _source_directory_open_flags()
    try:
        destination_root.mkdir(mode=0o700)
        destination_fd = os.open(destination_root, source_flags)
    except OSError:
        raise UnsupportedHarborTaskError(
            "Harbor task snapshot destination could not be created safely"
        ) from None
    try:
        for task_name in task_names:
            _copy_source_directory(
                source_fd,
                destination_fd,
                task_name,
                source_device=source_metadata.st_dev,
            )
        final_source_metadata = os.fstat(source_fd)
        if _metadata_changed(source_metadata, final_source_metadata):
            raise UnsupportedHarborTaskError(
                "local Harbor dataset root changed while its task snapshot was copied"
            )
    finally:
        os.close(destination_fd)


def _discover_local_task_names(
    source_fd: int,
    source_metadata: os.stat_result,
) -> tuple[str, ...]:
    try:
        names = sorted(os.listdir(source_fd))
    except OSError:
        raise UnsupportedHarborTaskError("local Harbor dataset cannot be enumerated") from None
    task_names: list[str] = []
    for name in names:
        try:
            entry = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except OSError:
            raise UnsupportedHarborTaskError(
                "local Harbor dataset entry changed during task discovery"
            ) from None
        if stat.S_ISLNK(entry.st_mode):
            raise UnsupportedHarborTaskError("local dataset child cannot be a symbolic link")
        if not stat.S_ISDIR(entry.st_mode):
            continue
        directory_fd, _ = _open_verified_source_entry(
            source_fd,
            name,
            expect_directory=True,
            source_device=source_metadata.st_dev,
        )
        try:
            try:
                task_config = os.stat(
                    "task.toml",
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                continue
            except OSError:
                raise UnsupportedHarborTaskError(
                    "local Harbor task configuration changed during discovery"
                ) from None
            if stat.S_ISLNK(task_config.st_mode):
                raise UnsupportedHarborTaskError(
                    "local Harbor task configuration cannot be a symbolic link"
                )
            if not stat.S_ISREG(task_config.st_mode) or task_config.st_nlink != 1:
                raise UnsupportedHarborTaskError(
                    "local Harbor task configuration must be a single-link regular file"
                )
            _require_owned_nonwritable_by_others(
                task_config,
                label="local Harbor task configuration",
            )
            task_names.append(name)
        finally:
            os.close(directory_fd)
    if _metadata_changed(source_metadata, os.fstat(source_fd)):
        raise UnsupportedHarborTaskError("local Harbor dataset root changed during task discovery")
    return tuple(task_names)


def _source_directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _require_secure_copy_primitives() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise RuntimeError("local Harbor task snapshots require POSIX no-follow file operations")
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd:
        raise RuntimeError("local Harbor task snapshots require fd-relative open and stat")
    if os.mkdir not in os.supports_dir_fd or os.listdir not in os.supports_fd:
        raise RuntimeError("local Harbor task snapshots require fd-relative directory traversal")


def _copy_source_directory(
    source_parent_fd: int,
    destination_parent_fd: int,
    name: str,
    *,
    source_device: int,
) -> None:
    source_fd, source_metadata = _open_verified_source_entry(
        source_parent_fd,
        name,
        expect_directory=True,
        source_device=source_device,
    )
    try:
        try:
            os.mkdir(name, mode=0o700, dir_fd=destination_parent_fd)
            destination_fd = os.open(
                name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=destination_parent_fd,
            )
        except OSError:
            raise UnsupportedHarborTaskError(
                "Harbor task snapshot directory could not be created safely"
            ) from None
        try:
            try:
                child_names = sorted(os.listdir(source_fd))
            except OSError:
                raise UnsupportedHarborTaskError(
                    "local Harbor task directory could not be enumerated safely"
                ) from None
            for child_name in child_names:
                try:
                    child_metadata = os.stat(
                        child_name,
                        dir_fd=source_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    raise UnsupportedHarborTaskError(
                        "local Harbor task entry changed while its snapshot was copied"
                    ) from None
                if stat.S_ISLNK(child_metadata.st_mode):
                    raise UnsupportedHarborTaskError(
                        "local Harbor task snapshot source cannot contain a symbolic link"
                    )
                if stat.S_ISDIR(child_metadata.st_mode):
                    _copy_source_directory(
                        source_fd,
                        destination_fd,
                        child_name,
                        source_device=source_device,
                    )
                elif stat.S_ISREG(child_metadata.st_mode):
                    _copy_source_file(
                        source_fd,
                        destination_fd,
                        child_name,
                        source_device=source_device,
                    )
                else:
                    raise UnsupportedHarborTaskError(
                        "local Harbor task snapshot source can contain only single-link regular "
                        "files and directories"
                    )
            if _metadata_changed(source_metadata, os.fstat(source_fd)):
                raise UnsupportedHarborTaskError(
                    "local Harbor task directory changed while its snapshot was copied"
                )
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def _copy_source_file(
    source_parent_fd: int,
    destination_parent_fd: int,
    name: str,
    *,
    source_device: int,
) -> None:
    source_fd, source_metadata = _open_verified_source_entry(
        source_parent_fd,
        name,
        expect_directory=False,
        source_device=source_device,
    )
    destination_fd: int | None = None
    try:
        try:
            destination_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                0o600 | (source_metadata.st_mode & 0o111),
                dir_fd=destination_parent_fd,
            )
            # Enforce the exact normalized mode after the process umask affected os.open.
            os.fchmod(destination_fd, 0o600 | (source_metadata.st_mode & 0o111))
            while chunk := os.read(source_fd, 1024 * 1024):
                _write_all(destination_fd, chunk)
        except OSError:
            raise UnsupportedHarborTaskError(
                "local Harbor task file could not be copied safely"
            ) from None
        if _metadata_changed(source_metadata, os.fstat(source_fd)):
            raise UnsupportedHarborTaskError(
                "local Harbor task file changed while its snapshot was copied"
            )
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def _open_verified_source_entry(
    parent_fd: int,
    name: str,
    *,
    expect_directory: bool,
    source_device: int,
) -> tuple[int, os.stat_result]:
    opened_fd: int | None = None
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        if expect_directory:
            flags |= os.O_DIRECTORY
        else:
            # A regular source swapped to a FIFO must fail after fstat instead of blocking the
            # evaluator indefinitely while open waits for a writer.
            flags |= os.O_NONBLOCK
        opened_fd = os.open(name, flags, dir_fd=parent_fd)
        opened = os.fstat(opened_fd)
    except OSError:
        if opened_fd is not None:
            os.close(opened_fd)
        raise UnsupportedHarborTaskError(
            "local Harbor task entry could not be opened without following links"
        ) from None
    expected_type = stat.S_ISDIR if expect_directory else stat.S_ISREG
    unsafe = (
        (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        or opened.st_dev != source_device
        or not expected_type(opened.st_mode)
        or (not expect_directory and opened.st_nlink != 1)
    )
    if unsafe:
        os.close(opened_fd)
        raise UnsupportedHarborTaskError(
            "local Harbor task entry changed identity or crossed its source filesystem"
        )
    try:
        _require_owned_nonwritable_by_others(opened, label="local Harbor task entry")
    except BaseException:
        os.close(opened_fd)
        raise
    return opened_fd, opened


def _require_owned_nonwritable_by_others(metadata: os.stat_result, *, label: str) -> None:
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
        raise UnsupportedHarborTaskError(
            f"{label} must be owned by the evaluator user and not group/world writable"
        )


def _require_private_owned_directory(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise UnsupportedHarborTaskError(f"{label} cannot be a symbolic link")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError:
        raise UnsupportedHarborTaskError(f"{label} metadata is unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsupportedHarborTaskError(f"{label} must be a regular directory")
    _require_owned_nonwritable_by_others(metadata, label=label)


def _write_all(destination_fd: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(destination_fd, remaining)
        if written <= 0:
            raise OSError("short write while copying Harbor task snapshot")
        remaining = remaining[written:]


def _metadata_changed(before: os.stat_result, after: os.stat_result) -> bool:
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    return any(getattr(before, field) != getattr(after, field) for field in fields)


def _task_snapshot_digest(root: Path) -> str:
    hasher = hashlib.sha256()
    _update_snapshot_digest_part(hasher, b"wmh-harbor-task-snapshot-v2")
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    except OSError:
        raise UnsupportedHarborTaskError("Harbor task snapshot cannot be enumerated") from None
    for path in paths:
        if path.is_symlink():
            raise UnsupportedHarborTaskError("Harbor task snapshot cannot contain symbolic links")
        relative = path.relative_to(root).as_posix().encode()
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise UnsupportedHarborTaskError(
                "Harbor task snapshot metadata is unavailable"
            ) from None
        if stat.S_ISDIR(metadata.st_mode):
            _update_snapshot_digest_part(hasher, b"directory")
            _update_snapshot_digest_part(hasher, relative)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsupportedHarborTaskError(
                "Harbor task snapshot can contain only regular files and directories"
            )
        _update_snapshot_digest_part(hasher, b"file")
        _update_snapshot_digest_part(hasher, relative)
        # Snapshot publication makes any source-executable file executable for all container
        # users, so identity binds that normalized semantic bit rather than host ownership bits.
        _update_snapshot_digest_part(
            hasher,
            (0o111 if metadata.st_mode & 0o111 else 0).to_bytes(2, "big"),
        )
        # File content is streamed rather than materialized, so frame it with the stable byte
        # count before hashing. Without this boundary, a file's suffix can encode the next
        # path/mode record and make two different trees share a digest without breaking SHA-256.
        _update_snapshot_digest_part(hasher, metadata.st_size.to_bytes(8, "big"))
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    hasher.update(chunk)
                final_metadata = os.fstat(handle.fileno())
        except OSError:
            raise UnsupportedHarborTaskError("Harbor task snapshot file cannot be read") from None
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(metadata, field) != getattr(final_metadata, field) for field in stable_fields
        ):
            raise UnsupportedHarborTaskError("Harbor task snapshot changed while it was hashed")
    return "sha256:" + hasher.hexdigest()


def _update_snapshot_digest_part(hasher: _DigestWriter, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def _make_task_snapshot_read_only(root: Path) -> None:
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        mode = path.stat(follow_symlinks=False).st_mode
        if path.is_dir():
            # Harbor may upload the frozen tree into a task or verifier that runs as non-root.
            # Private 0700 snapshot parents retain host isolation; the published tree itself must
            # preserve read/traverse semantics for every configured container user.
            path.chmod(0o555)
        else:
            path.chmod(0o555 if mode & 0o111 else 0o444)


def _validate_existing_dataset_snapshot(path: Path, *, expected_digest: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise UnsupportedHarborTaskError("Harbor task snapshot path is not a regular directory")
    try:
        paths = (path, *path.rglob("*"))
        for item in paths:
            metadata = item.stat(follow_symlinks=False)
            if metadata.st_uid != os.geteuid():
                raise UnsupportedHarborTaskError(
                    "Harbor task snapshot must remain owned by the evaluator user"
                )
            if metadata.st_mode & 0o222:
                raise UnsupportedHarborTaskError("Harbor task snapshot must remain read-only")
    except OSError:
        raise UnsupportedHarborTaskError(
            "Harbor task snapshot permissions are unavailable"
        ) from None
    actual = _task_snapshot_digest(path)
    if actual != expected_digest:
        raise UnsupportedHarborTaskError("Harbor task snapshot digest does not match its path")


def _remove_temporary_snapshot(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    try:
        for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif not path.is_symlink():
                path.chmod(0o600)
        root.chmod(0o700)
        shutil.rmtree(root)
    except OSError:
        # Pending roots are never used by Harbor. A later maintenance sweep may remove one that
        # survived an interruption; failure to clean it does not weaken the published snapshot.
        return


def _validate_prebuilt_local_task_tree(task_dir: Path) -> None:
    try:
        validate_prebuilt_image_task_tree(task_dir)
    except PrebuiltImageTaskBoundaryError as exc:
        raise UnsupportedHarborTaskError(
            f"local Harbor task {task_dir.name!r} violates the prebuilt-image policy: {exc}"
        ) from None


def _validate_task_tree_for_host_reads(
    task_dir: Path,
    *,
    extra_instruction_paths: tuple[Path, ...] | list[Path],
) -> None:
    """Require a regular, symlink-free task tree before trusted host reads or hashes it."""
    requested_root = task_dir.expanduser()
    if requested_root.is_symlink():
        raise UnsupportedHarborTaskError(
            f"Harbor task root cannot be a symbolic link: {requested_root}"
        )
    if not requested_root.is_dir():
        raise UnsupportedHarborTaskError(f"Harbor task root is not a directory: {requested_root}")
    for path in requested_root.rglob("*"):
        if path.is_symlink():
            raise UnsupportedHarborTaskError(
                f"Harbor task tree cannot contain a symbolic link: {path}"
            )
        if not path.is_dir() and not path.is_file():
            raise UnsupportedHarborTaskError(
                f"Harbor task tree can contain only regular files and directories: {path}"
            )

    root = requested_root.resolve()
    _require_bounded_host_text_file(root / "task.toml", label="task configuration")
    _require_bounded_host_text_file(root / "instruction.md", label="task instruction")
    if len(extra_instruction_paths) > _MAX_EXTRA_INSTRUCTION_FILES:
        raise UnsupportedHarborTaskError(
            "Harbor task has too many extra instruction files for bounded host ingestion"
        )
    for path in extra_instruction_paths:
        requested = path.expanduser()
        if requested.is_symlink():
            raise UnsupportedHarborTaskError(
                f"extra task instruction cannot be a symbolic link: {requested}"
            )
        try:
            resolved = requested.resolve(strict=True)
        except OSError as exc:
            raise UnsupportedHarborTaskError(
                f"extra task instruction is unavailable: {requested}"
            ) from exc
        if not resolved.is_relative_to(root):
            raise UnsupportedHarborTaskError(
                f"extra task instruction must remain inside the task root: {requested}"
            )
        _require_bounded_host_text_file(resolved, label="extra task instruction")


def _require_bounded_host_text_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise UnsupportedHarborTaskError(f"{label} must be a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise UnsupportedHarborTaskError(f"{label} is unavailable: {path}") from exc
    if size > _MAX_TASK_HOST_TEXT_BYTES:
        raise UnsupportedHarborTaskError(
            f"{label} exceeds the {_MAX_TASK_HOST_TEXT_BYTES}-byte host-read limit: {path}"
        )


def harbor_run_config_digest(spec: HarborJobSpec, agent_config_digest: str) -> str:
    """Hash logical run semantics and controlled concurrency without leaking host paths.

    Resolved per-task inputs live in each cell's trial-lock digest. Job name and jobs directory
    are deliberately absent because they only change storage location.
    Artifact collection is bound because its commands, transfers, and failures can change the
    observed run. Trial and per-agent concurrency remain because throttling, contention, and
    timeout behavior can change observed outcomes. Retries are bound at zero until an atomic
    per-attempt ledger exists.
    """
    payload = {
        "schema_version": 2,
        "evaluator": "wmh-harbor",
        "evaluator_version": HARBOR_EVALUATOR_VERSION,
        "harbor_version": SUPPORTED_HARBOR_VERSION,
        "agent_version": WMH_PI_AGENT_VERSION,
        "agent_config_digest": agent_config_digest,
        "task_environment": spec.environment_backend.value,
        "task_source_policy": (
            "prebuilt-image-only"
            if spec.environment_backend is HarborEnvironmentBackend.LOCAL
            else "e2b-image-or-dockerfile"
        ),
        "attempts_per_task": spec.n_attempts,
        "trial_concurrency": spec.n_concurrent_trials,
        "agent_concurrency": spec.agent_n_concurrent,
        "artifact_paths": list(spec.artifact_paths),
        "retry": {
            "max_retries": spec.max_retries,
            "exceptions": sorted(spec.retry_exceptions) if spec.max_retries else [],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _load_and_validate_tasks(
    job: Job,
    *,
    environment_backend: HarborEnvironmentBackend,
) -> dict[TaskIdType, Task]:
    tasks: dict[TaskIdType, Task] = {}
    for trial_config in job._trial_configs:
        task_id = trial_config.task.get_task_id()
        if task_id in tasks:
            continue
        try:
            download = job._task_download_results[task_id]
        except KeyError:
            raise RuntimeError(
                f"Harbor did not retain the prepared task {task_id.get_name()!r}"
            ) from None
        _validate_task_tree_for_host_reads(
            download.path,
            extra_instruction_paths=trial_config.extra_instruction_paths,
        )
        if environment_backend is HarborEnvironmentBackend.LOCAL:
            _validate_prebuilt_local_task_tree(download.path)
        task = Task(
            download.path,
            extra_instruction_paths=trial_config.extra_instruction_paths,
            disable_verification=job.config.verifier.disable,
        )
        try:
            validate_task_credential_boundary(task)
        except TaskCredentialBoundaryError as exc:
            raise UnsupportedHarborTaskError(
                f"task {trial_config.task.source!r}/{task_id.get_name()!r} crosses the "
                f"host credential boundary: {exc}"
            ) from None
        if task.has_steps:
            raise UnsupportedHarborTaskError(
                f"task {trial_config.task.source!r}/{task_id.get_name()!r} uses Harbor "
                "multi-step execution; this evaluator records one candidate trace and outcome "
                "per trial, so it refuses to publish ambiguous step evidence"
            )
        if task_has_any_separate_verifier(task.config):
            raise UnsupportedHarborTaskError(
                f"task {trial_config.task.source!r}/{task_id.get_name()!r} uses a separate "
                "verifier environment; Harbor 0.18 suppresses verifier-environment stop "
                "failures, so this evaluator refuses to produce ambiguous scores"
            )
        compose_sources = tuple(
            path
            for path in (
                task.paths.environment_dir / "docker-compose.yaml",
                task.paths.environment_dir / "docker-compose.yml",
            )
            if path.exists() or path.is_symlink()
        )
        if environment_backend is HarborEnvironmentBackend.E2B and compose_sources:
            raise UnsupportedHarborTaskError(
                f"task {trial_config.task.source!r}/{task_id.get_name()!r} uses task-authored "
                "Docker Compose; Harbor 0.18 E2B ignores Compose semantics, so this evaluator "
                "requires a Dockerfile/image-only task for E2B"
            )
        docker_image = task.config.environment.docker_image
        dockerfile = task.paths.environment_dir / "Dockerfile"
        if (
            environment_backend is HarborEnvironmentBackend.E2B
            and not (docker_image and docker_image.strip())
            and not dockerfile.is_file()
        ):
            raise UnsupportedHarborTaskError(
                f"task {trial_config.task.source!r}/{task_id.get_name()!r} has no "
                "[environment].docker_image or environment/Dockerfile; Harbor 0.18 E2B "
                "requires one immutable environment definition"
            )
        tasks[task_id] = task
    return tasks


def _prepare_trials(
    job: Job,
    job_lock: JobLock,
    tasks: dict[TaskIdType, Task],
    *,
    agent_config_digest: str,
) -> list[_PreparedTrial]:
    prepared_trials: list[_PreparedTrial] = []
    for trial_config, trial_lock in zip(
        job._trial_configs,
        job_lock.trials,
        strict=True,
    ):
        _validate_prepared_agent(trial_config, agent_config_digest)
        task_id = trial_config.task.get_task_id()
        task = tasks[task_id]
        task_identity = task_id.get_name()
        task_key = _dataset_qualified_task_key(
            source=trial_lock.task.source,
            task_identity=task_identity,
            task_checksum=trial_lock.task.digest,
        )
        prepared_trials.append(
            _PreparedTrial(
                config=trial_config,
                task_key=task_key,
                task_name=task.name,
                task_identity=task_identity,
                task_checksum=trial_lock.task.digest,
                task_source=trial_lock.task.source,
                task_instruction=_bound_task_instruction(task.instruction),
                trial_lock_digest=harbor_trial_lock_digest(trial_lock),
            )
        )
    return prepared_trials


def _bound_task_instruction(instruction: str) -> str:
    """Retain deterministic proposer evidence without growing the trusted manifest unboundedly."""
    normalized = normalize_durable_text(instruction)
    if len(normalized) <= MAX_BENCHMARK_TASK_INSTRUCTION_CHARS:
        return normalized
    marker = f"\n...[task instruction truncated; original_chars={len(normalized)}]...\n"
    retained = MAX_BENCHMARK_TASK_INSTRUCTION_CHARS - len(marker)
    head = retained // 2
    tail = retained - head
    return normalized[:head] + marker + normalized[-tail:]


def _build_manifest(
    job_name: str,
    prepared_trials: list[_PreparedTrial],
    *,
    identity: BenchmarkRunIdentity,
    agent_config_digest: str,
) -> HarborTrialManifest:
    attempts: defaultdict[str, int] = defaultdict(int)
    entries: list[HarborTrialManifestEntry] = []
    for prepared in sorted(
        prepared_trials,
        key=lambda item: (item.task_key, item.config.trial_name),
    ):
        attempts[prepared.task_key] += 1
        entries.append(
            HarborTrialManifestEntry(
                cell=BenchmarkCell(
                    task_key=prepared.task_key,
                    task_name=prepared.task_name,
                    attempt=attempts[prepared.task_key],
                    config_digest=prepared.trial_lock_digest,
                ),
                trial_name=prepared.config.trial_name,
                task_identity=prepared.task_identity,
                task_checksum=prepared.task_checksum,
                task_source=prepared.task_source,
                task_instruction=prepared.task_instruction,
                trial_lock_digest=prepared.trial_lock_digest,
            )
        )
    return HarborTrialManifest(
        job_name=job_name,
        identity=identity,
        agent_config_digest=agent_config_digest,
        entries=entries,
    )


def _restore_manifest_trial_names(
    job: Job,
    prepared_trials: list[_PreparedTrial],
    manifest: HarborTrialManifest,
) -> None:
    """Bind Harbor's regenerated remaining configs back to immutable manifest cells."""
    if len(prepared_trials) != len(manifest.entries):
        raise StaleHarborJobError(
            "Harbor resume prepared a different number of trials than the trusted manifest"
        )

    entries_by_name = {entry.trial_name: entry for entry in manifest.entries}
    remaining_config_ids = {id(config) for config in job._remaining_trial_configs}
    remaining_prepared: list[_PreparedTrial] = []
    completed_names: set[str] = set()

    for prepared in prepared_trials:
        if id(prepared.config) in remaining_config_ids:
            remaining_prepared.append(prepared)
            continue
        entry = entries_by_name.get(prepared.config.trial_name)
        if entry is None or _manifest_entry_immutable_key(entry) != prepared.immutable_key:
            raise StaleHarborJobError(
                "Harbor resume completed evidence does not match the trusted trial manifest"
            )
        completed_names.add(entry.trial_name)

    remaining_entries = [
        entry for entry in manifest.entries if entry.trial_name not in completed_names
    ]
    prepared_by_key: defaultdict[
        tuple[str, str, str, str, str | None, str, str], list[_PreparedTrial]
    ] = defaultdict(list)
    entries_by_key: defaultdict[
        tuple[str, str, str, str, str | None, str, str], list[HarborTrialManifestEntry]
    ] = defaultdict(list)
    for prepared in remaining_prepared:
        prepared_by_key[prepared.immutable_key].append(prepared)
    for entry in remaining_entries:
        entries_by_key[_manifest_entry_immutable_key(entry)].append(entry)

    if set(prepared_by_key) != set(entries_by_key):
        raise StaleHarborJobError("Harbor resume task identities or resolved trial locks changed")

    assignments: list[tuple[_PreparedTrial, HarborTrialManifestEntry]] = []
    for immutable_key, prepared_group in prepared_by_key.items():
        entry_group = entries_by_key[immutable_key]
        if len(prepared_group) != len(entry_group):
            raise StaleHarborJobError(
                "Harbor resume prepared a different number of attempts for a trusted task"
            )
        assignments.extend(
            zip(
                sorted(prepared_group, key=lambda item: item.config.trial_name),
                sorted(entry_group, key=lambda item: (item.cell.attempt, item.trial_name)),
                strict=True,
            )
        )

    restored_names = completed_names | {entry.trial_name for _, entry in assignments}
    if restored_names != set(entries_by_name):
        raise StaleHarborJobError("Harbor resume could not restore every trusted trial name")

    for prepared, entry in assignments:
        prepared.config.trial_name = entry.trial_name


def _manifest_entry_immutable_key(
    entry: HarborTrialManifestEntry,
) -> tuple[str, str, str, str, str | None, str, str]:
    return (
        entry.cell.task_key,
        entry.cell.task_name,
        entry.task_identity,
        entry.task_checksum,
        entry.task_source,
        entry.task_instruction,
        entry.trial_lock_digest,
    )


def _validate_existing_job_lock(job_dir: Path, expected: JobLock) -> None:
    lock_path = job_dir / "lock.json"
    try:
        existing = JobLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StaleHarborJobError(f"Harbor job lock is unreadable or invalid: {lock_path}") from exc
    if existing != expected:
        raise StaleHarborJobError(
            f"Harbor job lock does not match the newly resolved task inputs: {lock_path}"
        )


def _validate_prepared_agent(trial_config: TrialConfig, expected_digest: str) -> None:
    agent = trial_config.agent
    if agent.import_path != WmhPiAgent.import_path():
        raise RuntimeError(f"Harbor changed the pinned agent import path to {agent.import_path!r}")
    actual_digest = harbor_agent_config_digest(agent)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Harbor changed the pinned agent configuration: {actual_digest} != {expected_digest}"
        )


def _dataset_qualified_task_key(
    *,
    source: str | None,
    task_identity: str,
    task_checksum: str,
) -> str:
    payload = {
        "source": source,
        "task_identity": task_identity,
        "task_checksum": task_checksum,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _inspect_existing_job(
    job_dir: Path,
    *,
    expected_identity: BenchmarkRunIdentity,
    expected_agent_digest: str,
    expected_job_config: JobConfig,
) -> HarborTrialManifest | None:
    if job_dir.is_symlink():
        raise StaleHarborJobError(f"Harbor job directory cannot be a symlink: {job_dir}")
    if not job_dir.exists():
        return None
    if not job_dir.is_dir():
        raise StaleHarborJobError(f"Harbor job path is not a directory: {job_dir}")

    manifest_path = job_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise StaleHarborJobError(
            f"Harbor job directory already exists without {_MANIFEST_FILENAME}: {job_dir}"
        )
    manifest = _read_manifest(manifest_path)
    if (
        manifest.job_name != job_dir.name
        or manifest.identity != expected_identity
        or manifest.agent_config_digest != expected_agent_digest
    ):
        raise StaleHarborJobError(
            f"Harbor job directory manifest does not match this evaluation: {job_dir}"
        )
    reserved_trial_names = sorted(
        entry.trial_name for entry in manifest.entries if entry.trial_name in _JOB_ROOT_FILES
    )
    if reserved_trial_names:
        raise StaleHarborJobError(
            f"Harbor manifest trial names collide with job evidence: {reserved_trial_names}"
        )

    required_files = ("config.json", "lock.json", "result.json")
    unsafe_root_files = sorted(
        name
        for name in required_files
        if (job_dir / name).is_symlink() or not (job_dir / name).is_file()
    )
    if unsafe_root_files:
        raise StaleHarborJobError(
            "Harbor job directory is incomplete or unsafe; required regular files "
            f"{unsafe_root_files}: {job_dir}"
        )
    entries_by_name = {entry.trial_name: entry for entry in manifest.entries}
    trial_dirs: list[tuple[Path, HarborTrialManifestEntry]] = []
    for child in job_dir.iterdir():
        if child.is_symlink():
            raise StaleHarborJobError(f"Harbor job directory contains a symlink: {child}")
        if child.is_dir():
            entry = entries_by_name.get(child.name)
            if entry is None:
                raise StaleHarborJobError(
                    f"Harbor job directory contains an unplanned trial: {child}"
                )
            _validate_existing_trial_layout(child)
            trial_dirs.append((child, entry))
        elif child.name not in _JOB_ROOT_FILES:
            raise StaleHarborJobError(f"Harbor job directory contains an unexpected file: {child}")

    result_path = job_dir / "result.json"
    try:
        existing_job_config = JobConfig.model_validate_json(
            (job_dir / "config.json").read_text(encoding="utf-8")
        )
        job_result = JobResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        JobLock.model_validate_json((job_dir / "lock.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StaleHarborJobError(
            f"Harbor job config, result, or lock is unreadable or invalid: {job_dir}"
        ) from exc
    # Compare Python-mode values so semantically unordered Harbor sets (notably retry exception
    # names) do not become order-sensitive merely because JSON represents them as arrays.
    if existing_job_config.model_dump() != expected_job_config.model_dump():
        raise StaleHarborJobError(f"Harbor job config does not match this evaluation: {job_dir}")

    for trial_dir, entry in trial_dirs:
        trial_result_path = trial_dir / "result.json"
        if trial_result_path.exists():
            try:
                TrialResult.model_validate_json(trial_result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValidationError) as exc:
                raise StaleHarborJobError(
                    f"completed Harbor trial result is unreadable or invalid: {trial_dir}"
                ) from exc
        else:
            _validate_incomplete_trial_evidence(
                job_dir,
                trial_dir,
                entry,
                job_id=job_result.id,
                expected_agent_digest=expected_agent_digest,
            )

    try:
        load_harbor_job_result(job_dir, manifest)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise StaleHarborJobError(
            f"Harbor job evidence does not match the trusted manifest: {job_dir}"
        ) from exc
    return manifest


def _validate_existing_trial_layout(trial_dir: Path) -> None:
    evidence_paths = {
        name: trial_dir / name for name in ("config.json", "result.json", "lock.json")
    }
    if any(path.is_symlink() for path in evidence_paths.values()):
        raise StaleHarborJobError(
            f"Harbor job directory contains an incomplete or unsafe trial directory: {trial_dir}"
        )

    result_exists = evidence_paths["result.json"].exists()
    required = ("config.json", "result.json", "lock.json") if result_exists else ("lock.json",)
    if any(not evidence_paths[name].is_file() for name in required):
        raise StaleHarborJobError(
            f"Harbor job directory contains an incomplete or unsafe trial directory: {trial_dir}"
        )
    config_path = evidence_paths["config.json"]
    if config_path.exists() and not config_path.is_file():
        raise StaleHarborJobError(
            f"Harbor job directory contains an incomplete or unsafe trial directory: {trial_dir}"
        )


def _validate_incomplete_trial_evidence(
    job_dir: Path,
    trial_dir: Path,
    entry: HarborTrialManifestEntry,
    *,
    job_id: UUID,
    expected_agent_digest: str,
) -> None:
    lock_path = trial_dir / "lock.json"
    try:
        lock = TrialLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StaleHarborJobError(
            f"incomplete Harbor trial has an unreadable or invalid lock: {trial_dir}"
        ) from exc
    if (
        harbor_trial_lock_digest(lock) != entry.trial_lock_digest
        or lock.task.digest != entry.task_checksum
        or lock.task.source != entry.task_source
        or harbor_agent_config_digest(lock.agent) != expected_agent_digest
    ):
        raise StaleHarborJobError(
            f"incomplete Harbor trial lock does not match its trusted cell: {trial_dir}"
        )

    expected_task_key = _dataset_qualified_task_key(
        source=entry.task_source,
        task_identity=entry.task_identity,
        task_checksum=entry.task_checksum,
    )
    if expected_task_key != entry.cell.task_key:
        raise StaleHarborJobError(
            f"incomplete Harbor trial manifest cell is internally inconsistent: {trial_dir}"
        )

    config_path = trial_dir / "config.json"
    if not config_path.exists():
        return
    try:
        config = TrialConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        task_identity = config.task.get_task_id().get_name()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise StaleHarborJobError(
            f"incomplete Harbor trial has an unreadable or invalid config: {trial_dir}"
        ) from exc

    mismatched_locked_fields = [
        name
        for name, actual, expected in (
            ("install_only", config.install_only, lock.install_only),
            ("timeout_multiplier", config.timeout_multiplier, lock.timeout_multiplier),
            (
                "agent_timeout_multiplier",
                config.agent_timeout_multiplier,
                lock.agent_timeout_multiplier,
            ),
            (
                "verifier_timeout_multiplier",
                config.verifier_timeout_multiplier,
                lock.verifier_timeout_multiplier,
            ),
            (
                "agent_setup_timeout_multiplier",
                config.agent_setup_timeout_multiplier,
                lock.agent_setup_timeout_multiplier,
            ),
            (
                "environment_build_timeout_multiplier",
                config.environment_build_timeout_multiplier,
                lock.environment_build_timeout_multiplier,
            ),
            ("agent", config.agent, lock.agent),
            ("environment", config.environment, lock.environment),
            ("verifier", config.verifier, lock.verifier),
        )
        if actual != expected
    ]
    if (
        config.trial_name != entry.trial_name
        or task_identity != entry.task_identity
        or config.task.source != entry.task_source
        or config.trials_dir.expanduser().resolve() != job_dir.resolve()
        or config.job_id != job_id
        or mismatched_locked_fields
    ):
        raise StaleHarborJobError(
            f"incomplete Harbor trial config does not match its trusted lock: {trial_dir}"
        )


def _persist_or_compare_manifest(
    job_dir: Path,
    manifest: HarborTrialManifest,
    existing: HarborTrialManifest | None,
) -> None:
    manifest_path = job_dir / _MANIFEST_FILENAME
    if existing is not None:
        if existing != manifest:
            raise StaleHarborJobError(
                f"Harbor job directory resolved to a different trial manifest: {job_dir}"
            )
        return

    payload = manifest.model_dump_json(indent=2) + "\n"
    try:
        descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raced = _read_manifest(manifest_path)
        if raced != manifest:
            raise StaleHarborJobError(
                f"Harbor manifest was concurrently created with different contents: {job_dir}"
            ) from None
        return
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        raise


def _read_manifest(path: Path) -> HarborTrialManifest:
    if path.is_symlink() or not path.is_file():
        raise StaleHarborJobError(f"Harbor manifest must be a regular file: {path}")
    try:
        return HarborTrialManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as exc:
        raise StaleHarborJobError(f"Harbor manifest is unreadable or invalid: {path}") from exc


def _atomic_replace_job_result(path: Path, payload: str) -> None:
    """Replace a root job result without leaving crash debris inside the job directory."""
    job_dir = path.parent
    jobs_dir = job_dir.parent
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{job_dir.name}.{path.name}-",
        dir=jobs_dir,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _exclusive_job_run_lock(
    jobs_dir: Path,
    job_name: str,
) -> AbstractContextManager[None]:
    """Hold a crash-safe nonblocking lease that prevents duplicate paid job execution."""
    lock_path = harbor_job_lease_path(jobs_dir, job_name)
    return exclusive_posix_file_lease(
        lock_path,
        unsupported_error=RuntimeError(
            "Harbor evaluation job leases currently require POSIX file locking"
        ),
        irregular_file_error=OSError(f"Harbor evaluation lock is not a regular file: {lock_path}"),
        contention_error=ConcurrentHarborJobError(
            f"another process is already evaluating Harbor job {job_name!r} in {jobs_dir}"
        ),
    )


def harbor_job_lease_path(jobs_dir: Path, job_name: str) -> Path:
    """Return the canonical sibling lease path for one Harbor job name."""
    _validate_job_name(job_name)
    return jobs_dir.expanduser().resolve() / f".{job_name}.wmh-eval.lock"


def _validate_job_name(job_name: str) -> None:
    if (
        job_name in {".", "..", _TASK_SNAPSHOT_ROOT}
        or "\0" in job_name
        or Path(job_name).is_absolute()
        or "/" in job_name
        or "\\" in job_name
    ):
        raise ValueError("job_name must be a single safe path component")
