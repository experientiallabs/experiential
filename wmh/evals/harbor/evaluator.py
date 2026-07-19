"""Reusable Harbor 0.18 evaluator for ground-truth WMH benchmark runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import override
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
from pydantic import ValidationError

from wmh.evals.benchmark import (
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
    TaskCredentialBoundaryError,
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
_JOB_ROOT_FILES = frozenset(
    {_MANIFEST_FILENAME, "config.json", "job.log", "lock.json", "result.json"}
)


class StaleHarborJobError(RuntimeError):
    """A pre-existing Harbor job directory cannot be proved to match this evaluation."""


class ConcurrentHarborJobError(RuntimeError):
    """Another process already holds the exclusive execution lease for this Harbor job."""


class UnsupportedHarborTaskError(ValueError):
    """A task uses Harbor behavior whose failures cannot yet be observed safely."""


class UnsupportedHarborMetricError(ValueError):
    """A Harbor metric could execute untrusted code on the credential-bearing host."""


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
    trial_lock_digest: str

    @property
    def immutable_key(self) -> tuple[str, str, str, str, str | None, str]:
        return (
            self.task_key,
            self.task_name,
            self.task_identity,
            self.task_checksum,
            self.task_source,
            self.trial_lock_digest,
        )


class HarborEvaluator:
    """Evaluate one immutable harness candidate over a fixed Harbor task matrix."""

    def __init__(
        self,
        spec: HarborJobSpec,
        provider_config: ProviderConfig,
        *,
        runner_image: str = PI_CONTAINER_IMAGE,
        turn_timeout_s: float = 300.0,
    ) -> None:
        validate_pi_container_image(runner_image)
        if not math.isfinite(turn_timeout_s) or turn_timeout_s <= 0:
            raise ValueError("turn_timeout_s must be finite and positive")
        validated_spec = HarborJobSpec.model_validate(spec.model_dump())
        _validate_job_name(validated_spec.job_name)
        self._spec = validated_spec.model_copy(
            update={"jobs_dir": validated_spec.jobs_dir.expanduser().resolve()}, deep=True
        )
        self._provider_config = provider_config.model_copy(deep=True)
        self._runner_image = runner_image
        self._turn_timeout_s = turn_timeout_s
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
            job_lock = _build_prepared_job_lock(job)
            tasks = _load_and_validate_tasks(
                job,
                environment_backend=self._spec.environment_backend,
            )
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
        if self._runner_ready:
            return
        await asyncio.to_thread(
            verify_container_pi_runner_ready,
            image=self._runner_image,
        )
        self._runner_ready = True

    def _build_agent(self, candidate: HarnessDoc) -> AgentConfig:
        model_name = f"{self._provider_config.kind.value}/{self._provider_config.model}"
        return AgentConfig(
            import_path=WmhPiAgent.import_path(),
            model_name=model_name,
            n_concurrent=self._spec.agent_n_concurrent,
            kwargs={
                "harness": candidate.model_dump(mode="json"),
                "provider_config": self._provider_config.model_dump(mode="json"),
                "runner_image": self._runner_image,
                "turn_timeout_s": self._turn_timeout_s,
            },
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
        "schema_version": 1,
        "evaluator": "wmh-harbor",
        "harbor_version": SUPPORTED_HARBOR_VERSION,
        "agent_version": WMH_PI_AGENT_VERSION,
        "agent_config_digest": agent_config_digest,
        "task_environment": spec.environment_backend.value,
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
                trial_lock_digest=harbor_trial_lock_digest(trial_lock),
            )
        )
    return prepared_trials


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
        tuple[str, str, str, str, str | None, str], list[_PreparedTrial]
    ] = defaultdict(list)
    entries_by_key: defaultdict[
        tuple[str, str, str, str, str | None, str], list[HarborTrialManifestEntry]
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
) -> tuple[str, str, str, str, str | None, str]:
    return (
        entry.cell.task_key,
        entry.cell.task_name,
        entry.task_identity,
        entry.task_checksum,
        entry.task_source,
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
    if existing_job_config.model_dump(mode="json") != expected_job_config.model_dump(mode="json"):
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
        job_name in {".", ".."}
        or "\0" in job_name
        or Path(job_name).is_absolute()
        or "/" in job_name
        or "\\" in job_name
    ):
        raise ValueError("job_name must be a single safe path component")
