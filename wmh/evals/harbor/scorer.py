"""Strict Harbor job projection into immutable WMH harness scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol, Self, TypeVar
from uuid import UUID, uuid4

import harbor
from harbor import Job
from harbor.models.agent.name import AgentName
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.lock import JobLock, TrialLock
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult
from pydantic import BaseModel

from wmh.evals.harbor.agent import (
    WMH_HARBOR_AGENT_IMPORT_PATH,
    WMH_HARBOR_AGENT_VERSION,
    WmhHarborAgent,
)
from wmh.evals.harbor.tasks import (
    HarborTaskIdentity,
    ResolvedHarborTaskSet,
    resolve_harbor_task_set,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import resolve_e2b_template
from wmh.harness.scoring import (
    ArtifactReader,
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.providers.base import ProviderConfig

_INDEX_PATH = "raw/index.json"
_REQUIRED_JOB_FILES = frozenset({"config.json", "lock.json", "result.json", "job.log"})
_REQUIRED_TRIAL_FILES = frozenset({"config.json", "lock.json", "result.json", "trial.log"})
_HARBOR_SCORER_VERSION = "1"
_CHECKSUM_PATTERN = r"^[0-9a-f]{64}$"
_ModelT = TypeVar("_ModelT", bound=BaseModel)


@dataclass(frozen=True)
class HarborRun:
    """One completed official Harbor job and its raw output directory."""

    result: JobResult
    job_dir: Path


class HarborRunner(Protocol):
    """Synchronous execution seam for an official Harbor job."""

    def run(self, config: JobConfig) -> HarborRun: ...


class HarborJobRunner:
    """Run Harbor's async Python API from synchronous optimizer code."""

    def run(self, config: JobConfig) -> HarborRun:
        async def run_job() -> HarborRun:
            job = await Job.create(config)
            result = await job.run()
            return HarborRun(result=result, job_dir=job.job_dir)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_job())
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(run_job())).result()


class HarborScorer:
    """Evaluate exact harness candidates through Harbor's official verifier lifecycle."""

    def __init__(
        self,
        *,
        job_config: JobConfig,
        task_set: ResolvedHarborTaskSet,
        provider_config: ProviderConfig,
        reward_key: str,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        runner: HarborRunner | None = None,
    ) -> None:
        if len(job_config.datasets) != 1 or job_config.tasks:
            raise ValueError("HarborScorer requires exactly one dataset and no direct tasks")
        if len(job_config.agents) != 1:
            raise ValueError("HarborScorer requires exactly one agent template")
        if job_config.retry.max_retries != 0:
            raise ValueError("HarborScorer does not support hidden Harbor retries")
        if job_config.install_only:
            raise ValueError("HarborScorer cannot use an install-only Harbor job")
        if job_config.verifier.disable:
            raise ValueError("HarborScorer requires Harbor verification")
        dataset = job_config.datasets[0]
        if any(
            value is not None
            for value in (dataset.task_names, dataset.exclude_task_names, dataset.n_tasks)
        ):
            raise ValueError("HarborScorer does not accept preconfigured dataset filters")
        if job_config.extra_instruction_paths:
            raise ValueError("HarborScorer does not support unhashed extra instruction files")
        if job_config.environment.extra_docker_compose:
            raise ValueError("HarborScorer does not support unhashed extra compose files")
        template = job_config.agents[0]
        if any(
            (
                template.name not in (None, AgentName.ORACLE.value),
                template.import_path is not None,
                template.model_name is not None,
                bool(template.skills),
                bool(template.env),
                bool(template.mcp_servers),
                bool(template.kwargs),
            )
        ):
            raise ValueError(
                "HarborScorer owns agent identity, model, skills, environment, and kwargs"
            )
        if not reward_key:
            raise ValueError("reward_key must be nonempty")
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        effective_agent_concurrency = template.n_concurrent or job_config.n_concurrent_trials
        if harness_backend == "local" and effective_agent_concurrency > 1:
            raise ValueError(
                "local harness execution requires agent concurrency 1; use E2B for parallel runs"
            )
        _validate_dataset_origin(
            job_config.datasets[0],
            task_set.requested_dataset_config(),
        )
        resolved_dataset = task_set.resolved_dataset_config()
        self._job_config = JobConfig.model_validate(
            job_config.model_copy(
                update={"datasets": [resolved_dataset], "tasks": []},
                deep=True,
            ).model_dump(mode="python")
        )
        self._task_set = task_set
        self._provider_config = ProviderConfig.model_validate(
            provider_config.model_dump(mode="python")
        )
        self._reward_key = reward_key
        self._harness_backend = harness_backend
        self._pi_transport = "ssh" if harness_backend == "local" else None
        if harness_backend == "local":
            from wmh.harness.pi_runtime import PI_RUNNER_DIR, PI_RUNNER_HOST

            self._local_runner_identity = {
                "transport": self._pi_transport,
                "host": PI_RUNNER_HOST,
                "runner_dir": PI_RUNNER_DIR,
            }
        else:
            self._local_runner_identity = None
        if harness_backend == "e2b":
            effective_template = resolve_e2b_template(e2b_template)
            self._e2b_template = effective_template if effective_template is not None else ""
        else:
            self._e2b_template = None
        self._runner = runner or HarborJobRunner()

    @classmethod
    async def create(
        cls,
        *,
        job_config: JobConfig,
        task_ids: tuple[str, ...],
        provider_config: ProviderConfig,
        reward_key: str,
        harness_backend: Literal["local", "e2b"] = "local",
        e2b_template: str | None = None,
        runner: HarborRunner | None = None,
    ) -> Self:
        """Resolve exact task bytes before constructing a scorer that can incur spend."""
        if len(job_config.datasets) != 1 or job_config.tasks:
            raise ValueError("HarborScorer requires exactly one dataset and no direct tasks")
        task_set = await resolve_harbor_task_set(job_config.datasets[0], task_ids)
        return cls(
            job_config=job_config,
            task_set=task_set,
            provider_config=provider_config,
            reward_key=reward_key,
            harness_backend=harness_backend,
            e2b_template=e2b_template,
            runner=runner,
        )

    def context(self) -> ScoreContext:
        """Build the frozen context expected from exact resolved Harbor task identities."""
        return self._context(self._task_set.identities)

    def request(self, *, attempts: int) -> ScoreRequest:
        """Build the only exact score request accepted by this resolved scorer."""
        return ScoreRequest(
            context=self.context(),
            task_ids=self._task_set.task_ids,
            attempts=attempts,
        )

    def _context(self, task_identities: Mapping[str, HarborTaskIdentity]) -> ScoreContext:
        if not task_identities:
            raise ValueError("task_identities must be nonempty")
        identities = {
            task_id: HarborTaskIdentity.model_validate(identity)
            for task_id, identity in task_identities.items()
        }
        dataset = self._job_config.datasets[0].model_dump(
            mode="python",
            exclude={"task_names", "exclude_task_names", "n_tasks"},
        )
        task_payload = {
            "dataset": dataset,
            "tasks": [
                {"task_id": task_id, **identity.model_dump(mode="json")}
                for task_id, identity in sorted(identities.items())
            ],
        }
        evaluator_payload = {
            "harbor_version": harbor.__version__,
            "adapter": {
                "agent_version": WMH_HARBOR_AGENT_VERSION,
                "scorer_version": _HARBOR_SCORER_VERSION,
            },
            "reward_key": self._reward_key,
            "verifier": self._job_config.verifier.model_dump(mode="python"),
        }
        execution_payload = {
            "job": self._job_config.model_dump(
                mode="python",
                exclude={
                    "job_name",
                    "jobs_dir",
                    "n_attempts",
                    "datasets",
                    "tasks",
                    "agents",
                    "verifier",
                    "quiet",
                    "debug",
                },
            ),
            "agent_template": self._job_config.agents[0].model_dump(
                mode="python",
                exclude={"name", "import_path", "model_name", "kwargs", "env", "skills"},
            ),
            "provider": self._provider_config.model_dump(mode="python"),
            "harness_backend": self._harness_backend,
            "e2b_template": self._e2b_template,
            "local_runner": self._local_runner_identity,
        }
        return ScoreContext(
            task_set_digest=_digest_json(task_payload),
            evaluator_digest=_digest_json(evaluator_payload),
            execution_config_digest=_digest_json(execution_payload),
        )

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
        """Run and project one candidate, raising unless every requested cell is scoreable."""
        if request != self.request(attempts=request.attempts):
            raise ValueError("score request differs from the scorer's resolved task set")
        self._task_set.verify()
        config = self._candidate_job(candidate, request=request)
        run = self._runner.run(config)
        grouped = self._validate_run(
            run,
            candidate=candidate,
            request=request,
            expected_config=config,
        )
        manifests, reader, source_paths, observed_identities = _collect_artifacts(
            run,
            expected_config=config,
            grouped=grouped,
        )
        if self._context(observed_identities) != request.context:
            raise ValueError("Harbor resolved task or execution context differs from the request")
        trial_names = {trial.trial_name for trials in grouped.values() for trial in trials}
        shared_paths = {
            artifact_path
            for source_path, artifact_path in source_paths.items()
            if source_path.split("/", 1)[0] not in trial_names
        }
        cells: list[ScoreCell] = []
        for task_id in request.task_ids:
            for attempt, trial in enumerate(grouped[task_id], 1):
                score = _official_reward(trial, reward_key=self._reward_key)
                trial_prefix = f"{trial.trial_name}/"
                cell_paths = {
                    _INDEX_PATH,
                    *shared_paths,
                    *(
                        artifact_path
                        for source_path, artifact_path in source_paths.items()
                        if source_path.startswith(trial_prefix)
                    ),
                }
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        score=score,
                        passed=score == 1.0,
                        summary=_trial_summary(trial),
                        artifact_paths=tuple(sorted(cell_paths)),
                    )
                )
        report = HarnessScoreReport(
            source_run_id=str(run.result.id),
            candidate_doc_hash=candidate.doc_hash,
            request=request,
            cells=tuple(cells),
            artifacts=tuple(manifests),
        )
        return HarnessScore(report=report, artifacts=reader)

    def _candidate_job(self, candidate: HarnessDoc, *, request: ScoreRequest) -> JobConfig:
        template = self._job_config.agents[0]
        direct_tasks = [
            task.model_copy(update={"source": None}, deep=True)
            for task in self._task_set.task_configs()
        ]
        agent = template.model_copy(
            update={
                "name": None,
                "import_path": WMH_HARBOR_AGENT_IMPORT_PATH,
                "model_name": f"{self._provider_config.kind.value}/{self._provider_config.model}",
                "skills": [],
                "env": {},
                "mcp_servers": [],
                "kwargs": {
                    "harness": candidate.model_dump(mode="json"),
                    "provider_config": self._provider_config.model_dump(mode="json"),
                    "harness_backend": self._harness_backend,
                    "e2b_template": self._e2b_template,
                    "pi_transport": self._pi_transport,
                },
            },
            deep=True,
        )
        return JobConfig.model_validate(
            self._job_config.model_copy(
                update={
                    "job_name": f"wmh-{candidate.doc_hash[:12]}-{uuid4().hex[:12]}",
                    "n_attempts": request.attempts,
                    "quiet": True,
                    "retry": RetryConfig(max_retries=0),
                    "datasets": [],
                    "tasks": direct_tasks,
                    "agents": [agent],
                },
                deep=True,
            ).model_dump(mode="python")
        )

    def _validate_run(
        self,
        run: HarborRun,
        *,
        candidate: HarnessDoc,
        request: ScoreRequest,
        expected_config: JobConfig,
    ) -> dict[str, list[TrialResult]]:
        result = run.result
        expected_count = len(request.task_ids) * request.attempts
        if result.finished_at is None:
            raise ValueError("Harbor job did not finish")
        if result.stats.n_retries != 0:
            raise ValueError("Harbor job used retries")
        if result.n_total_trials != expected_count or len(result.trial_results) != expected_count:
            raise ValueError(
                "Harbor trial count does not match the requested matrix: "
                f"expected {expected_count}, declared {result.n_total_trials}, "
                f"returned {len(result.trial_results)}"
            )
        names = [trial.trial_name for trial in result.trial_results]
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise ValueError(f"Harbor returned duplicate trial names: {duplicates}")
        grouped: defaultdict[str, list[TrialResult]] = defaultdict(list)
        checksums: dict[str, str] = {}
        for trial in result.trial_results:
            task_id = trial.task_id.get_name()
            if task_id not in request.task_ids:
                raise ValueError(f"Harbor returned an unexpected task {task_id!r}")
            if trial.finished_at is None:
                raise ValueError(f"Harbor trial {trial.trial_name!r} did not finish")
            if trial.step_results is not None:
                raise ValueError("HarborScorer currently supports single-step trials only")
            if trial.verifier is None or trial.verifier.finished_at is None:
                raise ValueError(f"Harbor trial {trial.trial_name!r} did not finish verification")
            prior_checksum = checksums.setdefault(task_id, trial.task_checksum)
            if trial.task_checksum != prior_checksum:
                raise ValueError(f"Harbor returned inconsistent checksums for task {task_id!r}")
            _validate_candidate_trial(
                trial,
                candidate=candidate,
                provider_config=self._provider_config,
                harness_backend=self._harness_backend,
                e2b_template=self._e2b_template,
                pi_transport=self._pi_transport,
                expected_config=expected_config,
                job_id=result.id,
                job_dir=run.job_dir,
            )
            _official_reward(trial, reward_key=self._reward_key)
            grouped[task_id].append(trial)
        missing_tasks = sorted(set(request.task_ids) - set(grouped))
        wrong_counts = {
            task_id: len(grouped[task_id])
            for task_id in request.task_ids
            if len(grouped[task_id]) != request.attempts
        }
        if missing_tasks or wrong_counts:
            raise ValueError(
                f"Harbor task matrix is incomplete: missing={missing_tasks}, counts={wrong_counts}"
            )
        return {
            task_id: sorted(grouped[task_id], key=lambda trial: trial.trial_name)
            for task_id in request.task_ids
        }


class _HarborArtifactReader:
    def __init__(self, sources: Mapping[str, bytes | Path]) -> None:
        self._sources = dict(sources)

    def read_bytes(self, path: str) -> bytes:
        source = self._sources[path]
        return source if isinstance(source, bytes) else source.read_bytes()


def _collect_artifacts(
    run: HarborRun,
    *,
    expected_config: JobConfig,
    grouped: Mapping[str, list[TrialResult]],
) -> tuple[
    list[EvaluationArtifact],
    ArtifactReader,
    dict[str, str],
    dict[str, HarborTaskIdentity],
]:
    job_dir = run.job_dir
    expected_job_dir = expected_config.jobs_dir / expected_config.job_name
    if job_dir.resolve() != expected_job_dir.resolve():
        raise ValueError("Harbor runner returned a different job directory than configured")
    if job_dir.is_symlink():
        raise ValueError("Harbor job directory cannot be a symlink")
    root = job_dir.resolve()
    if not root.is_dir():
        raise ValueError("Harbor job directory does not exist")
    expected_trials = {trial.trial_name: trial for trials in grouped.values() for trial in trials}
    for trial_name in expected_trials:
        _validate_trial_name(trial_name)
    files = _regular_files(root, set(expected_trials))
    relative_files = {path.relative_to(root).as_posix(): path for path in files}
    missing_job = sorted(_REQUIRED_JOB_FILES - relative_files.keys())
    if missing_job:
        raise ValueError(f"Harbor job evidence is missing required files: {missing_job}")
    for trial_name in expected_trials:
        missing = sorted(
            name for name in _REQUIRED_TRIAL_FILES if f"{trial_name}/{name}" not in relative_files
        )
        if missing:
            raise ValueError(f"Harbor trial {trial_name!r} is missing required files: {missing}")

    disk_config = _read_model(relative_files["config.json"], JobConfig)
    if _canonical_model(disk_config) != _canonical_model(expected_config):
        raise ValueError("Harbor job config evidence differs from the executed config")
    _read_model(relative_files["result.json"], JobResult)
    disk_result_json = _read_json(relative_files["result.json"])
    returned_result_json = json.loads(run.result.model_dump_json(exclude={"trial_results"}))
    if disk_result_json != returned_result_json:
        raise ValueError("Harbor job result evidence differs from the returned result")
    job_lock = _read_model(relative_files["lock.json"], JobLock)
    if job_lock.schema_version != 2:
        raise ValueError("Harbor job lock has an unsupported schema version")
    if job_lock.n_concurrent_trials != expected_config.n_concurrent_trials:
        raise ValueError("Harbor job lock has different concurrency")
    if _canonical_model(job_lock.retry) != _canonical_model(expected_config.retry):
        raise ValueError("Harbor job lock has a different retry policy")
    if job_lock.harbor.version is not None and job_lock.harbor.version != harbor.__version__:
        raise ValueError("Harbor job lock was produced by a different Harbor version")

    trial_locks: dict[str, TrialLock] = {}
    identities: defaultdict[str, set[HarborTaskIdentity]] = defaultdict(set)
    for trial_name, returned_trial in expected_trials.items():
        prefix = f"{trial_name}/"
        _read_model(relative_files[prefix + "result.json"], TrialResult)
        if _read_json(relative_files[prefix + "result.json"]) != json.loads(
            returned_trial.model_dump_json()
        ):
            raise ValueError(f"Harbor trial {trial_name!r} result evidence differs")
        disk_trial_config = _read_model(relative_files[prefix + "config.json"], TrialConfig)
        if _canonical_model(disk_trial_config) != _canonical_model(returned_trial.config):
            raise ValueError(f"Harbor trial {trial_name!r} config evidence differs")
        trial_lock = _read_model(relative_files[prefix + "lock.json"], TrialLock)
        _validate_trial_lock(trial_lock, returned_trial)
        trial_locks[trial_name] = trial_lock
        if re.fullmatch(_CHECKSUM_PATTERN, returned_trial.task_checksum) is None:
            raise ValueError(f"Harbor trial {trial_name!r} has an invalid task checksum")
        identities[returned_trial.task_id.get_name()].add(
            HarborTaskIdentity(
                trial_checksum=returned_trial.task_checksum,
                lock_digest=trial_lock.task.digest,
            )
        )

    expected_lock_records = Counter(_canonical_model(lock) for lock in trial_locks.values())
    observed_lock_records = Counter(_canonical_model(lock) for lock in job_lock.trials)
    if observed_lock_records != expected_lock_records:
        raise ValueError("Harbor job lock does not match the completed trial locks")
    inconsistent = sorted(task_id for task_id, values in identities.items() if len(values) != 1)
    if inconsistent:
        raise ValueError(f"Harbor returned inconsistent task identities: {inconsistent}")
    observed_identities = {task_id: next(iter(values)) for task_id, values in identities.items()}

    manifests: list[EvaluationArtifact] = []
    sources: dict[str, bytes | Path] = {}
    source_paths: dict[str, str] = {}
    index_entries: list[dict[str, object]] = []
    for index, (source_path, path) in enumerate(sorted(relative_files.items()), 1):
        artifact_path = f"raw/{index:06d}"
        content = path.read_bytes()
        artifact = EvaluationArtifact.from_bytes(path=artifact_path, content=content)
        manifests.append(artifact)
        sources[artifact_path] = path
        source_paths[source_path] = artifact_path
        index_entries.append(
            {
                "source_path": source_path,
                "artifact_path": artifact_path,
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
            }
        )
    index_content = json.dumps(
        {"schema_version": 1, "files": index_entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifests.append(
        EvaluationArtifact.from_bytes(
            path=_INDEX_PATH,
            content=index_content,
            media_type="application/json",
        )
    )
    sources[_INDEX_PATH] = index_content
    return manifests, _HarborArtifactReader(sources), source_paths, observed_identities


def _regular_files(root: Path, trial_names: set[str]) -> list[Path]:
    files: list[Path] = []

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda path: path.name):
            if child.is_symlink():
                raise ValueError(f"Harbor evidence contains a symlink: {child.relative_to(root)}")
            if child.is_dir():
                visit(child)
            elif child.is_file():
                files.append(child)
            else:
                raise ValueError(
                    f"Harbor evidence contains a non-regular file: {child.relative_to(root)}"
                )

    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if child.is_symlink():
            raise ValueError(f"Harbor evidence contains a symlink: {child.relative_to(root)}")
        if child.is_dir():
            if child.name not in trial_names:
                raise ValueError(f"Harbor evidence contains an unexpected directory: {child.name}")
            visit(child)
        elif child.is_file():
            files.append(child)
        else:
            raise ValueError(
                f"Harbor evidence contains a non-regular file: {child.relative_to(root)}"
            )
    missing_dirs = sorted(name for name in trial_names if not (root / name).is_dir())
    if missing_dirs:
        raise ValueError(f"Harbor evidence is missing trial directories: {missing_dirs}")
    return files


def _validate_trial_name(trial_name: str) -> None:
    if (
        not trial_name
        or trial_name in {".", ".."}
        or "/" in trial_name
        or "\\" in trial_name
        or Path(trial_name).is_absolute()
    ):
        raise ValueError(f"Harbor returned an unsafe trial name: {trial_name!r}")


def _read_model(path: Path, model: type[_ModelT]) -> _ModelT:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Harbor evidence file {path.name!r} is invalid") from error


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Harbor evidence file {path.name!r} is invalid") from error


def _canonical_model(model: BaseModel) -> str:
    return json.dumps(
        _normalize_json(model.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _normalize_json(value: object) -> object:
    if isinstance(value, BaseModel):
        return _normalize_json(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_json(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _validate_trial_lock(lock: TrialLock, trial: TrialResult) -> None:
    config = trial.config
    task_id = trial.task_id.get_name()
    if lock.schema_version != 1:
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has an unsupported schema")
    task_config = config.task
    task_type = (
        "package"
        if task_config.is_package_task()
        else "git"
        if task_config.is_git_task()
        else "local"
    )
    expected_task_provenance = (
        task_id,
        task_type,
        task_config.source,
        task_config.path,
        task_config.git_url,
        task_config.git_commit_id,
    )
    observed_task_provenance = (
        lock.task.name,
        lock.task.type,
        lock.task.source,
        lock.task.path,
        lock.task.git_url,
        lock.task.git_commit_id,
    )
    if observed_task_provenance != expected_task_provenance:
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has different task provenance")
    scalar_fields = (
        "install_only",
        "timeout_multiplier",
        "agent_timeout_multiplier",
        "verifier_timeout_multiplier",
        "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier",
    )
    if any(getattr(lock, field) != getattr(config, field) for field in scalar_fields):
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has different timeouts")
    if (
        _canonical_model(lock.agent) != _canonical_model(config.agent)
        or _canonical_model(lock.environment) != _canonical_model(config.environment)
        or _canonical_model(lock.verifier) != _canonical_model(config.verifier)
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has different execution config")
    locked_instruction_paths = tuple(str(item.path) for item in (lock.extra_instructions or []))
    if locked_instruction_paths != tuple(str(path) for path in config.extra_instruction_paths):
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has different instructions")
    if lock.skills:
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has unexpected skills")
    if lock.extra_docker_compose is not None:
        raise ValueError(f"Harbor trial {trial.trial_name!r} lock has unexpected compose files")


def _validate_dataset_origin(requested: DatasetConfig, resolved: DatasetConfig) -> None:
    if requested.model_dump(mode="json") != resolved.model_dump(mode="json"):
        raise ValueError("resolved Harbor task set came from a different dataset selector")


def _official_reward(trial: TrialResult, *, reward_key: str) -> float:
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        raise ValueError(
            f"Harbor trial {trial.trial_name!r} is missing official reward {reward_key!r}"
        )
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Harbor reward {reward_key!r} must be numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Harbor reward {reward_key!r} must be finite and in [0, 1]")
    return score


def _validate_candidate_trial(
    trial: TrialResult,
    *,
    candidate: HarnessDoc,
    provider_config: ProviderConfig,
    harness_backend: str,
    e2b_template: str | None,
    pi_transport: str | None,
    expected_config: JobConfig,
    job_id: UUID,
    job_dir: Path,
) -> None:
    if (
        trial.agent_info.name != WmhHarborAgent.name()
        or trial.agent_info.version != WMH_HARBOR_AGENT_VERSION
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} did not run the WMH agent bridge")
    config = trial.config.agent
    if config.import_path != WMH_HARBOR_AGENT_IMPORT_PATH:
        raise ValueError(f"Harbor trial {trial.trial_name!r} has the wrong agent import path")
    try:
        configured_candidate = HarnessDoc.model_validate(config.kwargs["harness"])
        configured_provider = ProviderConfig.model_validate(config.kwargs["provider_config"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Harbor trial {trial.trial_name!r} has invalid WMH agent config"
        ) from error
    if configured_candidate.doc_hash != candidate.doc_hash:
        raise ValueError(f"Harbor trial {trial.trial_name!r} ran a different harness document")
    if (
        configured_provider != provider_config
        or config.kwargs.get("harness_backend") != harness_backend
        or config.kwargs.get("e2b_template") != e2b_template
        or config.kwargs.get("pi_transport") != pi_transport
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} ran a different execution config")
    expected_agent = expected_config.agents[0]
    if _canonical_model(config) != _canonical_model(expected_agent):
        raise ValueError(f"Harbor trial {trial.trial_name!r} has a different agent config")
    trial_config = trial.config
    scalar_fields = (
        "install_only",
        "timeout_multiplier",
        "agent_timeout_multiplier",
        "verifier_timeout_multiplier",
        "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier",
    )
    if any(
        getattr(trial_config, field) != getattr(expected_config, field) for field in scalar_fields
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} has different job timeouts")
    if (
        trial_config.job_id != job_id
        or trial_config.trials_dir.resolve() != job_dir.resolve()
        or trial_config.task.get_task_id() != trial.task_id
        or _canonical_model(trial_config.environment)
        != _canonical_model(expected_config.environment)
        or _canonical_model(trial_config.verifier) != _canonical_model(expected_config.verifier)
        or trial_config.artifacts != expected_config.artifacts
        or trial_config.extra_instruction_paths != expected_config.extra_instruction_paths
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} has a different resolved job config")
    model_info = trial.agent_info.model_info
    if (
        model_info is None
        or model_info.provider != provider_config.kind.value
        or model_info.name != provider_config.model
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} reports a different model")
    contexts = [trial.agent_result] if trial.agent_result is not None else []
    contexts.extend(
        step.agent_result for step in trial.step_results or [] if step.agent_result is not None
    )
    if not contexts or any(
        context.metadata is None or context.metadata.get("candidate_doc_hash") != candidate.doc_hash
        for context in contexts
    ):
        raise ValueError(f"Harbor trial {trial.trial_name!r} lacks candidate identity evidence")


def _trial_summary(trial: TrialResult) -> str:
    exception = trial.exception_info
    return "completed" if exception is None else f"completed with {exception.exception_type}"


def _digest_json(value: object) -> str:
    encoded = json.dumps(
        _normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
