"""Tests for strict projection of Harbor job evidence into harness scores."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

import harbor
import pytest
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.lock import (
    AgentSkillLock,
    ExtraDockerComposeLock,
    JobLock,
    TaskLock,
    TrialLock,
)
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import (
    AgentInfo,
    ExceptionInfo,
    ModelInfo,
    TimingInfo,
    TrialResult,
)
from harbor.models.verifier.result import VerifierResult
from harbor.tasks.client import TaskDownloadResult

import wmh.evals.harbor.scorer as scorer_module
from wmh.evals.harbor.agent import (
    MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    WMH_HARBOR_AGENT_IMPORT_PATH,
    WMH_HARBOR_AGENT_VERSION,
    WmhHarborAgent,
)
from wmh.evals.harbor.scorer import (
    WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH,
    HarborJobRunner,
    HarborRun,
    HarborScorer,
    HarborTaskIdentity,
    _validate_trial_lock,
)
from wmh.evals.harbor.tasks import ResolvedHarborTaskSet, resolve_harbor_task_set
from wmh.harness.doc import HarnessDoc
from wmh.harness.pi_e2b import DEFAULT_EVAL_EPISODE_TIMEOUT_S
from wmh.harness.scoring import ScoreRequest, score_harness
from wmh.providers.base import ProviderConfig, ProviderKind

_JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
_OPAQUE_SUFFIXES = ("a7Hm2Ks", "m4Vx8Pa", "z9Tc3Wb")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(task_id: str) -> HarborTaskIdentity:
    return HarborTaskIdentity(
        trial_checksum=_sha256(f"trial:{task_id}"),
        lock_digest=f"sha256:{_sha256(f'lock:{task_id}')}",
    )


class _Runner:
    def __init__(self, run: HarborRun, task_set: ResolvedHarborTaskSet) -> None:
        self.run_result = run
        self.task_set = task_set
        self.configs: list[JobConfig] = []

    def run(self, config: JobConfig) -> HarborRun:
        self.configs.append(config)
        source = self.run_result.job_dir
        destination = config.jobs_dir / config.job_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        task_configs = {config.get_task_id().get_name(): config for config in config.tasks}
        identities = self.task_set.identities
        for trial in self.run_result.result.trial_results:
            task_id = trial.task_id.get_name()
            identity = identities.get(task_id, _identity(task_id))
            if task_id in task_configs:
                trial.config.task = task_configs[task_id].model_copy(deep=True)
                trial.task_id = trial.config.task.get_task_id()
            trial.config.environment = config.environment.model_copy(deep=True)
            trial.task_checksum = identity.trial_checksum
            trial.config.trials_dir = destination
            trial.trial_uri = destination.joinpath(trial.trial_name).resolve().as_uri()
            trial_dir = destination / trial.trial_name
            (trial_dir / "result.json").write_text(trial.model_dump_json(), encoding="utf-8")
            (trial_dir / "config.json").write_text(trial.config.model_dump_json(), encoding="utf-8")
            (trial_dir / "lock.json").write_text(
                _trial_lock(trial, identity=identity).model_dump_json(),
                encoding="utf-8",
            )
        self.run_result = HarborRun(result=self.run_result.result, job_dir=destination)
        (destination / "config.json").write_text(config.model_dump_json(), encoding="utf-8")
        locks = [
            TrialLock.model_validate_json(
                (destination / trial.trial_name / "lock.json").read_text(encoding="utf-8")
            )
            for trial in self.run_result.result.trial_results
        ]
        job_lock = JobLock(
            n_concurrent_trials=config.n_concurrent_trials,
            retry=config.retry,
            trials=locks,
        )
        (destination / "lock.json").write_text(job_lock.model_dump_json(), encoding="utf-8")
        return self.run_result


def _provider(model: str = "worker-model") -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region="us-west-2")


def _job_config(tmp_path: Path, *, backend: EnvironmentType = EnvironmentType.DOCKER) -> JobConfig:
    return JobConfig(
        job_name="template",
        jobs_dir=tmp_path,
        n_attempts=1,
        n_concurrent_trials=4,
        retry=RetryConfig(max_retries=0),
        environment=EnvironmentConfig(type=backend),
        agents=[AgentConfig(n_concurrent=1)],
        datasets=[DatasetConfig(path=tmp_path / "tasks")],
    )


def _ensure_task_dataset(dataset_path: Path) -> None:
    for task_id in ("task-a", "task-b", "task-c"):
        task_dir = dataset_path / task_id
        if task_dir.exists():
            continue
        (task_dir / "environment").mkdir(parents=True)
        (task_dir / "tests").mkdir()
        (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.19\n", encoding="utf-8")
        (task_dir / "tests" / "test.sh").write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
        (task_dir / "instruction.md").write_text(f"Complete {task_id}.\n", encoding="utf-8")
        (task_dir / "task.toml").write_text('version = "1.0"\n\n[environment]\n', encoding="utf-8")


def _executed_agent(
    candidate: HarnessDoc,
    provider: ProviderConfig,
    job_config: JobConfig,
    *,
    harness_backend: str,
    e2b_template: str | None = None,
    environment_command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
) -> AgentConfig:
    return job_config.agents[0].model_copy(
        update={
            "name": None,
            "import_path": WMH_HARBOR_AGENT_IMPORT_PATH,
            "model_name": f"{provider.kind.value}/{provider.model}",
            "skills": [],
            "env": {},
            "mcp_servers": [],
            "kwargs": {
                "harness": candidate.model_dump(mode="json"),
                "provider_config": provider.model_dump(mode="json"),
                "harness_backend": harness_backend,
                "e2b_template": e2b_template,
                "command_timeout_sec": environment_command_timeout_sec,
                "episode_timeout_sec": float(episode_timeout_sec),
            },
        },
        deep=True,
    )


def _trial(
    job_dir: Path,
    task_id: str,
    replicate: int,
    *,
    score: float | int | None,
    reward_key: str = "reward",
    exception: str | None = None,
    candidate: HarnessDoc | None = None,
    provider: ProviderConfig | None = None,
    job_config: JobConfig | None = None,
    harness_backend: Literal["local", "e2b"] = "local",
    e2b_template: str | None = None,
    environment_command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
) -> TrialResult:
    candidate = candidate or HarnessDoc.baseline()
    provider = provider or _provider()
    job_config = job_config or _job_config(job_dir.parent)
    name = f"{task_id}__{_OPAQUE_SUFFIXES[replicate - 1]}"
    task_path = job_dir.parent / "tasks" / task_id
    dataset_path = job_config.datasets[0].path
    config = TrialConfig(
        task=TaskConfig(
            path=task_path,
            source=dataset_path.name if dataset_path is not None else "tasks",
        ),
        trial_name=name,
        trials_dir=job_dir,
        install_only=job_config.install_only,
        timeout_multiplier=job_config.timeout_multiplier,
        agent_timeout_multiplier=job_config.agent_timeout_multiplier,
        verifier_timeout_multiplier=job_config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=job_config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=job_config.environment_build_timeout_multiplier,
        agent=_executed_agent(
            candidate,
            provider,
            job_config,
            harness_backend=harness_backend,
            e2b_template=e2b_template,
            environment_command_timeout_sec=environment_command_timeout_sec,
            episode_timeout_sec=episode_timeout_sec,
        ),
        environment=job_config.environment,
        verifier=job_config.verifier,
        artifacts=job_config.artifacts,
        extra_instruction_paths=job_config.extra_instruction_paths,
        job_id=_JOB_ID,
    )
    verifier = (
        None if score is None else VerifierResult.model_construct(rewards={reward_key: score})
    )
    error = (
        None
        if exception is None
        else ExceptionInfo(
            exception_type=exception,
            exception_message="failed",
            exception_traceback="trace",
            occurred_at=datetime.now(UTC),
        )
    )
    now = datetime.now(UTC)
    return TrialResult(
        task_name=task_id,
        trial_name=name,
        trial_uri=f"file://{job_dir / name}",
        task_id=config.task.get_task_id(),
        source="dataset",
        task_checksum=_identity(task_id).trial_checksum,
        config=config,
        agent_info=AgentInfo(
            name=WmhHarborAgent.name(),
            version=WMH_HARBOR_AGENT_VERSION,
            model_info=ModelInfo(name=provider.model, provider=provider.kind.value),
        ),
        agent_result=AgentContext(metadata={"candidate_doc_hash": candidate.doc_hash}),
        verifier_result=verifier,
        exception_info=error,
        started_at=now,
        finished_at=now,
        verifier=TimingInfo(started_at=now, finished_at=now),
    )


def _trial_lock(
    trial: TrialResult,
    *,
    identity: HarborTaskIdentity | None = None,
) -> TrialLock:
    config = trial.config
    identity = identity or _identity(trial.task_id.get_name())
    return TrialLock(
        task=TaskLock(
            name=trial.task_id.get_name(),
            type="local",
            digest=identity.lock_digest,
            source=config.task.source,
            path=config.task.path,
        ),
        install_only=config.install_only,
        timeout_multiplier=config.timeout_multiplier,
        agent_timeout_multiplier=config.agent_timeout_multiplier,
        verifier_timeout_multiplier=config.verifier_timeout_multiplier,
        agent_setup_timeout_multiplier=config.agent_setup_timeout_multiplier,
        environment_build_timeout_multiplier=config.environment_build_timeout_multiplier,
        agent=config.agent,
        environment=config.environment,
        verifier=config.verifier,
    )


def _run(tmp_path: Path, trials: list[TrialResult], *, finished: bool = True) -> HarborRun:
    job_dir = tmp_path / "completed-job"
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(JobConfig().model_dump_json(), encoding="utf-8")
    (job_dir / "job.log").write_text("job log\n", encoding="utf-8")
    for trial in trials:
        trial_dir = job_dir / trial.trial_name
        if trial_dir.exists():
            continue
        (trial_dir / "agent").mkdir(parents=True)
        (trial_dir / "verifier").mkdir()
        (trial_dir / "artifacts").mkdir()
        (trial_dir / "steps" / "prepare" / "agent").mkdir(parents=True)
        (trial_dir / "result.json").write_text(trial.model_dump_json(), encoding="utf-8")
        (trial_dir / "config.json").write_text(trial.config.model_dump_json(), encoding="utf-8")
        (trial_dir / "lock.json").write_text(_trial_lock(trial).model_dump_json(), encoding="utf-8")
        (trial_dir / "trial.log").write_text("trial log\n", encoding="utf-8")
        (trial_dir / "agent" / "trace.bin").write_bytes(b"trace\x00")
        reward = trial.verifier_result or VerifierResult(rewards=None)
        (trial_dir / "verifier" / "reward.json").write_text(
            reward.model_dump_json(), encoding="utf-8"
        )
        (trial_dir / "artifacts" / "output.txt").write_text("artifact\n", encoding="utf-8")
        (trial_dir / "steps" / "prepare" / "agent" / "step.log").write_text(
            "step\n", encoding="utf-8"
        )
        if trial.exception_info is not None:
            (trial_dir / "exception.txt").write_text("exception\n", encoding="utf-8")
    now = datetime.now(UTC)
    result = JobResult(
        id=_JOB_ID,
        started_at=now,
        finished_at=now if finished else None,
        n_total_trials=len(trials),
        stats=JobStats.from_trial_results(trials, n_total_trials=len(trials)),
        trial_results=trials,
    )
    (job_dir / "result.json").write_text(
        result.model_dump_json(exclude={"trial_results"}), encoding="utf-8"
    )
    (job_dir / "lock.json").write_text(
        JobLock(
            n_concurrent_trials=4,
            retry=RetryConfig(max_retries=0),
            trials=[_trial_lock(trial) for trial in trials],
        ).model_dump_json(),
        encoding="utf-8",
    )
    return HarborRun(result=result, job_dir=job_dir)


def _scorer(
    tmp_path: Path,
    run: HarborRun,
    *,
    job_config: JobConfig | None = None,
    provider: ProviderConfig | None = None,
    reward_key: str = "reward",
    harness_backend: Literal["local", "e2b"] = "local",
    e2b_template: str | None = None,
    environment_command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    episode_timeout_sec: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
) -> tuple[HarborScorer, _Runner]:
    job_config = job_config or _job_config(tmp_path)
    dataset_path = job_config.datasets[0].path
    assert dataset_path is not None
    _ensure_task_dataset(dataset_path)
    task_set = asyncio.run(resolve_harbor_task_set(job_config.datasets[0], ("task-a", "task-b")))
    runner = _Runner(run, task_set)
    scorer = HarborScorer(
        job_config=job_config,
        task_set=task_set,
        provider_config=provider or _provider(),
        reward_key=reward_key,
        environment_command_timeout_sec=environment_command_timeout_sec,
        episode_timeout_sec=episode_timeout_sec,
        harness_backend=harness_backend,
        e2b_template=e2b_template,
        runner=runner,
    )
    return scorer, runner


def _request(scorer: HarborScorer, *, attempts: int = 2) -> ScoreRequest:
    return scorer.request(attempts=attempts)


def test_scorer_projects_shuffled_opaque_replicates_and_injects_candidate(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "completed-job"
    job_config = _job_config(tmp_path, backend=EnvironmentType.E2B)
    candidate = HarnessDoc.baseline("candidate")
    trials = [
        _trial(job_dir, "task-b", 2, score=0, candidate=candidate, job_config=job_config),
        _trial(job_dir, "task-a", 2, score=0.5, candidate=candidate, job_config=job_config),
        _trial(job_dir, "task-b", 1, score=1, candidate=candidate, job_config=job_config),
        _trial(job_dir, "task-a", 1, score=1, candidate=candidate, job_config=job_config),
    ]
    scorer, runner = _scorer(
        tmp_path,
        _run(tmp_path, trials),
        job_config=job_config,
        harness_backend="local",
    )

    scored = score_harness(scorer, candidate, request=_request(scorer))

    assert [(cell.task_id, cell.attempt, cell.score) for cell in scored.report.cells] == [
        ("task-a", 1, 1.0),
        ("task-a", 2, 0.5),
        ("task-b", 1, 1.0),
        ("task-b", 2, 0.0),
    ]
    assert scored.report.score == pytest.approx(0.625)
    assert scored.report.candidate_doc_hash == candidate.doc_hash
    config = runner.configs[0]
    assert config.n_attempts == 2
    assert config.environment.type is None
    assert config.environment.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
    assert config.datasets == []
    assert [task.get_task_id().get_name() for task in config.tasks] == [
        "task-a",
        "task-b",
    ]
    assert all(task.source is None for task in config.tasks)
    assert config.agents[0].import_path == WMH_HARBOR_AGENT_IMPORT_PATH
    assert config.agents[0].kwargs["harness"] == candidate.model_dump(mode="json")
    assert config.agents[0].kwargs["harness_backend"] == "local"
    assert config.agents[0].kwargs["command_timeout_sec"] == MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
    assert "secret" not in json.dumps(config.agents[0].kwargs).lower()


def test_scorer_rewrites_builtin_e2b_environment_without_losing_options(
    tmp_path: Path,
) -> None:
    environment = EnvironmentConfig(
        type=EnvironmentType.E2B,
        force_build=True,
        override_cpus=2,
        override_memory_mb=2048,
        env={"VISIBLE": "value"},
        kwargs={"option": "value"},
    )
    config = _job_config(tmp_path, backend=EnvironmentType.E2B).model_copy(
        update={"environment": environment}
    )
    scorer, _ = _scorer(tmp_path, _run(tmp_path, []), job_config=config)

    rewritten = scorer._job_config.environment
    assert rewritten.type is None
    assert rewritten.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
    assert rewritten.model_dump(exclude={"type", "import_path"}) == environment.model_dump(
        exclude={"type", "import_path"}
    )


def test_scorer_rejects_ambiguous_builtin_and_custom_e2b_environment(
    tmp_path: Path,
) -> None:
    config = _job_config(tmp_path, backend=EnvironmentType.E2B).model_copy(
        update={
            "environment": EnvironmentConfig(
                type=EnvironmentType.E2B,
                import_path="package.module:CustomE2BEnvironment",
            )
        }
    )

    with pytest.raises(ValueError, match="E2B environment.*import_path"):
        _scorer(tmp_path, _run(tmp_path, []), job_config=config)


def test_e2b_create_policy_is_bound_only_when_an_e2b_path_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local, _ = _scorer(tmp_path / "local", _run(tmp_path / "local", []))
    task_e2b, _ = _scorer(
        tmp_path / "task-e2b",
        _run(tmp_path / "task-e2b", []),
        job_config=_job_config(tmp_path / "task-e2b", backend=EnvironmentType.E2B),
    )
    worker_e2b, _ = _scorer(
        tmp_path / "worker-e2b",
        _run(tmp_path / "worker-e2b", []),
        harness_backend="e2b",
        e2b_template="runner-template",
    )
    local_digest = local.context().execution_config_digest
    task_digest = task_e2b.context().execution_config_digest
    worker_digest = worker_e2b.context().execution_config_digest

    monkeypatch.setattr(
        scorer_module,
        "e2b_create_rate_policy_payload",
        lambda: {
            "schema_version": 1,
            "provider": "e2b",
            "operation": "sandbox_create",
            "maximum_dispatches": 3,
            "period_milliseconds": 1000,
            "maximum_wait_seconds": 60.0,
        },
    )

    assert local.context().execution_config_digest == local_digest
    assert task_e2b.context().execution_config_digest != task_digest
    assert worker_e2b.context().execution_config_digest != worker_digest


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "unfinished"])
def test_scorer_rejects_incomplete_or_nonexact_runs(tmp_path: Path, case: str) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-a", 2, score=1),
        _trial(job_dir, "task-b", 1, score=1),
        _trial(job_dir, "task-b", 2, score=1),
    ]
    if case == "missing":
        trials.pop()
    elif case == "extra":
        trials.append(_trial(job_dir, "task-c", 1, score=1))
    elif case == "duplicate":
        trials[-1] = trials[0].model_copy(deep=True)
    run = _run(tmp_path, trials, finished=case != "unfinished")
    scorer, _ = _scorer(tmp_path, run)

    with pytest.raises(ValueError, match="finish|task|trial|matrix|count"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer))


def test_scorer_requires_finished_trials_and_verification(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    trials[0].finished_at = None
    scorer, _ = _scorer(tmp_path, _run(tmp_path, trials))
    with pytest.raises(ValueError, match="did not finish"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer, attempts=1))


def test_official_zero_with_agent_exception_is_scoreable_but_missing_reward_is_not(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=0, exception="AgentTimeoutError"),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    scorer, _ = _scorer(tmp_path, _run(tmp_path, trials))
    request = _request(scorer, attempts=1)

    scored = score_harness(scorer, HarnessDoc.baseline(), request=request)
    assert scored.report.cells[0].score == 0.0
    assert scored.report.cells[0].passed is False

    missing_path = tmp_path / "missing"
    missing_job_dir = missing_path / "completed-job"
    missing_trials = [
        _trial(missing_job_dir, "task-a", 1, score=None, exception="TimeoutException"),
        _trial(missing_job_dir, "task-b", 1, score=1),
    ]
    missing_scorer, _ = _scorer(missing_path, _run(missing_path, missing_trials))
    with pytest.raises(ValueError, match="reward"):
        score_harness(
            missing_scorer,
            HarnessDoc.baseline(),
            request=_request(missing_scorer, attempts=1),
        )


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_scorer_rejects_invalid_official_rewards(tmp_path: Path, invalid: object) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=cast("float | int | None", invalid)),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    scorer, _ = _scorer(tmp_path, _run(tmp_path, trials))
    with pytest.raises(ValueError, match="numeric|finite"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer, attempts=1))


def test_scorer_uses_only_the_configured_official_reward_key(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=0.25, reward_key="graded"),
        _trial(job_dir, "task-b", 1, score=0.75, reward_key="graded"),
    ]
    scorer, _ = _scorer(tmp_path, _run(tmp_path, trials), reward_key="graded")
    scored = score_harness(
        scorer,
        HarnessDoc.baseline(),
        request=_request(scorer, attempts=1),
    )
    assert [cell.score for cell in scored.report.cells] == [0.25, 0.75]

    wrong_path = tmp_path / "wrong"
    wrong_job_dir = wrong_path / "completed-job"
    wrong_trials = [
        _trial(wrong_job_dir, "task-a", 1, score=0.25, reward_key="graded"),
        _trial(wrong_job_dir, "task-b", 1, score=0.75, reward_key="graded"),
    ]
    wrong, _ = _scorer(wrong_path, _run(wrong_path, wrong_trials))
    with pytest.raises(ValueError, match="reward"):
        score_harness(wrong, HarnessDoc.baseline(), request=_request(wrong, attempts=1))


def test_scorer_preserves_every_raw_job_and_trial_evidence_file(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=0),
    ]
    run = _run(tmp_path, trials)
    large = b"event\n" * 100_001
    (run.job_dir / trials[0].trial_name / "agent" / "large.bin").write_bytes(large)
    scorer, runner = _scorer(tmp_path, run)

    scored = score_harness(
        scorer,
        HarnessDoc.baseline(),
        request=_request(scorer, attempts=1),
    )
    index = json.loads(scored.artifacts.read_bytes("raw/index.json"))
    indexed_sources = {entry["source_path"]: entry["artifact_path"] for entry in index["files"]}
    materialized_job_dir = runner.run_result.job_dir
    expected_sources = {
        path.relative_to(materialized_job_dir).as_posix()
        for path in materialized_job_dir.rglob("*")
        if path.is_file()
    }
    assert indexed_sources.keys() == expected_sources
    large_path = indexed_sources[f"{trials[0].trial_name}/agent/large.bin"]
    assert scored.artifacts.read_bytes(large_path) == large
    referenced = {path for cell in scored.report.cells for path in cell.artifact_paths}
    assert {artifact.path for artifact in scored.report.artifacts} == referenced


def test_scorer_rejects_symlinked_evidence(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    run = _run(tmp_path, trials)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (run.job_dir / trials[0].trial_name / "agent" / "escape").symlink_to(outside)
    scorer, _ = _scorer(tmp_path, run)

    with pytest.raises(ValueError, match="symlink"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer, attempts=1))


@pytest.mark.parametrize("target", ["job", "trial"])
def test_scorer_rejects_result_evidence_that_disagrees_with_returned_models(
    tmp_path: Path,
    target: str,
) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    run = _run(tmp_path, trials)
    if target == "job":
        path = run.job_dir / "result.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["n_total_trials"] = 99
        path.write_text(json.dumps(data), encoding="utf-8")
        scorer, _ = _scorer(tmp_path, run)
    else:

        class _TamperingRunner(_Runner):
            def run(self, config: JobConfig) -> HarborRun:
                materialized = super().run(config)
                path = materialized.job_dir / trials[0].trial_name / "result.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                data["task_checksum"] = "0" * 64
                path.write_text(json.dumps(data), encoding="utf-8")
                return materialized

        config = _job_config(tmp_path)
        dataset_path = config.datasets[0].path
        assert dataset_path is not None
        _ensure_task_dataset(dataset_path)
        task_set = asyncio.run(resolve_harbor_task_set(config.datasets[0], ("task-a", "task-b")))
        runner = _TamperingRunner(run, task_set)
        scorer = HarborScorer(
            job_config=config,
            task_set=task_set,
            provider_config=_provider(),
            reward_key="reward",
            runner=runner,
        )

    with pytest.raises(ValueError, match="evidence differs"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer, attempts=1))


def test_scorer_rejects_candidate_attestation_or_task_identity_drift(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    assert trials[0].agent_result is not None
    trials[0].agent_result.metadata = {"candidate_doc_hash": "sha256:" + "0" * 64}
    scorer, _ = _scorer(tmp_path, _run(tmp_path, trials))
    with pytest.raises(ValueError, match="candidate identity"):
        score_harness(scorer, HarnessDoc.baseline(), request=_request(scorer, attempts=1))

    drift_path = tmp_path / "drift"
    drift_job_dir = drift_path / "completed-job"
    drift_trials = [
        _trial(drift_job_dir, "task-a", 1, score=1),
        _trial(drift_job_dir, "task-b", 1, score=1),
    ]
    drift_scorer, drift_runner = _scorer(drift_path, _run(drift_path, drift_trials))
    request = drift_scorer.request(attempts=1)
    (drift_path / "tasks" / "task-a" / "instruction.md").write_text(
        "changed after resolution\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed on disk"):
        score_harness(drift_scorer, HarnessDoc.baseline(), request=request)
    assert drift_runner.configs == []


def test_trial_lock_binds_full_task_provenance(tmp_path: Path) -> None:
    trial = _trial(tmp_path / "job", "task-a", 1, score=1)
    lock = _trial_lock(trial)
    mutations: dict[str, object] = {
        "type": "git",
        "source": "different-source",
        "path": tmp_path / "different-task",
        "git_url": "https://example.invalid/different.git",
        "git_commit_id": "a" * 40,
    }

    for field, value in mutations.items():
        changed = lock.model_copy(
            update={"task": lock.task.model_copy(update={field: value})},
            deep=True,
        )
        with pytest.raises(ValueError, match="different task provenance"):
            _validate_trial_lock(changed, trial)


def test_trial_lock_rejects_unexpected_schema_skills_or_compose(tmp_path: Path) -> None:
    trial = _trial(tmp_path / "job", "task-a", 1, score=1)
    lock = _trial_lock(trial)
    digest = "sha256:" + "a" * 64
    cases = [
        (lock.model_copy(update={"schema_version": 2}), "unsupported schema"),
        (
            lock.model_copy(
                update={
                    "skills": [AgentSkillLock(name="unexpected", source=tmp_path, digest=digest)]
                }
            ),
            "unexpected skills",
        ),
        (
            lock.model_copy(
                update={
                    "extra_docker_compose": [
                        ExtraDockerComposeLock(path=tmp_path / "compose.yml", digest=digest)
                    ]
                }
            ),
            "unexpected compose",
        ),
    ]

    for changed, match in cases:
        with pytest.raises(ValueError, match=match):
            _validate_trial_lock(changed, trial)


def test_scorer_rejects_a_stale_request_before_runner_spend(tmp_path: Path) -> None:
    scorer, runner = _scorer(tmp_path, _run(tmp_path, []))
    request = scorer.request(attempts=1)
    stale = request.model_copy(update={"task_ids": tuple(reversed(request.task_ids))})

    with pytest.raises(ValueError, match="score request differs"):
        score_harness(scorer, HarnessDoc.baseline(), request=stale)
    assert runner.configs == []


@pytest.mark.parametrize(
    ("requested", "different"),
    [
        (
            DatasetConfig(name="registry-dataset", version="v1"),
            DatasetConfig(name="registry-dataset", version="v2"),
        ),
        (
            DatasetConfig(name="org/dataset", ref="release-1"),
            DatasetConfig(name="org/dataset", ref="release-2"),
        ),
    ],
)
def test_scorer_binds_the_original_dataset_selector(
    tmp_path: Path,
    requested: DatasetConfig,
    different: DatasetConfig,
) -> None:
    task_path = tmp_path / "task-a"
    _ensure_task_dataset(tmp_path)
    task_set = ResolvedHarborTaskSet.from_tasks(
        requested_dataset=requested,
        resolved_dataset=requested,
        tasks=[
            (
                TaskConfig(path=task_path, source="dataset"),
                TaskDownloadResult(
                    path=task_path,
                    download_time_sec=0.0,
                    cached=False,
                ),
                _identity("task-a"),
            )
        ],
    )
    config = _job_config(tmp_path).model_copy(update={"datasets": [different]})

    with pytest.raises(ValueError, match="different dataset selector"):
        HarborScorer(
            job_config=config,
            task_set=task_set,
            provider_config=_provider(),
            reward_key="reward",
        )


def test_context_commits_to_tasks_evaluator_backend_model_and_harbor_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run(tmp_path, [])
    base, _ = _scorer(tmp_path, run)
    original = base.context()

    identity_path = tmp_path / "identity"
    identity_job = _job_config(identity_path)
    identity_dataset = identity_job.datasets[0].path
    assert identity_dataset is not None
    _ensure_task_dataset(identity_dataset)
    (identity_dataset / "task-a" / "instruction.md").write_text(
        "different task bytes\n", encoding="utf-8"
    )
    changed_identity, _ = _scorer(identity_path, run, job_config=identity_job)
    verifier_job = _job_config(tmp_path).model_copy(
        update={
            "verifier": _job_config(tmp_path).verifier.model_copy(
                update={"override_timeout_sec": 42.0}
            )
        }
    )
    changed_verifier, _ = _scorer(tmp_path / "verifier", run, job_config=verifier_job)
    changed_backend, _ = _scorer(
        tmp_path / "backend",
        run,
        harness_backend="e2b",
        e2b_template="runner-template",
    )
    changed_model, _ = _scorer(tmp_path / "model", run, provider=_provider("other-model"))
    changed_command_timeout, _ = _scorer(
        tmp_path / "command-timeout",
        run,
        environment_command_timeout_sec=17,
    )
    changed_task_backend, _ = _scorer(
        tmp_path / "task-backend",
        run,
        job_config=_job_config(tmp_path, backend=EnvironmentType.E2B),
    )
    monkeypatch.setattr(harbor, "__version__", "0.20.0+changed")
    changed_version = base.context()

    assert changed_identity.context().task_set_digest != original.task_set_digest
    assert changed_verifier.context().evaluator_digest != original.evaluator_digest
    assert changed_backend.context().execution_config_digest != original.execution_config_digest
    assert changed_model.context().execution_config_digest != original.execution_config_digest
    assert (
        changed_command_timeout.context().execution_config_digest
        != original.execution_config_digest
    )
    assert (
        changed_task_backend.context().execution_config_digest != original.execution_config_digest
    )
    assert changed_version.evaluator_digest != original.evaluator_digest


def test_environment_command_timeout_is_injected_and_execution_drift_is_rejected(
    tmp_path: Path,
) -> None:
    timeout_sec = 17
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(
            job_dir,
            task_id,
            1,
            score=1,
            environment_command_timeout_sec=timeout_sec,
        )
        for task_id in ("task-a", "task-b")
    ]
    scorer, runner = _scorer(
        tmp_path,
        _run(tmp_path, trials),
        environment_command_timeout_sec=timeout_sec,
    )

    score_harness(
        scorer,
        HarnessDoc.baseline(),
        request=scorer.request(attempts=1),
    )

    assert runner.configs[0].agents[0].kwargs["command_timeout_sec"] == timeout_sec

    drift_path = tmp_path / "drift"
    drift_job_dir = drift_path / "completed-job"
    drift_trials = [
        _trial(
            drift_job_dir,
            task_id,
            1,
            score=1,
            environment_command_timeout_sec=timeout_sec - 1,
        )
        for task_id in ("task-a", "task-b")
    ]
    drift_scorer, _ = _scorer(
        drift_path,
        _run(drift_path, drift_trials),
        environment_command_timeout_sec=timeout_sec,
    )
    with pytest.raises(ValueError, match="different agent config|execution config"):
        score_harness(
            drift_scorer,
            HarnessDoc.baseline(),
            request=drift_scorer.request(attempts=1),
        )


def test_episode_timeout_is_evidence_bound_injected_and_drift_checked(tmp_path: Path) -> None:
    timeout_sec = 12_000
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(
            job_dir,
            task_id,
            1,
            score=1,
            harness_backend="e2b",
            e2b_template="",
            episode_timeout_sec=timeout_sec,
        )
        for task_id in ("task-a", "task-b")
    ]
    scorer, runner = _scorer(
        tmp_path,
        _run(tmp_path, trials),
        harness_backend="e2b",
        e2b_template="",
        episode_timeout_sec=timeout_sec,
    )

    score_harness(scorer, HarnessDoc.baseline(), request=scorer.request(attempts=1))

    assert runner.configs[0].agents[0].kwargs["episode_timeout_sec"] == timeout_sec
    assert (
        scorer.context().execution_config_digest
        != _scorer(
            tmp_path / "default",
            _run(tmp_path / "default", []),
            harness_backend="e2b",
            e2b_template="",
        )[0]
        .context()
        .execution_config_digest
    )

    drift_path = tmp_path / "drift"
    drift_job_dir = drift_path / "completed-job"
    drift_trials = [
        _trial(
            drift_job_dir,
            task_id,
            1,
            score=1,
            harness_backend="e2b",
            e2b_template="",
            episode_timeout_sec=timeout_sec - 1,
        )
        for task_id in ("task-a", "task-b")
    ]
    drift_scorer, _ = _scorer(
        drift_path,
        _run(drift_path, drift_trials),
        harness_backend="e2b",
        e2b_template="",
        episode_timeout_sec=timeout_sec,
    )
    with pytest.raises(ValueError, match="different agent config|execution config"):
        score_harness(
            drift_scorer,
            HarnessDoc.baseline(),
            request=drift_scorer.request(attempts=1),
        )


@pytest.mark.parametrize("invalid", [True, 0, -1, float("inf")])
def test_scorer_rejects_invalid_episode_timeout(tmp_path: Path, invalid: object) -> None:
    with pytest.raises(ValueError, match="episode_timeout_sec"):
        _scorer(
            tmp_path,
            _run(tmp_path, []),
            harness_backend="e2b",
            episode_timeout_sec=cast("float", invalid),
        )


def test_scorer_rejects_nondefault_episode_timeout_for_local_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="episode_timeout_sec requires harness_backend='e2b'"):
        _scorer(
            tmp_path,
            _run(tmp_path, []),
            harness_backend="local",
            episode_timeout_sec=12_000,
        )


@pytest.mark.parametrize("invalid", [True, 0, 241])
def test_scorer_rejects_unbounded_environment_command_timeout(
    tmp_path: Path,
    invalid: object,
) -> None:
    with pytest.raises(ValueError, match="environment_command_timeout_sec"):
        _scorer(
            tmp_path,
            _run(tmp_path, []),
            environment_command_timeout_sec=cast("int", invalid),
        )


@pytest.mark.parametrize(
    ("configured_env", "expected_template"),
    [("template-a", "template-a"), (None, "")],
)
def test_e2b_template_is_resolved_once_and_passed_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_env: str | None,
    expected_template: str,
) -> None:
    if configured_env is None:
        monkeypatch.delenv("WMH_E2B_TEMPLATE", raising=False)
    else:
        monkeypatch.setenv("WMH_E2B_TEMPLATE", configured_env)
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(
            job_dir,
            task_id,
            1,
            score=1,
            harness_backend="e2b",
            e2b_template=expected_template,
        )
        for task_id in ("task-a", "task-b")
    ]
    scorer, runner = _scorer(
        tmp_path,
        _run(tmp_path, trials),
        harness_backend="e2b",
    )
    context = scorer.context()
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "template-changed-after-construction")

    score_harness(
        scorer,
        HarnessDoc.baseline(),
        request=scorer.request(attempts=1),
    )

    assert scorer.context() == context
    assert runner.configs[0].agents[0].kwargs["e2b_template"] == expected_template


def test_scorer_uses_unique_job_names(tmp_path: Path) -> None:
    job_dir = tmp_path / "completed-job"
    trials = [
        _trial(job_dir, "task-a", 1, score=1),
        _trial(job_dir, "task-b", 1, score=1),
    ]
    scorer, runner = _scorer(tmp_path, _run(tmp_path, trials))
    request = _request(scorer, attempts=1)

    score_harness(scorer, HarnessDoc.baseline(), request=request)
    score_harness(scorer, HarnessDoc.baseline(), request=request)

    assert len({config.job_name for config in runner.configs}) == 2


def test_sync_harbor_runner_works_inside_an_active_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    result = JobResult(
        id=_JOB_ID,
        started_at=now,
        finished_at=now,
        n_total_trials=0,
        stats=JobStats.from_trial_results([], n_total_trials=0),
        trial_results=[],
    )

    class _Job:
        job_dir = tmp_path / "job"

        async def run(self) -> JobResult:
            return result

    async def create(_config: JobConfig) -> _Job:
        return _Job()

    monkeypatch.setattr("wmh.evals.harbor.scorer.Job.create", create)

    async def run() -> HarborRun:
        return HarborJobRunner().run(_job_config(tmp_path))

    observed = asyncio.run(run())
    assert observed.result is result
    assert observed.job_dir == tmp_path / "job"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda config, path: config.model_copy(update={"retry": RetryConfig(max_retries=1)}),
            "retries",
        ),
        (
            lambda config, path: config.model_copy(
                update={
                    "datasets": [DatasetConfig(path=path / "a"), DatasetConfig(path=path / "b")]
                }
            ),
            "one dataset",
        ),
        (lambda config, path: config.model_copy(update={"install_only": True}), "install-only"),
        (
            lambda config, path: config.model_copy(
                update={"verifier": config.verifier.model_copy(update={"disable": True})}
            ),
            "verification",
        ),
        (
            lambda config, path: config.model_copy(
                update={"datasets": [config.datasets[0].model_copy(update={"n_tasks": 1})]}
            ),
            "filters",
        ),
        (
            lambda config, path: config.model_copy(
                update={"extra_instruction_paths": [path / "x"]}
            ),
            "instruction",
        ),
    ],
)
def test_scorer_rejects_unsupported_job_shapes(
    tmp_path: Path,
    mutate: Callable[[JobConfig, Path], JobConfig],
    match: str,
) -> None:
    run = _run(tmp_path, [])
    config = mutate(_job_config(tmp_path), tmp_path)
    with pytest.raises(ValueError, match=match):
        _scorer(tmp_path, run, job_config=config)


def test_local_harness_execution_rejects_parallel_agent_phases(tmp_path: Path) -> None:
    config = _job_config(tmp_path).model_copy(update={"agents": [AgentConfig(n_concurrent=2)]})

    with pytest.raises(ValueError, match="agent concurrency 1"):
        _scorer(tmp_path, _run(tmp_path, []), job_config=config)

    e2b_scorer, _ = _scorer(
        tmp_path / "e2b",
        _run(tmp_path / "e2b", []),
        job_config=config.model_copy(update={"jobs_dir": tmp_path / "e2b"}),
        harness_backend="e2b",
        e2b_template="immutable-template",
    )
    assert e2b_scorer.context().execution_config_digest.startswith("sha256:")
