"""Tests for projecting harbor job results into harness score reports (fakes only)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

import wmh.evals.harbor.scorer as scorer_module
from wmh.evals.harbor.agent import WMH_HARBOR_AGENT_IMPORT_PATH
from wmh.evals.harbor.e2b_template_policy import WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborScorer,
)
from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import RewardMode
from wmh.providers.base import ProviderConfig, ProviderKind

_JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
_SUFFIXES = ("a7Hm2Ks", "m4Vx8Pa", "z9Tc3Wb")


def _provider() -> ProviderConfig:
    return ProviderConfig(kind=ProviderKind.BEDROCK, model="worker-model", region="us-west-2")


def _tasks(tmp_path: Path, task_ids: tuple[str, ...]) -> list[TaskConfig]:
    return [TaskConfig(path=tmp_path / "tasks" / task_id, source="tasks") for task_id in task_ids]


def _job_template(
    tmp_path: Path, *, backend: EnvironmentType = EnvironmentType.DOCKER
) -> JobConfig:
    return JobConfig(
        job_name="template",
        jobs_dir=tmp_path / "jobs",
        n_concurrent_trials=4,
        environment=EnvironmentConfig(type=backend),
        agents=[AgentConfig()],
        datasets=[DatasetConfig(path=tmp_path / "tasks")],
    )


def _trial(
    tmp_path: Path,
    task_id: str,
    attempt: int,
    *,
    reward: float | None,
    exception: str | None = None,
) -> TrialResult:
    name = f"{task_id}__{_SUFFIXES[attempt - 1]}"
    now = datetime.now(UTC)
    return TrialResult(
        task_name=task_id,
        trial_name=name,
        trial_uri=f"file://{tmp_path}/{name}",
        task_id=TaskConfig(path=tmp_path / "tasks" / task_id).get_task_id(),
        source="tasks",
        task_checksum="c" * 64,
        config=TrialConfig(task=TaskConfig(path=tmp_path / "tasks" / task_id), job_id=_JOB_ID),
        agent_info=AgentInfo(
            name="wmh-harness",
            version="1",
            model_info=ModelInfo(name="worker-model", provider="bedrock"),
        ),
        verifier_result=(
            None if reward is None else VerifierResult.model_construct(rewards={"reward": reward})
        ),
        exception_info=(
            None
            if exception is None
            else ExceptionInfo(
                exception_type=exception,
                exception_message="failed",
                exception_traceback="trace",
                occurred_at=now,
            )
        ),
        started_at=now,
        finished_at=now,
    )


class _Runner:
    """Materializes trial dirs under the candidate's deterministic job dir, like harbor."""

    def __init__(self, trials: list[TrialResult]) -> None:
        self.trials = trials
        self.configs: list[JobConfig] = []

    def run(self, config: JobConfig) -> HarborRun:
        self.configs.append(config)
        job_dir = config.jobs_dir / config.job_name
        for trial in self.trials:
            trial_dir = job_dir / trial.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "result.json").write_text(trial.model_dump_json(), encoding="utf-8")
        now = datetime.now(UTC)
        result = JobResult(
            id=_JOB_ID,
            started_at=now,
            finished_at=now,
            n_total_trials=len(self.trials),
            stats=JobStats.from_trial_results(self.trials, n_total_trials=len(self.trials)),
            trial_results=self.trials,
        )
        return HarborRun(result=result, job_dir=job_dir)


def _scorer(
    tmp_path: Path,
    trials: list[TrialResult],
    *,
    task_ids: tuple[str, ...] = ("task-a", "task-b"),
    attempts: int = 1,
    reward_mode: RewardMode = "raw",
    agent_concurrency: int | None = None,
) -> tuple[HarborScorer, _Runner]:
    runner = _Runner(trials)
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, task_ids),
        provider_config=_provider(),
        reward_mode=reward_mode,
        attempts=attempts,
        harness_backend="e2b",
        e2b_template="pi-template",
        agent_concurrency=agent_concurrency,
        runner=runner,
    )
    return scorer, runner


def test_scorer_projects_rewards_and_injects_the_exact_candidate(tmp_path: Path) -> None:
    candidate = HarnessDoc.baseline("candidate")
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=0.0),
    ]
    scorer, runner = _scorer(tmp_path, trials)

    report = scorer.score(candidate)

    assert report.doc_hash == candidate.doc_hash
    assert report.score == 0.5
    by_task = report.by_task()
    assert by_task["task-a"][0].passed is True
    assert by_task["task-b"][0].passed is False
    assert by_task["task-a"][0].note == "completed"
    expected_dir = tmp_path / "jobs" / f"wmh-{candidate.doc_hash[:12]}"
    assert by_task["task-a"][0].artifact_dir == str(expected_dir / trials[0].trial_name)

    [config] = runner.configs
    # Deterministic per-candidate job dir: the harbor-native trial-resume key.
    assert config.jobs_dir / config.job_name == scorer.candidate_job_dir(candidate)
    assert config.job_name == f"wmh-{candidate.doc_hash[:12]}"
    [agent] = config.agents
    assert agent.import_path == WMH_HARBOR_AGENT_IMPORT_PATH
    assert agent.model_name == "bedrock/worker-model"
    assert agent.kwargs["harness"] == candidate.model_dump(mode="json")
    assert agent.kwargs["harness_backend"] == "e2b"
    assert agent.kwargs["e2b_template"] == "pi-template"
    assert agent.kwargs["episode_workers"] >= 2 * config.n_concurrent_trials
    assert config.retry == RetryConfig(max_retries=0)
    assert [task.get_task_id().get_name() for task in config.tasks] == ["task-a", "task-b"]
    assert config.datasets == []


def test_failed_trial_with_a_written_reward_is_a_scored_cell_not_an_infra_halt(
    tmp_path: Path,
) -> None:
    """AgentTimeoutError with reward 0 is a CANDIDATE outcome; a missing reward is not."""
    scored = [
        _trial(tmp_path, "task-a", 1, reward=0.0, exception="AgentTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, scored)
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0
    assert cell.passed is False
    assert cell.note == "completed with AgentTimeoutError"

    missing = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="VerifierTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, missing)
    with pytest.raises(HarborRewardMissingError, match="no verifier reward"):
        scorer.score(HarnessDoc.baseline())


def test_positive_binary_mode_passes_on_any_positive_reward_and_keeps_raw_values(
    tmp_path: Path,
) -> None:
    trials = [
        _trial(tmp_path, "task-a", 1, reward=0.25),
        _trial(tmp_path, "task-b", 1, reward=0.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, reward_mode="positive-binary")
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.25  # raw reward untouched
    assert cell.passed is True
    assert report.score == 0.5


def test_entry_prunes_only_unscoreable_trial_dirs(tmp_path: Path) -> None:
    """Missing/unparseable result.json or exception-without-reward dirs are pruned; scoreable
    completed trials survive so harbor's resume re-runs only what is broken."""
    candidate = HarnessDoc.baseline()
    scorer, runner = _scorer(
        tmp_path,
        [
            _trial(tmp_path, "task-a", 1, reward=1.0),
            _trial(tmp_path, "task-b", 1, reward=0.0),
        ],
    )
    job_dir = scorer.candidate_job_dir(candidate)

    # Distinct names from anything the fake runner writes, so survival/pruning is attributable
    # to the entry prune alone.
    keep = _trial(tmp_path, "task-a", 2, reward=0.0, exception="AgentTimeoutError")
    (job_dir / keep.trial_name).mkdir(parents=True)
    (job_dir / keep.trial_name / "result.json").write_text(keep.model_dump_json(), encoding="utf-8")
    unparseable = job_dir / "task-b__broken1"
    unparseable.mkdir()
    (unparseable / "result.json").write_text("{not json", encoding="utf-8")
    missing_result = job_dir / "task-b__broken2"
    missing_result.mkdir()
    crashed = _trial(tmp_path, "task-b", 3, reward=None, exception="RuntimeError")
    (job_dir / crashed.trial_name).mkdir()
    (job_dir / crashed.trial_name / "result.json").write_text(
        crashed.model_dump_json(), encoding="utf-8"
    )

    scorer.score(candidate)

    assert (job_dir / keep.trial_name).is_dir()  # written reward: a kept candidate outcome
    assert not unparseable.exists()
    assert not missing_result.exists()
    assert not (job_dir / crashed.trial_name).exists()
    assert runner.configs  # the job still ran after pruning


def test_backend_and_template_conflicts_are_rejected_not_rewritten(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, ("task-a",))
    with pytest.raises(ValueError, match="task_environment='e2b' was requested"):
        HarborScorer(
            job_template=_job_template(tmp_path, backend=EnvironmentType.DOCKER),
            tasks=tasks,
            provider_config=_provider(),
            task_environment="e2b",
            harness_backend="e2b",
            agent_concurrency=1,
        )
    with pytest.raises(ValueError, match="task_environment='docker' was requested"):
        HarborScorer(
            job_template=_job_template(tmp_path, backend=EnvironmentType.E2B),
            tasks=tasks,
            provider_config=_provider(),
            task_environment="docker",
            agent_concurrency=1,
        )
    template = _job_template(tmp_path)
    template.environment.import_path = "somewhere.else:Env"
    template.environment.type = None
    with pytest.raises(ValueError, match="owns task-environment routing"):
        HarborScorer(
            job_template=template,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )


def test_builtin_e2b_environment_is_routed_through_the_paced_subclass(tmp_path: Path) -> None:
    template = _job_template(tmp_path, backend=EnvironmentType.E2B)
    template.environment.kwargs = {"keep": "me"}
    scorer = HarborScorer(
        job_template=template,
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        task_environment="e2b",
        harness_backend="e2b",
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.environment.type is None
    assert config.environment.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
    assert config.environment.kwargs == {"keep": "me"}  # options survive the routing rewrite


def test_candidate_job_is_revalidated_after_model_copy(tmp_path: Path) -> None:
    """model_copy(update=) skips validation; the scorer must re-validate the exact config."""
    template = _job_template(tmp_path)
    template.agents = [AgentConfig(n_concurrent=4)]
    with pytest.raises(ValueError, match="cannot exceed"):
        HarborScorer(
            job_template=template,
            tasks=_tasks(tmp_path, ("task-a",)),
            provider_config=_provider(),
            harness_backend="e2b",
            # Re-validation catches the now-inconsistent concurrency pair.
            agent_concurrency=2,
        )


def test_template_ownership_and_local_concurrency_rules(tmp_path: Path) -> None:
    tasks = _tasks(tmp_path, ("task-a",))
    template = _job_template(tmp_path)
    template.retry = RetryConfig(max_retries=3)
    with pytest.raises(ValueError, match="through harbor_retries"):
        HarborScorer(
            job_template=template,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )
    with pytest.raises(ValueError, match="agent concurrency 1"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=tasks,
            provider_config=_provider(),
            harness_backend="local",
        )
    owned = _job_template(tmp_path)
    owned.agents = [AgentConfig(import_path="x.y:Z")]
    with pytest.raises(ValueError, match="owns agent identity"):
        HarborScorer(
            job_template=owned,
            tasks=tasks,
            provider_config=_provider(),
            agent_concurrency=1,
        )


def test_harbor_retries_thread_into_retry_config_with_default_exclusions(
    tmp_path: Path,
) -> None:
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        harbor_retries=2,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.retry.max_retries == 2
    # Harbor's default exclude list keeps candidate-outcome exceptions unretried.
    assert config.retry.exclude_exceptions is not None
    assert {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "VerifierOutputParseError",
    } <= config.retry.exclude_exceptions


def test_sync_runner_works_with_and_without_a_running_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _job_template(tmp_path)
    now = datetime.now(UTC)
    result = JobResult(
        id=_JOB_ID, started_at=now, finished_at=now, n_total_trials=0, stats=JobStats()
    )

    class _FakeJob:
        job_dir = tmp_path / "jobs" / "template"

        @classmethod
        async def create(cls, _config: JobConfig) -> _FakeJob:
            return cls()

        async def run(self) -> JobResult:
            return result

    monkeypatch.setattr(scorer_module, "Job", _FakeJob)
    runner = HarborJobRunner()
    assert runner.run(config).result is result

    async def nested() -> HarborRun:
        return runner.run(config)

    assert asyncio.run(nested()).result is result


def test_cancellation_is_observed_before_any_spend(tmp_path: Path) -> None:
    scorer, runner = _scorer(tmp_path, [])
    with pytest.raises(scorer_module.HarnessSearchCancelled):
        scorer.score(HarnessDoc.baseline(), should_cancel=lambda: True)
    assert runner.configs == []


def test_wmh_import_pulls_neither_harbor_nor_e2b() -> None:
    """The packaging contract of both extras: `import wmh` (and wmh.evals) stays clean, and
    even the harbor subpackage never imports the e2b SDK (that loads only through harbor's
    environment factory when a job actually targets E2B)."""
    code = (
        "import sys\n"
        "import wmh, wmh.evals\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('harbor', 'e2b')]\n"
        "assert not bad, f'eager optional imports: {bad}'\n"
        "import wmh.evals.harbor\n"
        "assert 'harbor' in sys.modules\n"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'e2b']\n"
        "assert not bad, f'harbor subpackage must not import the e2b SDK: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
