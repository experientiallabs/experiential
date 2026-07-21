"""Score harness candidates on real benchmark tasks through harbor.

`HarborScorer` implements the `wmh.harness.scoring.Scorer` protocol: one exact `HarnessDoc`
candidate becomes one harbor job (the WMH agent bridge + a pinned task list), harbor owns the
task environments and the verifier lifecycle, and the verifier rewards project into a
`ScoreReport`. Harbor's own job directory is the artifact record — each cell points at its trial
directory; nothing is re-read, re-hashed, or copied.

Two operational behaviors matter most here:

- **Trial-level resume.** Each candidate gets a deterministic job directory
  (`jobs_dir/wmh-<doc_hash12>`), and before running, trial directories whose result.json is
  missing/unparseable or that failed without a verifier reward are pruned. Harbor's native
  resume then keeps completed trials and re-runs only what is missing, so a crashed or
  interrupted boundary re-pays a handful of trials instead of the whole matrix.
- **Candidate outcomes vs infra failures.** A trial that raised (e.g. AgentTimeoutError) but
  still carries a written verifier reward is a CANDIDATE outcome: it becomes a scored cell with
  a note. The scorer raises (`HarborRewardMissingError`) only when verifier evidence or the
  configured reward is absent — that is an infrastructure failure no reward can stand in for.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from harbor import Job
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, RetryConfig
from harbor.models.job.result import JobResult
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.result import TrialResult

from wmh.evals.harbor.agent import (
    DEFAULT_EPISODE_WORKERS,
    MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    WMH_HARBOR_AGENT_IMPORT_PATH,
)
from wmh.evals.harbor.e2b_template_policy import WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.tasks import resolve_harbor_tasks
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import resolve_e2b_template
from wmh.harness.runtime import (
    DEFAULT_EVAL_EPISODE_TIMEOUT_S,
    HarnessSearchCancelled,
    validate_episode_timeout_s,
)
from wmh.harness.scoring import RewardMode, ScoreCell, ScoreReport, ScoreRequest, reward_passed
from wmh.providers.base import ProviderConfig

logger = logging.getLogger(__name__)

TaskEnvironment = Literal["docker", "e2b"]
HarnessBackend = Literal["local", "e2b"]


class HarborRewardMissingError(RuntimeError):
    """Verifier evidence or the configured reward key is absent from a finished trial.

    This is the one condition the scorer refuses to score around: a missing reward is an
    infrastructure failure (verifier never ran, reward file lost), not a candidate outcome.
    """


@dataclass(frozen=True)
class HarborRun:
    """One completed harbor job and its output directory."""

    result: JobResult
    job_dir: Path


class HarborRunner(Protocol):
    """Synchronous execution seam for one harbor job (fakes replace it in tests)."""

    def run(self, config: JobConfig) -> HarborRun: ...


class HarborJobRunner:
    """Run harbor's async Python API from synchronous optimizer code."""

    def run(self, config: JobConfig) -> HarborRun:
        async def run_job() -> HarborRun:
            job = await Job.create(config)
            result = await job.run()
            return HarborRun(result=result, job_dir=job.job_dir)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_job())
        # Called from inside an event loop (e.g. an async CLI): run the job on its own loop in
        # one worker thread instead of failing on nested asyncio.run.
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(run_job())).result()


class HarborScorer:
    """Evaluate exact harness candidates through harbor's verifier lifecycle."""

    def __init__(
        self,
        *,
        job_template: JobConfig,
        tasks: Sequence[TaskConfig],
        provider_config: ProviderConfig,
        reward_key: str = "reward",
        reward_mode: RewardMode = "raw",
        attempts: int = 1,
        task_environment: TaskEnvironment = "docker",
        harness_backend: HarnessBackend = "local",
        e2b_template: str | None = None,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        agent_concurrency: int | None = None,
        harbor_retries: int = 0,
        runner: HarborRunner | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("HarborScorer requires at least one resolved task")
        task_ids = [task.get_task_id().get_name() for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("HarborScorer requires unique task ids")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if not reward_key:
            raise ValueError("reward_key must be nonempty")
        if reward_mode not in ("raw", "positive-binary"):
            raise ValueError("reward_mode must be raw or positive-binary")
        if harness_backend not in ("local", "e2b"):
            raise ValueError("harness_backend must be local or e2b")
        if harness_backend == "local" and e2b_template is not None:
            raise ValueError("e2b_template requires harness_backend='e2b'")
        episode_timeout_s = validate_episode_timeout_s(episode_timeout_s)
        if harness_backend == "local" and episode_timeout_s != DEFAULT_EVAL_EPISODE_TIMEOUT_S:
            raise ValueError("episode_timeout_s requires harness_backend='e2b'")
        if isinstance(harbor_retries, bool) or not isinstance(harbor_retries, int):
            raise ValueError("harbor_retries must be a nonnegative integer")
        if harbor_retries < 0:
            raise ValueError("harbor_retries must be a nonnegative integer")
        if agent_concurrency is not None and (
            isinstance(agent_concurrency, bool)
            or not isinstance(agent_concurrency, int)
            or agent_concurrency < 1
        ):
            raise ValueError("agent_concurrency must be a positive integer")
        _validate_job_template(job_template)
        environment = _route_task_environment(job_template, task_environment)
        effective_concurrency = agent_concurrency or job_template.n_concurrent_trials
        if harness_backend == "local" and effective_concurrency > 1:
            raise ValueError(
                "local harness execution requires agent concurrency 1 (the local pi runner "
                "shares one runner dir); use harness_backend='e2b' for parallel trials"
            )
        # model_copy(update=) skips validation, so the copied config is re-validated as a whole.
        self._job_template = JobConfig.model_validate(
            job_template.model_copy(
                update={
                    "environment": environment,
                    "datasets": [],
                    "tasks": [],
                    "n_concurrent_trials": effective_concurrency,
                    "quiet": True,
                    "retry": RetryConfig(max_retries=harbor_retries),
                },
                deep=True,
            ).model_dump(mode="python")
        )
        self._tasks = [TaskConfig.model_validate(task.model_dump(mode="python")) for task in tasks]
        self._task_ids = tuple(task_ids)
        self._provider_config = ProviderConfig.model_validate(
            provider_config.model_dump(mode="python")
        )
        self._reward_key = reward_key
        self._reward_mode: RewardMode = reward_mode
        self._attempts = attempts
        self._harness_backend: HarnessBackend = harness_backend
        self._episode_timeout_s = episode_timeout_s
        self._command_timeout_sec = command_timeout_sec
        # The dedicated episode executor must never be smaller than agent concurrency (episodes
        # + uncancellable cleanup share it), or queued episodes burn harbor timeout budget.
        self._episode_workers = max(DEFAULT_EPISODE_WORKERS, 2 * effective_concurrency)
        if harness_backend == "e2b":
            resolved_template = resolve_e2b_template(e2b_template)
            # "" pins "no template" so the agent process cannot drift onto an ambient
            # $WMH_E2B_TEMPLATE that differs from what this scorer resolved.
            self._e2b_template: str | None = (
                resolved_template if resolved_template is not None else ""
            )
        else:
            self._e2b_template = None
        self._runner = runner or HarborJobRunner()

    @classmethod
    async def create(
        cls,
        job_template: JobConfig,
        task_ids: Sequence[str],
        *,
        provider_config: ProviderConfig,
        reward_key: str = "reward",
        reward_mode: RewardMode = "raw",
        attempts: int = 1,
        task_environment: TaskEnvironment = "docker",
        harness_backend: HarnessBackend = "local",
        e2b_template: str | None = None,
        episode_timeout_s: float = DEFAULT_EVAL_EPISODE_TIMEOUT_S,
        command_timeout_sec: int = MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
        agent_concurrency: int | None = None,
        harbor_retries: int = 0,
        runner: HarborRunner | None = None,
    ) -> Self:
        """Resolve the exact tasks and construct a scorer that can incur spend.

        `job_template` supplies the run directory (`jobs_dir`), the task environment config,
        and harbor tuning (timeouts, concurrency); it must carry exactly one dataset, no direct
        tasks, and an untouched default agent + retry config — the scorer owns those.
        """
        if len(job_template.datasets) != 1 or job_template.tasks:
            raise ValueError("HarborScorer requires exactly one dataset and no direct tasks")
        tasks = await resolve_harbor_tasks(job_template.datasets[0], task_ids)
        return cls(
            job_template=job_template,
            tasks=tasks,
            provider_config=provider_config,
            reward_key=reward_key,
            reward_mode=reward_mode,
            attempts=attempts,
            task_environment=task_environment,
            harness_backend=harness_backend,
            e2b_template=e2b_template,
            episode_timeout_s=episode_timeout_s,
            command_timeout_sec=command_timeout_sec,
            agent_concurrency=agent_concurrency,
            harbor_retries=harbor_retries,
            runner=runner,
        )

    @property
    def request(self) -> ScoreRequest:
        """The exact task-by-attempt matrix every `score` call evaluates."""
        return ScoreRequest(task_ids=self._task_ids, attempts=self._attempts)

    @property
    def reward_mode(self) -> RewardMode:
        """The frozen reward interpretation this scorer applies."""
        return self._reward_mode

    def candidate_job_dir(self, doc: HarnessDoc) -> Path:
        """The deterministic job directory one candidate's trials live in (resume key)."""
        return self._job_template.jobs_dir / f"wmh-{doc.doc_hash[:12]}"

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        """Run one candidate through harbor and project its verifier rewards."""
        if should_cancel is not None and should_cancel():
            raise HarnessSearchCancelled("harbor scoring cancelled before job start")
        config = self._candidate_job(doc)
        pruned = _prune_invalid_trial_dirs(
            config.jobs_dir / config.job_name, reward_key=self._reward_key
        )
        if pruned:
            logger.info(
                "pruned %d invalid trial dir(s) under %s; harbor resume re-runs only those",
                pruned,
                config.jobs_dir / config.job_name,
            )
        run = self._runner.run(config)
        return self._project(doc, run)

    def _candidate_job(self, doc: HarnessDoc) -> JobConfig:
        template_agent = self._job_template.agents[0]
        agent = template_agent.model_copy(
            update={
                "name": None,
                "import_path": WMH_HARBOR_AGENT_IMPORT_PATH,
                "model_name": f"{self._provider_config.kind.value}/{self._provider_config.model}",
                "skills": [],
                "env": {},
                "mcp_servers": [],
                "kwargs": {
                    "harness": doc.model_dump(mode="json"),
                    "provider_config": self._provider_config.model_dump(mode="json"),
                    "harness_backend": self._harness_backend,
                    "e2b_template": self._e2b_template,
                    "command_timeout_sec": self._command_timeout_sec,
                    "episode_timeout_sec": self._episode_timeout_s,
                    "episode_workers": self._episode_workers,
                },
            },
            deep=True,
        )
        config = self._job_template.model_copy(
            update={
                # Deterministic (NOT uuid-suffixed): rescoring the same candidate resumes its
                # completed trials through harbor's native trial resume.
                "job_name": f"wmh-{doc.doc_hash[:12]}",
                "n_attempts": self._attempts,
                "tasks": [task.model_copy(deep=True) for task in self._tasks],
                "agents": [agent],
            },
            deep=True,
        )
        # model_copy(update=) skips validation: re-validate the exact config harbor will run.
        return JobConfig.model_validate(config.model_dump(mode="python"))

    def _project(self, doc: HarnessDoc, run: HarborRun) -> ScoreReport:
        result = run.result
        if result.finished_at is None:
            raise ValueError("harbor job did not finish")
        expected = len(self._task_ids) * self._attempts
        if len(result.trial_results) != expected:
            raise ValueError(
                f"harbor returned {len(result.trial_results)} trials; expected {expected}"
            )
        grouped: defaultdict[str, list[TrialResult]] = defaultdict(list)
        for trial in result.trial_results:
            task_id = trial.task_id.get_name()
            if task_id not in self._task_ids:
                raise ValueError(f"harbor returned an unexpected task {task_id!r}")
            if trial.finished_at is None:
                raise ValueError(f"harbor trial {trial.trial_name!r} did not finish")
            grouped[task_id].append(trial)
        wrong_counts = {
            task_id: len(grouped[task_id])
            for task_id in self._task_ids
            if len(grouped[task_id]) != self._attempts
        }
        if wrong_counts:
            raise ValueError(f"harbor task matrix is incomplete: counts={wrong_counts}")

        cells: list[ScoreCell] = []
        for task_id in self._task_ids:
            trials = sorted(grouped[task_id], key=lambda trial: trial.trial_name)
            for attempt, trial in enumerate(trials, 1):
                reward = _official_reward(trial, reward_key=self._reward_key)
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        reward=reward,
                        passed=reward_passed(reward, self._reward_mode),
                        artifact_dir=str(run.job_dir / trial.trial_name),
                        note=_trial_note(trial),
                    )
                )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self.request,
            reward_mode=self._reward_mode,
            cells=tuple(cells),
        )


def _validate_job_template(job_template: JobConfig) -> None:
    """Reject template shapes the scorer would otherwise have to silently rewrite."""
    if len(job_template.agents) != 1:
        raise ValueError("HarborScorer requires exactly one agent template")
    template = job_template.agents[0]
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
            "HarborScorer owns agent identity, model, skills, environment, and kwargs; "
            "leave the template agent unset"
        )
    if job_template.install_only:
        raise ValueError("HarborScorer cannot use an install-only harbor job")
    if job_template.verifier.disable:
        raise ValueError("HarborScorer requires harbor verification")
    if job_template.retry != RetryConfig():
        raise ValueError(
            "HarborScorer owns the retry policy; configure retries through harbor_retries"
        )


def _route_task_environment(job_template: JobConfig, task_environment: TaskEnvironment) -> object:
    """Validate the template/backend combination; rewrite type -> import_path only for E2B.

    Conflicts are REJECTED, never rewritten: a docker template with a requested e2b task
    environment (or vice versa) means the caller's config and intent disagree.
    """
    environment = job_template.environment
    if task_environment not in ("docker", "e2b"):
        raise ValueError("task_environment must be docker or e2b")
    if environment.import_path is not None:
        if environment.type is not None:
            raise ValueError("harbor environment cannot set both type and import_path")
        if (
            task_environment == "e2b"
            and environment.import_path == WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
        ):
            return environment.model_copy(deep=True)
        raise ValueError(
            "HarborScorer owns task-environment routing; set environment.type and pass "
            "task_environment instead of import_path"
        )
    if task_environment == "docker":
        if environment.type is not EnvironmentType.DOCKER:
            raise ValueError(
                f"job template declares a {environment.type} task environment but "
                "task_environment='docker' was requested; make them agree"
            )
        return environment.model_copy(deep=True)
    if environment.type is not EnvironmentType.E2B:
        raise ValueError(
            f"job template declares a {environment.type} task environment but "
            "task_environment='e2b' was requested; make them agree"
        )
    # The consistent combination: route harbor's built-in E2B type through WMH's paced
    # subclass (qualified aliases, single-submit builds, create pacing). Options survive.
    return environment.model_copy(
        update={"type": None, "import_path": WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH},
        deep=True,
    )


def _prune_invalid_trial_dirs(job_dir: Path, *, reward_key: str) -> int:
    """Delete trial dirs harbor's resume would either crash on or wrongly keep.

    Harbor keeps any trial whose result.json parses; a trial that died with exception_info and
    no verifier reward would therefore be "kept" as an unscoreable cell forever. Pruning it (and
    unreadable ones) makes harbor re-run exactly those trials: cheap trial-level resume of a
    crashed or interrupted boundary.
    """
    if not job_dir.is_dir():
        return 0
    pruned = 0
    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        if _trial_dir_is_scoreable(trial_dir, reward_key=reward_key):
            continue
        shutil.rmtree(trial_dir)
        pruned += 1
    return pruned


def _trial_dir_is_scoreable(trial_dir: Path, *, reward_key: str) -> bool:
    result_path = trial_dir / "result.json"
    try:
        trial = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if trial.exception_info is None:
        return True
    return _trial_reward(trial, reward_key=reward_key) is not None


def _trial_reward(trial: TrialResult, *, reward_key: str) -> float | None:
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        return None
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        return None
    return reward


def _official_reward(trial: TrialResult, *, reward_key: str) -> float:
    """The verifier's written reward; absence is the one unscoreable (infra) condition."""
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        raise HarborRewardMissingError(
            f"harbor trial {trial.trial_name!r} has no verifier reward {reward_key!r}; "
            "the verifier never produced evidence, so this cell cannot be scored"
        )
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"harbor reward {reward_key!r} must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"harbor reward {reward_key!r} must be finite and in [0, 1]")
    return reward


def _trial_note(trial: TrialResult) -> str:
    exception = trial.exception_info
    return "completed" if exception is None else f"completed with {exception.exception_type}"
