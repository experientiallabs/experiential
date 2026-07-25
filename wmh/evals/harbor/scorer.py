"""Score harness candidates on real benchmark tasks through harbor.

`HarborScorer` implements the `wmh.harness.scoring.Scorer` protocol: one exact `HarnessDoc`
candidate becomes one harbor job (the WMH agent bridge + a pinned task list), harbor owns the
task environments and the verifier lifecycle, and the verifier rewards project into a
`ScoreReport`. Harbor's own job directory is the artifact record, and each cell points at its
trial directory; nothing is re-read, re-hashed, or copied.

Two operational behaviors matter most here:

- **Trial-level resume.** Each candidate gets a deterministic job directory
  (`jobs_dir/wmh-<doc_hash12>`), and before running, trial directories whose result.json is
  missing/unparseable or that failed without a verifier reward are pruned. Harbor's native
  resume then keeps completed trials and re-runs only what is missing, so a crashed or
  interrupted boundary re-pays a handful of trials instead of the whole matrix.
- **Candidate outcomes vs infra failures.** A trial that raised (e.g. AgentTimeoutError) but
  still carries a written verifier reward is a CANDIDATE outcome: it becomes a scored cell with
  a note. Absent that reward there is nothing to score, so search mode raises
  (`HarborRewardMissingError`) and evaluation-tolerant mode records the cell as an
  infrastructure failure whose 0.0 is an explicit stand-in. The rule that decides it is a
  single measurement question, not the shape of the exception: in `missing_reward="zero"` mode a
  cell is `infra_failed` exactly when the verifier wrote no reward for the configured key, so a
  verifier that timed out on work the agent really submitted can never be reported as a definite
  task failure (see `_trial_outcome`).

Each cell also carries the trial's CTRF test breakdown when the verifier wrote one
(`wmh.evals.harbor.ctrf`), so a caller can read a graded test-pass score BESIDE the binary reward.
The reward is unchanged and remains the benchmark's own verdict; the breakdown is None (never 0.0)
whenever no report exists, so a graded rate can exclude it the way a solve rate excludes an
ungradeable trial.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import threading
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
from harbor.models.trial.config import AgentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult

from wmh.core.types import JsonObject
from wmh.evals.harbor.agent import (
    DEFAULT_EPISODE_WORKERS,
    MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC,
    WMH_HARBOR_AGENT_IMPORT_PATH,
)
from wmh.evals.harbor.ctrf import read_trial_graded_tests
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
MissingRewardMode = Literal["raise", "zero"]
"""How a trial with no verifier reward scores: "raise" aborts (candidate search
semantics: an unscoreable candidate is an infra failure), "zero" records a
stand-in 0.0 flagged `infra_failed` with an auditable note (distillation evals:
an ungradeable trial is an UNKNOWN outcome, kept out of every solve rate)."""

# In-process registry of job dirs with a score() in flight: the entry prune is destructive, so
# concurrent scores of the same candidate must be rejected, not interleaved.
_ACTIVE_GUARD = threading.Lock()
_ACTIVE_JOB_DIRS: set[Path] = set()

# Agent kwargs the scorer computes and owns; `extra_agent_kwargs` may extend the kwargs dict but
# never silently override these, or a custom agent would run a different candidate/provider than
# the one this scorer reports on.
_SCORER_OWNED_AGENT_KWARGS = frozenset(
    {
        "harness",
        "provider_config",
        "harness_backend",
        "e2b_template",
        "command_timeout_sec",
        "episode_timeout_sec",
        "episode_workers",
        "context_window",
    }
)


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


@dataclass(frozen=True)
class _TrialOutcome:
    """How one finished trial projects into a cell: its reward, note, and measurement status."""

    reward: float
    infra_failed: bool
    note: str


class HarborRunner(Protocol):
    """Synchronous execution seam for one harbor job (fakes replace it in tests)."""

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun: ...


class HarborJobRunner:
    """Run harbor's async Python API from synchronous optimizer code.

    `should_cancel` is polled every `poll_interval_s` while the job runs; observing it cancels
    the harbor job task (harbor cancels its in-flight trials and persists what it can; the
    scorer's entry prune makes the interrupted boundary resumable) and raises
    `HarnessSearchCancelled`, so a multi-hour boundary stays cancellable through the Scorer
    contract instead of only at its edges.
    """

    def __init__(self, *, poll_interval_s: float = 2.0) -> None:
        if not poll_interval_s > 0:
            raise ValueError("poll_interval_s must be positive")
        self._poll_interval_s = poll_interval_s

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun:
        async def run_job() -> HarborRun:
            job = await Job.create(config)
            job_task = asyncio.ensure_future(job.run())
            if should_cancel is None:
                return HarborRun(result=await job_task, job_dir=job.job_dir)
            while True:
                done, _pending = await asyncio.wait({job_task}, timeout=self._poll_interval_s)
                if done:
                    return HarborRun(result=job_task.result(), job_dir=job.job_dir)
                if should_cancel():
                    job_task.cancel()
                    await asyncio.wait({job_task})
                    if not job_task.cancelled():
                        job_task.exception()  # consume; cancellation is the outcome
                    raise HarnessSearchCancelled("harbor scoring cancelled mid-job")

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
        agent_import_path: str = WMH_HARBOR_AGENT_IMPORT_PATH,
        extra_agent_kwargs: JsonObject | None = None,
        missing_reward: MissingRewardMode = "raise",
        context_window: int | None = None,
        runner: HarborRunner | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("HarborScorer requires at least one resolved task")
        if missing_reward not in ("raise", "zero"):
            raise ValueError("missing_reward must be raise or zero")
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
        # Both harness backends honor the wall budget now: the local SSH transport applies it as
        # the remote node timeout instead of the fixed 300s it used to hardcode.
        episode_timeout_s = validate_episode_timeout_s(episode_timeout_s)
        if isinstance(harbor_retries, bool) or not isinstance(harbor_retries, int):
            raise ValueError("harbor_retries must be a nonnegative integer")
        if harbor_retries < 0:
            raise ValueError("harbor_retries must be a nonnegative integer")
        if not agent_import_path or ":" not in agent_import_path:
            raise ValueError(
                f"agent_import_path must be a 'module:Class' import path, got "
                f"{agent_import_path!r}; use the exported constant of the agent bridge "
                "(e.g. WMH_HARBOR_AGENT_IMPORT_PATH)"
            )
        overridden = _SCORER_OWNED_AGENT_KWARGS & set(extra_agent_kwargs or {})
        if overridden:
            raise ValueError(
                f"extra_agent_kwargs may not override the scorer-owned agent kwargs "
                f"{sorted(overridden)}; configure those through the scorer's own parameters"
            )
        if agent_concurrency is not None and (
            isinstance(agent_concurrency, bool)
            or not isinstance(agent_concurrency, int)
            or agent_concurrency < 1
        ):
            raise ValueError("agent_concurrency must be a positive integer")
        if (
            isinstance(command_timeout_sec, bool)
            or not isinstance(command_timeout_sec, int)
            or not 1 <= command_timeout_sec <= MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC
        ):
            raise ValueError(
                "command_timeout_sec must be an integer in "
                f"[1, {MAX_ENVIRONMENT_COMMAND_TIMEOUT_SEC}]"
            )
        _validate_job_template(job_template)
        environment = _route_task_environment(job_template, task_environment)
        effective_concurrency = agent_concurrency or job_template.n_concurrent_trials
        if harness_backend == "local" and effective_concurrency > 1:
            raise ValueError(
                "local harness execution requires agent concurrency 1 (the local pi runner "
                "shares one runner dir); use harness_backend='e2b' for parallel trials"
            )
        # model_copy(update=) skips validation, so the copied config is re-validated as a whole.
        self._job_template = _revalidated_job_config(
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
            )
        )
        self._tasks = [TaskConfig.model_validate(task.model_dump(mode="python")) for task in tasks]
        self._task_ids = tuple(task_ids)
        self._provider_config = ProviderConfig.model_validate(
            provider_config.model_dump(mode="python")
        )
        self._reward_key = reward_key
        self._reward_mode: RewardMode = reward_mode
        self._attempts = attempts
        self._agent_import_path = agent_import_path
        self._extra_agent_kwargs: JsonObject = dict(extra_agent_kwargs or {})
        self._missing_reward: MissingRewardMode = missing_reward
        self._harness_backend: HarnessBackend = harness_backend
        self._episode_timeout_s = episode_timeout_s
        self._context_window = context_window
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
        agent_import_path: str = WMH_HARBOR_AGENT_IMPORT_PATH,
        extra_agent_kwargs: JsonObject | None = None,
        missing_reward: MissingRewardMode = "raise",
        context_window: int | None = None,
        runner: HarborRunner | None = None,
    ) -> Self:
        """Resolve the exact tasks and construct a scorer that can incur spend.

        `job_template` supplies the run directory (`jobs_dir`), the task environment config,
        and harbor tuning (timeouts, concurrency); it must carry exactly one dataset, no direct
        tasks, and an untouched default agent + retry config (the scorer owns those).
        `agent_import_path` plus `extra_agent_kwargs` route trials through a custom agent
        bridge (e.g. the distill collector's token-recording subclass); the defaults preserve
        the standard WMH agent, and extra kwargs may never shadow the scorer-owned ones.
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
            agent_import_path=agent_import_path,
            extra_agent_kwargs=extra_agent_kwargs,
            missing_reward=missing_reward,
            context_window=context_window,
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

    @property
    def task_pins(self) -> dict[str, str]:
        """One stable provenance pin per resolved task, keyed by task id.

        Git tasks pin their resolved commit and package tasks their name@ref, so a caller can
        record the exact task identity a run was scored against and detect a dataset that
        re-resolves differently on resume. Local-path tasks pin only their resolved path (the
        weaker identity is deliberate: hashing arbitrary task dirs is not this scorer's job).
        """
        pins: dict[str, str] = {}
        for task in self._tasks:
            task_id = task.get_task_id().get_name()
            if task.is_package_task():
                pins[task_id] = f"package:{task.name}@{task.ref or 'latest'}"
            elif task.is_git_task():
                pins[task_id] = f"git:{task.git_url}@{task.git_commit_id}"
            else:
                pins[task_id] = f"path:{task.path}"
        return pins

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
        job_dir = (config.jobs_dir / config.job_name).resolve()
        # The entry prune is destructive; two concurrent scores of one candidate in this
        # process would delete each other's in-flight trials through the shared job dir.
        with _ACTIVE_GUARD:
            if job_dir in _ACTIVE_JOB_DIRS:
                raise RuntimeError(
                    f"candidate {doc.doc_hash[:12]} is already being scored (job dir {job_dir}); "
                    "wait for the in-flight score to finish"
                )
            _ACTIVE_JOB_DIRS.add(job_dir)
        try:
            _assert_job_dir_resumable(job_dir, config)
            pruned = _prune_invalid_trial_dirs(job_dir, reward_key=self._reward_key)
            if pruned:
                logger.info(
                    "pruned %d invalid trial dir(s) under %s; harbor resume re-runs only those",
                    pruned,
                    job_dir,
                )
            run = self._runner.run(config, should_cancel=should_cancel)
        finally:
            with _ACTIVE_GUARD:
                _ACTIVE_JOB_DIRS.discard(job_dir)
        return self._project(doc, run)

    def _candidate_job(self, doc: HarnessDoc) -> JobConfig:
        template_agent = self._job_template.agents[0]
        # Constructor, not model_copy(update=): construction runs AgentConfig's validators.
        agent_fields = {name: getattr(template_agent, name) for name in AgentConfig.model_fields}
        agent_fields.update(
            {
                "name": None,
                "import_path": self._agent_import_path,
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
                    "context_window": self._context_window,
                    # Custom-bridge kwargs; collisions with the scorer-owned keys above
                    # were rejected at construction, so this merge cannot shadow them.
                    **self._extra_agent_kwargs,
                },
            }
        )
        agent = AgentConfig(**agent_fields)
        config = self._job_template.model_copy(
            update={
                # Deterministic (NOT uuid-suffixed): rescoring the same candidate resumes its
                # completed trials through harbor's native trial resume.
                "job_name": f"wmh-{doc.doc_hash[:12]}",
                "n_attempts": self._attempts,
                # source names the dataset a task came from, but candidate jobs carry no
                # datasets: harbor's Job._refresh_metrics_for_eval indexes its metrics
                # defaultdict by source, creating an empty entry for the unknown name that
                # its display hook later crashes on with IndexError. Adhoc tasks (source
                # None) use the always-present "adhoc" metrics entry.
                "tasks": [
                    task.model_copy(deep=True, update={"source": None}) for task in self._tasks
                ],
                "agents": [agent],
            },
            deep=True,
        )
        # model_copy(update=) skips validation: re-validate the exact config harbor will run.
        return _revalidated_job_config(config)

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
                outcome = self._trial_outcome(trial)
                trial_dir = run.job_dir / trial.trial_name
                cells.append(
                    ScoreCell(
                        task_id=task_id,
                        attempt=attempt,
                        reward=outcome.reward,
                        passed=reward_passed(outcome.reward, self._reward_mode),
                        artifact_dir=str(trial_dir),
                        note=outcome.note,
                        infra_failed=outcome.infra_failed,
                        # Read for every trial, including infra failures: the report is evidence,
                        # and whether it counts is an aggregation decision (a graded rate excludes
                        # `infra_failed` cells exactly as a solve rate does). A trial whose verifier
                        # died wrote no report anyway, so this is None there.
                        tests=read_trial_graded_tests(trial_dir),
                    )
                )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=self.request,
            reward_mode=self._reward_mode,
            cells=tuple(cells),
        )

    def _trial_outcome(self, trial: TrialResult) -> _TrialOutcome:
        """Classify one finished trial: a measured candidate outcome or an ungradeable failure.

        Evaluation-tolerant mode (`missing_reward="zero"`, the distillation evals) reports a
        cell as `infra_failed` exactly when the verifier wrote no reward for the configured key,
        which reaches this method in two shapes that must be treated identically:

        - Nothing was written and nothing explains it: `_official_reward` raises. The runner
          died before verification (no sandbox, dead transport, rate limit).
        - Nothing was written and a terminal verifier failure explains it
          (`_UNGRADEABLE_VERIFIER_EXCEPTIONS`): `_official_reward` substitutes a stand-in 0.0.
          The live case was two of 48 TerminalBench-2 probe trials whose E2B command stream
          stalled until `VerifierTimeoutError` after the agent had submitted real work (5,039
          and 12,459 sampled tokens); recorded as definite failures they held the measured
          solve rate at 20.8% (10/48) when the gradeable denominator was 46.

        Search mode is deliberately unchanged: it raises on the first shape and scores the
        substituted 0.0 on the second, so a deterministic verifier failure cannot wedge a
        boundary in a raise -> prune -> identical re-run loop.
        """
        try:
            reward = _official_reward(trial, reward_key=self._reward_key)
        except HarborRewardMissingError:
            if self._missing_reward == "raise":
                raise
            # The 0.0 keeps advantage estimation defined; `infra_failed` keeps it out of every
            # reported solve rate, and the note keeps the cause auditable per cell.
            return self._ungradeable_outcome(trial, reward=0.0)
        if (
            self._missing_reward == "zero"
            and _ungradeable_verifier_cause(trial, reward_key=self._reward_key) is not None
        ):
            return self._ungradeable_outcome(trial, reward=reward)
        return _TrialOutcome(reward=reward, infra_failed=False, note=_trial_note(trial))

    def _ungradeable_outcome(self, trial: TrialResult, *, reward: float) -> _TrialOutcome:
        """One trial the verifier never graded, as an auditable stand-in cell."""
        exception = trial.exception_info
        cause = exception.exception_type if exception else "no exception info"
        logger.warning(
            "harbor trial %s: the VERIFIER produced no evidence for reward key %r (%s), so this "
            "trial's reward is a stand-in; counting it as an infrastructure failure EXCLUDED "
            "from the solve rate, not as a task failure",
            trial.trial_name,
            self._reward_key,
            cause,
        )
        return _TrialOutcome(
            reward=reward,
            infra_failed=True,
            note=f"infra-failure: {cause}; no verifier evidence, excluded from solve rate",
        )


def _revalidated_job_config(config: JobConfig) -> JobConfig:
    """Re-run JobConfig validation without serializing env-bearing sections.

    model_copy(update=) skips validation, and a model_dump round-trip is NOT safe here:
    harbor's EnvironmentConfig/VerifierConfig env field serializers templatize and redact
    sensitive-named values on every dump (harbor.utils.env.templatize_sensitive_env) with no
    disabling context, so dumping would silently corrupt a literal secret whose value differs
    from os.environ. Reconstructing from field values re-runs every JobConfig validator while
    nested models pass through by reference, never through their serializers.
    """
    return JobConfig(**{name: getattr(config, name) for name in JobConfig.model_fields})


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


def _assert_job_dir_resumable(job_dir: Path, config: JobConfig) -> None:
    """Refuse to touch a job dir whose recorded config differs from the one about to run.

    Harbor itself refuses to resume such a dir (FileExistsError), so pruning it first would
    only destroy transcripts of a run that can never be resumed by this config anyway.
    JobConfig equality ignores job_name/debug, matching harbor's own resume check.
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return
    try:
        existing = JobConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(
            f"candidate job dir {job_dir} has an unreadable config.json; refusing to prune or "
            "resume it. Move or delete the directory to rerun this candidate"
        ) from error
    if existing != config:
        raise ValueError(
            f"candidate job dir {job_dir} was produced by a different job config; harbor would "
            "refuse to resume it. Move or delete the directory to rerun this candidate"
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
    result_path = TrialPaths(trial_dir).result_path
    try:
        trial = TrialResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if trial.exception_info is None:
        return True
    if trial.exception_info.exception_type in _UNGRADEABLE_VERIFIER_EXCEPTIONS:
        # `_official_reward` substitutes a 0.0 for these, so the trial projects into a cell
        # (an `infra_failed` one under evaluation-tolerant scoring); re-running a deterministic
        # verifier failure would loop forever without changing the outcome.
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


_UNGRADEABLE_VERIFIER_EXCEPTIONS = frozenset(
    {
        # The verifier's own wall clock expired before it wrote a reward. Live cause on
        # TerminalBench-2: an E2B command-stream stall (asyncio.CancelledError inside harbor's
        # `harbor/environments/e2b.py exec` -> `command_handle.wait`) on trials whose agent had
        # already submitted, so nothing about the agent's work was ever measured.
        "VerifierTimeoutError",
        # The test script exited without writing /verifier/reward.{txt,json}: no grade exists,
        # and nothing in the trial says whether the submitted work was right or wrong.
        "RewardFileNotFoundError",
        # The reward file is there but empty (a write cut off mid-flight): same, no grade.
        "RewardFileEmptyError",
        # The reward file's contents do not parse as a float/JSON: the grader's output is
        # corrupt, which is a statement about the grader, not about the agent.
        "VerifierOutputParseError",
    }
)
"""Terminal verifier failures that leave the trial with NO grade for anyone to read.

Three roles, all of them following from that one fact:

- `_official_reward` substitutes a stand-in 0.0 instead of raising, so a deterministic verifier
  failure cannot wedge a candidate boundary in a raise -> prune -> identical re-run loop.
- `_trial_dir_is_scoreable` keeps such a trial on resume, for the same reason.
- Evaluation-tolerant scoring flags the cell `infra_failed`, because "the grader never spoke" is
  an UNKNOWN outcome and reporting it as a task failure biases every solve rate downward.

Deaths BEFORE the verifier are deliberately absent (`AgentSetupTimeoutError`,
`EnvironmentStartTimeoutError`, a failed sandbox build, an upload/download failure, anything the
WMH bridge itself raises): they leave no reward at all, so the missing-reward branch of
`_trial_outcome` already classifies them from the evidence, while listing them here would make the
entry prune KEEP a transient environment failure instead of re-running it.

Pinned against harbor's own retry-exclude vocabulary in `scorer_test.py`, together with
`_GRADED_AGENT_EXCEPTIONS`, so a new harbor error type cannot silently join the wrong side."""

_GRADED_AGENT_EXCEPTIONS = frozenset(
    {
        # The agent ran out of ITS wall budget. harbor swallows this inside `_run_agent` and
        # still verifies, so the trial carries a real grade of the work the agent managed: a
        # legitimate benchmark outcome that STAYS in the solve-rate denominator, and one already
        # visible as the `cancelled-by-harbor-timeout` stop reason behind `scaffold_loss_rate`.
        "AgentTimeoutError",
        # The remaining four are `NonZeroAgentExitCodeError` subclasses harbor also swallows in
        # `_run_agent` before verifying, all raised by harbor's own installed CLI agents (WMH's
        # Python bridge never raises them). Classified anyway so a harbor upgrade cannot move one
        # onto the ungradeable side unnoticed.
        "ApiUsageLimitError",
        "AgentSafetyRefusalError",
        "AgentAuthenticationError",
        "ModelNotFoundError",
    }
)
"""Terminal agent-side failures that still reach the verifier, so a written grade is real.

These are never reclassified as infrastructure: an agent that exhausted its budget produced a
measurable outcome. When one of them leaves no reward at all the verifier still failed to grade
the trial, and the missing-reward branch of `_trial_outcome` flags that on the evidence itself."""


def _official_reward(trial: TrialResult, *, reward_key: str) -> float:
    """The verifier's reward; a terminal verifier failure substitutes 0.0, anything else raises.

    A substituted 0.0 is NOT a measurement (`_ungradeable_verifier_cause` is how a caller tells
    the two apart); it exists so search-mode scoring stays terminating.
    """
    verifier = trial.verifier_result
    rewards = None if verifier is None else verifier.rewards
    if rewards is None or reward_key not in rewards:
        exception = trial.exception_info
        if exception is not None and exception.exception_type in _UNGRADEABLE_VERIFIER_EXCEPTIONS:
            return 0.0
        available = sorted(rewards or {})
        raise HarborRewardMissingError(
            f"harbor trial {trial.trial_name!r} has no verifier reward {reward_key!r} "
            f"(available reward keys: {available or 'none'}); either the verifier never "
            "produced evidence or reward_key is misconfigured for this task set"
        )
    value = rewards[reward_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"harbor reward {reward_key!r} must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"harbor reward {reward_key!r} must be finite and in [0, 1]")
    return reward


def _ungradeable_verifier_cause(trial: TrialResult, *, reward_key: str) -> str | None:
    """The exception type behind a reward the verifier never wrote, or None if it wrote one.

    `_official_reward` substitutes a stand-in 0.0 for every cause in
    `_UNGRADEABLE_VERIFIER_EXCEPTIONS`, so a cell can otherwise carry a reward no grader ever
    produced. This is what separates that stand-in from a measured 0.0.
    """
    if _trial_reward(trial, reward_key=reward_key) is not None:
        return None
    exception = trial.exception_info
    if exception is None or exception.exception_type not in _UNGRADEABLE_VERIFIER_EXCEPTIONS:
        return None
    return exception.exception_type


def _trial_note(trial: TrialResult) -> str:
    exception = trial.exception_info
    return "completed" if exception is None else f"completed with {exception.exception_type}"
