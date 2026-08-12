"""Tests for projecting harbor job results into harness score reports (fakes only)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from harbor.agents.installed import base as harbor_installed_agent
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, ModelInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial import errors as harbor_trial_errors
from harbor.verifier import verifier as harbor_verifier

import wmo.runtime.evaluation.harbor.scorer as scorer_module
from wmo.common.providers.base import ProviderConfig, ProviderKind
from wmo.runtime.evaluation.harbor.agent import WMO_HARBOR_AGENT_IMPORT_PATH
from wmo.runtime.evaluation.harbor.e2b_template_policy import WMO_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
from wmo.runtime.evaluation.harbor.scorer import (
    HarborJobRunner,
    HarborRewardMissingError,
    HarborRun,
    HarborScorer,
    MissingRewardMode,
)
from wmo.runtime.harness.doc import HarnessDoc
from wmo.runtime.harness.scoring import GradedTests, RewardMode

_JOB_ID = UUID("00000000-0000-4000-8000-000000000001")
_SUFFIXES = ("a7Hm2Ks", "m4Vx8Pa", "z9Tc3Wb", "q6Rn5Jd")


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
            name="wmo-harness",
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


def _write_ctrf(trial_dir: Path, passed: int, failed: int) -> None:
    """Write one trial's CTRF report the way harbor's pytest verifier does."""
    report = {
        "results": {
            "tool": {"name": "pytest", "version": "8.4.1"},
            "summary": {
                "tests": passed + failed,
                "passed": passed,
                "failed": failed,
                "skipped": 0,
                "pending": 0,
                "other": 0,
            },
            "tests": [
                {"name": f"test_outputs.py::test_{index}", "status": "passed"}
                for index in range(passed)
            ]
            + [
                {"name": f"test_outputs.py::test_f{index}", "status": "failed"}
                for index in range(failed)
            ],
        }
    }
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "ctrf.json").write_text(json.dumps(report), encoding="utf-8")


class _Runner:
    """Materializes trial dirs under the candidate's deterministic job dir, like harbor."""

    def __init__(
        self, trials: list[TrialResult], ctrf: dict[str, tuple[int, int]] | None = None
    ) -> None:
        self.trials = trials
        self.configs: list[JobConfig] = []
        self.ctrf = ctrf or {}
        """Per-trial-name (passed, failed) test counts to write as a CTRF report, like a pytest
        verifier does; a trial absent from this mapping gets no report at all."""

    def run(
        self,
        config: JobConfig,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> HarborRun:
        del should_cancel
        self.configs.append(config)
        job_dir = config.jobs_dir / config.job_name
        for trial in self.trials:
            trial_dir = job_dir / trial.trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            (trial_dir / "result.json").write_text(trial.model_dump_json(), encoding="utf-8")
            counts = self.ctrf.get(trial.trial_name)
            if counts is not None:
                _write_ctrf(trial_dir, *counts)
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
    missing_reward: MissingRewardMode = "raise",
    ctrf: dict[str, tuple[int, int]] | None = None,
) -> tuple[HarborScorer, _Runner]:
    runner = _Runner(trials, ctrf)
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, task_ids),
        provider_config=_provider(),
        reward_mode=reward_mode,
        attempts=attempts,
        harness_backend="e2b",
        e2b_template="pi-template",
        agent_concurrency=agent_concurrency,
        missing_reward=missing_reward,
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
    expected_dir = tmp_path / "jobs" / f"wmo-{candidate.doc_hash[:12]}"
    assert by_task["task-a"][0].artifact_dir == str(expected_dir / trials[0].trial_name)

    [config] = runner.configs
    # Deterministic per-candidate job dir: the harbor-native trial-resume key.
    assert config.jobs_dir / config.job_name == scorer.candidate_job_dir(candidate)
    assert config.job_name == f"wmo-{candidate.doc_hash[:12]}"
    [agent] = config.agents
    assert agent.import_path == WMO_HARBOR_AGENT_IMPORT_PATH
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
        _trial(tmp_path, "task-a", 1, reward=None, exception="RuntimeError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, missing)
    with pytest.raises(HarborRewardMissingError, match="no verifier reward"):
        scorer.score(HarnessDoc.baseline())


def test_missing_reward_zero_mode_scores_the_trial_failed_instead_of_halting(
    tmp_path: Path,
) -> None:
    """Distillation evals: a runner that died before verification is an INFRA failure.

    The search default stays strict (previous test); "zero" scores the cell 0.0 with an auditable
    note so a single transient sandbox death cannot abort a long training run at its final eval.
    The cell is also flagged `infra_failed`: the 0.0 is a stand-in that keeps advantage estimation
    defined, and reporting it as a task failure is how three Super `student-before` baselines were
    published as 0.0% from 51/51 rate-limited trials.
    """
    trials = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="RuntimeError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, missing_reward="zero")
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0
    assert cell.passed is False
    assert cell.note == (
        "infra-failure: RuntimeError; no verifier evidence, excluded from solve rate"
    )
    assert cell.infra_failed is True
    # A trial the verifier really scored is never flagged, whatever its reward.
    assert report.by_task()["task-b"][0].infra_failed is False
    assert report.by_task()["task-b"][0].reward == 1.0


def test_search_mode_still_scores_terminal_verifier_failures_zero_instead_of_halting(
    tmp_path: Path,
) -> None:
    """Search mode is unchanged by the ungradeable classification.

    A terminal verifier failure keeps its stand-in 0.0 and its plain note here, because raising
    would wedge the boundary in a raise/prune/identical-re-run loop: the entry prune keeps such a
    trial dir precisely so harbor does not re-run a deterministic verifier failure forever.
    `infra_failed` stays False because search-mode reports never carry it (see `ScoreCell`)."""
    trials = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="VerifierTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, reward_mode="positive-binary")
    report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0
    assert cell.passed is False
    assert cell.note == "completed with VerifierTimeoutError"
    assert cell.infra_failed is False
    assert report.score == 0.5

    misconfigured = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, misconfigured)
    scorer._reward_key = "grade"  # a clean trial + wrong key is a config error, not infra
    with pytest.raises(HarborRewardMissingError, match=r"available reward keys: \['reward'\]"):
        scorer.score(HarnessDoc.baseline())


@pytest.mark.parametrize(
    "exception",
    sorted(scorer_module._UNGRADEABLE_VERIFIER_EXCEPTIONS),
)
def test_an_ungradeable_verifier_failure_is_an_infra_failure_not_a_task_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    exception: str,
) -> None:
    """The live measurement bug: a stand-in 0.0 read as a definite task failure.

    Exact shape from a 48-episode TerminalBench-2 probe, twice (4.2%):
    `stop_reason=submitted`, thousands of sampled tokens, `reward=0.0`, `passed=False`,
    `infra_failed=False`, and a `result.json` whose only explanation is
    `VerifierTimeoutError: Verifier execution timed out after 900.0 seconds` over an E2B
    command-stream stall. The agent submitted real work and nobody graded it, so the outcome is
    UNKNOWN; recorded as a failure it held the probe's solve rate at 10/48 = 20.8% when its
    gradeable denominator was 46.
    """
    trials = [
        _trial(tmp_path, "task-a", 1, reward=None, exception=exception),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, trials, missing_reward="zero")
    with caplog.at_level("WARNING"):
        report = scorer.score(HarnessDoc.baseline())
    cell = report.by_task()["task-a"][0]
    assert cell.reward == 0.0  # a stand-in, so advantage estimation stays defined
    assert cell.passed is False
    assert cell.infra_failed is True
    assert cell.note == (
        f"infra-failure: {exception}; no verifier evidence, excluded from solve rate"
    )
    # Greppable in a 26-hour run, and it names the verifier as the thing that failed.
    assert "the VERIFIER produced no evidence for reward key 'reward'" in caplog.text
    assert trials[0].trial_name in caplog.text
    assert cell.artifact_dir.endswith(trials[0].trial_name)
    assert report.by_task()["task-b"][0].infra_failed is False


def test_an_agent_timeout_stays_a_measured_result_in_the_solve_rate_denominator(
    tmp_path: Path,
) -> None:
    """AgentTimeoutError is NOT reclassified, in either mode.

    harbor swallows it inside `_run_agent` and still verifies, so the trial carries a real grade
    of the work the agent managed inside its wall budget: a legitimate benchmark outcome, already
    visible through `stop_reason` / `scaffold_loss_rate`. Only the absence of a grade makes a
    trial ungradeable, so a passing agent-timeout trial still counts as a pass.
    """
    trials = [
        _trial(tmp_path, "task-a", 1, reward=0.0, exception="AgentTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0, exception="AgentTimeoutError"),
    ]
    scorer, _runner = _scorer(tmp_path, trials, missing_reward="zero")
    report = scorer.score(HarnessDoc.baseline())
    assert [cell.infra_failed for cell in report.cells] == [False, False]
    assert [cell.note for cell in report.cells] == ["completed with AgentTimeoutError"] * 2
    assert report.score == 0.5

    # The complement: an agent timeout whose verifier ALSO produced nothing has no grade behind
    # it either (harbor records only the first exception), so it is infra like any other.
    ungraded = [
        _trial(tmp_path, "task-a", 1, reward=None, exception="AgentTimeoutError"),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _runner = _scorer(tmp_path, ungraded, missing_reward="zero")
    report = scorer.score(HarnessDoc.baseline())
    assert report.by_task()["task-a"][0].infra_failed is True


def test_eval_tolerant_infra_failure_means_exactly_no_verifier_reward(tmp_path: Path) -> None:
    """The whole rule, stated as one invariant over a mixed batch.

    `infra_failed` is not a shape of exception, it is the answer to "did the verifier write a
    reward for this trial?". Anything else needs a second reason to be right, and the two
    previously wrong answers (a rate-limited trial reported as 0%, an ungradeable submission
    reported as a failure) were exactly the cases where the shape and the evidence disagreed.
    """
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),  # graded, clean
        _trial(tmp_path, "task-b", 1, reward=0.0, exception="AgentTimeoutError"),  # graded
        _trial(tmp_path, "task-c", 1, reward=None, exception="VerifierTimeoutError"),  # ungraded
        _trial(tmp_path, "task-d", 1, reward=None, exception="RuntimeError"),  # never ran
    ]
    scorer, _runner = _scorer(
        tmp_path,
        trials,
        task_ids=("task-a", "task-b", "task-c", "task-d"),
        missing_reward="zero",
    )
    report = scorer.score(HarnessDoc.baseline())
    graded = {
        trial.trial_name: trial.verifier_result is not None for trial in trials
    }  # the evidence itself
    assert {Path(cell.artifact_dir).name: cell.infra_failed for cell in report.cells} == {
        name: not was_graded for name, was_graded in graded.items()
    }


def test_cells_carry_the_graded_test_breakdown_beside_the_binary_reward(tmp_path: Path) -> None:
    """The graded score rides along per cell; the reward and `passed` are untouched.

    Four trials, four states: a partial pass the binary reward calls a total failure, a full pass,
    a graded trial whose verifier wrote no CTRF report, and one whose verifier timed out (no reward
    and no report). The last two carry `tests=None`, which is an ABSENT measurement, not a 0.0.
    """
    trials = [
        _trial(tmp_path, "task-a", 1, reward=0.0),
        _trial(tmp_path, "task-b", 1, reward=1.0),
        _trial(tmp_path, "task-c", 1, reward=0.0),
        _trial(tmp_path, "task-d", 1, reward=None, exception="VerifierTimeoutError"),
    ]
    scorer, _runner = _scorer(
        tmp_path,
        trials,
        task_ids=("task-a", "task-b", "task-c", "task-d"),
        missing_reward="zero",
        ctrf={trials[0].trial_name: (1, 1), trials[1].trial_name: (2, 0)},
    )

    report = scorer.score(HarnessDoc.baseline())

    cells = {cell.task_id: cell for cell in report.cells}
    assert (cells["task-a"].reward, cells["task-a"].graded_score) == (0.0, 0.5)
    assert cells["task-a"].tests == GradedTests(passed=1, resolved=2)
    assert (cells["task-b"].reward, cells["task-b"].graded_score) == (1.0, 1.0)
    # A reward with no report, and no reward with no report: both have no graded score at all.
    assert (cells["task-c"].reward, cells["task-c"].graded_score) == (0.0, None)
    assert cells["task-c"].infra_failed is False
    assert cells["task-d"].graded_score is None
    assert cells["task-d"].infra_failed is True
    # The headline is unchanged by any of this: binary is still 1 of 3 gradeable trials.
    assert report.score == 0.25  # equal task weight over the four requested tasks


def test_terminal_harbor_exceptions_are_classified_on_exactly_one_side() -> None:
    """Pin both sides of the taxonomy against harbor's own vocabulary.

    The classification is matched as STRINGS against `exception_info.exception_type`, so a typo
    or an upstream rename silently sends a trial to the wrong side; a harbor upgrade that adds a
    terminal exception must fail here until someone decides whether it means "we could not grade
    this" or "the agent genuinely did not finish".
    """
    ungradeable = scorer_module._UNGRADEABLE_VERIFIER_EXCEPTIONS
    graded = scorer_module._GRADED_AGENT_EXCEPTIONS
    assert ungradeable == {
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
    assert not ungradeable & graded
    # harbor's default retry-exclude set is its own list of terminal (never-retried) trial
    # failures, and it is exactly the union of the two sides.
    assert ungradeable | graded == RetryConfig().exclude_exceptions
    # Every name must resolve to a real harbor exception class, or the match never fires.
    modules = (harbor_trial_errors, harbor_verifier, harbor_installed_agent)
    for name in ungradeable | graded:
        assert any(isinstance(getattr(module, name, None), type) for module in modules), name
    # Pre-verifier deaths belong to NEITHER side: they leave no reward at all, so the
    # missing-reward branch already flags them as infra, and adding them to the ungradeable set
    # would make the entry prune KEEP a transient environment failure instead of re-running it.
    assert not (ungradeable | graded) & {
        harbor_trial_errors.AgentSetupTimeoutError.__name__,
        harbor_trial_errors.EnvironmentStartTimeoutError.__name__,
    }


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
    verifier_timeout = _trial(tmp_path, "task-b", 4, reward=None, exception="VerifierTimeoutError")
    (job_dir / verifier_timeout.trial_name).mkdir()
    (job_dir / verifier_timeout.trial_name / "result.json").write_text(
        verifier_timeout.model_dump_json(), encoding="utf-8"
    )

    scorer.score(candidate)

    assert (job_dir / keep.trial_name).is_dir()  # written reward: a kept candidate outcome
    assert not unparseable.exists()
    assert not missing_result.exists()
    assert not (job_dir / crashed.trial_name).exists()
    # Outcome-shaped verifier failure: scored 0, so re-running it would loop forever.
    assert (job_dir / verifier_timeout.trial_name).is_dir()
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
    assert config.environment.import_path == WMO_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH
    assert config.environment.kwargs == {"keep": "me"}  # options survive the routing rewrite


def test_sensitive_env_values_survive_scorer_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """harbor redacts sensitive-named env values on every model_dump when the literal differs
    from os.environ; the scorer's re-validation must never route env-bearing sections through
    that serializer."""
    monkeypatch.delenv("TASK_API_KEY", raising=False)
    monkeypatch.delenv("GRADER_TOKEN", raising=False)
    template = _job_template(tmp_path)
    template.environment.env = {"TASK_API_KEY": "literal-secret-12345"}
    template.verifier.env = {"GRADER_TOKEN": "grader-secret-6789"}
    scorer = HarborScorer(
        job_template=template,
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert config.environment.env == {"TASK_API_KEY": "literal-secret-12345"}
    assert config.verifier.env == {"GRADER_TOKEN": "grader-secret-6789"}


def test_candidate_job_tasks_drop_their_dataset_source(tmp_path: Path) -> None:
    """Candidate jobs pin tasks directly with no datasets; a surviving dataset source name
    poisons harbor's metrics defaultdict (empty entry for the unknown dataset) and crashes
    its per-trial display hook with IndexError."""
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a", "task-b")),
        provider_config=_provider(),
        agent_concurrency=1,
        runner=_Runner([]),
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    assert [task.source for task in config.tasks] == [None, None]


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
    with pytest.raises(ValueError, match="command_timeout_sec must be an integer"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=tasks,
            provider_config=_provider(),
            harness_backend="e2b",
            agent_concurrency=1,
            command_timeout_sec=0,
        )


def test_custom_agent_import_path_and_extra_kwargs_thread_into_the_job(tmp_path: Path) -> None:
    """A custom bridge retains its extra kwargs next to every scorer-owned setting."""
    scorer = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        e2b_template="pi-template",
        agent_concurrency=1,
        agent_import_path="example.agents:CustomAgent",
        extra_agent_kwargs={"trace_sink_dir": "/runs/traces"},
        runner=_Runner([]),
    )
    candidate = HarnessDoc.baseline()
    config = scorer._candidate_job(candidate)
    [agent] = config.agents
    assert agent.import_path == "example.agents:CustomAgent"
    assert agent.kwargs["trace_sink_dir"] == "/runs/traces"
    assert agent.kwargs["harness"] == candidate.model_dump(mode="json")
    assert agent.kwargs["harness_backend"] == "e2b"
    assert agent.kwargs["e2b_template"] == "pi-template"


def test_agent_model_name_overrides_the_provider_provenance_label(tmp_path: Path) -> None:
    """Harbor's terminus-2 reads AgentConfig.model_name as a REAL model identity.

    It picks its renderer and tokenizer from it. The default WMO bridge keeps the provider
    identity as provenance.
    """
    default = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        runner=_Runner([]),
    )
    [agent] = default._candidate_job(HarnessDoc.baseline()).agents
    assert agent.model_name == "bedrock/worker-model"

    overridden = HarborScorer(
        job_template=_job_template(tmp_path),
        tasks=_tasks(tmp_path, ("task-a",)),
        provider_config=_provider(),
        harness_backend="e2b",
        agent_concurrency=1,
        agent_import_path="harbor.agents.terminus_2.terminus_2:Terminus2",
        agent_model_name="Qwen/Qwen3-8B",
        runner=_Runner([]),
    )
    [agent] = overridden._candidate_job(HarnessDoc.baseline()).agents
    assert agent.model_name == "Qwen/Qwen3-8B"
    # Provider identity remains in the kwargs for provenance checks.
    assert agent.kwargs["provider_config"]["model"] == "worker-model"

    with pytest.raises(ValueError, match="agent_model_name must be a nonempty string"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=_tasks(tmp_path, ("task-a",)),
            provider_config=_provider(),
            harness_backend="e2b",
            agent_concurrency=1,
            agent_model_name="",
        )


def test_extra_agent_kwargs_and_import_path_are_validated(tmp_path: Path) -> None:
    """Extras may extend the agent kwargs but never shadow the scorer-owned ones."""
    with pytest.raises(ValueError, match=r"scorer-owned agent kwargs \['harness'\]"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=_tasks(tmp_path, ("task-a",)),
            provider_config=_provider(),
            harness_backend="e2b",
            agent_concurrency=1,
            extra_agent_kwargs={"harness": {}, "token_sink_dir": "/tmp/x"},
        )
    with pytest.raises(ValueError, match="module:Class"):
        HarborScorer(
            job_template=_job_template(tmp_path),
            tasks=_tasks(tmp_path, ("task-a",)),
            provider_config=_provider(),
            harness_backend="e2b",
            agent_concurrency=1,
            agent_import_path="not-an-import-path",
        )


def test_create_threads_agent_import_path_and_extra_kwargs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_resolve(
        _dataset: DatasetConfig,
        task_ids: list[str],
    ) -> list[TaskConfig]:
        return _tasks(tmp_path, tuple(task_ids))

    monkeypatch.setattr(scorer_module, "resolve_harbor_tasks", fake_resolve)
    scorer = asyncio.run(
        HarborScorer.create(
            _job_template(tmp_path),
            ["task-a"],
            provider_config=_provider(),
            harness_backend="e2b",
            e2b_template="pi-template",
            agent_concurrency=1,
            agent_import_path="example.agents:CustomAgent",
            extra_agent_kwargs={"trace_sink_dir": "/runs/traces"},
        )
    )
    config = scorer._candidate_job(HarnessDoc.baseline())
    [agent] = config.agents
    assert agent.import_path == "example.agents:CustomAgent"
    assert agent.kwargs["trace_sink_dir"] == "/runs/traces"


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


def test_concurrent_scores_of_one_candidate_are_rejected(tmp_path: Path) -> None:
    """The entry prune is destructive; a second in-process score of the same doc must not be
    able to delete the first one's in-flight trials."""
    candidate = HarnessDoc.baseline()
    trials = [
        _trial(tmp_path, "task-a", 1, reward=1.0),
        _trial(tmp_path, "task-b", 1, reward=1.0),
    ]
    scorer, _inner = _scorer(tmp_path, trials)
    reentered: list[str] = []

    class _ReentrantRunner(_Runner):
        def run(
            self,
            config: JobConfig,
            *,
            should_cancel: Callable[[], bool] | None = None,
        ) -> HarborRun:
            with pytest.raises(RuntimeError, match="already being scored"):
                scorer.score(candidate)
            reentered.append(config.job_name)
            return super().run(config)

    scorer._runner = _ReentrantRunner(trials)
    report = scorer.score(candidate)
    assert reentered == [f"wmo-{candidate.doc_hash[:12]}"]
    assert report.doc_hash == candidate.doc_hash
    # The guard releases after the score; the same candidate is scoreable again.
    scorer._runner = _Runner(trials)
    scorer.score(candidate)


def test_job_dir_with_a_different_recorded_config_is_never_pruned(tmp_path: Path) -> None:
    """Harbor refuses to resume a dir whose config changed; pruning it first would only
    destroy transcripts. Raise before touching anything."""
    candidate = HarnessDoc.baseline()
    scorer, runner = _scorer(
        tmp_path,
        [
            _trial(tmp_path, "task-a", 1, reward=1.0),
            _trial(tmp_path, "task-b", 1, reward=1.0),
        ],
    )
    job_dir = scorer.candidate_job_dir(candidate)
    other = scorer._candidate_job(candidate).model_copy(update={"n_attempts": 7})
    job_dir.mkdir(parents=True)
    (job_dir / "config.json").write_text(other.model_dump_json(), encoding="utf-8")
    broken = job_dir / "task-a__stale00"
    broken.mkdir()
    (broken / "result.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="different job config"):
        scorer.score(candidate)
    assert broken.is_dir()  # nothing was deleted
    assert runner.configs == []  # and nothing ran


def test_should_cancel_is_polled_during_the_running_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-hour harbor boundary must be cancellable mid-job through the Scorer contract."""
    polled = threading.Event()

    class _HangingJob:
        job_dir = tmp_path / "jobs" / "wmo-hanging"

        @classmethod
        async def create(cls, _config: JobConfig) -> _HangingJob:
            return cls()

        async def run(self) -> JobResult:
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    def cancel_requested() -> bool:
        polled.set()
        return True

    monkeypatch.setattr(scorer_module, "Job", _HangingJob)
    runner = scorer_module.HarborJobRunner(poll_interval_s=0.01)
    with pytest.raises(scorer_module.HarnessSearchCancelled, match="mid-job"):
        runner.run(_job_template(tmp_path), should_cancel=cancel_requested)
    assert polled.is_set()


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


def test_wmo_import_pulls_neither_harbor_nor_e2b() -> None:
    """The packaging contract keeps optional imports out of the core packages.

    Importing `wmo` and `wmo.simulation.evaluation` stays clean, and even the harbor subpackage
    never imports the e2b SDK (that loads only through harbor's
    environment factory when a job actually targets E2B)."""
    code = (
        "import sys\n"
        "import wmo, wmo.simulation.evaluation\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('harbor', 'e2b')]\n"
        "assert not bad, f'eager optional imports: {bad}'\n"
        "import wmo.runtime.evaluation.harbor\n"
        "assert 'harbor' in sys.modules\n"
        "bad = [m for m in sys.modules if m.split('.')[0] == 'e2b']\n"
        "assert not bad, f'harbor subpackage must not import the e2b SDK: {bad}'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
