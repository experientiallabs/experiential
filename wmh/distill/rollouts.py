"""Collect one distillation training step's rollouts as real harbor trials.

`collect_rollouts` turns one train batch (task ids x group size) into scored
harbor trials of the pi agent sampling the Tinker student, with every trial's
exact token spans captured to a per-step sink directory and joined back into
`TrialRecord`s. Two directory choices are load-bearing:

- The harbor jobs dir is FRESH per training step (`run_dir/harbor/step-NNNN`):
  harbor job dirs are keyed by the candidate doc hash only, so reusing one
  jobs dir across steps would resume step N's completed trials as step N+1's
  "results", carrying tokens sampled from the previous weights.
- Token sinks live OUTSIDE the trial dirs (`run_dir/tokens/step-NNNN`): the
  scorer's entry prune deletes invalid trial dirs wholesale before re-running
  them, and a sink inside the trial dir would vanish with it.
- A job dir left by a PREVIOUS SESSION under different sampler weights is
  wiped before scoring (`_wipe_stale_policy_dir`): sampler paths carry a
  per-session nonce, so such a dir can never satisfy the scorer's strict
  job-config resume check, and its trials sampled a policy this session did
  not restore. Wiping makes a crash-resumed step re-run whole from the
  current weights. A dir whose recorded provider matches (the teacher's
  stable identity) is kept for harbor's native trial-level resume.

The harbor SDK is an optional extra imported lazily here, the same contract
as the CLI's harbor commands; `import wmh.distill.rollouts` succeeds without
it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from wmh.core.types import JsonObject
from wmh.distill.config import DistillConfig
from wmh.distill.tokens import TrialRecord, assemble_trial_records
from wmh.harness.doc import HarnessDoc
from wmh.harness.runtime import StopReason
from wmh.providers.base import ProviderConfig

logger = logging.getLogger(__name__)

MISSING_HARBOR_EXTRA = (
    "the harbor SDK is not installed; distillation rollouts run as real harbor trials. "
    "Run `uv sync --extra harbor` (or `pip install 'world-model-harness[harbor]'`) and retry"
)

E2B_SANDBOXES_PER_TRIAL = 2
"""Concurrent E2B sandboxes one `harbor.backend = "e2b"` trial holds at once.

`collect_rollouts` selects `task_environment="e2b"` AND `harness_backend="e2b"` together, so a
running trial occupies harbor's task environment sandbox plus the pooled sandbox hosting the pi
harness process. Capacity planning (`wmh.cli.harness_distill`) multiplies by this, because a run
that reserves only one slot per trial starves at exactly half its configured concurrency."""


def _recorded_provider_config(config_path: Path) -> JsonObject | None:
    """The provider config a persisted harbor job config ran with, or None.

    None means the file is unreadable or not shaped like a scorer-produced
    JobConfig dump; callers leave those dirs alone so the scorer can raise its
    own actionable error instead of evidence being destroyed silently.
    """
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    agents = payload.get("agents")
    if not isinstance(agents, list) or not agents or not isinstance(agents[0], dict):
        return None
    kwargs = agents[0].get("kwargs")
    if not isinstance(kwargs, dict):
        return None
    recorded = kwargs.get("provider_config")
    return recorded if isinstance(recorded, dict) else None


def _wipe_stale_policy_dir(
    candidate_dir: Path, provider_config: ProviderConfig, token_sink_dir: Path
) -> bool:
    """Wipe a candidate job dir recorded under different provider weights.

    Sampler paths carry a per-session nonce, so a job dir left by a previous
    session (a crash mid-batch, then `--resume`) can never satisfy the
    scorer's strict job-config resume check, and its completed trials sampled
    a policy this session did not restore, so resuming them would mix
    policies. Deleting the dir (and the batch's token sinks with it) makes the
    batch re-run whole from the current weights: correct, at the price of
    re-running that batch's completed trials.

    A recorded provider that MATCHES the current one (the teacher's stable
    identity, whose baseline eval legitimately resumes across sessions) is
    left for harbor's native trial-level resume, as is an unreadable config
    (the scorer raises its own actionable error for those).

    Args:
        candidate_dir: The scorer's deterministic per-candidate job dir.
        provider_config: The provider the batch is about to sample.
        token_sink_dir: The batch's token sink dir, wiped and recreated
            alongside the job dir so no stale spans survive.

    Returns:
        True when the directory was wiped.
    """
    recorded = _recorded_provider_config(candidate_dir / "config.json")
    if recorded is None or recorded == provider_config.model_dump(mode="json"):
        return False
    logger.warning(
        "harbor job dir %s was produced under provider %r, not the current %r; "
        "wiping it (and its token sinks) so the batch re-runs from the current "
        "weights instead of resuming another policy's trials",
        candidate_dir,
        recorded.get("model"),
        provider_config.model,
    )
    shutil.rmtree(candidate_dir)
    shutil.rmtree(token_sink_dir, ignore_errors=True)
    token_sink_dir.mkdir(parents=True, exist_ok=True)
    return True


class RolloutStats(BaseModel):
    """Aggregate health metrics for one collected rollout batch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trials: int = Field(ge=0)
    trials_with_spans: int = Field(ge=0)
    solve_rate: float = Field(ge=0.0, le=1.0)
    """Fraction of EXECUTED trials whose verifier reward passed (metrics/gating signal).

    Infrastructure failures are excluded from the denominator: a trial with no verifier evidence
    behind its stand-in 0.0 is an UNKNOWN outcome, and counting it as a task failure biases this
    rate in whichever direction the failures fell. Both directions have been measured: three Super
    `student-before` baselines were reported as 0.0% from 51/51 rate-limited trials, and 2 of 48
    TerminalBench-2 probe trials whose verifier timed out on submitted work held a probe at 20.8%
    when its gradeable denominator was 46. 0.0 when nothing executed (`executed_trials == 0`),
    which callers must treat as a null measurement rather than a score."""

    graded_solve_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Mean GRADED test-pass score over the trials that have one: the power metric beside
    `solve_rate`.

    Same trials, read at test resolution instead of the benchmark's one bit (see
    `wmh.harness.scoring.GradedTests`). `solve_rate` stays the headline because binary IS the
    benchmark's definition of success; this exists because a small binary holdout cannot resolve a
    real effect. On the 48-episode TerminalBench-2 probe it reads 0.319 against a 0.217 binary solve
    rate, and it moves 2 of 12 tasks off a flat 0.00 where no improvement could ever have shown up.

    Denominator: trials that are NOT `infra_failed` AND carry a readable test report
    (`graded_trials`). A verifier that timed out wrote no report either, so such a trial is excluded
    here exactly as it is from `solve_rate`, never averaged in as 0.0. 0.0 when there are none,
    which callers must read as a null measurement, not a score.

    Coarse, not continuous: the probe's tasks carried 1 to 6 tests, so most trial scores are 0, 1/2,
    or 1, and a single-test task is exactly as binary as its reward."""

    graded_trials: int = Field(default=0, ge=0)
    """Gradeable trials that carried a readable test report: the `graded_solve_rate` denominator.

    Below `executed_trials` means some graded trials produced a reward but no parseable test report,
    so the two rates ran over different trial sets; the gap is what says so instead of a fabricated
    0.0 hiding it."""

    raw_solve_rate: float = Field(ge=0.0, le=1.0)
    """Passing trials over ALL trials, infra failures included.

    What the training path actually optimizes against (`missing_reward="zero"` makes a dead trial a
    failed trial for advantage estimation), kept alongside `solve_rate` so the two can be compared
    instead of one silently standing in for the other."""

    executed_trials: int = Field(ge=0)
    """Trials that produced verifier evidence (`trials` minus `infra_failed_trials`)."""

    infra_failed_trials: int = Field(ge=0)
    """Trials with no verifier evidence, so no measured outcome.

    One count over two causes, which share this denominator treatment but not their diagnosis:
    the agent never ran (sandbox creation, rate limit, transport death) or the agent ran and was
    never graded (verifier timeout, unwritten/unparseable reward file). The per-cell note the
    scorer writes (`infra-failure: <exception type>; ...`) is what tells them apart."""

    empty_span_trials: int = Field(ge=0)
    """Trials that recorded no token span (died before the first completion)."""

    stop_reason_counts: dict[str, int] = Field(default_factory=dict)
    """How many trials ended on each recorded stop reason (`"unknown"` when no trace was readable).

    The per-reason breakdown behind `scaffold_loss_rate`: `max_turns` (turn cap), `budget` (wall
    clock), `no_tool_call`, `output_truncated`, `unparsed_tool_call`, `provider_error`, and
    `submitted`."""

    scaffold_loss_rate: float = Field(ge=0.0, le=1.0)
    """Share of trials WITH A READABLE STOP REASON that never reached an explicit `submit`.

    The headline number of the pi/Nemotron-3 scaffold audit: it was 88.8% for Super and 92.2% for
    Ultra, every one of those trials scored reward 0, and nothing in the metrics surfaced it. An
    episode the harness cut off measures where the guillotine fell, not what the model can do, so
    this belongs beside every solve rate. 0.0 when no trial reported a stop reason.

    Deliberately NOT the `executed` denominator that `solve_rate` uses, because the two answer
    different questions. "Did the harness cut this episode off?" is answered by the stop reason
    alone and needs no verifier; "did the model solve it?" needs a grade. An episode that reached
    `submit` and then had its VERIFIER time out is a scaffold SUCCESS whose task outcome is
    unknown, so it belongs in this denominator (as a non-loss) while being excluded from
    `solve_rate`. Sharing one denominator conflated them and inflated this rate: on the 48-episode
    probe it read 15.22% over 46 gradeable trials when the true scaffold loss was 14.58% (7 of 48
    episodes reported a non-submit stop reason). A trial with no readable trace at all has no
    stop reason and is excluded from both."""

    # TODO: surface TokenRecorder.fallback_count (incremental-prompt re-renders) here;
    # it needs the per-trial span sink format to carry the counter across the
    # agent-process boundary, so it is not trivially wireable today.


UNKNOWN_STOP_REASON = "unknown"
"""Stop-reason bucket for a trial whose run trace was missing or unreadable."""


def rollout_stats(records: Sequence[TrialRecord]) -> RolloutStats:
    """The batch health stats over a set of assembled trial records.

    A pure function of the records, so a batch loaded back from persisted
    trial records (the warmup-trials manifest) reports the same stats its
    original collection did.

    Args:
        records: The batch's trial records.

    Returns:
        The aggregate stats. An empty batch, and a batch where every trial was
        an infrastructure failure, both report a 0.0 solve rate over zero
        executed trials: those are null measurements, and the counts are what
        distinguish them. `scaffold_loss_rate` runs over its own denominator
        (trials that reported a stop reason), so a batch whose agents all ran
        but whose verifiers all failed still reports a real scaffold rate.
        `graded_solve_rate` runs over `graded_trials` (gradeable trials with a
        readable test report) and is 0.0 over zero of them, likewise a null
        measurement rather than a score.
    """
    with_spans = sum(1 for record in records if record.spans)
    executed = [record for record in records if not record.infra_failed]
    # The graded rate's own denominator: gradeable trials whose verifier also left a readable test
    # report. A missing report is an absent measurement, so it is excluded rather than scored 0.0,
    # the same rule that keeps an ungradeable trial out of `solve_rate`.
    graded = [record.graded_score for record in executed if record.graded_score is not None]
    counts: dict[str, int] = {}
    for record in records:
        key = record.stop_reason or UNKNOWN_STOP_REASON
        counts[key] = counts.get(key, 0) + 1
    # The scaffold question ("did the harness cut this off?") is answered by the stop reason and
    # needs no verifier, so it gets its own denominator: every trial that reported one. Sharing
    # `executed` with solve_rate dropped submit-then-ungradeable episodes out of a rate they
    # belong in as successes, which inflated it.
    with_stop_reason = [record for record in records if record.stop_reason]
    submitted = sum(
        1 for record in with_stop_reason if record.stop_reason == StopReason.SUBMITTED.value
    )
    return RolloutStats(
        trials=len(records),
        trials_with_spans=with_spans,
        solve_rate=(
            sum(1 for record in executed if record.passed) / len(executed) if executed else 0.0
        ),
        graded_solve_rate=(sum(graded) / len(graded) if graded else 0.0),
        graded_trials=len(graded),
        raw_solve_rate=(
            sum(1 for record in records if record.passed) / len(records) if records else 0.0
        ),
        executed_trials=len(executed),
        infra_failed_trials=len(records) - len(executed),
        empty_span_trials=len(records) - with_spans,
        stop_reason_counts=dict(sorted(counts.items())),
        scaffold_loss_rate=(
            (len(with_stop_reason) - submitted) / len(with_stop_reason) if with_stop_reason else 0.0
        ),
    )


def collect_rollouts(
    step_index: int,
    task_ids: Sequence[str],
    cfg: DistillConfig,
    harness: HarnessDoc,
    provider_config: ProviderConfig,
    run_dir: Path,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[TrialRecord], RolloutStats]:
    """Run one train batch of harbor trials and join rewards with token spans.

    Each task in `task_ids` runs `cfg.train.group_size` attempts (the
    on-policy group), every trial sampling the student through the distill
    agent bridge, which enforces the tinker provider kind and records the
    trial's spans to `run_dir/tokens/step-NNNN/{trial_name}.jsonl`.

    This is a synchronous entry point, mirroring how the CLI drives the
    scorer: the async `HarborScorer.create` runs on its own loop and the
    (task x attempts) batch runs inside the blocking `score` call.

    Args:
        step_index: 0-based training step; keys the fresh jobs dir and the
            token sink dir.
        task_ids: Exact task ids for this batch (a subset of the train split).
        cfg: The validated run config (harbor template/backend/reward key,
            group size, trial concurrency).
        harness: The pinned harness document the pi agent runs.
        provider_config: The student provider config; `model` must point at
            the CURRENT sampler weights (`tinker://` path).
        run_dir: The distillation run directory.
        should_cancel: Optional cooperative cancellation poll, forwarded to
            the harbor runner.

    Returns:
        The `TrialRecord`s (one per task x attempt, in report order) and the
        batch stats. A trial with no sink file is recorded with empty spans
        and counted in `empty_span_trials`, never dropped: its reward is real
        batch signal and dropping it would silently bias the solve rate.

    Raises:
        ValueError: If `step_index` is negative or the harbor job template
            cannot be loaded/validated.
        ImportError: If the harbor extra is not installed.
    """
    if step_index < 0:
        raise ValueError(f"step_index must be >= 0, got {step_index}")
    try:
        import yaml
        from harbor.models.job.config import JobConfig

        from wmh.distill.agents import WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH
        from wmh.evals.harbor.scorer import HarborScorer
    except ImportError as error:
        raise ImportError(MISSING_HARBOR_EXTRA) from error

    step_name = f"step-{step_index:04d}"
    jobs_dir = run_dir / "harbor" / step_name
    token_sink_dir = run_dir / "tokens" / step_name

    template_path = Path(cfg.harbor.job_template)
    try:
        raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"cannot load the harbor job template from {template_path}: {error}; point "
            "harbor.job_template at a harbor JobConfig YAML/JSON file"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"the harbor job template in {template_path} must be a mapping")
    try:
        template = JobConfig.model_validate({**raw, "jobs_dir": str(jobs_dir)})
    except ValueError as error:
        raise ValueError(
            f"invalid harbor job template {template_path}: {error}; fix the template "
            "so it validates as a harbor JobConfig"
        ) from error

    backend = cfg.harbor.backend
    token_sink_dir.mkdir(parents=True, exist_ok=True)
    scorer = asyncio.run(
        HarborScorer.create(
            template,
            list(task_ids),
            provider_config=provider_config,
            reward_key=cfg.harbor.reward_key,
            attempts=cfg.train.group_size,
            task_environment="e2b" if backend == "e2b" else "docker",
            harness_backend=backend,
            # The local pi runner shares one runner dir, so local concurrency is
            # pinned to 1; e2b parallelizes up to the configured trial concurrency.
            agent_concurrency=1 if backend == "local" else cfg.train.trial_concurrency,
            harbor_retries=cfg.harbor.retries,
            agent_import_path=WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH,
            extra_agent_kwargs={"token_sink_dir": str(token_sink_dir)},
            # TerminalBench-2 tasks compile toolchains and boot VMs. Without this the episode
            # inherited the 300s eval default and killed 31% of trials on the wall clock.
            episode_timeout_s=cfg.rollout.episode_timeout_s,
            # The runner calibrates pi's context guard to this instead of assuming 128k.
            context_window=cfg.rollout.context_budget_tokens,
            # A trial with no verifier evidence (the agent never ran, or the verifier never
            # graded it) keeps a stand-in 0.0 so advantage estimation stays defined, and is
            # flagged `infra_failed` so it never enters a reported solve rate. Never a reason
            # to abort the run.
            missing_reward="zero",
        )
    )
    _wipe_stale_policy_dir(scorer.candidate_job_dir(harness), provider_config, token_sink_dir)
    logger.info(
        "collecting rollouts for %s: %d task(s) x %d attempt(s), backend %s -> %s",
        step_name,
        len(list(task_ids)),
        cfg.train.group_size,
        backend,
        jobs_dir,
    )
    report = scorer.score(harness, should_cancel=should_cancel)
    records = assemble_trial_records(report.cells, token_sink_dir)
    stats = rollout_stats(records)
    if stats.empty_span_trials:
        logger.warning(
            "%d/%d trial(s) in %s produced no token spans (the agent died before its "
            "first student completion); they carry reward signal but no training data",
            stats.empty_span_trials,
            stats.trials,
            step_name,
        )
    if stats.infra_failed_trials:
        logger.warning(
            "%d/%d trial(s) in %s never produced verifier evidence (no sandbox, a rate limit, a "
            "dead transport, or a VERIFIER that never graded the work); they are EXCLUDED from "
            "the reported solve rate (%d executed) and carry a stand-in 0.0 reward for advantage "
            "estimation only; grep the cells' `infra-failure:` notes for the exact causes",
            stats.infra_failed_trials,
            stats.trials,
            step_name,
            stats.executed_trials,
        )
    if stats.graded_trials < stats.executed_trials:
        logger.warning(
            "%d/%d gradeable trial(s) in %s carry a verifier reward but no readable CTRF test "
            "report, so they are EXCLUDED from the graded solve rate (%.3f over %d trial(s)) "
            "instead of counted as 0.0; the binary solve rate still covers all %d",
            stats.executed_trials - stats.graded_trials,
            stats.executed_trials,
            step_name,
            stats.graded_solve_rate,
            stats.graded_trials,
            stats.executed_trials,
        )
    if stats.scaffold_loss_rate > 0:
        logger.warning(
            "scaffold loss rate %.1f%% in %s: %d/%d executed trial(s) never reached an "
            "explicit submit (stop reasons %s); those rewards measure where the harness cut "
            "the episode off, not model capability",
            100.0 * stats.scaffold_loss_rate,
            step_name,
            round(stats.scaffold_loss_rate * stats.executed_trials),
            stats.executed_trials,
            stats.stop_reason_counts,
        )
    return records, stats
