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
from wmh.harness.runtime import DEFAULT_EVAL_EPISODE_TIMEOUT_S
from wmh.providers.base import ProviderConfig

logger = logging.getLogger(__name__)

MISSING_HARBOR_EXTRA = (
    "the harbor SDK is not installed; distillation rollouts run as real harbor trials. "
    "Run `uv sync --extra harbor` (or `pip install 'world-model-harness[harbor]'`) and retry"
)


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
    """Fraction of trials whose verifier reward passed (metrics/gating signal)."""

    empty_span_trials: int = Field(ge=0)
    """Trials that recorded no token span (died before the first completion)."""

    mean_sampled_tokens: float = Field(default=0.0, ge=0.0)
    p50_sampled_tokens: int = Field(default=0, ge=0)
    p99_sampled_tokens: int = Field(default=0, ge=0)
    max_sampled_tokens: int = Field(default=0, ge=0)
    """Per-trial totals of sampled tokens: the length distribution, not just its mean.

    These exist because generation length is an UNSUPERVISED degree of freedom in
    a pure-KL objective and it drifts. A sibling lane's run collapsed from 2,866
    to 49 mean generated tokens (59x) with entropy below 0.2, while every
    alignment and teacher-transfer metric it logged stayed inside its healthy
    band -- coverage and projection accuracy are structurally blind to length
    collapse. The distribution matters more than the mean: a mean can fall
    because the policy became efficient or because it went bimodal (one mode
    terminating immediately, one never terminating), and only the percentiles
    separate those."""

    entropy_estimate: float = Field(default=0.0, ge=0.0)
    """Mean of `-sampled_logprobs` over every sampled token in the batch.

    At temperature 1.0 this is an unbiased single-sample Monte Carlo estimate of
    the policy's per-token entropy, since `H(pi) = E_{x~pi}[-log pi(x)]` and the
    rollouts ARE draws from `pi`. It is therefore free -- the sampler already
    records a logprob per token, so no distribution and no extra forward pass is
    needed. **Only valid at temperature 1.0**: any other sampling temperature
    makes the draws come from a tempered distribution while the logprobs remain
    the untempered ones, and the estimate is biased. Read it as a collapse
    tripwire, not as a calibrated entropy."""

    trials_without_delta: int = Field(default=0, ge=0)
    """Trials where at least one span has `delta_messages is None`.

    This is the cross-tokenizer kill switch made visible.
    `reconstruct_conversation` returns None if ANY span in a trial lost its
    canonical messages, so one re-render fallback anywhere in an episode
    discards that whole episode's teacher signal -- and it does so silently, in
    the sense that the step still reports datums > 0 with coverage near zero
    rather than raising. A recorded live batch fragmented 100 of 108 datums.
    Counting it here replaces the older plan of plumbing
    `TokenRecorder.fallback_count` across the agent-process boundary, because
    `delta_messages` is already in the sink format and is the exact field the
    consumer gates on."""


def _percentile(sorted_values: Sequence[int], fraction: float) -> int:
    """The nearest-rank percentile of an already-sorted sequence.

    Nearest-rank rather than interpolated: these are token counts used as
    tripwires, and an interpolated p99 of a 5-element batch invents a value no
    rollout had. An empty sequence reports 0.

    Args:
        sorted_values: Values in ascending order.
        fraction: The percentile as a fraction in [0, 1].

    Returns:
        The value at the nearest rank, or 0 when there are no values.
    """
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def rollout_stats(records: Sequence[TrialRecord]) -> RolloutStats:
    """The batch health stats over a set of assembled trial records.

    A pure function of the records, so a batch loaded back from persisted
    trial records (the warmup-trials manifest) reports the same stats its
    original collection did.

    Args:
        records: The batch's trial records.

    Returns:
        The aggregate stats; an empty batch reports zero trials and a 0.0
        solve rate.
    """
    with_spans = sum(1 for record in records if record.spans)
    per_trial = sorted(
        sum(len(span.sampled_token_ids) for span in record.spans)
        for record in records
        if record.spans
    )
    logprobs = [
        value
        for record in records
        for span in record.spans
        for value in span.sampled_logprobs
    ]
    return RolloutStats(
        trials=len(records),
        trials_with_spans=with_spans,
        solve_rate=(
            sum(1 for record in records if record.passed) / len(records) if records else 0.0
        ),
        empty_span_trials=len(records) - with_spans,
        mean_sampled_tokens=(sum(per_trial) / len(per_trial) if per_trial else 0.0),
        p50_sampled_tokens=_percentile(per_trial, 0.50),
        p99_sampled_tokens=_percentile(per_trial, 0.99),
        max_sampled_tokens=(per_trial[-1] if per_trial else 0),
        # Negated: sampled logprobs are <= 0, and entropy is the mean of their
        # magnitudes. An empty batch reports 0.0 rather than nan so the metric
        # stays plottable across a step that collected nothing.
        entropy_estimate=(-sum(logprobs) / len(logprobs) if logprobs else 0.0),
        trials_without_delta=sum(
            1
            for record in records
            if record.spans and any(span.delta_messages is None for span in record.spans)
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
            # The local runner rejects any non-default episode wall, so only the
            # e2b backend carries the configured value through.
            episode_timeout_s=(
                DEFAULT_EVAL_EPISODE_TIMEOUT_S
                if backend == "local"
                else cfg.rollout.episode_timeout_s
            ),
            harbor_retries=cfg.harbor.retries,
            agent_import_path=WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH,
            extra_agent_kwargs={"token_sink_dir": str(token_sink_dir)},
            # A trial that dies before producing verifier evidence is a failed
            # trial for distillation purposes, never a reason to abort the run.
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
    return records, stats
