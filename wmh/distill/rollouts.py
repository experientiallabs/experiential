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
    """Fraction of trials whose verifier reward passed (metrics/gating signal)."""

    empty_span_trials: int = Field(ge=0)
    """Trials that recorded no token span (died before the first completion)."""

    # TODO: surface TokenRecorder.fallback_count (incremental-prompt re-renders) here;
    # it needs the per-trial span sink format to carry the counter across the
    # agent-process boundary, so it is not trivially wireable today.


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
    return RolloutStats(
        trials=len(records),
        trials_with_spans=with_spans,
        solve_rate=(
            sum(1 for record in records if record.passed) / len(records) if records else 0.0
        ),
        empty_span_trials=len(records) - with_spans,
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
