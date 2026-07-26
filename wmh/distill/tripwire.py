"""Degeneration tripwires: entropy and generation length, relative to baseline.

Reverse-KL distillation can degenerate in ways a KL curve cannot show, because
KL falls while the policy collapses. Two failures are on record in sibling
lanes:

- Length/mode collapse: generation length fell 50x (2,866 to about 50 tokens),
  task accuracy fell 0.813 to 0.596, and entropy breached the lane's
  pre-registered floor at steps 17 to 23; the run was killed at step 30/200.
- The mirror pathology, reported by a second lane: "pure KL gives EOS no
  gradient, so the student never learns to stop", i.e. episodes that run to
  the turn or token cap instead of submitting.

This module measures the two statistics that make both visible and decides
warn/kill against a baseline THIS run measured, from data the training step
already holds (the student's own recorded spans): no extra sampling, no extra
spend.

Why relative and never absolute. The sibling lane's rule was "entropy < 0.2
nats means collapse". Our healthy, untrained baseline is 0.181 nats/token
(`PROBE_BASELINE_ENTROPY_NATS`, pooled over 356,122 sampled tokens of a live
48-episode TerminalBench-2 probe on Super-120B base weights at temperature
0.7), so that rule would fire at step 0, before a single gradient step.
Terminal-command tokens are much more predictable than math reasoning, and a
sampled-token entropy estimate taken at temperature 0.7 is biased downward
besides. An always-firing tripwire gets muted, so an absolute threshold here
would be worse than no threshold at all: every bound in `TripwireConfig` is a
fraction of the measured baseline, and it must stay that way.

Why batch-pooled and never per-episode. On that same probe the healthiest
single episode already sits at entropy 0.082 nats/token and episode lengths
span 349 to 30,869 sampled tokens. Any per-episode rule (or a mean of
per-episode ratios) would fire on healthy runs constantly, so both statistics
pool over the whole batch: total sampled logprob over total sampled tokens, and
total sampled tokens over episodes.

What the defaults cost in false alarms. Resampling the probe's own 47 healthy
episodes into batches (200,000 draws per shape) gives the pooled ratio a
completely stationary policy would report:

    episodes/batch   length ratio p0.1 / min   entropy ratio p0.1 / min
    32 (8 tasks x 4)      0.62 / 0.47               0.86 / 0.81
    16                    0.50 / 0.36               0.80 / 0.70
     8                    0.37 / 0.21               0.71 / 0.57

At the headline batch shape the length WARN (0.5) has probability 1e-5 per step
and the length KILL (0.25) never occurred in 200,000 draws; no draw at any shape
put the entropy ratio under 0.5, let alone the 0.3 kill bound. Requiring
`kill_consecutive_steps` in a row squares an already negligible number. The
tripwire tightens as the batch shrinks, though, so a step whose batch was
decimated by infrastructure failures is the one case worth reading beside its
`trials` and `empty_span_trials` counts before believing a length warning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wmh.distill.config import TripwireConfig
from wmh.distill.tokens import TrialRecord

logger = logging.getLogger(__name__)

TripwireMetric = Literal["entropy_per_token", "mean_generation_tokens"]
"""The two statistics under tripwire, named exactly as the metrics-row keys."""

TripwireLevel = Literal["warn", "kill"]


class PolicyHealth(BaseModel):
    """One rollout batch's batch-pooled entropy proxy and generation length."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episodes: int = Field(ge=0)
    """Episodes that recorded at least one sampled token.

    Episodes with no spans are excluded from BOTH denominators on purpose: a
    trial that died at sandbox creation or lost its transport generated nothing,
    so counting it as a zero-length episode would drag the length statistic down
    for an infrastructure outage and could kill a healthy policy. The
    all-empty-batch abort (`MAX_CONSECUTIVE_EMPTY_STEPS`) owns that failure
    mode, and `RolloutStats.empty_span_trials` counts it in the same row."""

    sampled_tokens: int = Field(ge=0)
    """Sampled (loss-position) tokens pooled over those episodes."""

    entropy_per_token: float | None
    """`-mean(sampled_logprobs)` over every sampled token in the batch.

    The tokens were drawn from the policy that scored them, so `-log p(x)` with
    `x ~ p` is an unbiased single-sample estimator of that policy's entropy
    `H(p) = E_{x~p}[-log p(x)]`; pooling over the batch averages many such
    samples. Two caveats to read it with:

    - Temperature. Rollouts sample at `sampling.temperature` (0.7 for the
      headline run), which concentrates the distribution the tokens come from,
      so this sits BELOW the T=1 entropy: treat it as a lower bound and never
      compare it against a number measured at another temperature.
    - It is a policy-shape statistic, not a per-token uncertainty: a batch
      dominated by long, highly predictable tool output reads low without
      anything being wrong, which is exactly why the tripwire compares it only
      against this run's own baseline.

    None when the batch recorded no sampled token at all."""

    mean_generation_tokens: float | None
    """Sampled tokens per episode, pooled (`sampled_tokens / episodes`).

    None when no episode recorded a sampled token."""


def policy_health(records: Sequence[TrialRecord]) -> PolicyHealth:
    """Pool one rollout batch's recorded spans into the two tripwire statistics.

    Measured from the raw `TrialRecord` spans rather than from the built
    `TrainDatum`s, though the datum builder's loss positions are exactly these
    sampled tokens (its loss mask is 1.0 on sampled tokens and 0.0 on context).
    The reason is drops: `build_datums` discards whole episodes that overflow
    the context budget or exceed `train.max_datum_tokens`, which are precisely
    the LONGEST episodes, so measuring generation length after the drops would
    deflate it in exactly the tail that matters and could kill a healthy run.

    Args:
        records: The batch's assembled trial records.

    Returns:
        The batch-pooled health; both statistics are None when no episode
        recorded a sampled token (an all-empty batch, whose own abort path
        handles it).
    """
    episodes = 0
    sampled_tokens = 0
    logprob_total = 0.0
    for record in records:
        episode_tokens = sum(len(span.sampled_token_ids) for span in record.spans)
        if episode_tokens == 0:
            continue
        episodes += 1
        sampled_tokens += episode_tokens
        logprob_total += sum(sum(span.sampled_logprobs) for span in record.spans)
    if episodes == 0 or sampled_tokens == 0:
        return PolicyHealth(
            episodes=episodes,
            sampled_tokens=sampled_tokens,
            entropy_per_token=None,
            mean_generation_tokens=None,
        )
    return PolicyHealth(
        episodes=episodes,
        sampled_tokens=sampled_tokens,
        entropy_per_token=-logprob_total / sampled_tokens,
        mean_generation_tokens=sampled_tokens / episodes,
    )


class TripwireBaseline(BaseModel):
    """The reference the tripwires compare every later step against.

    Captured from the run's FIRST training step and persisted in the run's
    checkpoint manifest, because a resumed run that re-baselines against an
    already-degenerated policy has a blind tripwire: the collapse becomes the
    new normal and nothing fires again.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0)
    """The 0-based training step measured, so a resumed run can tell whether
    the current step IS the baseline step (which never fires against itself)."""

    entropy_per_token: float = Field(gt=0.0)
    mean_generation_tokens: float = Field(gt=0.0)
    episodes: int = Field(ge=1)
    """Episodes behind the measurement, for reading the row's confidence."""

    sampled_tokens: int = Field(ge=1)


def capture_baseline(step: int, health: PolicyHealth) -> TripwireBaseline | None:
    """Turn a first step's measured health into the run's baseline.

    Args:
        step: The 0-based training step being measured.
        health: That step's batch-pooled health.

    Returns:
        The baseline, or None when this step cannot serve as one: no episode
        sampled anything, or a statistic came out non-positive (an entropy of
        exactly 0.0 would make every later ratio undefined). The caller leaves
        the tripwire unarmed and tries again at the next step.
    """
    entropy = health.entropy_per_token
    length = health.mean_generation_tokens
    if entropy is None or length is None or entropy <= 0.0 or length <= 0.0:
        return None
    return TripwireBaseline(
        step=step,
        entropy_per_token=entropy,
        mean_generation_tokens=length,
        episodes=health.episodes,
        sampled_tokens=health.sampled_tokens,
    )


class TripwireBreach(BaseModel):
    """One metric under one threshold at one step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: TripwireMetric
    level: TripwireLevel
    baseline: float
    value: float
    ratio: float
    """`value / baseline`; the tripwire is a bound on this, never on `value`."""

    threshold_frac: float
    """The configured fraction the ratio fell to or below."""

    config_field: str
    """The `[tripwire]` key that set `threshold_frac`, so a breach message says
    exactly which knob to move."""

    def describe(self) -> str:
        """One line naming the metric, its baseline, the value, and the ratio."""
        return (
            f"{self.metric} {self.value:.4g} is {self.ratio:.2f}x this run's baseline "
            f"{self.baseline:.4g} (measured at its first training step), at or under the "
            f"{self.level} threshold {self.threshold_frac:.2f} ({self.config_field})"
        )


def metric_ratio(value: float | None, baseline: float | None) -> float | None:
    """`value / baseline`, or None when either side is missing or unusable."""
    if value is None or baseline is None or baseline <= 0.0:
        return None
    return value / baseline


class _MetricReading(BaseModel):
    """One metric's current value beside the baseline and bounds it answers to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: TripwireMetric
    value: float | None
    baseline: float
    warn_frac: float
    kill_frac: float
    warn_field: str
    kill_field: str


def _readings(
    cfg: TripwireConfig, baseline: TripwireBaseline, health: PolicyHealth
) -> list[_MetricReading]:
    """Both metrics wired to their configured fractions, in report order."""
    return [
        _MetricReading(
            metric="entropy_per_token",
            value=health.entropy_per_token,
            baseline=baseline.entropy_per_token,
            warn_frac=cfg.entropy_warn_frac,
            kill_frac=cfg.entropy_kill_frac,
            warn_field="tripwire.entropy_warn_frac",
            kill_field="tripwire.entropy_kill_frac",
        ),
        _MetricReading(
            metric="mean_generation_tokens",
            value=health.mean_generation_tokens,
            baseline=baseline.mean_generation_tokens,
            warn_frac=cfg.length_warn_frac,
            kill_frac=cfg.length_kill_frac,
            warn_field="tripwire.length_warn_frac",
            kill_field="tripwire.length_kill_frac",
        ),
    ]


def evaluate_breaches(
    cfg: TripwireConfig, baseline: TripwireBaseline, health: PolicyHealth
) -> list[TripwireBreach]:
    """Compare one step's health against the baseline fractions.

    A metric with no measurement (None) is skipped rather than treated as zero:
    a batch that sampled nothing is an infrastructure failure, and the
    all-empty-batch abort owns it. The boundary belongs to the breach side (a
    ratio exactly at the fraction fires), so a threshold set to a value that was
    actually observed is never silently inert.

    Args:
        cfg: The run's `[tripwire]` section (its `enabled` flag is the caller's
            business; this function always reports what it sees).
        baseline: The run's captured baseline.
        health: The step's batch-pooled health.

    Returns:
        At most one breach per metric, the more severe level when both bounds
        are crossed; an empty list when nothing breached.
    """
    breaches: list[TripwireBreach] = []
    for reading in _readings(cfg, baseline, health):
        ratio = metric_ratio(reading.value, reading.baseline)
        if reading.value is None or ratio is None:
            continue
        if ratio <= reading.kill_frac:
            level: TripwireLevel = "kill"
            threshold, field = reading.kill_frac, reading.kill_field
        elif ratio <= reading.warn_frac:
            level = "warn"
            threshold, field = reading.warn_frac, reading.warn_field
        else:
            continue
        breaches.append(
            TripwireBreach(
                metric=reading.metric,
                level=level,
                baseline=reading.baseline,
                value=reading.value,
                ratio=ratio,
                threshold_frac=threshold,
                config_field=field,
            )
        )
    return breaches


def health_summary(health: PolicyHealth, baseline: TripwireBaseline | None) -> str:
    """The progress-line fragment for one step's health.

    Args:
        health: The step's batch-pooled health.
        baseline: The run's baseline, when one is armed.

    Returns:
        A compact "entropy X (Rx baseline), gen Y tok/episode (Rx)" fragment,
        with `n/a` in place of a statistic the batch could not measure and the
        ratios omitted until a baseline exists.
    """
    parts: list[str] = []
    entropy_ratio = metric_ratio(
        health.entropy_per_token, baseline.entropy_per_token if baseline else None
    )
    length_ratio = metric_ratio(
        health.mean_generation_tokens, baseline.mean_generation_tokens if baseline else None
    )
    entropy = health.entropy_per_token
    length = health.mean_generation_tokens
    parts.append("entropy/token n/a" if entropy is None else f"entropy/token {entropy:.4f}")
    if entropy_ratio is not None:
        parts[-1] += f" ({entropy_ratio:.2f}x baseline)"
    parts.append("gen tokens/episode n/a" if length is None else f"gen tokens/episode {length:.0f}")
    if length_ratio is not None:
        parts[-1] += f" ({length_ratio:.2f}x)"
    return ", ".join(parts)
