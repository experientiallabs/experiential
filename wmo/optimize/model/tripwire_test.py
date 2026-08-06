"""Tests for the degeneration tripwires: pooling, baseline capture, breaches.

The probe numbers asserted here (0.181 nats/token pooled, 7,577 sampled tokens
per episode, the per-episode extremes) are the measured baseline from a live
48-episode TerminalBench-2 probe on Super-120B base weights at temperature 0.7
(`.wmo/distill-runs/probe-scaffold/tokens/step-0000/*.jsonl`, 47 episodes with
spans, 356,122 sampled tokens). That directory is a local, gitignored artifact
root, so the statistics are reproduced here as literals: these tests pin the
rule that a HEALTHY run at those numbers trips nothing.
"""

from __future__ import annotations

import pytest

from wmo.common.providers.tinker import TokenSpan
from wmo.optimize.model.config import (
    PROBE_BASELINE_ENTROPY_NATS,
    PROBE_BASELINE_EPISODE_TOKENS,
    TripwireConfig,
)
from wmo.optimize.model.tokens import TrialRecord
from wmo.optimize.model.tripwire import (
    PolicyHealth,
    TripwireBaseline,
    capture_baseline,
    evaluate_breaches,
    health_summary,
    metric_ratio,
    policy_health,
)

# The sibling lane's pre-registered absolute rule, kept here only to prove it
# would fire on our own healthy baseline.
SIBLING_ABSOLUTE_ENTROPY_FLOOR = 0.2


def _record(
    trial_name: str, *, tokens: int, entropy: float, calls: int = 1, passed: bool = True
) -> TrialRecord:
    """One episode whose pooled entropy proxy is exactly `entropy`.

    Args:
        trial_name: The trial identity.
        tokens: Sampled tokens per call.
        entropy: The per-token `-logprob` every sampled token carries.
        calls: Sampling calls in the episode (spans are prefix-extending).
        passed: The verifier outcome (never read by the tripwires).

    Returns:
        The trial record.
    """
    spans: list[TokenSpan] = []
    prompt = [1, 2, 3]
    for call_index in range(calls):
        sampled = [10 + call_index] * tokens
        spans.append(
            TokenSpan(
                call_index=call_index,
                prompt_token_ids=list(prompt),
                sampled_token_ids=sampled,
                sampled_logprobs=[-entropy] * tokens,
            )
        )
        prompt = [*prompt, *sampled, 4]
    return TrialRecord(
        task_id=trial_name.split("__")[0],
        attempt=1,
        trial_name=trial_name,
        reward=1.0 if passed else 0.0,
        passed=passed,
        spans=spans,
        stop_reason="submitted",
        artifact_dir=f"/tmp/{trial_name}",
    )


def _empty_record(trial_name: str) -> TrialRecord:
    """A trial that died before its first completion (no spans at all)."""
    return TrialRecord(
        task_id=trial_name,
        attempt=1,
        trial_name=trial_name,
        reward=0.0,
        passed=False,
        spans=[],
        infra_failed=True,
        artifact_dir=f"/tmp/{trial_name}",
    )


def _baseline(
    *,
    step: int = 0,
    entropy: float = PROBE_BASELINE_ENTROPY_NATS,
    tokens: float = float(PROBE_BASELINE_EPISODE_TOKENS),
) -> TripwireBaseline:
    return TripwireBaseline(
        step=step,
        entropy_per_token=entropy,
        mean_generation_tokens=tokens,
        episodes=47,
        sampled_tokens=356122,
    )


# -- pooling ---------------------------------------------------------------------------------


def test_policy_health_pools_over_the_batch_not_per_episode() -> None:
    """Token weighting, not an average of per-episode averages.

    The probe's healthiest single episode already reads 0.082 nats/token and its
    lengths span 349 to 30,869 tokens, so a per-episode rule would fire on
    healthy runs constantly. Here a long flat episode and a short peaked one
    pool to the token-weighted mean (0.1), well away from the per-episode mean
    of the same two numbers (0.55).
    """
    health = policy_health(
        [
            _record("long", tokens=900, entropy=0.05),
            _record("short", tokens=100, entropy=1.0),
        ]
    )
    assert health.episodes == 2
    assert health.sampled_tokens == 1000
    assert health.entropy_per_token == pytest.approx((900 * 0.05 + 100 * 1.0) / 1000)
    assert health.mean_generation_tokens == pytest.approx(500.0)
    per_episode_mean = (0.05 + 1.0) / 2
    assert health.entropy_per_token is not None
    assert health.entropy_per_token < per_episode_mean


def test_policy_health_sums_every_call_in_an_episode() -> None:
    """An episode's length is all of its sampling calls, not just the first."""
    health = policy_health([_record("multi", tokens=40, entropy=0.2, calls=3)])
    assert health.episodes == 1
    assert health.sampled_tokens == 120
    assert health.mean_generation_tokens == pytest.approx(120.0)
    assert health.entropy_per_token == pytest.approx(0.2)


def test_policy_health_excludes_span_less_episodes_from_both_denominators() -> None:
    """A trial that generated nothing is an infrastructure failure, not a short
    episode: counting it would halve the length statistic for an E2B outage and
    could kill a perfectly healthy policy."""
    healthy = [_record("a", tokens=500, entropy=0.2), _record("b", tokens=500, entropy=0.2)]
    dead = [_empty_record("c"), _empty_record("d")]
    assert policy_health(healthy) == policy_health([*healthy, *dead])


def test_policy_health_of_an_all_empty_batch_measures_nothing() -> None:
    health = policy_health([_empty_record("a"), _empty_record("b")])
    assert health.episodes == 0
    assert health.sampled_tokens == 0
    assert health.entropy_per_token is None
    assert health.mean_generation_tokens is None


def test_policy_health_of_no_records_measures_nothing() -> None:
    health = policy_health([])
    assert health.entropy_per_token is None
    assert health.mean_generation_tokens is None


# -- baseline capture ------------------------------------------------------------------------


def test_capture_baseline_records_the_step_and_both_statistics() -> None:
    health = policy_health([_record("a", tokens=200, entropy=0.3)])
    baseline = capture_baseline(4, health)
    assert baseline is not None
    assert baseline.step == 4
    assert baseline.entropy_per_token == pytest.approx(0.3)
    assert baseline.mean_generation_tokens == pytest.approx(200.0)
    assert baseline.episodes == 1
    assert baseline.sampled_tokens == 200


def test_capture_baseline_refuses_an_unmeasurable_step() -> None:
    """No baseline from a batch that sampled nothing: the tripwire stays unarmed
    and the next step retries, rather than anchoring on a dead batch."""
    assert capture_baseline(0, policy_health([_empty_record("a")])) is None


def test_capture_baseline_refuses_a_zero_entropy_step() -> None:
    """A fully deterministic batch would make every later ratio undefined."""
    assert capture_baseline(0, policy_health([_record("a", tokens=50, entropy=0.0)])) is None


# -- breach evaluation -----------------------------------------------------------------------


def test_no_breach_at_the_baseline_itself() -> None:
    baseline = _baseline()
    health = PolicyHealth(
        episodes=47,
        sampled_tokens=356122,
        entropy_per_token=baseline.entropy_per_token,
        mean_generation_tokens=baseline.mean_generation_tokens,
    )
    assert evaluate_breaches(TripwireConfig(), baseline, health) == []


def test_measured_probe_baselines_trip_nothing() -> None:
    """The sanity check that keeps the tripwire usable: a run sitting exactly at
    the measured baseline, and one drifting mildly around it, must be silent.

    An absolute threshold would not be: the sibling lane's pre-registered
    "entropy below 0.2 nats means collapse" is ABOVE our healthy untrained
    0.181, so it would have fired at step 0 of this run, before a single
    gradient step, and been muted before it could ever catch a real collapse.
    """
    assert PROBE_BASELINE_ENTROPY_NATS < SIBLING_ABSOLUTE_ENTROPY_FLOOR
    cfg = TripwireConfig()
    baseline = _baseline()
    # The drift cases are the 1-in-1000 tails of resampling the probe's own 47
    # episodes into 32-episode batches (the headline batch shape): pooled length
    # 0.62x and pooled entropy 0.86x of baseline. See the bootstrap recorded in
    # the `wmo.optimize.model.tripwire` module docstring.
    for entropy, tokens in (
        (PROBE_BASELINE_ENTROPY_NATS, float(PROBE_BASELINE_EPISODE_TOKENS)),
        (0.200, 5543.0),  # the probe's per-episode mean entropy at its p50 length
        (0.156, 4700.0),  # both at their measured p0.1 of batch resampling noise
        (0.135, 4000.0),  # entropy at the per-episode p10, length past p0.1
    ):
        health = PolicyHealth(
            episodes=8,
            sampled_tokens=int(tokens * 8),
            entropy_per_token=entropy,
            mean_generation_tokens=tokens,
        )
        assert evaluate_breaches(cfg, baseline, health) == [], (entropy, tokens)


def test_a_single_flat_probe_episode_pooled_into_a_batch_trips_nothing() -> None:
    """The probe's most predictable episode (0.082 nats/token) beside its
    shortest (349 tokens): pooled with ordinary episodes they stay clear of both
    warn bounds, which is the whole reason the statistics are batch-pooled."""
    records = [
        _record("flat", tokens=349, entropy=0.082),
        *[_record(f"ordinary-{i}", tokens=5543, entropy=0.184) for i in range(7)],
    ]
    breaches = evaluate_breaches(TripwireConfig(), _baseline(), policy_health(records))
    assert breaches == []


def test_warn_fires_at_the_warn_fraction() -> None:
    """The boundary belongs to the breach side, so a threshold set at an
    observed value is never silently inert."""
    cfg = TripwireConfig()
    baseline = _baseline()
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=8000,
        entropy_per_token=baseline.entropy_per_token * cfg.entropy_warn_frac,
        mean_generation_tokens=baseline.mean_generation_tokens,
    )
    (breach,) = evaluate_breaches(cfg, baseline, health)
    assert breach.metric == "entropy_per_token"
    assert breach.level == "warn"
    assert breach.ratio == pytest.approx(cfg.entropy_warn_frac)
    assert breach.threshold_frac == pytest.approx(cfg.entropy_warn_frac)
    assert breach.config_field == "tripwire.entropy_warn_frac"
    # The message names the metric, the baseline, the value and the ratio.
    described = breach.describe()
    assert "entropy_per_token" in described
    assert "0.0905" in described
    assert "0.181" in described
    assert "0.50x" in described


def test_just_above_the_warn_fraction_is_silent() -> None:
    cfg = TripwireConfig()
    baseline = _baseline()
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=8000,
        entropy_per_token=baseline.entropy_per_token * (cfg.entropy_warn_frac + 0.01),
        mean_generation_tokens=baseline.mean_generation_tokens * (cfg.length_warn_frac + 0.01),
    )
    assert evaluate_breaches(cfg, baseline, health) == []


def test_kill_level_replaces_the_warn_for_the_same_metric() -> None:
    cfg = TripwireConfig()
    baseline = _baseline()
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=8000,
        entropy_per_token=baseline.entropy_per_token * cfg.entropy_kill_frac,
        mean_generation_tokens=baseline.mean_generation_tokens,
    )
    (breach,) = evaluate_breaches(cfg, baseline, health)
    assert breach.level == "kill"
    assert breach.config_field == "tripwire.entropy_kill_frac"


def test_length_collapse_breaches_independently_of_entropy() -> None:
    """The mirror pathology (EOS never learned, or answers collapsing to
    nothing) shows in length alone; entropy can stay put."""
    cfg = TripwireConfig()
    baseline = _baseline()
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=400,
        entropy_per_token=baseline.entropy_per_token,
        mean_generation_tokens=50.0,
    )
    (breach,) = evaluate_breaches(cfg, baseline, health)
    assert breach.metric == "mean_generation_tokens"
    assert breach.level == "kill"
    assert breach.config_field == "tripwire.length_kill_frac"


def test_the_sibling_lanes_collapse_would_have_been_caught() -> None:
    """Their measured collapse, expressed against OUR baseline: entropy 0.05
    nats/token and generation length falling to about 50 tokens. Both must read
    kill, or the tripwire does not do the job it was added for."""
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=400,
        entropy_per_token=0.05,
        mean_generation_tokens=50.0,
    )
    breaches = evaluate_breaches(TripwireConfig(), _baseline(), health)
    assert [breach.metric for breach in breaches] == [
        "entropy_per_token",
        "mean_generation_tokens",
    ]
    assert {breach.level for breach in breaches} == {"kill"}


def test_unmeasured_metrics_never_breach() -> None:
    """An all-empty batch is the empty-batch abort's business; a None must not
    be read as a zero."""
    health = PolicyHealth(
        episodes=0, sampled_tokens=0, entropy_per_token=None, mean_generation_tokens=None
    )
    assert evaluate_breaches(TripwireConfig(), _baseline(), health) == []


def test_looser_fractions_are_honored() -> None:
    """The thresholds are config, not constants: a run may widen them."""
    cfg = TripwireConfig(entropy_warn_frac=0.2, entropy_kill_frac=0.1)
    baseline = _baseline()
    health = PolicyHealth(
        episodes=8,
        sampled_tokens=8000,
        entropy_per_token=baseline.entropy_per_token * 0.4,
        mean_generation_tokens=baseline.mean_generation_tokens,
    )
    assert evaluate_breaches(cfg, baseline, health) == []


# -- helpers ---------------------------------------------------------------------------------


def test_metric_ratio_guards_missing_and_degenerate_baselines() -> None:
    assert metric_ratio(0.09, 0.18) == pytest.approx(0.5)
    assert metric_ratio(None, 0.18) is None
    assert metric_ratio(0.09, None) is None
    assert metric_ratio(0.09, 0.0) is None


def test_health_summary_carries_both_metrics_and_their_ratios() -> None:
    health = PolicyHealth(
        episodes=8, sampled_tokens=8000, entropy_per_token=0.0905, mean_generation_tokens=3788.5
    )
    text = health_summary(health, _baseline())
    assert "entropy/token 0.0905 (0.50x baseline)" in text
    assert "gen tokens/episode 3788 (0.50x)" in text


def test_health_summary_without_a_baseline_omits_the_ratios() -> None:
    health = PolicyHealth(
        episodes=8, sampled_tokens=8000, entropy_per_token=0.181, mean_generation_tokens=1000.0
    )
    text = health_summary(health, None)
    assert "0.181" in text
    assert "x baseline" not in text


def test_health_summary_of_an_unmeasurable_batch_says_so() -> None:
    health = PolicyHealth(
        episodes=0, sampled_tokens=0, entropy_per_token=None, mean_generation_tokens=None
    )
    assert health_summary(health, _baseline()) == "entropy/token n/a, gen tokens/episode n/a"


# The Qwen3.5-9B <- Qwen3.6-27B run of 2026-07-26, which is why the ceilings exist. Ratios
# are its measured `entropy_ratio` and `generation_tokens_ratio` per step, against its own
# step-0 baseline of 0.513 nats/token.
REFUTATION_ENTROPY_RATIOS = (1.00, 1.28, 1.49, 1.68)
REFUTATION_LENGTH_RATIOS = (1.00, 2.31, 3.24, 5.33)
REFUTATION_BASELINE_ENTROPY = 0.513
REFUTATION_BASELINE_LENGTH = 10522.703125


def _refutation_health(entropy_ratio: float, length_ratio: float) -> PolicyHealth:
    """One step of the refutation run, expressed as its measured ratios."""
    length = REFUTATION_BASELINE_LENGTH * length_ratio
    return PolicyHealth(
        episodes=64,
        sampled_tokens=int(length * 64),
        entropy_per_token=REFUTATION_BASELINE_ENTROPY * entropy_ratio,
        mean_generation_tokens=length,
    )


def _refutation_baseline() -> TripwireBaseline:
    """That run's step-0 baseline."""
    return TripwireBaseline(
        step=0,
        entropy_per_token=REFUTATION_BASELINE_ENTROPY,
        mean_generation_tokens=REFUTATION_BASELINE_LENGTH,
        episodes=64,
        sampled_tokens=673453,
    )


def test_the_runaway_that_had_to_be_stopped_by_hand_now_kills_itself() -> None:
    """The regression this whole two-sided change exists for.

    Under downside-only bounds this run's four steps produced ZERO breaches while
    context overflow discarded 22 of 64 episodes and a third of the trainable datums,
    so a human had to read the metrics and stop it. With ceilings it must warn early
    and reach kill level on its own.
    """
    cfg = TripwireConfig()
    baseline = _refutation_baseline()
    levels: list[tuple[int, list[tuple[str, str, str]]]] = []
    for step, (entropy_ratio, length_ratio) in enumerate(
        zip(REFUTATION_ENTROPY_RATIOS, REFUTATION_LENGTH_RATIOS, strict=True)
    ):
        breaches = evaluate_breaches(cfg, baseline, _refutation_health(entropy_ratio, length_ratio))
        levels.append((step, [(b.metric, b.level, b.direction) for b in breaches]))

    assert levels[0][1] == [], "step 0 IS the baseline and must never fire against itself"
    assert ("mean_generation_tokens", "warn", "ceiling") in levels[1][1], (
        "2.31x length at step 1 is where a human should have been asked to look"
    )
    kill_steps = [step for step, bs in levels if any(level == "kill" for _, level, _ in bs)]
    assert kill_steps, "the run that had to be stopped by hand must now stop itself"
    assert kill_steps[0] <= 2, (
        "kill must land by step 2, the step where overflow drops reached 14 of 64 and the "
        f"batch began collapsing; got first kill at step {kill_steps[0]}"
    )


def test_the_ceilings_stay_silent_on_the_healthy_probe() -> None:
    """A ceiling that fires on healthy data is worse than no ceiling.

    Same drift cases as the floor test, plus their upside mirrors: the 32-episode
    resampling put pooled length noise at 0.62x on the downside, whose reciprocal
    1.61x is the upside a stationary policy can show.
    """
    cfg = TripwireConfig()
    baseline = _baseline()
    for entropy, tokens in (
        (PROBE_BASELINE_ENTROPY_NATS, float(PROBE_BASELINE_EPISODE_TOKENS)),
        (0.181 * 1.40, PROBE_BASELINE_EPISODE_TOKENS * 1.61),
        (0.181 * 1.20, PROBE_BASELINE_EPISODE_TOKENS * 1.90),
    ):
        health = PolicyHealth(
            episodes=32,
            sampled_tokens=int(tokens * 32),
            entropy_per_token=entropy,
            mean_generation_tokens=tokens,
        )
        assert evaluate_breaches(cfg, baseline, health) == [], (entropy, tokens)


def test_ceiling_kill_outranks_a_floor_warn_on_the_other_metric() -> None:
    """Both directions can fire in one step, each on its own metric."""
    baseline = _baseline()
    health = PolicyHealth(
        episodes=32,
        sampled_tokens=1_000_000,
        entropy_per_token=baseline.entropy_per_token * 0.4,  # floor warn
        mean_generation_tokens=baseline.mean_generation_tokens * 3.5,  # ceiling kill
    )
    breaches = {b.metric: b for b in evaluate_breaches(TripwireConfig(), baseline, health)}
    assert breaches["entropy_per_token"].level == "warn"
    assert breaches["entropy_per_token"].direction == "floor"
    assert breaches["mean_generation_tokens"].level == "kill"
    assert breaches["mean_generation_tokens"].direction == "ceiling"


def test_a_ceiling_breach_reads_as_above_not_under() -> None:
    """The breach message names the direction, or it reads as its own opposite."""
    baseline = _baseline()
    health = PolicyHealth(
        episodes=32,
        sampled_tokens=1_000_000,
        entropy_per_token=baseline.entropy_per_token,
        mean_generation_tokens=baseline.mean_generation_tokens * 2.2,
    )
    (breach,) = evaluate_breaches(TripwireConfig(), baseline, health)
    assert "at or above" in breach.describe()
    assert "length_warn_mult" in breach.describe()


def test_a_kill_ceiling_under_its_warn_ceiling_is_rejected() -> None:
    """The comparison INVERTS between floors and ceilings; getting it wrong would
    abort a run at a ratio it never warned about."""
    with pytest.raises(ValueError, match="length_kill_mult"):
        TripwireConfig(length_warn_mult=3.0, length_kill_mult=2.0)
    with pytest.raises(ValueError, match="entropy_kill_mult"):
        TripwireConfig(entropy_warn_mult=2.0, entropy_kill_mult=1.5)
