"""Tests for cost projection and budget metering, with hand-computed numbers."""

import pytest

from wmh.distill.config import (
    DistillConfig,
    EvalConfig,
    GateConfig,
    HarborConfig,
    PricingConfig,
    RolloutConfig,
    SamplingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
    WarmupConfig,
)
from wmh.distill.cost import (
    METER_NAMES,
    BudgetExhausted,
    BudgetMeter,
    CostEstimate,
    estimate_run_cost,
)

FULL_PRICING = PricingConfig(
    student_prefill=1.0, student_sample=2.0, student_train=4.0, teacher_prefill=10.0
)


def _config(pricing: PricingConfig | None = None) -> DistillConfig:
    """A small config whose projections are easy to hand-compute.

    Heuristics resolve to: avg_turns = ceil(4 * 0.5) = 2, sampled per turn =
    min(128, 512) = 128, episode tokens = min(65536, 2048 + 2 * (1024 + 128))
    = 4352, sampled per episode = 256, prefill per episode = 4096.
    """
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-4B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-235B-A22B-Instruct-2507"),
        harbor=HarborConfig(job_template="jobs/tb2.yaml"),
        rollout=RolloutConfig(max_turns=4),
        train=TrainConfig(steps=2, tasks_per_batch=3, group_size=2),
        sampling=SamplingConfig(max_tokens=128),
        eval=EvalConfig(every=1, tasks=2, k=1),
        gate=GateConfig(k=2),
        pricing=pricing if pricing is not None else PricingConfig(),
    )


def _tokens(estimate: CostEstimate) -> dict[str, int]:
    return {line.meter: line.tokens for line in estimate.lines}


# --- estimate_run_cost -------------------------------------------------------


def test_estimate_hand_computed_tokens() -> None:
    # Episodes: train = 2 steps x min(3, 5) tasks x 2 group = 12;
    # evals = (2 // 1) x min(2, 5) x 1 = 4; gate attempts = 3 holdout x k=2 = 6;
    # student baselines (before + after) = 12; teacher baseline = 6.
    # Student episodes = 12 + 4 + 12 = 28. Warmup off by default: 0 episodes.
    estimate = estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.train_episodes == 12
    assert estimate.eval_episodes == 4
    assert estimate.baseline_episodes == 18
    assert estimate.warmup_episodes == 0
    assert _tokens(estimate) == {
        "student_prefill": 28 * 4096,
        "student_sample": 28 * 256,
        "student_train": 12 * 4352,
        "teacher_prefill": (12 + 6) * 4352,
    }


def test_estimate_includes_warmup_teacher_episodes() -> None:
    # Warmup on: teacher episodes = 5 train tasks x 3 rollouts_per_task = 15,
    # all charged to teacher_prefill at full episode tokens (the same
    # teacher-in-harness approximation as the gate baseline). Student meters
    # are untouched: warmup samples the TEACHER, and its SFT train tokens are
    # not projected (they depend on the unknown pass rate).
    cfg = _config().model_copy(update={"warmup": WarmupConfig(steps=2, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    baseline = estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.warmup_episodes == 15
    tokens = _tokens(estimate)
    base_tokens = _tokens(baseline)
    assert tokens["teacher_prefill"] == base_tokens["teacher_prefill"] + 15 * 4352
    for meter in ("student_prefill", "student_sample", "student_train"):
        assert tokens[meter] == base_tokens[meter]


def test_estimate_warmup_steps_zero_means_no_warmup_episodes() -> None:
    # rollouts_per_task alone must not add episodes: steps = 0 disables warmup.
    cfg = _config().model_copy(update={"warmup": WarmupConfig(steps=0, rollouts_per_task=3)})
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.warmup_episodes == 0
    assert _tokens(estimate) == _tokens(
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=3)
    )


def test_estimate_hand_computed_usd() -> None:
    estimate = estimate_run_cost(_config(FULL_PRICING), n_train_tasks=5, n_holdout_tasks=3)
    usd = {line.meter: line.usd for line in estimate.lines}
    assert usd["student_prefill"] == pytest.approx(114688 / 1e6 * 1.0)
    assert usd["student_sample"] == pytest.approx(7168 / 1e6 * 2.0)
    assert usd["student_train"] == pytest.approx(52224 / 1e6 * 4.0)
    assert usd["teacher_prefill"] == pytest.approx(78336 / 1e6 * 10.0)
    assert estimate.priced_usd == pytest.approx(1.12128)
    assert estimate.is_fully_priced()
    assert estimate.unpriced_meters == []


def test_estimate_unpriced_meters_surface_as_none_lines() -> None:
    partial = PricingConfig(student_sample=2.0)
    estimate = estimate_run_cost(_config(partial), n_train_tasks=5, n_holdout_tasks=3)
    by_meter = {line.meter: line for line in estimate.lines}
    assert by_meter["student_sample"].usd == pytest.approx(7168 / 1e6 * 2.0)
    for meter in ("student_prefill", "student_train", "teacher_prefill"):
        assert by_meter[meter].usd is None
        assert by_meter[meter].price_per_mtok is None
    assert not estimate.is_fully_priced()
    assert set(estimate.unpriced_meters) == {
        "student_prefill",
        "student_train",
        "teacher_prefill",
    }
    # priced_usd still totals the priced lines so the CLI can show a floor.
    assert estimate.priced_usd == pytest.approx(0.014336)


def test_estimate_lines_cover_every_meter_in_order() -> None:
    estimate = estimate_run_cost(_config(), n_train_tasks=1, n_holdout_tasks=0)
    assert tuple(line.meter for line in estimate.lines) == METER_NAMES


def test_estimate_clamps_batch_to_train_split() -> None:
    # tasks_per_batch = 3 but only 1 train task: train episodes = 2 x 1 x 2 = 4;
    # evals = 2 x min(2, 1) x 1 = 2; no holdout -> student episodes = 6.
    estimate = estimate_run_cost(_config(), n_train_tasks=1, n_holdout_tasks=0)
    assert estimate.train_episodes == 4
    assert estimate.eval_episodes == 2
    assert estimate.baseline_episodes == 0
    assert _tokens(estimate) == {
        "student_prefill": 6 * 4096,
        "student_sample": 6 * 256,
        "student_train": 4 * 4352,
        "teacher_prefill": 4 * 4352,
    }


def test_estimate_eval_every_zero_means_no_interim_evals() -> None:
    cfg = _config().model_copy(deep=True)
    cfg.eval.every = 0
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert estimate.eval_episodes == 0


def test_estimate_context_budget_caps_episode_tokens() -> None:
    cfg = _config().model_copy(deep=True)
    cfg.rollout.context_budget_tokens = 2048
    # episode tokens = min(2048, 4352) = 2048; sampled = min(256, 2048) = 256;
    # prefill = 1792. Student episodes stay 28, train episodes 12.
    estimate = estimate_run_cost(cfg, n_train_tasks=5, n_holdout_tasks=3)
    assert _tokens(estimate) == {
        "student_prefill": 28 * 1792,
        "student_sample": 28 * 256,
        "student_train": 12 * 2048,
        "teacher_prefill": 18 * 2048,
    }


def test_estimate_rejects_bad_split_sizes() -> None:
    with pytest.raises(ValueError, match="n_train_tasks must be >= 1"):
        estimate_run_cost(_config(), n_train_tasks=0, n_holdout_tasks=3)
    with pytest.raises(ValueError, match="n_holdout_tasks must be >= 0"):
        estimate_run_cost(_config(), n_train_tasks=5, n_holdout_tasks=-1)


# --- BudgetMeter --------------------------------------------------------------


def test_meter_accumulates_tokens_and_usd() -> None:
    meter = BudgetMeter(FULL_PRICING)
    meter.charge("teacher_prefill", 500_000)
    meter.charge("teacher_prefill", 250_000)
    meter.charge("student_sample", 1_000_000)
    assert meter.tokens("teacher_prefill") == 750_000
    assert meter.tokens("student_sample") == 1_000_000
    assert meter.tokens("student_prefill") == 0
    # 0.75M x $10 + 1M x $2 = $9.50, hand-computed.
    assert meter.spent_usd == pytest.approx(9.5)


def test_meter_unpriced_meter_counts_tokens_but_no_usd() -> None:
    meter = BudgetMeter(PricingConfig(student_sample=2.0))
    meter.charge("teacher_prefill", 1_000_000)
    assert meter.tokens("teacher_prefill") == 1_000_000
    assert meter.spent_usd == pytest.approx(0.0)
    meter.charge("student_sample", 1_000_000)
    assert meter.spent_usd == pytest.approx(2.0)


def test_meter_check_raises_when_cap_exceeded() -> None:
    meter = BudgetMeter(FULL_PRICING, max_usd=5.0)
    meter.charge("student_sample", 2_000_000)  # $4.00
    meter.check()  # under the cap
    meter.charge("student_sample", 1_000_000)  # $6.00 total
    with pytest.raises(BudgetExhausted) as excinfo:
        meter.check()
    assert excinfo.value.spent_usd == pytest.approx(6.0)
    assert excinfo.value.max_usd == pytest.approx(5.0)
    message = str(excinfo.value)
    assert "$6.00" in message
    assert "$5.00" in message
    assert "resume" in message


def test_meter_exactly_at_cap_is_not_exhausted() -> None:
    meter = BudgetMeter(FULL_PRICING, max_usd=2.0)
    meter.charge("student_sample", 1_000_000)  # exactly $2.00
    meter.check()


def test_meter_without_cap_never_raises() -> None:
    meter = BudgetMeter(FULL_PRICING)
    meter.charge("teacher_prefill", 10_000_000_000)
    meter.check()


def test_meter_rejects_negative_charge() -> None:
    meter = BudgetMeter(FULL_PRICING)
    with pytest.raises(ValueError, match="negative token count"):
        meter.charge("student_train", -1)


def test_meter_lines_mirror_the_estimate_shape() -> None:
    meter = BudgetMeter(PricingConfig(student_train=4.0))
    meter.charge("student_train", 250_000)
    lines = {line.meter: line for line in meter.lines()}
    assert tuple(line.meter for line in meter.lines()) == METER_NAMES
    assert lines["student_train"].tokens == 250_000
    assert lines["student_train"].usd == pytest.approx(1.0)
    assert lines["teacher_prefill"].tokens == 0
    assert lines["teacher_prefill"].usd is None
