"""Cost projection and budget metering for one distillation run.

`estimate_run_cost` turns the run config plus the task split sizes into
per-meter token projections priced from the `[pricing]` section, for the CLI's
cost-confirm prompt. `BudgetMeter` then accumulates the ACTUAL token counts as
the run spends, and `check()` enforces the optional `[budget] max_usd` hard
cap by raising `BudgetExhausted` (the loop saves state and prints the resume
command on that error).

The projection is a deliberately simple, documented heuristic: episode counts
come exactly from the config (steps x tasks x group size, plus interim evals
and the gate/baseline episodes), and per-episode tokens come from a turns x
tokens-per-turn model capped by the rollout context budget. Meters mirror the
four `[pricing]` fields; a meter without a price surfaces as a None-usd line
so the CLI can warn instead of silently under-reporting.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wmh.distill.config import DistillConfig, PricingConfig

logger = logging.getLogger(__name__)

MeterName = Literal["student_prefill", "student_sample", "student_train", "teacher_prefill"]

METER_NAMES: tuple[MeterName, ...] = (
    "student_prefill",
    "student_sample",
    "student_train",
    "teacher_prefill",
)

_TOKENS_PER_USD_UNIT = 1_000_000
"""Prices in `[pricing]` are USD per million tokens."""

_AVG_TURN_FRACTION = 0.5
"""Episodes are assumed to use half the configured turn cap on average."""

_SAMPLED_TOKENS_PER_TURN = 512
"""Assumed sampled (assistant/tool-call) tokens per agent turn."""

_OBSERVATION_TOKENS_PER_TURN = 1024
"""Assumed prompt growth per turn (tool results and scaffolding)."""

_BASE_PROMPT_TOKENS = 2048
"""Assumed initial prompt (system prompt, task instruction, tool schemas)."""


class BudgetExhausted(RuntimeError):
    """Raised by `BudgetMeter.check` when actual spend exceeds the hard cap."""

    def __init__(self, spent_usd: float, max_usd: float) -> None:
        self.spent_usd = spent_usd
        self.max_usd = max_usd
        super().__init__(
            f"budget exhausted: ${spent_usd:.2f} spent against the ${max_usd:.2f} "
            "cap (budget.max_usd); the run saves its training state on this error, "
            "so raise budget.max_usd in the run config and resume the run to continue"
        )


class CostLine(BaseModel):
    """One meter's token projection (or actuals) with its optional price."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meter: MeterName
    tokens: int = Field(ge=0)
    price_per_mtok: float | None
    """USD per million tokens from `[pricing]`; None means unpriced."""

    usd: float | None
    """tokens x price; None when the meter is unpriced (CLI warns on these)."""


class CostEstimate(BaseModel):
    """Per-meter projections for one run, plus the episode counts behind them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: list[CostLine]
    train_episodes: int = Field(ge=0)
    eval_episodes: int = Field(ge=0)
    baseline_episodes: int = Field(ge=0)
    """Gate/baseline episodes: student before + student after + teacher-in-harness."""

    @property
    def priced_usd(self) -> float:
        """Total USD over the priced lines only."""
        return sum(line.usd for line in self.lines if line.usd is not None)

    @property
    def unpriced_meters(self) -> list[MeterName]:
        """Meters with no `[pricing]` entry, for the CLI's warning."""
        return [line.meter for line in self.lines if line.usd is None]

    def is_fully_priced(self) -> bool:
        """Whether every meter carries a price, so `priced_usd` is the whole run."""
        return not self.unpriced_meters


def _meter_price(pricing: PricingConfig, meter: MeterName) -> float | None:
    if meter == "student_prefill":
        return pricing.student_prefill
    if meter == "student_sample":
        return pricing.student_sample
    if meter == "student_train":
        return pricing.student_train
    return pricing.teacher_prefill


def _line(pricing: PricingConfig, meter: MeterName, tokens: int) -> CostLine:
    price = _meter_price(pricing, meter)
    usd = tokens / _TOKENS_PER_USD_UNIT * price if price is not None else None
    return CostLine(meter=meter, tokens=tokens, price_per_mtok=price, usd=usd)


def estimate_run_cost(cfg: DistillConfig, n_train_tasks: int, n_holdout_tasks: int) -> CostEstimate:
    """Project the run's per-meter token volumes and price them.

    Episode counts (exact, from the config):

    - train: `steps x min(tasks_per_batch, n_train_tasks) x group_size`
    - interim evals: `steps // eval.every` evals (0 when eval.every is 0) of
      `min(eval.tasks, n_train_tasks) x eval.k` student episodes each
    - gate/baseline: student-before and student-after at
      `n_holdout x gate.k` each, plus one teacher-in-harness baseline at
      `n_holdout x gate.k`

    Per-episode tokens (heuristic; module constants document the assumptions):

    - `avg_turns = max(1, ceil(rollout.max_turns x 0.5))`
    - `sampled = avg_turns x min(sampling.max_tokens, 512)`
    - `episode_tokens = min(rollout.context_budget_tokens,
      2048 + avg_turns x (1024 + sampled_per_turn))`: the episode's final
      unique sequence length under the prefix property
    - `prefill = episode_tokens - sampled`

    Meter mapping: every student episode charges `prefill` to student_prefill
    and `sampled` to student_sample; every train episode additionally charges
    `episode_tokens` to student_train (forward_backward over the full datum)
    and `episode_tokens` to teacher_prefill (the teacher scores each episode's
    full sequence once, append-only single datum). Teacher-in-harness baseline
    episodes charge their full `episode_tokens` to teacher_prefill as an
    approximation: the config carries no teacher sampling meter, so their
    sampled tokens are priced at the teacher's prefill rate.

    Args:
        cfg: The validated run config.
        n_train_tasks: Size of the train task split (must be >= 1).
        n_holdout_tasks: Size of the holdout task split (>= 0; 0 skips the
            gate/baseline episodes entirely).

    Returns:
        The estimate, one line per meter in `METER_NAMES` order.

    Raises:
        ValueError: If the split sizes are out of range.
    """
    if n_train_tasks < 1:
        raise ValueError(
            f"n_train_tasks must be >= 1, got {n_train_tasks}; a distillation run "
            "needs a non-empty train task split"
        )
    if n_holdout_tasks < 0:
        raise ValueError(f"n_holdout_tasks must be >= 0, got {n_holdout_tasks}")

    avg_turns = max(1, math.ceil(cfg.rollout.max_turns * _AVG_TURN_FRACTION))
    sampled_per_turn = min(cfg.sampling.max_tokens, _SAMPLED_TOKENS_PER_TURN)
    episode_tokens = min(
        cfg.rollout.context_budget_tokens,
        _BASE_PROMPT_TOKENS + avg_turns * (_OBSERVATION_TOKENS_PER_TURN + sampled_per_turn),
    )
    sampled_tokens = min(avg_turns * sampled_per_turn, episode_tokens)
    prefill_tokens = episode_tokens - sampled_tokens

    tasks_per_step = min(cfg.train.tasks_per_batch, n_train_tasks)
    train_episodes = cfg.train.steps * tasks_per_step * cfg.train.group_size
    interim_evals = cfg.train.steps // cfg.eval.every if cfg.eval.every > 0 else 0
    eval_episodes = interim_evals * min(cfg.eval.tasks, n_train_tasks) * cfg.eval.k
    gate_attempts = n_holdout_tasks * cfg.gate.k
    student_baseline_episodes = 2 * gate_attempts  # student-before + student-after
    teacher_baseline_episodes = gate_attempts

    student_episodes = train_episodes + eval_episodes + student_baseline_episodes
    projections: dict[MeterName, int] = {
        "student_prefill": student_episodes * prefill_tokens,
        "student_sample": student_episodes * sampled_tokens,
        "student_train": train_episodes * episode_tokens,
        "teacher_prefill": (train_episodes + teacher_baseline_episodes) * episode_tokens,
    }
    estimate = CostEstimate(
        lines=[_line(cfg.pricing, meter, projections[meter]) for meter in METER_NAMES],
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        baseline_episodes=student_baseline_episodes + teacher_baseline_episodes,
    )
    logger.debug(
        "cost estimate: %d train + %d eval + %d baseline episode(s), priced $%.2f%s",
        estimate.train_episodes,
        estimate.eval_episodes,
        estimate.baseline_episodes,
        estimate.priced_usd,
        f" (unpriced meters: {', '.join(estimate.unpriced_meters)})"
        if estimate.unpriced_meters
        else "",
    )
    return estimate


class BudgetMeter:
    """Accumulates actual metered tokens and enforces the hard USD cap.

    Args:
        pricing: The `[pricing]` section; unpriced meters accumulate tokens
            but contribute no USD (mirroring the estimate's None lines).
        max_usd: The `[budget] max_usd` hard cap; None disables enforcement.
    """

    def __init__(self, pricing: PricingConfig, max_usd: float | None = None) -> None:
        self._pricing = pricing
        self._max_usd = max_usd
        self._tokens: dict[MeterName, int] = {meter: 0 for meter in METER_NAMES}
        self._spent_usd = 0.0

    def charge(self, meter: MeterName, tokens: int) -> None:
        """Record actual token usage against one meter.

        Args:
            meter: Which meter the tokens belong to.
            tokens: The token count to add (>= 0).

        Raises:
            ValueError: If `tokens` is negative.
        """
        if tokens < 0:
            raise ValueError(f"cannot charge a negative token count ({tokens}) to {meter}")
        self._tokens[meter] += tokens
        price = _meter_price(self._pricing, meter)
        if price is not None:
            self._spent_usd += tokens / _TOKENS_PER_USD_UNIT * price

    def check(self) -> None:
        """Enforce the hard cap against the priced spend so far.

        Raises:
            BudgetExhausted: When the cap is set and the spend exceeds it.
        """
        if self._max_usd is not None and self._spent_usd > self._max_usd:
            raise BudgetExhausted(self._spent_usd, self._max_usd)

    def tokens(self, meter: MeterName) -> int:
        """Actual tokens charged to one meter so far."""
        return self._tokens[meter]

    @property
    def spent_usd(self) -> float:
        """Priced USD spend so far (unpriced meters contribute nothing)."""
        return self._spent_usd

    def lines(self) -> list[CostLine]:
        """The actuals in the same line shape the estimate uses, for reporting."""
        return [_line(self._pricing, meter, self._tokens[meter]) for meter in METER_NAMES]
