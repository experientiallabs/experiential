"""Deterministic paired benchmark blocks and task-clustered panel analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from enum import StrEnum
from fractions import Fraction
from statistics import fmean
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)

PAIRED_ANALYSIS_VERSION: Literal["2"] = "2"


class PairedArm(StrEnum):
    """One arm of a paired benchmark comparison."""

    BASELINE = "baseline"
    CANDIDATE = "candidate"


class PairedPanelPlan(BaseModel):
    """Opaque panel identity and its predeclared repeated-attempt count."""

    model_config = ConfigDict(frozen=True)

    panel_member: str = Field(min_length=1)
    attempts: StrictInt = Field(ge=1)

    @field_validator("panel_member")
    @classmethod
    def _require_canonical_member(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("panel_member cannot have surrounding whitespace")
        return value

    @field_validator("attempts", mode="before")
    @classmethod
    def _reject_boolean_attempts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("panel attempts cannot be boolean")
        return value


class BoundedMeanBet(BaseModel):
    """One immutable component of a finite-sample bounded-mean e-value mixture."""

    model_config = ConfigDict(frozen=True)

    fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    weight: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("fraction", "weight", mode="before")
    @classmethod
    def _reject_boolean_values(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("bounded-mean bet values cannot be boolean")
        return value


class PairedBlock(BaseModel):
    """Frozen execution order for both fresh-sandbox arms of one paired cell."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(min_length=1)
    panel_member: str = Field(min_length=1)
    attempt: StrictInt = Field(ge=1)
    first_arm: PairedArm

    @property
    def key(self) -> tuple[str, str, int]:
        """Return the order-independent identity of this paired cell."""
        return self.task_id, self.panel_member, self.attempt


class PairedEvaluationDesign(BaseModel):
    """Complete immutable matrix, schedule, and statistical decision thresholds."""

    model_config = ConfigDict(frozen=True)

    analysis_version: Literal["2"] = PAIRED_ANALYSIS_VERSION
    task_ids: tuple[str, ...]
    panel: tuple[PairedPanelPlan, ...]
    bounded_mean_bets: tuple[BoundedMeanBet, ...]
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    randomization_samples: StrictInt = Field(ge=999)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_panel_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    minimum_member_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    noninferiority_margin: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    blocks: tuple[PairedBlock, ...]

    @field_validator("randomization_samples", mode="before")
    @classmethod
    def _reject_boolean_counts(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("paired design counts cannot be boolean")
        return value

    @field_validator(
        "alpha",
        "minimum_panel_delta",
        "minimum_member_delta",
        "noninferiority_margin",
        mode="before",
    )
    @classmethod
    def _reject_boolean_thresholds(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("paired design thresholds cannot be boolean")
        return value

    @field_validator("schedule_seed", "analysis_seed")
    @classmethod
    def _require_canonical_seed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("paired design seeds cannot have surrounding whitespace")
        return value

    @property
    def digest(self) -> str:
        """Return the canonical identity of the complete schedule and decision rule."""
        return _canonical_digest(self.model_dump(mode="json"))

    @property
    def panel_members(self) -> tuple[str, ...]:
        """Return panel identities in their frozen canonical order."""
        return tuple(plan.panel_member for plan in self.panel)

    @property
    def attempts_by_member(self) -> dict[str, int]:
        """Return a copy of repeated-attempt counts keyed by panel identity."""
        return {plan.panel_member: plan.attempts for plan in self.panel}

    @classmethod
    def create(
        cls,
        *,
        task_ids: tuple[str, ...],
        panel: tuple[PairedPanelPlan, ...],
        bounded_mean_bets: tuple[BoundedMeanBet, ...],
        schedule_seed: str,
        analysis_seed: str,
        randomization_samples: int,
        minimum_panel_delta: float,
        minimum_member_delta: float,
        noninferiority_margin: float,
        alpha: float = 0.05,
    ) -> PairedEvaluationDesign:
        """Create the canonical balanced AB/BA schedule for one declared matrix."""
        canonical_tasks = _canonical_names(task_ids, label="task_ids")
        canonical_panel = _canonical_panel(panel)
        canonical_bets = _canonical_bounded_mean_bets(bounded_mean_bets)
        blocks = _scheduled_blocks(
            canonical_tasks,
            canonical_panel,
            seed=schedule_seed,
        )
        return cls(
            task_ids=canonical_tasks,
            panel=canonical_panel,
            bounded_mean_bets=canonical_bets,
            schedule_seed=schedule_seed,
            analysis_seed=analysis_seed,
            randomization_samples=randomization_samples,
            alpha=alpha,
            minimum_panel_delta=minimum_panel_delta,
            minimum_member_delta=minimum_member_delta,
            noninferiority_margin=noninferiority_margin,
            blocks=blocks,
        )

    @model_validator(mode="after")
    def _validate_frozen_schedule(self) -> Self:
        if not self.task_ids or not self.panel:
            raise ValueError("paired evaluation needs at least one task and panel member")
        if _divide_float_downward(self.alpha, max(2, len(self.panel))) == 0.0:
            raise ValueError("alpha is too small for the frozen two-sided and memberwise tests")
        if self.task_ids != _canonical_names(self.task_ids, label="task_ids"):
            raise ValueError("task_ids must be unique and in canonical order")
        if self.panel != _canonical_panel(self.panel):
            raise ValueError("panel must be unique and in canonical order")
        if self.bounded_mean_bets != _canonical_bounded_mean_bets(self.bounded_mean_bets):
            raise ValueError("bounded-mean bets must be unique, normalized, and in canonical order")
        expected = _scheduled_blocks(
            self.task_ids,
            self.panel,
            seed=self.schedule_seed,
        )
        if self.blocks != expected:
            raise ValueError("paired blocks do not match the frozen schedule_seed")
        return self


class PairedBlockOutcome(BaseModel):
    """Admitted binary outcomes for both arms of one fresh-sandbox block."""

    model_config = ConfigDict(frozen=True)

    block: PairedBlock
    baseline_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    candidate_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("baseline_reward", "candidate_reward", mode="before")
    @classmethod
    def _require_binary_reward(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0, 1):
            raise ValueError("paired analysis rewards must be binary")
        return float(value)


class ClusterInterval(BaseModel):
    """Finite-sample bounded-mean interval over complete task clusters."""

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class PanelMemberAnalysis(BaseModel):
    """Effect and adjusted noninferiority evidence for one frozen panel member."""

    model_config = ConfigDict(frozen=True)

    panel_member: str = Field(min_length=1)
    delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    raw_positive_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    holm_positive_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    simultaneous_lower_bound: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    raw_noninferiority_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    holm_noninferiority_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PairedAnalysisReport(BaseModel):
    """Task-clustered effect estimates, primary endpoint, and diagnostics."""

    model_config = ConfigDict(frozen=True)

    analysis_version: Literal["2"]
    design_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    outcome_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    panel_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    members: tuple[PanelMemberAnalysis, ...]
    label_swap_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    panel_mean_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cluster_interval: ClusterInterval
    panel_lift_passed: bool
    member_lifts_passed: bool
    member_positive_passed: bool
    member_intervals_passed: bool
    label_swap_passed: bool
    panel_mean_passed: bool
    cluster_interval_passed: bool
    noninferiority_passed: bool

    @property
    def member_deltas(self) -> dict[str, float]:
        """Return a copy of member effect estimates keyed by frozen panel identity."""
        return {member.panel_member: member.delta for member in self.members}

    @property
    def raw_noninferiority_p(self) -> dict[str, float]:
        """Return unadjusted noninferiority p-values keyed by panel identity."""
        return {member.panel_member: member.raw_noninferiority_p for member in self.members}

    @property
    def raw_positive_p(self) -> dict[str, float]:
        """Return unadjusted positive-lift p-values keyed by panel identity."""
        return {member.panel_member: member.raw_positive_p for member in self.members}

    @property
    def holm_positive_p(self) -> dict[str, float]:
        """Return Holm-adjusted positive-lift p-values keyed by panel identity."""
        return {member.panel_member: member.holm_positive_p for member in self.members}

    @property
    def simultaneous_lower_bounds(self) -> dict[str, float]:
        """Return Bonferroni-simultaneous one-sided lower bounds by panel identity."""
        return {member.panel_member: member.simultaneous_lower_bound for member in self.members}

    @property
    def holm_noninferiority_p(self) -> dict[str, float]:
        """Return Holm-adjusted noninferiority p-values keyed by panel identity."""
        return {member.panel_member: member.holm_noninferiority_p for member in self.members}

    @property
    def passed(self) -> bool:
        """Return whether every lane clears its point floor and simultaneous lower bound."""
        return self.member_lifts_passed and self.member_intervals_passed


def analyze_paired_outcomes(
    design: PairedEvaluationDesign,
    outcomes: list[PairedBlockOutcome],
) -> PairedAnalysisReport:
    """Analyze one exact paired matrix under its frozen-roster independence assumptions.

    Every primary member test uses one attempt-averaged observation per frozen task.
    Arbitrary dependence among repeated attempts within a task is therefore allowed;
    finite-sample validity requires only independence across the frozen task clusters.
    Attempts improve the precision of each task mean but never inflate the primary
    sample size. The result targets equal-task expected success on the fixed roster
    and makes no unobserved-task population claim.
    """
    ordered = _validate_and_order_outcomes(design, outcomes)
    task_member_deltas = _task_member_deltas(design, ordered)
    member_deltas = {
        member: fmean(task_member_deltas[(task, member)] for task in design.task_ids)
        for member in design.panel_members
    }
    task_panel_deltas = {
        task: fmean(task_member_deltas[(task, member)] for member in design.panel_members)
        for task in design.task_ids
    }
    panel_delta = fmean(task_panel_deltas.values())
    label_swap_p = _one_sided_sign_flip_p(
        tuple(task_panel_deltas[task] for task in design.task_ids),
        samples=design.randomization_samples,
        seed=_domain_seed(design.analysis_seed, "panel-randomization"),
    )
    panel_task_deltas = tuple(task_panel_deltas[task] for task in design.task_ids)
    panel_mean_p = _bounded_mean_p(
        panel_task_deltas,
        null_mean=0.0,
        bets=design.bounded_mean_bets,
    )
    cluster_interval = _bounded_mean_interval(
        panel_task_deltas,
        alpha=design.alpha,
        bets=design.bounded_mean_bets,
    )
    member_task_deltas = {
        member: tuple(task_member_deltas[(task, member)] for task in design.task_ids)
        for member in design.panel_members
    }
    raw_positive = {
        member: _bounded_mean_p(
            member_task_deltas[member],
            null_mean=0.0,
            bets=design.bounded_mean_bets,
        )
        for member in design.panel_members
    }
    holm_positive = _holm_adjust(raw_positive)
    member_alpha = _divide_float_downward(design.alpha, len(design.panel_members))
    member_lower_bounds = {
        member: _bounded_mean_lower_bound(
            member_task_deltas[member],
            alpha=member_alpha,
            bets=design.bounded_mean_bets,
        )
        for member in design.panel_members
    }
    raw_noninferiority = {
        member: _bounded_mean_p(
            member_task_deltas[member],
            null_mean=-design.noninferiority_margin,
            bets=design.bounded_mean_bets,
        )
        for member in design.panel_members
    }
    holm = _holm_adjust(raw_noninferiority)
    panel_lift_passed = panel_delta >= design.minimum_panel_delta
    member_lifts_passed = all(
        delta >= design.minimum_member_delta for delta in member_deltas.values()
    )
    member_positive_passed = all(value < design.alpha for value in holm_positive.values())
    member_intervals_passed = all(value > 0.0 for value in member_lower_bounds.values())
    label_swap_passed = label_swap_p < design.alpha
    panel_mean_passed = panel_mean_p < design.alpha
    cluster_interval_passed = cluster_interval.lower > 0.0
    noninferiority_passed = all(value < design.alpha for value in holm.values())
    return PairedAnalysisReport(
        analysis_version=PAIRED_ANALYSIS_VERSION,
        design_digest=design.digest,
        outcome_digest=_canonical_digest([outcome.model_dump(mode="json") for outcome in ordered]),
        panel_delta=panel_delta,
        members=tuple(
            PanelMemberAnalysis(
                panel_member=member,
                delta=member_deltas[member],
                raw_positive_p=raw_positive[member],
                holm_positive_p=holm_positive[member],
                simultaneous_lower_bound=member_lower_bounds[member],
                raw_noninferiority_p=raw_noninferiority[member],
                holm_noninferiority_p=holm[member],
            )
            for member in design.panel_members
        ),
        label_swap_p=label_swap_p,
        panel_mean_p=panel_mean_p,
        cluster_interval=cluster_interval,
        panel_lift_passed=panel_lift_passed,
        member_lifts_passed=member_lifts_passed,
        member_positive_passed=member_positive_passed,
        member_intervals_passed=member_intervals_passed,
        label_swap_passed=label_swap_passed,
        panel_mean_passed=panel_mean_passed,
        cluster_interval_passed=cluster_interval_passed,
        noninferiority_passed=noninferiority_passed,
    )


def _canonical_names(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    if not values or any(
        not isinstance(value, str) or not value.strip() or value != value.strip()
        for value in values
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates}")
    return tuple(sorted(values))


def _canonical_panel(panel: tuple[PairedPanelPlan, ...]) -> tuple[PairedPanelPlan, ...]:
    if not panel:
        raise ValueError("panel must contain at least one member")
    duplicates = sorted(
        name for name, count in Counter(plan.panel_member for plan in panel).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"panel contains duplicate members: {duplicates}")
    return tuple(sorted(panel, key=lambda plan: plan.panel_member))


def _canonical_bounded_mean_bets(
    bets: tuple[BoundedMeanBet, ...],
) -> tuple[BoundedMeanBet, ...]:
    if not bets:
        raise ValueError("bounded-mean evidence needs at least one frozen bet")
    fractions = [bet.fraction for bet in bets]
    duplicates = sorted(fraction for fraction, count in Counter(fractions).items() if count > 1)
    if duplicates:
        raise ValueError(f"bounded-mean bets contain duplicate fractions: {duplicates}")
    if sum((Fraction.from_float(bet.weight) for bet in bets), start=Fraction()) != 1:
        raise ValueError("bounded-mean bet weights must sum to one")
    return tuple(sorted(bets, key=lambda bet: bet.fraction))


def _scheduled_blocks(
    task_ids: tuple[str, ...],
    panel: tuple[PairedPanelPlan, ...],
    *,
    seed: str,
) -> tuple[PairedBlock, ...]:
    if not seed:
        raise ValueError("schedule seed must be non-empty")
    blocks: list[PairedBlock] = []
    for plan in panel:
        member = plan.panel_member
        candidate_extra_tasks: frozenset[str] = frozenset()
        if plan.attempts % 2:
            ranked_tasks = sorted(
                task_ids,
                key=lambda task_id: _digest_bytes(
                    seed,
                    "odd-attempt-task-order",
                    member,
                    task_id,
                ),
            )
            candidate_extra_count = len(task_ids) // 2
            if len(task_ids) % 2:
                candidate_extra_count += _digest_bytes(seed, "odd-attempt-extra-arm", member)[0] & 1
            candidate_extra_tasks = frozenset(ranked_tasks[:candidate_extra_count])
        for task_id in task_ids:
            ranked_attempts = sorted(
                range(1, plan.attempts + 1),
                key=lambda attempt: _digest_bytes(
                    seed,
                    "attempt-order",
                    member,
                    task_id,
                    str(attempt),
                ),
            )
            candidate_first_count = plan.attempts // 2 + (task_id in candidate_extra_tasks)
            candidate_first_attempts = frozenset(ranked_attempts[:candidate_first_count])
            blocks.extend(
                PairedBlock(
                    task_id=task_id,
                    panel_member=member,
                    attempt=attempt,
                    first_arm=(
                        PairedArm.CANDIDATE
                        if attempt in candidate_first_attempts
                        else PairedArm.BASELINE
                    ),
                )
                for attempt in range(1, plan.attempts + 1)
            )
    return tuple(blocks)


def _validate_and_order_outcomes(
    design: PairedEvaluationDesign,
    outcomes: list[PairedBlockOutcome],
) -> tuple[PairedBlockOutcome, ...]:
    counts = Counter(outcome.block.key for outcome in outcomes)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"paired outcome matrix contains duplicate cells: {duplicates}")
    expected = {block.key: block for block in design.blocks}
    observed = {outcome.block.key: outcome for outcome in outcomes}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"paired outcome matrix differs from design: missing={missing}, extra={extra}"
        )
    mismatched = sorted(key for key, outcome in observed.items() if outcome.block != expected[key])
    if mismatched:
        raise ValueError(f"paired outcome execution order differs from design: {mismatched}")
    return tuple(observed[block.key] for block in design.blocks)


def _task_member_deltas(
    design: PairedEvaluationDesign,
    outcomes: tuple[PairedBlockOutcome, ...],
) -> dict[tuple[str, str], float]:
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for outcome in outcomes:
        grouped[(outcome.block.task_id, outcome.block.panel_member)].append(
            outcome.candidate_reward - outcome.baseline_reward
        )
    expected_attempts = design.attempts_by_member
    if any(len(values) != expected_attempts[member] for (_task, member), values in grouped.items()):
        raise ValueError("paired outcome task/member groups omit planned attempts")
    return {key: fmean(values) for key, values in grouped.items()}


def _one_sided_sign_flip_p(
    task_deltas: tuple[float, ...],
    *,
    samples: int,
    seed: int,
) -> float:
    observed = fmean(task_deltas)
    if len(task_deltas) <= 20:
        total = 1 << len(task_deltas)
        at_least_observed = 0
        for mask in range(total):
            statistic = fmean(
                delta if mask & (1 << index) else -delta for index, delta in enumerate(task_deltas)
            )
            at_least_observed += statistic >= observed - 1e-15
        return at_least_observed / total
    generator = random.Random(seed)
    at_least_observed = 0
    for _ in range(samples):
        statistic = fmean(delta if generator.getrandbits(1) else -delta for delta in task_deltas)
        at_least_observed += statistic >= observed - 1e-15
    return (at_least_observed + 1) / (samples + 1)


def _bounded_mean_p(
    task_deltas: tuple[float, ...],
    *,
    null_mean: float,
    bets: tuple[BoundedMeanBet, ...],
) -> float:
    """Test the composite null that the average task mean is at most ``null_mean``.

    Each supplied delta must be an independent observation in [-1, 1]. The
    caller may supply task-cluster deltas or fixed-roster fresh-attempt deltas
    according to its declared estimand. The preregistered mixture is an e-value,
    so ``min(1, 1 / e_value)`` is a finite-sample p-value without a symmetry or
    identical-distribution assumption.
    """
    log_e_value = _bounded_mean_log_e(
        task_deltas,
        null_mean=null_mean,
        bets=bets,
    )
    if null_mean == -1.0:
        return 1.0 if all(delta == -1.0 for delta in task_deltas) else 0.0
    nominal_p = (
        1.0 if log_e_value <= 0.0 else 0.0 if math.isinf(log_e_value) else math.exp(-log_e_value)
    )
    exact_e_value = _bounded_mean_exact_e(
        task_deltas,
        null_mean=null_mean,
        bets=bets,
    )
    if exact_e_value <= 1:
        return 1.0
    certified_p = _fraction_to_float_ceiling(1 / exact_e_value)
    return max(nominal_p, certified_p)


def _bounded_mean_interval(
    task_deltas: tuple[float, ...],
    *,
    alpha: float,
    bets: tuple[BoundedMeanBet, ...],
) -> ClusterInterval:
    """Invert two one-sided bounded-mean e-tests with Bonferroni coverage."""
    tail_alpha = _divide_float_downward(alpha, 2)
    lower = _bounded_mean_lower_bound(
        task_deltas,
        alpha=tail_alpha,
        bets=bets,
    )
    upper = -_bounded_mean_lower_bound(
        tuple(-delta for delta in task_deltas),
        alpha=tail_alpha,
        bets=bets,
    )
    return ClusterInterval(lower=lower, upper=upper)


def _bounded_mean_lower_bound(
    task_deltas: tuple[float, ...],
    *,
    alpha: float,
    bets: tuple[BoundedMeanBet, ...],
) -> float:
    """Return a lower endpoint rounded below the one-sided rejection boundary."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("bounded-mean test alpha must be between zero and one")
    rejection_log_e = -math.log(alpha)
    rejected_mean = math.nextafter(-1.0, 0.0)
    nonrejected_mean = 1.0
    if not _bounded_mean_exact_rejects(
        task_deltas,
        null_mean=rejected_mean,
        alpha=alpha,
        bets=bets,
    ):
        return -1.0
    for _ in range(107):
        midpoint = (rejected_mean + nonrejected_mean) / 2.0
        if midpoint in (rejected_mean, nonrejected_mean):
            break
        if (
            _bounded_mean_log_e(
                task_deltas,
                null_mean=midpoint,
                bets=bets,
            )
            > rejection_log_e
        ):
            rejected_mean = midpoint
        else:
            nonrejected_mean = midpoint
    candidate = max(-1.0, math.nextafter(rejected_mean, -math.inf))
    if candidate == -1.0 or _bounded_mean_exact_rejects(
        task_deltas,
        null_mean=candidate,
        alpha=alpha,
        bets=bets,
    ):
        return candidate

    exact_rejected = math.nextafter(-1.0, 0.0)
    exact_nonrejected = candidate
    for _ in range(107):
        midpoint = (exact_rejected + exact_nonrejected) / 2.0
        if midpoint in (exact_rejected, exact_nonrejected):
            break
        if _bounded_mean_exact_rejects(
            task_deltas,
            null_mean=midpoint,
            alpha=alpha,
            bets=bets,
        ):
            exact_rejected = midpoint
        else:
            exact_nonrejected = midpoint
    return max(-1.0, math.nextafter(exact_rejected, -math.inf))


def _bounded_mean_log_e(
    task_deltas: tuple[float, ...],
    *,
    null_mean: float,
    bets: tuple[BoundedMeanBet, ...],
) -> float:
    if not task_deltas:
        raise ValueError("bounded-mean evidence needs at least one task delta")
    if any(not math.isfinite(delta) or not -1.0 <= delta <= 1.0 for delta in task_deltas):
        raise ValueError("bounded-mean task deltas must be finite and in [-1, 1]")
    if not math.isfinite(null_mean) or not -1.0 <= null_mean <= 1.0:
        raise ValueError("bounded-mean null must be finite and in [-1, 1]")
    if null_mean == -1.0:
        return 0.0 if all(delta == -1.0 for delta in task_deltas) else math.inf
    canonical_bets = _canonical_bounded_mean_bets(bets)
    component_logs: list[float] = []
    denominator = 1.0 + null_mean
    for bet in canonical_bets:
        component_log = math.log(bet.weight)
        for delta in task_deltas:
            factor = (1.0 - bet.fraction) + bet.fraction * (1.0 + delta) / denominator
            if factor == 0.0:
                component_log = -math.inf
                break
            component_log += math.log(factor)
        component_logs.append(component_log)
    return _log_sum_exp(component_logs)


def _bounded_mean_exact_e(
    task_deltas: tuple[float, ...],
    *,
    null_mean: float,
    bets: tuple[BoundedMeanBet, ...],
) -> Fraction:
    """Return the exact e-value over the binary-float inputs for safe rounding."""
    if null_mean == -1.0:
        raise ValueError("the negative-one support null is handled before exact e-value evaluation")
    one = Fraction(1)
    null_fraction = Fraction.from_float(null_mean)
    denominator = one + null_fraction
    e_value = Fraction()
    for bet in _canonical_bounded_mean_bets(bets):
        fraction = Fraction.from_float(bet.fraction)
        component = Fraction.from_float(bet.weight)
        for delta in task_deltas:
            delta_fraction = Fraction.from_float(delta)
            factor = (one - fraction) + fraction * (one + delta_fraction) / denominator
            component *= factor
        e_value += component
    return e_value


def _bounded_mean_exact_rejects(
    task_deltas: tuple[float, ...],
    *,
    null_mean: float,
    alpha: float,
    bets: tuple[BoundedMeanBet, ...],
) -> bool:
    e_value = _bounded_mean_exact_e(
        task_deltas,
        null_mean=null_mean,
        bets=bets,
    )
    return e_value * Fraction.from_float(alpha) > 1


def _fraction_to_float_ceiling(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    return rounded


def _fraction_to_float_floor(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) > value:
        rounded = math.nextafter(rounded, -math.inf)
    return rounded


def _divide_float_downward(value: float, divisor: int) -> float:
    """Divide a binary float by an integer without rounding the result upward."""
    return _fraction_to_float_floor(Fraction.from_float(value) / divisor)


def _log_sum_exp(values: list[float]) -> float:
    largest = max(values)
    if largest == -math.inf:
        return -math.inf
    return largest + math.log(math.fsum(math.exp(value - largest) for value in values))


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        scaled = Fraction.from_float(value) * (count - index)
        multiplied_upward = min(1.0, _fraction_to_float_ceiling(scaled))
        running = max(running, multiplied_upward)
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _domain_seed(seed: str, domain: str) -> int:
    return int.from_bytes(_digest_bytes(seed, domain)[:8], "big")


def _digest_bytes(*parts: str) -> bytes:
    hasher = hashlib.sha256()
    for part in parts:
        encoded = part.encode()
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.digest()


def _canonical_digest(value: JsonValue) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
