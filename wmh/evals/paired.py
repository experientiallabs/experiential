"""Deterministic paired benchmark blocks and task-clustered panel analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from enum import StrEnum
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

PAIRED_ANALYSIS_VERSION: Literal["1"] = "1"


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

    task_ids: tuple[str, ...]
    panel: tuple[PairedPanelPlan, ...]
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    randomization_samples: StrictInt = Field(ge=999)
    bootstrap_samples: StrictInt = Field(ge=999)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_panel_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    minimum_member_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    noninferiority_margin: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    blocks: tuple[PairedBlock, ...]

    @field_validator("randomization_samples", "bootstrap_samples", mode="before")
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
        schedule_seed: str,
        analysis_seed: str,
        randomization_samples: int,
        bootstrap_samples: int,
        minimum_panel_delta: float,
        minimum_member_delta: float,
        noninferiority_margin: float,
        alpha: float = 0.05,
    ) -> PairedEvaluationDesign:
        """Create the canonical balanced AB/BA schedule for one declared matrix."""
        canonical_tasks = _canonical_names(task_ids, label="task_ids")
        canonical_panel = _canonical_panel(panel)
        blocks = _scheduled_blocks(
            canonical_tasks,
            canonical_panel,
            seed=schedule_seed,
        )
        return cls(
            task_ids=canonical_tasks,
            panel=canonical_panel,
            schedule_seed=schedule_seed,
            analysis_seed=analysis_seed,
            randomization_samples=randomization_samples,
            bootstrap_samples=bootstrap_samples,
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
        if self.task_ids != _canonical_names(self.task_ids, label="task_ids"):
            raise ValueError("task_ids must be unique and in canonical order")
        if self.panel != _canonical_panel(self.panel):
            raise ValueError("panel must be unique and in canonical order")
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
    """Frozen percentile interval from resampling complete tasks."""

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class PanelMemberAnalysis(BaseModel):
    """Effect and adjusted noninferiority evidence for one frozen panel member."""

    model_config = ConfigDict(frozen=True)

    panel_member: str = Field(min_length=1)
    delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    raw_noninferiority_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    holm_noninferiority_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PairedAnalysisReport(BaseModel):
    """Task-clustered effect estimates and every predeclared decision gate."""

    model_config = ConfigDict(frozen=True)

    analysis_version: Literal["1"]
    design_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    outcome_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    panel_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    members: tuple[PanelMemberAnalysis, ...]
    randomization_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    cluster_interval: ClusterInterval
    panel_lift_passed: bool
    member_lifts_passed: bool
    randomization_passed: bool
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
    def holm_noninferiority_p(self) -> dict[str, float]:
        """Return Holm-adjusted noninferiority p-values keyed by panel identity."""
        return {member.panel_member: member.holm_noninferiority_p for member in self.members}

    @property
    def passed(self) -> bool:
        """Return whether the result clears every frozen success criterion."""
        return all(
            (
                self.panel_lift_passed,
                self.member_lifts_passed,
                self.randomization_passed,
                self.cluster_interval_passed,
                self.noninferiority_passed,
            )
        )


def analyze_paired_outcomes(
    design: PairedEvaluationDesign,
    outcomes: list[PairedBlockOutcome],
) -> PairedAnalysisReport:
    """Analyze one exact paired matrix with task-level clustering across attempts and models."""
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
    randomization_p = _one_sided_sign_flip_p(
        tuple(task_panel_deltas[task] for task in design.task_ids),
        samples=design.randomization_samples,
        seed=_domain_seed(design.analysis_seed, "panel-randomization"),
    )
    cluster_interval = _cluster_interval(
        tuple(task_panel_deltas[task] for task in design.task_ids),
        samples=design.bootstrap_samples,
        alpha=design.alpha,
        seed=_domain_seed(design.analysis_seed, "panel-bootstrap"),
    )
    raw_noninferiority = {
        member: _one_sided_sign_flip_p(
            tuple(
                task_member_deltas[(task, member)] + design.noninferiority_margin
                for task in design.task_ids
            ),
            samples=design.randomization_samples,
            seed=_domain_seed(design.analysis_seed, f"noninferiority:{member}"),
        )
        for member in design.panel_members
    }
    holm = _holm_adjust(raw_noninferiority)
    panel_lift_passed = panel_delta >= design.minimum_panel_delta
    member_lifts_passed = all(
        delta >= design.minimum_member_delta for delta in member_deltas.values()
    )
    randomization_passed = randomization_p < design.alpha
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
                raw_noninferiority_p=raw_noninferiority[member],
                holm_noninferiority_p=holm[member],
            )
            for member in design.panel_members
        ),
        randomization_p=randomization_p,
        cluster_interval=cluster_interval,
        panel_lift_passed=panel_lift_passed,
        member_lifts_passed=member_lifts_passed,
        randomization_passed=randomization_passed,
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
        ranked_tasks = sorted(
            task_ids,
            key=lambda task_id: _digest_bytes(seed, "task-order", member, task_id),
        )
        candidate_first_count = len(task_ids) // 2
        if len(task_ids) % 2:
            candidate_first_count += _digest_bytes(seed, "odd-arm", member)[0] & 1
        candidate_first_tasks = frozenset(ranked_tasks[:candidate_first_count])
        for task_id in task_ids:
            first_arm = (
                PairedArm.CANDIDATE if task_id in candidate_first_tasks else PairedArm.BASELINE
            )
            blocks.extend(
                PairedBlock(
                    task_id=task_id,
                    panel_member=member,
                    attempt=attempt,
                    first_arm=first_arm,
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


def _cluster_interval(
    task_deltas: tuple[float, ...],
    *,
    samples: int,
    alpha: float,
    seed: int,
) -> ClusterInterval:
    generator = random.Random(seed)
    task_count = len(task_deltas)
    statistics = sorted(
        fmean(task_deltas[generator.randrange(task_count)] for _ in range(task_count))
        for _ in range(samples)
    )
    lower_index = max(0, math.ceil((alpha / 2) * samples) - 1)
    upper_index = min(samples - 1, math.ceil((1 - alpha / 2) * samples) - 1)
    return ClusterInterval(lower=statistics[lower_index], upper=statistics[upper_index])


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
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
