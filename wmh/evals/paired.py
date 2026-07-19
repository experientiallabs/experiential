"""Deterministic paired benchmark blocks and fixed-roster panel analysis."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from scipy.stats import t as student_t

PAIRED_ANALYSIS_VERSION: Literal["5"] = "5"
PAIRED_PRIMARY_ESTIMAND: Literal[
    "fixed-roster-equal-task-conditional-expected-paired-reward-delta"
] = "fixed-roster-equal-task-conditional-expected-paired-reward-delta"
PAIRED_PRIMARY_EVIDENCE_METHOD: Literal[
    "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
] = "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
PAIRED_SEMANTIC_CLUSTER_SENSITIVITY_METHOD: Literal[
    "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
] = "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
PAIRED_MODEL_BASED_DIAGNOSTIC_METHOD: Literal[
    "leave-one-semantic-cluster-out-jackknife-student-t"
] = "leave-one-semantic-cluster-out-jackknife-student-t"
PAIRED_PRIMARY_COMBINATION_RULE: Literal["intersection-union-all-lanes"] = (
    "intersection-union-all-lanes"
)


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


class PairedTaskPlan(BaseModel):
    """One task identity and its predeclared semantic sensitivity cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)

    @field_validator("task_id", "group_id")
    @classmethod
    def _require_canonical_identity(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("paired task identities cannot have surrounding whitespace")
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

    analysis_version: Literal["5"] = PAIRED_ANALYSIS_VERSION
    primary_estimand: Literal[
        "fixed-roster-equal-task-conditional-expected-paired-reward-delta"
    ] = PAIRED_PRIMARY_ESTIMAND
    primary_evidence_method: Literal[
        "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
    ] = PAIRED_PRIMARY_EVIDENCE_METHOD
    semantic_cluster_sensitivity_method: Literal[
        "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
    ] = PAIRED_SEMANTIC_CLUSTER_SENSITIVITY_METHOD
    model_based_diagnostic_method: Literal["leave-one-semantic-cluster-out-jackknife-student-t"] = (
        PAIRED_MODEL_BASED_DIAGNOSTIC_METHOD
    )
    primary_combination_rule: Literal["intersection-union-all-lanes"] = (
        PAIRED_PRIMARY_COMBINATION_RULE
    )
    tasks: tuple[PairedTaskPlan, ...]
    panel: tuple[PairedPanelPlan, ...]
    primary_e_value_bets: tuple[BoundedMeanBet, ...]
    schedule_seed: str = Field(min_length=1)
    analysis_seed: str = Field(min_length=1)
    randomization_samples: StrictInt = Field(ge=999)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, allow_inf_nan=False)
    minimum_equal_task_member_delta: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
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
        "minimum_equal_task_member_delta",
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
    def task_ids(self) -> tuple[str, ...]:
        """Return task identities in their frozen canonical order."""
        return tuple(task.task_id for task in self.tasks)

    @property
    def group_ids_by_task(self) -> dict[str, str]:
        """Return a copy of score-independent cluster identities keyed by task."""
        return {task.task_id: task.group_id for task in self.tasks}

    @property
    def group_ids(self) -> tuple[str, ...]:
        """Return unique score-independent clusters in canonical order."""
        return tuple(sorted({task.group_id for task in self.tasks}))

    @property
    def attempts_by_member(self) -> dict[str, int]:
        """Return a copy of repeated-attempt counts keyed by panel identity."""
        return {plan.panel_member: plan.attempts for plan in self.panel}

    @classmethod
    def create(
        cls,
        *,
        tasks: tuple[PairedTaskPlan, ...],
        panel: tuple[PairedPanelPlan, ...],
        primary_e_value_bets: tuple[BoundedMeanBet, ...],
        schedule_seed: str,
        analysis_seed: str,
        randomization_samples: int,
        minimum_equal_task_member_delta: float,
        noninferiority_margin: float,
        alpha: float = 0.05,
    ) -> PairedEvaluationDesign:
        """Create the canonical balanced AB/BA schedule for one declared matrix."""
        canonical_tasks = _canonical_tasks(tasks)
        canonical_panel = _canonical_panel(panel)
        canonical_bets = _canonical_bounded_mean_bets(primary_e_value_bets)
        blocks = _scheduled_blocks(
            tuple(task.task_id for task in canonical_tasks),
            canonical_panel,
            seed=schedule_seed,
        )
        return cls(
            tasks=canonical_tasks,
            panel=canonical_panel,
            primary_e_value_bets=canonical_bets,
            schedule_seed=schedule_seed,
            analysis_seed=analysis_seed,
            randomization_samples=randomization_samples,
            alpha=alpha,
            minimum_equal_task_member_delta=minimum_equal_task_member_delta,
            noninferiority_margin=noninferiority_margin,
            blocks=blocks,
        )

    @model_validator(mode="after")
    def _validate_frozen_schedule(self) -> Self:
        if not self.tasks or not self.panel:
            raise ValueError("paired evaluation needs at least one task and panel member")
        if _divide_float_downward(self.alpha, max(2, len(self.panel))) == 0.0:
            raise ValueError("alpha is too small for the frozen primary and secondary bounds")
        if self.tasks != _canonical_tasks(self.tasks):
            raise ValueError("paired tasks must be unique and in canonical order")
        if len(self.group_ids) < 2:
            raise ValueError("paired diagnostics need at least two semantic sensitivity clusters")
        if self.panel != _canonical_panel(self.panel):
            raise ValueError("panel must be unique and in canonical order")
        if self.primary_e_value_bets != _canonical_bounded_mean_bets(self.primary_e_value_bets):
            raise ValueError(
                "primary e-value bets must be unique, normalized, and in canonical order"
            )
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


class BoundedMeanInterval(BaseModel):
    """Finite-sample bounded-mean interval over independent bounded observations."""

    model_config = ConfigDict(frozen=True)

    lower: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    upper: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)


class PanelMemberAnalysis(BaseModel):
    """Fixed-roster primary evidence, cluster sensitivity, and diagnostics for one lane."""

    model_config = ConfigDict(frozen=True)

    panel_member: str = Field(min_length=1)
    equal_task_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    primary_positive_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    primary_lower_bound: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    semantic_cluster_sensitivity_positive_p: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    semantic_cluster_sensitivity_lower_bound: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    model_based_jackknife_standard_error: float = Field(ge=0.0, allow_inf_nan=False)
    model_based_jackknife_degrees_of_freedom: StrictInt = Field(ge=1)
    model_based_jackknife_positive_p: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    model_based_jackknife_lower_bound: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    model_based_jackknife_bonferroni_lower_bound: float = Field(
        ge=-1.0,
        le=1.0,
        allow_inf_nan=False,
    )
    secondary_noninferiority_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    secondary_holm_noninferiority_p: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )


class PairedAnalysisReport(BaseModel):
    """Finite-sample primary evidence, effect-size endpoints, and diagnostics."""

    model_config = ConfigDict(frozen=True)

    analysis_version: Literal["5"]
    primary_estimand: Literal["fixed-roster-equal-task-conditional-expected-paired-reward-delta"]
    primary_evidence_method: Literal[
        "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
    ]
    semantic_cluster_sensitivity_method: Literal[
        "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
    ]
    model_based_diagnostic_method: Literal["leave-one-semantic-cluster-out-jackknife-student-t"]
    primary_combination_rule: Literal["intersection-union-all-lanes"]
    design_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    outcome_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    equal_task_panel_delta: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    members: tuple[PanelMemberAnalysis, ...]
    model_based_label_swap_p: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    equal_task_member_lifts_passed: bool
    member_primary_bounds_passed: bool
    member_semantic_cluster_sensitivity_bounds_positive: bool
    member_model_based_jackknife_bounds_passed: bool
    member_model_based_jackknife_bonferroni_bounds_passed: bool
    model_based_label_swap_passed: bool
    secondary_noninferiority_passed: bool

    @property
    def equal_task_member_deltas(self) -> dict[str, float]:
        """Return observed equal-task effects on the fixed roster by lane identity."""
        return {member.panel_member: member.equal_task_delta for member in self.members}

    @property
    def semantic_cluster_sensitivity_positive_p(self) -> dict[str, float]:
        """Return conservative semantic-cluster sensitivity p-values by lane."""
        return {
            member.panel_member: member.semantic_cluster_sensitivity_positive_p
            for member in self.members
        }

    @property
    def semantic_cluster_sensitivity_lower_bounds(self) -> dict[str, float]:
        """Return conservative semantic-cluster sensitivity lower bounds by lane."""
        return {
            member.panel_member: member.semantic_cluster_sensitivity_lower_bound
            for member in self.members
        }

    @property
    def secondary_noninferiority_p(self) -> dict[str, float]:
        """Return secondary noninferiority p-values keyed by panel identity."""
        return {member.panel_member: member.secondary_noninferiority_p for member in self.members}

    @property
    def primary_positive_p(self) -> dict[str, float]:
        """Return primary fixed-roster positive-lift p-values by lane identity."""
        return {member.panel_member: member.primary_positive_p for member in self.members}

    @property
    def primary_lower_bounds(self) -> dict[str, float]:
        """Return finite-sample fixed-roster lower bounds by lane identity."""
        return {member.panel_member: member.primary_lower_bound for member in self.members}

    @property
    def model_based_jackknife_lower_bounds(self) -> dict[str, float]:
        """Return unadjusted model-based jackknife bounds by lane identity."""
        return {
            member.panel_member: member.model_based_jackknife_lower_bound for member in self.members
        }

    @property
    def model_based_jackknife_bonferroni_lower_bounds(self) -> dict[str, float]:
        """Return model-based Bonferroni jackknife bounds by lane identity."""
        return {
            member.panel_member: member.model_based_jackknife_bonferroni_lower_bound
            for member in self.members
        }

    @property
    def model_based_jackknife_positive_p(self) -> dict[str, float]:
        """Return model-based jackknife p-values by lane identity."""
        return {
            member.panel_member: member.model_based_jackknife_positive_p for member in self.members
        }

    @property
    def secondary_holm_noninferiority_p(self) -> dict[str, float]:
        """Return Holm-adjusted secondary noninferiority p-values by panel identity."""
        return {
            member.panel_member: member.secondary_holm_noninferiority_p for member in self.members
        }

    @property
    def passed(self) -> bool:
        """Return the primary IUT plus equal-task observed effect-size decision."""
        return self.equal_task_member_lifts_passed and self.member_primary_bounds_passed


@dataclass(frozen=True)
class _JackknifeStudentTInference:
    """One model-based leave-one-semantic-cluster-out Student-t result."""

    standard_error: float
    degrees_of_freedom: int
    positive_p: float
    lower_bound: float


def paired_primary_decision_passed(
    design: PairedEvaluationDesign,
    task_deltas_by_member: tuple[tuple[float, ...], ...],
) -> bool:
    """Evaluate only the exact frozen v5 primary decision.

    Rows must follow ``design.panel_members`` and columns must follow
    ``design.task_ids``. This compact surface lets locked operating-characteristic
    simulations exercise the production effect floor, one-sided bounded-mean
    rejection, and all-lane intersection rule without constructing diagnostics or
    synthetic block records. It is equivalent to ``analyze_paired_outcomes(...).passed``
    for a complete matrix with the supplied task means.
    """
    if len(task_deltas_by_member) != len(design.panel_members):
        raise ValueError("primary decision rows must exactly match the frozen lane set")
    if any(len(row) != len(design.task_ids) for row in task_deltas_by_member):
        raise ValueError("primary decision columns must exactly match the frozen task roster")
    if any(
        not math.isfinite(delta) or not -1.0 <= delta <= 1.0
        for row in task_deltas_by_member
        for delta in row
    ):
        raise ValueError("primary decision task deltas must be finite and in [-1, 1]")
    return all(
        fmean(row) >= design.minimum_equal_task_member_delta
        and _bounded_mean_exact_rejects(
            row,
            null_mean=0.0,
            alpha=design.alpha,
            bets=design.primary_e_value_bets,
        )
        for row in task_deltas_by_member
    )


def analyze_paired_outcomes(
    design: PairedEvaluationDesign,
    outcomes: list[PairedBlockOutcome],
) -> PairedAnalysisReport:
    """Analyze one complete fixed-roster paired matrix.

    For a lane, the primary observations are the complete planned per-task paired-
    attempt mean deltas in [-1, 1]. Attempts may be arbitrarily dependent within a
    task, but the complete task outcome vectors must be mutually independent. The
    estimand is the equal-task mean of expected rerun deltas conditional on these
    exact held-out task identities, the frozen harnesses, and the execution contract.
    It is not a future-task, task-population, or finite-all-benchmark estimand.

    For each frozen bet, task independence factors the expected product. Under the
    weak null that the equal-task average expectation is at most ``m``, AM-GM bounds
    that product by one. The frozen mixture is therefore an e-value even when task
    distributions are nonidentical. Inverting its one-sided test gives the finite-
    sample primary lower bound.

    A conservative sensitivity allows arbitrary dependence within predeclared
    semantic clusters and assumes only cluster independence. If cluster ``g`` has
    equal-task weight ``w_g`` and mean ``X_g``, it uses scale
    ``c_g = w_g / max_h(w_h)`` and factor
    ``1 + f*c_g*(X_g-m)/(1+m)``. The weighted null makes the average expected
    factor at most one because ``sum_g c_g*(E[X_g]-m)`` equals
    ``(theta-m)/max_h(w_h)`` for the equal-task mean ``theta``. Independence and
    AM-GM again yield an e-value. Failure of this stricter sensitivity is
    inconclusive and does not alter the primary decision.

    Every lane must clear its own unadjusted ``alpha`` primary bound. This
    intersection-union rule controls the all-lanes claim at ``alpha`` without lane
    independence or multiplicity correction. Every lane must also clear the frozen
    observed equal-task effect floor. Jackknife Student-t, its Bonferroni variant,
    and label swapping are explicitly model-based secondary diagnostics and make no
    finite-sample alpha-control claim.

    Validity additionally requires a fixed horizon and frozen bets; complete,
    score-blind admission of every planned pair; no score-adaptive missingness or
    retry; fresh isolated arm requests and sandboxes; and whole-pair score-blind
    handling of allowlisted infrastructure failures. Common shocks across tasks,
    shared mutable state, provider drift, or correlated infrastructure incidents
    invalidate the primary task-independence claim.
    """
    ordered = _validate_and_order_outcomes(design, outcomes)
    task_member_deltas = _task_member_deltas(design, ordered)
    equal_task_member_deltas = {
        member: fmean(task_member_deltas[(task, member)] for task in design.task_ids)
        for member in design.panel_members
    }
    task_panel_deltas = {
        task: fmean(task_member_deltas[(task, member)] for member in design.panel_members)
        for task in design.task_ids
    }
    equal_task_panel_delta = fmean(task_panel_deltas.values())
    model_based_label_swap_p = _one_sided_sign_flip_p(
        _group_sum_deltas(
            design,
            {task: task_panel_deltas[task] for task in design.task_ids},
        ),
        samples=design.randomization_samples,
        seed=_domain_seed(design.analysis_seed, "panel-randomization"),
    )
    member_task_deltas = {
        member: tuple(task_member_deltas[(task, member)] for task in design.task_ids)
        for member in design.panel_members
    }
    primary_decision_passed = paired_primary_decision_passed(
        design,
        tuple(member_task_deltas[member] for member in design.panel_members),
    )
    member_semantic_observations = {
        member: _semantic_group_observations(
            design,
            {task: task_member_deltas[(task, member)] for task in design.task_ids},
        )
        for member in design.panel_members
    }
    primary_positive = {
        member: _bounded_mean_p(
            member_task_deltas[member],
            null_mean=0.0,
            bets=design.primary_e_value_bets,
        )
        for member in design.panel_members
    }
    primary_lower_bounds = {
        member: _bounded_mean_lower_bound(
            member_task_deltas[member],
            alpha=design.alpha,
            bets=design.primary_e_value_bets,
        )
        for member in design.panel_members
    }
    semantic_cluster_sensitivity_positive = {
        member: _bounded_mean_p(
            member_semantic_observations[member][0],
            null_mean=0.0,
            bets=design.primary_e_value_bets,
            observation_scales=member_semantic_observations[member][1],
        )
        for member in design.panel_members
    }
    semantic_cluster_sensitivity_lower_bounds = {
        member: _bounded_mean_lower_bound(
            member_semantic_observations[member][0],
            alpha=design.alpha,
            bets=design.primary_e_value_bets,
            observation_scales=member_semantic_observations[member][1],
        )
        for member in design.panel_members
    }
    model_based_jackknife = {
        member: _jackknife_student_t_inference(
            member_task_deltas[member],
            design=design,
            alpha=design.alpha,
        )
        for member in design.panel_members
    }
    model_based_bonferroni_alpha = _divide_float_downward(
        design.alpha,
        len(design.panel_members),
    )
    model_based_jackknife_bonferroni_lower_bounds = {
        member: _jackknife_student_t_lower_bound(
            equal_task_member_deltas[member],
            standard_error=model_based_jackknife[member].standard_error,
            degrees_of_freedom=model_based_jackknife[member].degrees_of_freedom,
            alpha=model_based_bonferroni_alpha,
        )
        for member in design.panel_members
    }
    secondary_noninferiority = {
        member: _bounded_mean_p(
            member_task_deltas[member],
            null_mean=-design.noninferiority_margin,
            bets=design.primary_e_value_bets,
        )
        for member in design.panel_members
    }
    secondary_holm_noninferiority = _holm_adjust(secondary_noninferiority)
    equal_task_member_lifts_passed = all(
        delta >= design.minimum_equal_task_member_delta
        for delta in equal_task_member_deltas.values()
    )
    member_primary_bounds_passed = all(value > 0.0 for value in primary_lower_bounds.values())
    if primary_decision_passed != (equal_task_member_lifts_passed and member_primary_bounds_passed):
        raise RuntimeError("exact primary rejection and inverted lower bound disagree")
    member_semantic_cluster_sensitivity_bounds_positive = all(
        value > 0.0 for value in semantic_cluster_sensitivity_lower_bounds.values()
    )
    member_model_based_jackknife_bounds_passed = all(
        inference.lower_bound > 0.0 for inference in model_based_jackknife.values()
    )
    member_model_based_jackknife_bonferroni_bounds_passed = all(
        value > 0.0 for value in model_based_jackknife_bonferroni_lower_bounds.values()
    )
    model_based_label_swap_passed = model_based_label_swap_p < design.alpha
    secondary_noninferiority_passed = all(
        value < design.alpha for value in secondary_holm_noninferiority.values()
    )
    return PairedAnalysisReport(
        analysis_version=PAIRED_ANALYSIS_VERSION,
        primary_estimand=design.primary_estimand,
        primary_evidence_method=design.primary_evidence_method,
        semantic_cluster_sensitivity_method=design.semantic_cluster_sensitivity_method,
        model_based_diagnostic_method=design.model_based_diagnostic_method,
        primary_combination_rule=design.primary_combination_rule,
        design_digest=design.digest,
        outcome_digest=_canonical_digest([outcome.model_dump(mode="json") for outcome in ordered]),
        equal_task_panel_delta=equal_task_panel_delta,
        members=tuple(
            PanelMemberAnalysis(
                panel_member=member,
                equal_task_delta=equal_task_member_deltas[member],
                primary_positive_p=primary_positive[member],
                primary_lower_bound=primary_lower_bounds[member],
                semantic_cluster_sensitivity_positive_p=(
                    semantic_cluster_sensitivity_positive[member]
                ),
                semantic_cluster_sensitivity_lower_bound=(
                    semantic_cluster_sensitivity_lower_bounds[member]
                ),
                model_based_jackknife_standard_error=(model_based_jackknife[member].standard_error),
                model_based_jackknife_degrees_of_freedom=(
                    model_based_jackknife[member].degrees_of_freedom
                ),
                model_based_jackknife_positive_p=(model_based_jackknife[member].positive_p),
                model_based_jackknife_lower_bound=(model_based_jackknife[member].lower_bound),
                model_based_jackknife_bonferroni_lower_bound=(
                    model_based_jackknife_bonferroni_lower_bounds[member]
                ),
                secondary_noninferiority_p=secondary_noninferiority[member],
                secondary_holm_noninferiority_p=(secondary_holm_noninferiority[member]),
            )
            for member in design.panel_members
        ),
        model_based_label_swap_p=model_based_label_swap_p,
        equal_task_member_lifts_passed=equal_task_member_lifts_passed,
        member_primary_bounds_passed=member_primary_bounds_passed,
        member_semantic_cluster_sensitivity_bounds_positive=(
            member_semantic_cluster_sensitivity_bounds_positive
        ),
        member_model_based_jackknife_bounds_passed=(member_model_based_jackknife_bounds_passed),
        member_model_based_jackknife_bonferroni_bounds_passed=(
            member_model_based_jackknife_bonferroni_bounds_passed
        ),
        model_based_label_swap_passed=model_based_label_swap_passed,
        secondary_noninferiority_passed=secondary_noninferiority_passed,
    )


def _canonical_tasks(tasks: tuple[PairedTaskPlan, ...]) -> tuple[PairedTaskPlan, ...]:
    if not tasks:
        raise ValueError("paired tasks must contain at least one task")
    duplicates = sorted(
        task_id for task_id, count in Counter(task.task_id for task in tasks).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"paired tasks contain duplicate task_ids: {duplicates}")
    return tuple(sorted(tasks, key=lambda task: task.task_id))


def _semantic_group_observations(
    design: PairedEvaluationDesign,
    task_deltas: dict[str, float],
) -> tuple[tuple[float, ...], tuple[Fraction, ...]]:
    """Return group means and scales that preserve the equal-task target.

    With ``w_g = n_g/N`` and ``c_g = n_g/n_max = w_g/w_max``, the scaled weak
    null is exactly ``sum_g c_g*(E[X_g]-m) = (theta-m)/w_max <= 0``, where
    ``theta = sum_g w_g*E[X_g]`` is the equal-task fixed-roster estimand.
    """
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for task in design.tasks:
        grouped[task.group_id].append(task_deltas[task.task_id])
    largest_group = max(len(values) for values in grouped.values())
    return (
        tuple(fmean(grouped[group_id]) for group_id in design.group_ids),
        tuple(Fraction(len(grouped[group_id]), largest_group) for group_id in design.group_ids),
    )


def _group_sum_deltas(
    design: PairedEvaluationDesign,
    task_deltas: dict[str, float],
) -> tuple[float, ...]:
    """Aggregate task deltas into canonical semantic-group totals.

    Every task appears in exactly one frozen semantic group. Group totals are
    emitted in ``design.group_ids`` order so leave-one-group-out diagnostics
    can combine them with the matching frozen group sizes without relying on
    dictionary iteration order.

    Args:
        design: Frozen task roster and score-independent semantic groups.
        task_deltas: Complete task-level paired deltas keyed by task identity.

    Returns:
        One exact floating-point sum per canonical semantic group.
    """
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for task in design.tasks:
        grouped[task.group_id].append(task_deltas[task.task_id])
    return tuple(math.fsum(grouped[group_id]) for group_id in design.group_ids)


def _jackknife_student_t_inference(
    task_deltas: tuple[float, ...],
    *,
    design: PairedEvaluationDesign,
    alpha: float,
) -> _JackknifeStudentTInference:
    """Compute the model-based leave-one-semantic-group-out diagnostic.

    The point estimate remains the equal-task mean. Each deleted estimate
    removes one complete score-independent semantic group and renormalizes by
    the number of remaining tasks. The usual delete-one-group jackknife
    variance and Student-t reference distribution are diagnostic only; they
    do not enter the finite-sample primary decision.

    Args:
        task_deltas: Complete task-level deltas in the design's canonical order.
        design: Frozen task roster and semantic-group assignment.
        alpha: One-sided diagnostic tail probability.

    Returns:
        Jackknife standard error, degrees of freedom, p-value, and lower bound.

    Raises:
        ValueError: If deltas do not fill the roster or fewer than two nonempty
            deletion groups remain available.
    """
    if len(task_deltas) != len(design.tasks):
        raise ValueError("jackknife task deltas must match the frozen design")
    task_values = dict(zip(design.task_ids, task_deltas, strict=True))
    estimate = fmean(task_deltas)
    total = math.fsum(task_deltas)
    deleted_estimates: list[float] = []
    for group_id in design.group_ids:
        group_tasks = tuple(task.task_id for task in design.tasks if task.group_id == group_id)
        remaining_count = len(task_deltas) - len(group_tasks)
        if remaining_count <= 0:
            raise ValueError("jackknife needs at least two non-empty independence clusters")
        deleted_estimates.append(
            (total - math.fsum(task_values[task_id] for task_id in group_tasks)) / remaining_count
        )
    group_count = len(deleted_estimates)
    deleted_mean = fmean(deleted_estimates)
    variance = (
        (group_count - 1)
        / group_count
        * math.fsum((value - deleted_mean) ** 2 for value in deleted_estimates)
    )
    standard_error = math.sqrt(max(0.0, variance))
    degrees_of_freedom = group_count - 1
    if standard_error == 0.0:
        positive_p = 0.0 if estimate > 0.0 else 1.0
    else:
        positive_p = float(student_t.sf(estimate / standard_error, degrees_of_freedom))
    return _JackknifeStudentTInference(
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        positive_p=positive_p,
        lower_bound=_jackknife_student_t_lower_bound(
            estimate,
            standard_error=standard_error,
            degrees_of_freedom=degrees_of_freedom,
            alpha=alpha,
        ),
    )


def _jackknife_student_t_lower_bound(
    estimate: float,
    *,
    standard_error: float,
    degrees_of_freedom: int,
    alpha: float,
) -> float:
    """Return a clipped one-sided Student-t lower endpoint.

    A zero standard error yields the point estimate exactly. Otherwise the
    endpoint subtracts the one-sided Student-t critical value times the
    jackknife standard error and clips the result to the reward-delta range.

    Args:
        estimate: Equal-task point estimate in the reward-delta range.
        standard_error: Nonnegative jackknife standard error.
        degrees_of_freedom: Positive number of semantic groups minus one.
        alpha: One-sided tail probability strictly between zero and one.

    Returns:
        Lower endpoint clipped to ``[-1, 1]``.

    Raises:
        ValueError: If alpha, standard error, or degrees of freedom are invalid.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("jackknife Student-t alpha must be between zero and one")
    if standard_error < 0.0 or not math.isfinite(standard_error):
        raise ValueError("jackknife standard error must be finite and non-negative")
    if degrees_of_freedom < 1:
        raise ValueError("jackknife Student-t needs at least one degree of freedom")
    critical_value = float(student_t.ppf(1.0 - alpha, degrees_of_freedom))
    return min(1.0, max(-1.0, estimate - critical_value * standard_error))


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


def _bounded_mean_observation_scales(
    observations: tuple[float, ...],
    scales: tuple[Fraction, ...] | None,
) -> tuple[Fraction, ...]:
    """Validate exact preregistered scales for a bounded-mean e-value."""
    canonical = scales if scales is not None else (Fraction(1),) * len(observations)
    if len(canonical) != len(observations):
        raise ValueError("bounded-mean observation scales must match the observations")
    if any(scale <= 0 or scale > 1 for scale in canonical):
        raise ValueError("bounded-mean observation scales must be in (0, 1]")
    return canonical


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
    observation_scales: tuple[Fraction, ...] | None = None,
) -> float:
    """Test a scaled average expectation of independent deltas against a weak null.

    Each fixed-horizon input is an independent, potentially nonidentically
    distributed observation in [-1, 1]. With exact scales ``c_i`` in ``(0, 1]``,
    a frozen bet uses ``1 + f*c_i*(X_i-m)/(1+m)``. Under the weak null
    ``sum(c_i*E[X_i-m]) <= 0``, the arithmetic mean of these nonnegative expected
    factors is at most one. Independence and AM-GM therefore bound their product
    by one. A frozen convex mixture preserves the e-value property. Thus
    ``min(1, 1 / e_value)`` is a finite-sample p-value without symmetry or
    identical-distribution assumptions. Unit scales target the equal-observation
    mean. Adaptive horizons, score-dependent admission, or post-outcome scales and
    bets are outside this guarantee.
    """
    log_e_value = _bounded_mean_log_e(
        task_deltas,
        null_mean=null_mean,
        bets=bets,
        observation_scales=observation_scales,
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
        observation_scales=observation_scales,
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
    observation_scales: tuple[Fraction, ...] | None = None,
) -> BoundedMeanInterval:
    """Invert two one-sided bounded-mean e-tests with Bonferroni coverage."""
    tail_alpha = _divide_float_downward(alpha, 2)
    lower = _bounded_mean_lower_bound(
        task_deltas,
        alpha=tail_alpha,
        bets=bets,
        observation_scales=observation_scales,
    )
    upper = -_bounded_mean_lower_bound(
        tuple(-delta for delta in task_deltas),
        alpha=tail_alpha,
        bets=bets,
        observation_scales=observation_scales,
    )
    return BoundedMeanInterval(lower=lower, upper=upper)


def _bounded_mean_lower_bound(
    task_deltas: tuple[float, ...],
    *,
    alpha: float,
    bets: tuple[BoundedMeanBet, ...],
    observation_scales: tuple[Fraction, ...] | None = None,
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
        observation_scales=observation_scales,
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
                observation_scales=observation_scales,
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
        observation_scales=observation_scales,
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
            observation_scales=observation_scales,
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
    observation_scales: tuple[Fraction, ...] | None = None,
) -> float:
    if not task_deltas:
        raise ValueError("bounded-mean evidence needs at least one task delta")
    if any(not math.isfinite(delta) or not -1.0 <= delta <= 1.0 for delta in task_deltas):
        raise ValueError("bounded-mean task deltas must be finite and in [-1, 1]")
    if not math.isfinite(null_mean) or not -1.0 <= null_mean <= 1.0:
        raise ValueError("bounded-mean null must be finite and in [-1, 1]")
    scales = _bounded_mean_observation_scales(task_deltas, observation_scales)
    if null_mean == -1.0:
        return 0.0 if all(delta == -1.0 for delta in task_deltas) else math.inf
    canonical_bets = _canonical_bounded_mean_bets(bets)
    component_logs: list[float] = []
    denominator = 1.0 + null_mean
    for bet in canonical_bets:
        component_log = math.log(bet.weight)
        for delta, scale in zip(task_deltas, scales, strict=True):
            scaled_fraction = bet.fraction * float(scale)
            factor = (1.0 - scaled_fraction) + scaled_fraction * (1.0 + delta) / denominator
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
    observation_scales: tuple[Fraction, ...] | None = None,
) -> Fraction:
    """Return the exact e-value over the binary-float inputs for safe rounding."""
    if null_mean == -1.0:
        raise ValueError("the negative-one support null is handled before exact e-value evaluation")
    one = Fraction(1)
    null_fraction = Fraction.from_float(null_mean)
    denominator = one + null_fraction
    scales = _bounded_mean_observation_scales(task_deltas, observation_scales)
    e_value = Fraction()
    for bet in _canonical_bounded_mean_bets(bets):
        fraction = Fraction.from_float(bet.fraction)
        component = Fraction.from_float(bet.weight)
        for delta, scale in zip(task_deltas, scales, strict=True):
            delta_fraction = Fraction.from_float(delta)
            factor = one + fraction * scale * (delta_fraction - null_fraction) / denominator
            component *= factor
        e_value += component
    return e_value


def _bounded_mean_exact_rejects(
    task_deltas: tuple[float, ...],
    *,
    null_mean: float,
    alpha: float,
    bets: tuple[BoundedMeanBet, ...],
    observation_scales: tuple[Fraction, ...] | None = None,
) -> bool:
    e_value = _bounded_mean_exact_e(
        task_deltas,
        null_mean=null_mean,
        bets=bets,
        observation_scales=observation_scales,
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
