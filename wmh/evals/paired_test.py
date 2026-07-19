"""Tests for deterministic paired benchmark design and fixed-roster analysis."""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction

import pytest
from pydantic import ValidationError

import wmh.evals.paired as paired_analysis
from wmh.evals.paired import (
    BoundedMeanBet,
    PairedArm,
    PairedBlockOutcome,
    PairedEvaluationDesign,
    PairedPanelPlan,
    PairedTaskPlan,
    _bounded_mean_interval,
    _bounded_mean_log_e,
    _bounded_mean_lower_bound,
    _bounded_mean_p,
    _divide_float_downward,
    _holm_adjust,
    analyze_paired_outcomes,
    paired_primary_decision_passed,
)


def _design(
    *,
    task_ids: tuple[str, ...] = ("task-a", "task-b", "task-c", "task-d"),
    panel_members: tuple[str, ...] = ("small", "medium", "large"),
    attempts: int = 2,
    attempts_by_member: dict[str, int] | None = None,
    primary_e_value_bets: tuple[BoundedMeanBet, ...] | None = None,
    noninferiority_margin: float = 0.02,
    grouped_tasks: tuple[PairedTaskPlan, ...] | None = None,
) -> PairedEvaluationDesign:
    attempt_counts = attempts_by_member or {
        panel_member: attempts for panel_member in panel_members
    }
    tasks = grouped_tasks or tuple(
        PairedTaskPlan(task_id=task_id, group_id=task_id) for task_id in task_ids
    )
    return PairedEvaluationDesign.create(
        tasks=tasks,
        panel=tuple(
            PairedPanelPlan(
                panel_member=panel_member,
                attempts=attempt_counts[panel_member],
            )
            for panel_member in panel_members
        ),
        primary_e_value_bets=primary_e_value_bets or (BoundedMeanBet(fraction=1.0, weight=1.0),),
        schedule_seed="schedule-v1",
        analysis_seed="analysis-v1",
        randomization_samples=2_000,
        minimum_equal_task_member_delta=0.03,
        noninferiority_margin=noninferiority_margin,
    )


def _outcomes(
    design: PairedEvaluationDesign,
    *,
    baseline: float,
    candidate: float,
) -> list[PairedBlockOutcome]:
    return [
        PairedBlockOutcome(
            block=block,
            baseline_reward=baseline,
            candidate_reward=candidate,
        )
        for block in design.blocks
    ]


def test_schedule_is_deterministic_complete_and_balanced_within_tasks() -> None:
    task_ids = tuple(f"task-{index}" for index in range(5))
    design = _design(task_ids=task_ids, attempts=3)
    repeated = _design(task_ids=task_ids, attempts=3)
    reordered = _design(
        task_ids=tuple(reversed(task_ids)),
        panel_members=("large", "medium", "small"),
        attempts=3,
    )

    assert design == repeated
    assert design == reordered
    assert design.digest == repeated.digest
    assert design.task_ids == tuple(sorted(task_ids))
    assert len(design.blocks) == 5 * 3 * 3
    for member in design.panel_members:
        first_arm_by_task: dict[str, Counter[PairedArm]] = {}
        for block in design.blocks:
            if block.panel_member == member:
                first_arm_by_task.setdefault(block.task_id, Counter())[block.first_arm] += 1
        assert all(
            abs(counts[PairedArm.BASELINE] - counts[PairedArm.CANDIDATE]) == 1
            for counts in first_arm_by_task.values()
        )
        total = sum(first_arm_by_task.values(), Counter())
        assert abs(total[PairedArm.BASELINE] - total[PairedArm.CANDIDATE]) == 1


def test_primary_bounded_mean_rejects_zero_se_five_point_counterexample() -> None:
    # Under the independent mean-zero null X_g=.05 with probability 1/1.05
    # and X_g=-1 otherwise. All 50 observations equal .05 with probability
    # (1/1.05)^50, so a zero jackknife SE cannot make that event significant.
    task_ids = tuple(f"task-{index:02d}" for index in range(50))
    design = _design(task_ids=task_ids, attempts=20)
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(block.attempt == 1),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert report.equal_task_member_deltas == {
        "large": pytest.approx(0.05),
        "medium": pytest.approx(0.05),
        "small": pytest.approx(0.05),
    }
    assert report.semantic_cluster_sensitivity_positive_p == report.primary_positive_p
    null_all_positive_probability = 1 / 1.05**50
    assert null_all_positive_probability > design.alpha
    assert all(
        value == pytest.approx(null_all_positive_probability)
        for value in report.primary_positive_p.values()
    )
    assert all(value <= 0.0 for value in report.primary_lower_bounds.values())
    assert all(value > 0.0 for value in report.model_based_jackknife_lower_bounds.values())
    assert all(
        value > 0.0 for value in report.model_based_jackknife_bonferroni_lower_bounds.values()
    )
    assert report.equal_task_member_lifts_passed is True
    assert report.member_primary_bounds_passed is False
    assert report.member_model_based_jackknife_bounds_passed is True
    assert report.member_model_based_jackknife_bonferroni_bounds_passed is True
    assert report.passed is False


def test_primary_bounded_mean_passes_strong_fixed_roster_effect() -> None:
    task_ids = tuple(f"task-{index:02d}" for index in range(50))
    design = _design(task_ids=task_ids, attempts=20)
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(block.attempt <= 2),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert all(value == pytest.approx(1 / 1.1**50) for value in report.primary_positive_p.values())
    assert all(value > 0.0 for value in report.primary_lower_bounds.values())
    assert report.equal_task_member_lifts_passed is True
    assert report.member_primary_bounds_passed is True
    assert report.passed is True

    compact = paired_primary_decision_passed(
        design,
        tuple(tuple(0.1 for _task in design.task_ids) for _member in design.panel_members),
    )
    assert compact is report.passed


def test_primary_decision_does_not_depend_on_rounded_bound_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _design(
        task_ids=("task-a", "task-b", "task-c", "task-d", "task-e"),
        attempts=1,
    )
    task_deltas = tuple(1.0 for _task in design.task_ids)
    assert (
        paired_primary_decision_passed(
            design,
            tuple(task_deltas for _member in design.panel_members),
        )
        is True
    )
    monkeypatch.setattr(
        paired_analysis,
        "_bounded_mean_lower_bound",
        lambda *_args, **_kwargs: 0.0,
    )

    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=1.0,
        )
        for block in design.blocks
    ]
    report = analyze_paired_outcomes(design, outcomes)

    assert all(value <= 0.0 for value in report.primary_lower_bounds.values())
    assert report.member_primary_bounds_passed is True
    assert report.passed is True


def test_primary_iut_uses_unadjusted_alpha_for_every_lane() -> None:
    task_ids = tuple(f"task-{index:02d}" for index in range(59))
    design = _design(task_ids=task_ids, attempts=1)
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(int(block.task_id.removeprefix("task-")) < 5),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert all(value == pytest.approx(1 / 32) for value in report.primary_positive_p.values())
    assert all(value > 0.0 for value in report.primary_lower_bounds.values())
    assert report.equal_task_member_lifts_passed is True
    assert report.member_primary_bounds_passed is True
    assert report.passed is True


def test_semantic_groups_affect_only_the_conservative_sensitivity() -> None:
    task_ids = tuple(f"task-{index:02d}" for index in range(59))
    grouped_tasks = tuple(
        PairedTaskPlan(
            task_id=task_id,
            group_id="shared-family" if index < 6 else task_id,
        )
        for index, task_id in enumerate(task_ids)
    )
    singleton = _design(task_ids=task_ids, attempts=1)
    grouped = _design(task_ids=task_ids, attempts=1, grouped_tasks=grouped_tasks)

    def outcomes(design: PairedEvaluationDesign) -> list[PairedBlockOutcome]:
        return [
            PairedBlockOutcome(
                block=block,
                baseline_reward=0.0,
                candidate_reward=float(int(block.task_id.removeprefix("task-")) < 6),
            )
            for block in design.blocks
        ]

    singleton_report = analyze_paired_outcomes(singleton, outcomes(singleton))
    grouped_report = analyze_paired_outcomes(grouped, outcomes(grouped))

    assert singleton_report.passed is True
    assert grouped_report.passed is True
    assert singleton_report.primary_lower_bounds == grouped_report.primary_lower_bounds
    assert singleton_report.member_semantic_cluster_sensitivity_bounds_positive is True
    assert grouped_report.member_semantic_cluster_sensitivity_bounds_positive is False
    assert all(
        member.model_based_jackknife_degrees_of_freedom == 53 for member in grouped_report.members
    )
    assert all(
        member.semantic_cluster_sensitivity_lower_bound < 0.0 for member in grouped_report.members
    )


def test_unequal_semantic_clusters_preserve_the_equal_task_estimand() -> None:
    # A naive equal-cluster product would be .8 * 2**10 and would reject even
    # though the observed equal-task effect is negative. The registered weights
    # instead scale every singleton cluster by (1/100)/(90/100) = 1/90.
    task_ids = tuple(f"task-{index:03d}" for index in range(100))
    tasks = tuple(
        PairedTaskPlan(
            task_id=task_id,
            group_id="large-family" if index < 90 else task_id,
        )
        for index, task_id in enumerate(task_ids)
    )
    design = _design(
        task_ids=task_ids,
        panel_members=("member",),
        attempts=5,
        grouped_tasks=tasks,
    )
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=float(
                int(block.task_id.removeprefix("task-")) < 90 and block.attempt == 1
            ),
            candidate_reward=float(int(block.task_id.removeprefix("task-")) >= 90),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)
    sensitivity_deltas = (-0.2,) + (1.0,) * 10
    sensitivity_scales = (Fraction(1),) + (Fraction(1, 90),) * 10
    equal_task_group_weights = (Fraction(90, 100),) + (Fraction(1, 100),) * 10
    expected_e_value = 0.8 * (91 / 90) ** 10

    assert report.equal_task_member_deltas == {"member": pytest.approx(-0.08)}
    assert float(
        sum(
            weight * Fraction.from_float(delta)
            for weight, delta in zip(equal_task_group_weights, sensitivity_deltas, strict=True)
        )
    ) == pytest.approx(report.equal_task_member_deltas["member"], abs=1e-15)
    assert 0.8 * 2**10 > 1 / design.alpha
    assert math.exp(
        _bounded_mean_log_e(
            sensitivity_deltas,
            null_mean=0.0,
            bets=design.primary_e_value_bets,
            observation_scales=sensitivity_scales,
        )
    ) == pytest.approx(expected_e_value)
    assert expected_e_value < 1.0
    assert report.semantic_cluster_sensitivity_positive_p == {"member": 1.0}
    assert report.semantic_cluster_sensitivity_lower_bounds["member"] < 0.0
    assert report.member_semantic_cluster_sensitivity_bounds_positive is False
    assert report.passed is False


def test_design_binds_semantic_clusters_and_requires_two_sensitivity_units() -> None:
    task_ids = ("task-a", "task-b", "task-c")
    singleton = _design(task_ids=task_ids)
    grouped = tuple(PairedTaskPlan(task_id=task_id, group_id="one-family") for task_id in task_ids)

    with pytest.raises(ValidationError, match="at least two semantic sensitivity clusters"):
        _design(task_ids=task_ids, grouped_tasks=grouped)

    regrouped = _design(
        task_ids=task_ids,
        grouped_tasks=(
            PairedTaskPlan(task_id="task-a", group_id="shared-family"),
            PairedTaskPlan(task_id="task-b", group_id="shared-family"),
            PairedTaskPlan(task_id="task-c", group_id="task-c"),
        ),
    )
    assert regrouped.task_ids == singleton.task_ids
    assert regrouped.digest != singleton.digest


def test_schedule_exactly_balances_even_attempt_counts_within_each_task() -> None:
    design = _design(
        task_ids=tuple(f"task-{index}" for index in range(7)),
        panel_members=("economy", "standard"),
        attempts_by_member={"economy": 20, "standard": 10},
    )

    counts: dict[tuple[str, str], Counter[PairedArm]] = {}
    for block in design.blocks:
        counts.setdefault((block.task_id, block.panel_member), Counter())[block.first_arm] += 1

    assert all(value[PairedArm.BASELINE] == value[PairedArm.CANDIDATE] for value in counts.values())


def test_schedule_balances_odd_attempt_extra_direction_across_tasks() -> None:
    design = _design(
        task_ids=tuple(f"task-{index}" for index in range(8)),
        panel_members=("premium",),
        attempts_by_member={"premium": 5},
    )

    per_task: dict[str, Counter[PairedArm]] = {}
    for block in design.blocks:
        per_task.setdefault(block.task_id, Counter())[block.first_arm] += 1

    assert Counter(
        PairedArm.CANDIDATE
        if counts[PairedArm.CANDIDATE] > counts[PairedArm.BASELINE]
        else PairedArm.BASELINE
        for counts in per_task.values()
    ) == {PairedArm.BASELINE: 4, PairedArm.CANDIDATE: 4}


def test_schedule_supports_predeclared_member_specific_attempt_counts() -> None:
    design = _design(
        task_ids=tuple(f"task-{index}" for index in range(9)),
        panel_members=("economy", "standard", "premium"),
        attempts_by_member={"economy": 20, "standard": 10, "premium": 5},
    )

    counts = Counter(block.panel_member for block in design.blocks)
    assert counts == {"economy": 180, "standard": 90, "premium": 45}
    assert design.attempts_by_member == {"economy": 20, "premium": 5, "standard": 10}

    per_task_member: dict[tuple[str, str], Counter[PairedArm]] = {}
    for block in design.blocks:
        per_task_member.setdefault((block.task_id, block.panel_member), Counter())[
            block.first_arm
        ] += 1
    for plan in design.panel:
        member_counts = {
            task: per_task_member[(task, plan.panel_member)] for task in design.task_ids
        }
        assert all(sum(value.values()) == plan.attempts for value in member_counts.values())
        assert all(
            abs(value[PairedArm.BASELINE] - value[PairedArm.CANDIDATE]) == plan.attempts % 2
            for value in member_counts.values()
        )
        total = sum(member_counts.values(), Counter())
        assert abs(total[PairedArm.BASELINE] - total[PairedArm.CANDIDATE]) <= 1


def test_design_rejects_a_schedule_that_differs_from_its_frozen_seed() -> None:
    design = _design()
    changed = list(design.blocks)
    changed[0] = changed[0].model_copy(
        update={
            "first_arm": (
                PairedArm.CANDIDATE
                if changed[0].first_arm is PairedArm.BASELINE
                else PairedArm.BASELINE
            )
        }
    )

    with pytest.raises(ValidationError, match="schedule_seed"):
        PairedEvaluationDesign.model_validate(
            {**design.model_dump(mode="json"), "blocks": [block.model_dump() for block in changed]}
        )


def test_primary_e_value_bets_are_frozen_canonical_and_part_of_the_digest() -> None:
    first = _design(
        primary_e_value_bets=(
            BoundedMeanBet(fraction=1.0, weight=0.4),
            BoundedMeanBet(fraction=0.25, weight=0.6),
        )
    )
    reordered = _design(
        primary_e_value_bets=(
            BoundedMeanBet(fraction=0.25, weight=0.6),
            BoundedMeanBet(fraction=1.0, weight=0.4),
        )
    )

    assert first == reordered
    assert first.digest == reordered.digest
    assert tuple(bet.fraction for bet in first.primary_e_value_bets) == (0.25, 1.0)
    changed = _design(
        primary_e_value_bets=(
            BoundedMeanBet(fraction=0.25, weight=0.4),
            BoundedMeanBet(fraction=1.0, weight=0.6),
        )
    )
    assert changed.digest != first.digest


def test_calibrated_binary_fraction_bet_mixture_is_bound_exactly() -> None:
    design = _design(
        task_ids=tuple(f"task-{index:02d}" for index in range(59)),
        attempts=20,
        primary_e_value_bets=(
            BoundedMeanBet(fraction=0.25, weight=1 / 16),
            BoundedMeanBet(fraction=0.5, weight=1 / 16),
            BoundedMeanBet(fraction=1.0, weight=7 / 8),
        ),
    )

    assert tuple(
        (Fraction.from_float(bet.fraction), Fraction.from_float(bet.weight))
        for bet in design.primary_e_value_bets
    ) == (
        (Fraction(1, 4), Fraction(1, 16)),
        (Fraction(1, 2), Fraction(1, 16)),
        (Fraction(1), Fraction(7, 8)),
    )
    assert design.attempts_by_member == {"large": 20, "medium": 20, "small": 20}
    assert design.minimum_equal_task_member_delta == 0.03


def test_primary_e_value_bets_reject_duplicates_and_unnormalized_weights() -> None:
    with pytest.raises(ValueError, match="duplicate fractions"):
        _design(
            primary_e_value_bets=(
                BoundedMeanBet(fraction=0.5, weight=0.5),
                BoundedMeanBet(fraction=0.5, weight=0.5),
            )
        )
    with pytest.raises(ValueError, match="sum to one"):
        _design(
            primary_e_value_bets=(
                BoundedMeanBet(fraction=0.25, weight=0.4),
                BoundedMeanBet(fraction=1.0, weight=0.4),
            )
        )
    with pytest.raises(ValueError, match="sum to one"):
        _design(
            primary_e_value_bets=(
                BoundedMeanBet(fraction=0.1, weight=0.07),
                BoundedMeanBet(fraction=0.5, weight=0.14),
                BoundedMeanBet(fraction=1.0, weight=0.79),
            )
        )
    with pytest.raises(ValidationError, match="cannot be boolean"):
        BoundedMeanBet(fraction=True, weight=1.0)


def test_design_rejects_alpha_that_underflows_its_adjusted_tests() -> None:
    design = _design()

    with pytest.raises(ValidationError, match="alpha is too small"):
        PairedEvaluationDesign.model_validate(
            {**design.model_dump(mode="json"), "alpha": math.ulp(0.0)}
        )


def test_analysis_rejects_missing_duplicate_and_nonbinary_cells() -> None:
    design = _design()
    outcomes = _outcomes(design, baseline=0.0, candidate=1.0)

    with pytest.raises(ValueError, match="missing"):
        analyze_paired_outcomes(design, outcomes[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_paired_outcomes(design, [*outcomes, outcomes[0]])
    with pytest.raises(ValidationError, match="binary"):
        PairedBlockOutcome(
            block=design.blocks[0],
            baseline_reward=0.5,
            candidate_reward=1.0,
        )


def test_uniform_lift_passes_every_frozen_criterion() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(12)))
    report = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=1.0),
    )

    assert report.equal_task_panel_delta == 1.0
    assert report.analysis_version == "5"
    assert (
        report.primary_estimand
        == "fixed-roster-equal-task-conditional-expected-paired-reward-delta"
    )
    assert (
        report.primary_evidence_method
        == "fixed-horizon-independent-task-bounded-mean-e-value-inverted-lower-bound"
    )
    assert (
        report.semantic_cluster_sensitivity_method
        == "weighted-semantic-cluster-bounded-mean-e-value-inverted-lower-bound"
    )
    assert (
        report.model_based_diagnostic_method == "leave-one-semantic-cluster-out-jackknife-student-t"
    )
    assert report.primary_combination_rule == "intersection-union-all-lanes"
    assert report.design_digest == design.digest
    assert report.outcome_digest.startswith("sha256:")
    assert report.equal_task_member_deltas == {
        "large": 1.0,
        "medium": 1.0,
        "small": 1.0,
    }
    assert report.semantic_cluster_sensitivity_positive_p == report.primary_positive_p
    assert report.primary_positive_p == {
        "large": pytest.approx(2.0**-12),
        "medium": pytest.approx(2.0**-12),
        "small": pytest.approx(2.0**-12),
    }
    assert all(value > 0.0 for value in report.primary_lower_bounds.values())
    assert report.model_based_label_swap_p < 0.05
    assert all(value > 0.0 for value in report.semantic_cluster_sensitivity_lower_bounds.values())
    assert all(value < 0.05 for value in report.secondary_holm_noninferiority_p.values())
    assert report.equal_task_member_lifts_passed is True
    assert report.member_primary_bounds_passed is True
    assert report.member_semantic_cluster_sensitivity_bounds_positive is True
    assert report.member_model_based_jackknife_bounds_passed is True
    assert report.member_model_based_jackknife_bonferroni_bounds_passed is True
    assert all(value > 0.0 for value in report.model_based_jackknife_lower_bounds.values())
    assert all(
        value > 0.0 for value in report.model_based_jackknife_bonferroni_lower_bounds.values()
    )
    assert report.model_based_label_swap_passed is True
    assert report.secondary_noninferiority_passed is True
    assert report.passed is True


def test_passed_uses_only_member_floors_and_primary_iut_bounds() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(12)))
    report = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=1.0),
    )

    diagnostics_failed = report.model_copy(
        update={
            "member_semantic_cluster_sensitivity_bounds_positive": False,
            "member_model_based_jackknife_bounds_passed": False,
            "member_model_based_jackknife_bonferroni_bounds_passed": False,
            "model_based_label_swap_passed": False,
            "secondary_noninferiority_passed": False,
        }
    )

    assert diagnostics_failed.equal_task_member_lifts_passed is True
    assert diagnostics_failed.member_primary_bounds_passed is True
    assert diagnostics_failed.passed is True
    assert (
        diagnostics_failed.model_copy(update={"equal_task_member_lifts_passed": False}).passed
        is False
    )
    assert (
        diagnostics_failed.model_copy(update={"member_primary_bounds_passed": False}).passed
        is False
    )


def test_panel_average_cannot_hide_a_member_below_the_lift_floor() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(12)), attempts=1)
    outcomes: list[PairedBlockOutcome] = []
    for block in design.blocks:
        candidate = 0.0 if block.panel_member == "small" else 1.0
        outcomes.append(
            PairedBlockOutcome(
                block=block,
                baseline_reward=0.0,
                candidate_reward=candidate,
            )
        )

    report = analyze_paired_outcomes(design, outcomes)

    assert report.equal_task_panel_delta > design.minimum_equal_task_member_delta
    assert report.equal_task_member_deltas["small"] == 0.0
    assert report.equal_task_member_lifts_passed is False
    assert report.member_primary_bounds_passed is False
    assert report.member_model_based_jackknife_bounds_passed is False
    assert report.member_model_based_jackknife_bonferroni_bounds_passed is False
    assert report.passed is False


def test_repeated_attempts_are_averaged_before_fixed_roster_inference() -> None:
    design = _design(
        task_ids=tuple(f"task-{index:02d}" for index in range(59)),
        attempts=25,
    )
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(block.attempt == 1),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert report.equal_task_panel_delta == pytest.approx(0.04)
    assert report.equal_task_member_lifts_passed is True
    assert report.member_primary_bounds_passed is False
    assert report.member_semantic_cluster_sensitivity_bounds_positive is False
    assert report.member_model_based_jackknife_bounds_passed is True
    assert report.member_model_based_jackknife_bonferroni_bounds_passed is True
    assert all(member.model_based_jackknife_degrees_of_freedom == 58 for member in report.members)
    assert report.passed is False


def test_no_lift_fails_every_positive_evidence_gate() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(12)))
    report = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=0.0),
    )

    assert report.equal_task_panel_delta == 0.0
    assert report.equal_task_member_lifts_passed is False
    assert report.member_primary_bounds_passed is False
    assert report.member_model_based_jackknife_bounds_passed is False
    assert report.member_model_based_jackknife_bonferroni_bounds_passed is False
    assert report.model_based_label_swap_passed is False
    assert report.member_semantic_cluster_sensitivity_bounds_positive is False
    assert report.passed is False


def test_exact_task_cluster_randomization_flips_all_panel_members_together() -> None:
    design = _design(
        task_ids=("task-a", "task-b"),
        panel_members=("small", "large"),
        attempts=1,
    )
    rewards = {
        ("task-a", "small"): (0.0, 1.0),
        ("task-a", "large"): (0.0, 1.0),
        ("task-b", "small"): (1.0, 0.0),
        ("task-b", "large"): (1.0, 0.0),
    }
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=rewards[(block.task_id, block.panel_member)][0],
            candidate_reward=rewards[(block.task_id, block.panel_member)][1],
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert report.equal_task_panel_delta == 0.0
    assert report.model_based_label_swap_p == 0.75


def test_bounded_mean_gate_rejects_the_sign_flip_rare_tail_counterexample() -> None:
    task_ids = tuple(f"task-{index:02d}" for index in range(59))
    design = _design(task_ids=task_ids, panel_members=("member",), attempts=49)
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(block.attempt == 1),
        )
        for block in design.blocks
    ]

    report = analyze_paired_outcomes(design, outcomes)

    assert report.equal_task_panel_delta == pytest.approx(1 / 49)
    assert report.model_based_label_swap_passed is True
    assert report.primary_positive_p["member"] == pytest.approx((49 / 50) ** 59)
    assert report.primary_lower_bounds["member"] <= 0.0
    assert report.passed is False


def test_noninferiority_uses_the_weak_mean_null_without_sign_flipping_margin() -> None:
    design = _design(task_ids=tuple(f"task-{index:02d}" for index in range(59)))
    report = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=0.0),
    )
    expected_raw = 0.98**59

    assert report.secondary_noninferiority_p == {
        "large": pytest.approx(expected_raw),
        "medium": pytest.approx(expected_raw),
        "small": pytest.approx(expected_raw),
    }
    assert report.secondary_holm_noninferiority_p == {
        "large": pytest.approx(3 * expected_raw),
        "medium": pytest.approx(3 * expected_raw),
        "small": pytest.approx(3 * expected_raw),
    }
    assert report.secondary_noninferiority_passed is False


def test_bounded_mean_evidence_handles_boundary_losses_and_log_space() -> None:
    losing = _design(task_ids=tuple(f"task-{index:03d}" for index in range(200)))
    losing_report = analyze_paired_outcomes(
        losing,
        _outcomes(losing, baseline=1.0, candidate=0.0),
    )
    assert all(value == 1.0 for value in losing_report.primary_positive_p.values())
    assert all(value == -1.0 for value in losing_report.primary_lower_bounds.values())

    winning_report = analyze_paired_outcomes(
        losing,
        _outcomes(losing, baseline=0.0, candidate=1.0),
    )
    assert all(
        value == pytest.approx(2.0**-200) for value in winning_report.primary_positive_p.values()
    )
    assert all(value > 0.0 for value in winning_report.primary_lower_bounds.values())


def test_bounded_mean_interval_rounds_outward_at_the_analytic_boundary() -> None:
    task_count = 12
    alpha = 0.05
    bets = (BoundedMeanBet(fraction=1.0, weight=1.0),)
    interval = _bounded_mean_interval((1.0,) * task_count, alpha=alpha, bets=bets)
    analytic_lower = 2.0 * (alpha / 2.0) ** (1.0 / task_count) - 1.0
    reflected = _bounded_mean_interval((-1.0,) * task_count, alpha=alpha, bets=bets)

    assert interval.lower <= analytic_lower
    assert analytic_lower - interval.lower < 1e-14
    assert reflected.upper >= -analytic_lower
    assert reflected.upper + analytic_lower < 1e-14


def test_noninferiority_supports_the_negative_one_boundary_null() -> None:
    design = _design(noninferiority_margin=1.0)
    all_losses = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=1.0, candidate=0.0),
    )
    ties = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=0.0),
    )

    assert all(value == 1.0 for value in all_losses.secondary_noninferiority_p.values())
    assert all_losses.secondary_noninferiority_passed is False
    assert all(value == 0.0 for value in ties.secondary_noninferiority_p.values())
    assert ties.secondary_noninferiority_passed is True


def test_bounded_mean_evidence_stays_finite_for_large_preregistered_mixtures() -> None:
    bets = (
        BoundedMeanBet(fraction=0.1, weight=0.2),
        BoundedMeanBet(fraction=0.5, weight=0.3),
        BoundedMeanBet(fraction=1.0, weight=0.5),
    )

    assert math.isfinite(_bounded_mean_log_e((1.0,) * 10_000, null_mean=0.0, bets=bets))
    assert _bounded_mean_log_e((-1.0,) * 10_000, null_mean=0.0, bets=bets) < 0.0
    single_bet = (BoundedMeanBet(fraction=1.0, weight=1.0),)
    assert _bounded_mean_p((1.0,) * 2_000, null_mean=0.0, bets=single_bet) == math.ulp(0.0)


def test_bounded_mean_evidence_rejects_invalid_preregistered_scales() -> None:
    bets = (BoundedMeanBet(fraction=1.0, weight=1.0),)

    with pytest.raises(ValueError, match="must match"):
        _bounded_mean_p(
            (0.0, 0.0),
            null_mean=-1.0,
            bets=bets,
            observation_scales=(Fraction(1),),
        )
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _bounded_mean_p(
            (0.0,),
            null_mean=0.0,
            bets=bets,
            observation_scales=(Fraction(0),),
        )


@pytest.mark.parametrize(
    ("null_mean", "low", "high"),
    (
        (0.0, -1.0, 1 / 49),
        (0.0, -1.0, 1.0),
        (0.0, -0.25, 0.75),
        (0.0, -0.1, 0.4),
        (-0.02, -1.0, 0.03),
    ),
)
def test_bounded_mean_exact_two_point_null_calibration_grid(
    null_mean: float,
    low: float,
    high: float,
) -> None:
    task_count = 59
    high_probability = (null_mean - low) / (high - low)
    bets = (
        BoundedMeanBet(fraction=0.1, weight=0.2),
        BoundedMeanBet(fraction=0.5, weight=0.3),
        BoundedMeanBet(fraction=1.0, weight=0.5),
    )
    rejection_probability = 0.0
    noncoverage_probability = 0.0
    for high_count in range(task_count + 1):
        probability = (
            math.comb(task_count, high_count)
            * high_probability**high_count
            * (1.0 - high_probability) ** (task_count - high_count)
        )
        deltas = (high,) * high_count + (low,) * (task_count - high_count)
        if _bounded_mean_p(deltas, null_mean=null_mean, bets=bets) < 0.05:
            rejection_probability += probability
        interval = _bounded_mean_interval(deltas, alpha=0.05, bets=bets)
        if not interval.lower <= null_mean <= interval.upper:
            noncoverage_probability += probability

    assert rejection_probability <= 0.05 + 1e-12
    assert noncoverage_probability <= 0.05 + 1e-12


def test_bounded_mean_nonidentical_null_calibration() -> None:
    high_probabilities = (0.99, 0.99, 0.01, 0.01, 0.8, 0.2, 0.7, 0.3)
    assert sum(2.0 * probability - 1.0 for probability in high_probabilities) == pytest.approx(0.0)
    bets = (
        BoundedMeanBet(fraction=0.1, weight=0.2),
        BoundedMeanBet(fraction=0.5, weight=0.3),
        BoundedMeanBet(fraction=1.0, weight=0.5),
    )
    rejection_probability = 0.0
    noncoverage_probability = 0.0
    for mask in range(1 << len(high_probabilities)):
        deltas: list[float] = []
        probability = 1.0
        for index, high_probability in enumerate(high_probabilities):
            high = bool(mask & (1 << index))
            deltas.append(1.0 if high else -1.0)
            probability *= high_probability if high else 1.0 - high_probability
        frozen_deltas = tuple(deltas)
        if _bounded_mean_p(frozen_deltas, null_mean=0.0, bets=bets) < 0.05:
            rejection_probability += probability
        interval = _bounded_mean_interval(frozen_deltas, alpha=0.05, bets=bets)
        if not interval.lower <= 0.0 <= interval.upper:
            noncoverage_probability += probability

    assert rejection_probability <= 0.05 + 1e-12
    assert noncoverage_probability <= 0.05 + 1e-12


def test_weighted_semantic_cluster_null_calibration() -> None:
    high_probabilities = (0.99, 0.01, 0.8, 0.2, 0.99, 0.01, 0.7, 0.3)
    scales = (
        Fraction(1),
        Fraction(1),
        Fraction(1),
        Fraction(1),
        Fraction(1, 2),
        Fraction(1, 2),
        Fraction(1, 4),
        Fraction(1, 4),
    )
    assert sum(
        float(scale) * (2.0 * probability - 1.0)
        for scale, probability in zip(scales, high_probabilities, strict=True)
    ) == pytest.approx(0.0)
    bets = (
        BoundedMeanBet(fraction=0.1, weight=0.2),
        BoundedMeanBet(fraction=0.5, weight=0.3),
        BoundedMeanBet(fraction=1.0, weight=0.5),
    )
    rejection_probability = 0.0
    lower_noncoverage_probability = 0.0
    for mask in range(1 << len(high_probabilities)):
        deltas: list[float] = []
        probability = 1.0
        for index, high_probability in enumerate(high_probabilities):
            high = bool(mask & (1 << index))
            deltas.append(1.0 if high else -1.0)
            probability *= high_probability if high else 1.0 - high_probability
        frozen_deltas = tuple(deltas)
        if (
            _bounded_mean_p(
                frozen_deltas,
                null_mean=0.0,
                bets=bets,
                observation_scales=scales,
            )
            < 0.05
        ):
            rejection_probability += probability
        if (
            _bounded_mean_lower_bound(
                frozen_deltas,
                alpha=0.05,
                bets=bets,
                observation_scales=scales,
            )
            > 0.0
        ):
            lower_noncoverage_probability += probability

    assert rejection_probability <= 0.05 + 1e-12
    assert lower_noncoverage_probability <= 0.05 + 1e-12


def test_holm_adjustment_is_order_invariant_and_stable_across_ties() -> None:
    first = _holm_adjust({"z": 0.02, "a": 0.01, "m": 0.01})
    second = _holm_adjust({"m": 0.01, "z": 0.02, "a": 0.01})

    rounded_up = math.nextafter(0.03, math.inf)
    assert first == second == {"a": rounded_up, "m": rounded_up, "z": rounded_up}


def test_alpha_division_is_rounded_downward() -> None:
    alpha = 0.05
    divisor = 7
    exact = Fraction.from_float(alpha) / divisor
    ordinary = alpha / divisor

    assert Fraction.from_float(ordinary) > exact
    adjusted = _divide_float_downward(alpha, divisor)
    assert Fraction.from_float(adjusted) <= exact
    assert Fraction.from_float(math.nextafter(adjusted, math.inf)) > exact


def test_holm_adjustment_rounds_integer_products_upward() -> None:
    raw_p = float.fromhex("0x1.111111111110fp-6")
    exact_scaled = Fraction.from_float(raw_p) * 3

    assert Fraction.from_float(raw_p * 3) < exact_scaled
    adjusted = _holm_adjust({"first": raw_p, "second": 0.2, "third": 0.3})
    certified = adjusted["first"]
    assert Fraction.from_float(certified) >= exact_scaled
    assert Fraction.from_float(math.nextafter(certified, -math.inf)) < exact_scaled


def test_analysis_is_reproducible_from_the_frozen_seed() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(24)), attempts=1)
    outcomes = [
        PairedBlockOutcome(
            block=block,
            baseline_reward=0.0,
            candidate_reward=float(sum(f"{block.task_id}:{block.panel_member}".encode()) % 3 != 0),
        )
        for block in design.blocks
    ]

    first = analyze_paired_outcomes(design, outcomes)
    second = analyze_paired_outcomes(design, list(reversed(outcomes)))

    assert first == second
