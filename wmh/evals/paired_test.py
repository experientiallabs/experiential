"""Tests for deterministic paired benchmark design and task-clustered analysis."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from wmh.evals.paired import (
    PairedArm,
    PairedBlockOutcome,
    PairedEvaluationDesign,
    PairedPanelPlan,
    analyze_paired_outcomes,
)


def _design(
    *,
    task_ids: tuple[str, ...] = ("task-a", "task-b", "task-c", "task-d"),
    panel_members: tuple[str, ...] = ("small", "medium", "large"),
    attempts: int = 2,
    attempts_by_member: dict[str, int] | None = None,
) -> PairedEvaluationDesign:
    attempt_counts = attempts_by_member or {
        panel_member: attempts for panel_member in panel_members
    }
    return PairedEvaluationDesign.create(
        task_ids=task_ids,
        panel=tuple(
            PairedPanelPlan(
                panel_member=panel_member,
                attempts=attempt_counts[panel_member],
            )
            for panel_member in panel_members
        ),
        schedule_seed="schedule-v1",
        analysis_seed="analysis-v1",
        randomization_samples=2_000,
        bootstrap_samples=2_000,
        minimum_panel_delta=0.05,
        minimum_member_delta=0.03,
        noninferiority_margin=0.02,
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


def test_schedule_is_deterministic_complete_and_balanced_across_tasks() -> None:
    task_ids = tuple(f"task-{index}" for index in range(5))
    design = _design(task_ids=task_ids, attempts=3)
    repeated = _design(task_ids=task_ids, attempts=3)

    assert design == repeated
    assert design.digest == repeated.digest
    assert len(design.blocks) == 5 * 3 * 3
    for member in design.panel_members:
        first_arm_by_task: dict[str, set[PairedArm]] = {}
        for block in design.blocks:
            if block.panel_member == member:
                first_arm_by_task.setdefault(block.task_id, set()).add(block.first_arm)
        assert all(len(arms) == 1 for arms in first_arm_by_task.values())
        counts = Counter(next(iter(arms)) for arms in first_arm_by_task.values())
        assert abs(counts[PairedArm.BASELINE] - counts[PairedArm.CANDIDATE]) == 1


def test_schedule_supports_predeclared_member_specific_attempt_counts() -> None:
    design = _design(
        panel_members=("economy", "standard", "premium"),
        attempts_by_member={"economy": 20, "standard": 10, "premium": 5},
    )

    counts = Counter(block.panel_member for block in design.blocks)
    assert counts == {"economy": 80, "standard": 40, "premium": 20}
    assert design.attempts_by_member == {"economy": 20, "premium": 5, "standard": 10}


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

    assert report.panel_delta == 1.0
    assert report.analysis_version == "1"
    assert report.design_digest == design.digest
    assert report.outcome_digest.startswith("sha256:")
    assert report.member_deltas == {"large": 1.0, "medium": 1.0, "small": 1.0}
    assert report.randomization_p < 0.05
    assert report.cluster_interval.lower == 1.0
    assert report.cluster_interval.upper == 1.0
    assert all(value < 0.05 for value in report.holm_noninferiority_p.values())
    assert report.panel_lift_passed is True
    assert report.member_lifts_passed is True
    assert report.randomization_passed is True
    assert report.cluster_interval_passed is True
    assert report.noninferiority_passed is True
    assert report.passed is True


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

    assert report.panel_delta > design.minimum_panel_delta
    assert report.member_deltas["small"] == 0.0
    assert report.member_lifts_passed is False
    assert report.passed is False


def test_no_lift_fails_every_positive_evidence_gate() -> None:
    design = _design(task_ids=tuple(f"task-{index}" for index in range(12)))
    report = analyze_paired_outcomes(
        design,
        _outcomes(design, baseline=0.0, candidate=0.0),
    )

    assert report.panel_delta == 0.0
    assert report.panel_lift_passed is False
    assert report.member_lifts_passed is False
    assert report.randomization_passed is False
    assert report.cluster_interval_passed is False
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

    assert report.panel_delta == 0.0
    assert report.randomization_p == 0.75


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
