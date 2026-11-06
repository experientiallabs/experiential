"""Difficulty tiers from synthetic judged outcomes, merged with nudge rungs, and replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.difficulty_curriculum import (
    DIFFICULTY_TYPE,
    ActorLadder,
    DifficultySpec,
    TaskAttempts,
    curriculum_plan,
    curriculum_summary,
    difficulty_band,
    difficulty_tiers,
    load_difficulty_curriculum,
    merge_curriculum,
    persist_difficulty_curriculum,
    task_difficulty,
    tier_label,
    wilson,
    with_task_type_difficulty,
)
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumStage
from exp.simulation.mining.environment_families import FamilyTask, StartingState
from exp.simulation.mining.task_types import FamilyTaskTypes, TaskType, TaskTypes, TaskTypeSpec

LADDER = ActorLadder(actors=("a1", "a2", "a3", "a4", "a5"))
G = 16


def outcomes(passed: int, unscored: int = 0, attempts: int = G) -> tuple[float | None, ...]:
    """Judged labels: ``passed`` passes, ``unscored`` unscored, the rest failures."""
    failed = attempts - passed - unscored
    return (1.0,) * passed + (None,) * unscored + (0.0,) * failed


def group(task: str, passes: tuple[int, ...]) -> list[TaskAttempts]:
    """One task's attempts by every ladder actor, with the given pass counts."""
    return [
        TaskAttempts(task_id=task, actor=actor, outcomes=outcomes(count))
        for actor, count in zip(LADDER.actors, passes, strict=True)
    ]


def synthetic() -> list[TaskAttempts]:
    """Easy, middle, hard and borderline tasks over the five-actor ladder."""
    return [
        *group("easy", (16, 16, 16, 16, 16)),
        *group("middle", (0, 1, 15, 16, 16)),
        *group("hard", (0, 0, 0, 0, 1)),
        *group("borderline", (2, 9, 11, 16, 16)),
    ]


def stage(ladder: str, position: int, strength: str) -> CurriculumStage:
    """An unbound nudge curriculum stage."""
    return CurriculumStage.model_validate(
        {
            "ladder": ladder,
            "scenario_id": f"{ladder}-{strength}",
            "position": position,
            "strength": strength,
        }
    )


def test_wilson_bounds_and_tier_labels() -> None:
    """Intervals widen with fewer attempts; labels run from weak passing to none passing."""
    assert wilson(0, 0, 1.96) == (0.0, 1.0)
    narrow, wide = wilson(8, 16, 1.96), wilson(2, 4, 1.96)
    assert wide[0] < narrow[0] < 0.5 < narrow[1] < wide[1]
    assert tier_label(0, LADDER) == "weak models pass"
    assert tier_label(2, LADDER) == "a3 and stronger pass"
    assert tier_label(4, LADDER) == "only the strongest (a5) passes"
    assert tier_label(5, LADDER) == "no model passes"


def test_tiers_come_from_judged_pass_rates_with_uncertainty() -> None:
    """Each task's tier is its weakest passing actor; borderline rates widen the range."""
    records = {r.task_id: r for r in difficulty_tiers(synthetic(), LADDER)}
    easy, middle, hard, border = (
        records[k].difficulty for k in ("easy", "middle", "hard", "borderline")
    )
    assert (easy.level, easy.label, easy.confident) == (0, "weak models pass", True)
    assert (middle.level, middle.low, middle.high, middle.confident) == (2, 2, 2, True)
    assert (hard.level, hard.label, hard.confident) == (5, "no model passes", True)
    assert border.level == 1 and border.low == 1 and border.high == 3 and not border.confident
    assert [r.task_id for r in difficulty_tiers(synthetic(), LADDER)] == [
        "easy",
        "borderline",
        "middle",
        "hard",
    ]
    rates = {a.actor: a for a in records["borderline"].actors}
    assert rates["a2"].verdict == "uncertain" and rates["a3"].verdict == "uncertain"
    assert rates["a4"].verdict == "pass" and rates["a1"].verdict == "fail"


def test_unscored_attempts_are_not_counted_and_thin_evidence_stays_uncertain() -> None:
    """Unscored attempts leave the rate alone; too few scored attempts widen both ways."""
    rows = [
        TaskAttempts(task_id="t", actor="a1", outcomes=outcomes(3, unscored=13)),
        TaskAttempts(task_id="t", actor="a2", outcomes=outcomes(16)),
    ]
    record = task_difficulty("t", rows, LADDER)
    first = record.actors[0]
    assert (first.scored, first.passed, first.rate, first.verdict) == (3, 3, 1.0, "uncertain")
    assert record.actors[2].attempts == 0 and record.actors[2].verdict == "uncertain"
    assert record.difficulty.level == 1  # a1 is unread at the point estimate; a2 passes
    assert record.difficulty.low == 0 and record.difficulty.high == 1


def test_inversions_and_invalid_attempts_are_reported() -> None:
    """A weaker actor confidently passing where a stronger fails is an inversion."""
    record = task_difficulty("t", group("t", (0, 16, 16, 0, 16)), LADDER)
    assert record.inversions == ("a2>a4", "a3>a4")
    with pytest.raises(ValueError, match="not on the ladder"):
        task_difficulty("t", [TaskAttempts(task_id="t", actor="zz", outcomes=())], LADDER)
    with pytest.raises(ValueError, match="two attempt groups"):
        task_difficulty("t", [*group("t", (1,) * 5), *group("t", (1,) * 5)], LADDER)
    with pytest.raises(ValueError, match="1.0, 0.0 or unscored"):
        TaskAttempts(task_id="t", actor="a1", outcomes=(0.5,))
    assert task_difficulty("t", group("t", (8,) * 5), LADDER).difficulty.level == 0
    strict = DifficultySpec(pass_rate=0.6)
    assert task_difficulty("t", group("t", (8,) * 5), LADDER, strict).difficulty.level == 5


def test_the_curriculum_merges_tiers_with_ladder_rungs() -> None:
    """Easier tasks come first; inside a task its rungs fade from strong to un-nudged."""
    records = difficulty_tiers(synthetic(), LADDER)
    curriculum = merge_curriculum(
        records,
        {
            "hard": [stage("lh", 2, "none"), stage("lh", 0, "strong"), stage("lh", 1, "weak")],
            "middle": [stage("lm", 0, "strong"), stage("lm", 1, "none")],
        },
        ladder=LADDER,
    )
    steps = [(e.task_id, e.strength) for e in curriculum.entries]
    assert steps == [
        ("easy", "none"),
        ("borderline", "none"),
        ("middle", "strong"),
        ("middle", "none"),
        ("hard", "strong"),
        ("hard", "weak"),
        ("hard", "none"),
    ]
    assert [e.order for e in curriculum.entries] == list(range(7))
    assert curriculum.entries[0].stage is None and curriculum.entries[2].stage is not None

    assert curriculum_plan(curriculum, rollouts_per_tier=2) is None  # nothing bound yet
    plan = curriculum_plan(curriculum, rollouts_per_tier=2, bound_only=False)
    assert plan is not None
    assert [s.scenario_id for s in plan.tier(0)] == ["lm-strong", "lh-strong"]
    assert [s.scenario_id for s in plan.tier(2)] == ["lh-weak"]
    assert [s.scenario_id for s in plan.tier(99)] == ["lm-none", "lh-none"]

    assert [e.task_id for e in difficulty_band(curriculum, 0, 3)] == ["easy"]
    assert {e.task_id for e in difficulty_band(curriculum, 6, 3)} == {"middle"}
    assert {e.task_id for e in difficulty_band(curriculum, 99, 3)} == {"hard"}
    summary = curriculum_summary(curriculum)
    assert summary["levels"] == {
        "weak models pass": 1,
        "a2 and stronger pass": 1,
        "a3 and stronger pass": 1,
        "no model passes": 1,
    }
    assert summary["uncertain"] == ["borderline"]
    with pytest.raises(ValueError, match="no difficulty"):
        merge_curriculum(records, {"ghost": [stage("g", 0, "none")]}, ladder=LADDER)


def test_task_types_receive_curriculum_difficulty_and_the_record_replays(tmp_path: Path) -> None:
    """A task type's difficulty fills from its tasks' tiers; the curriculum saves and loads."""
    curriculum = merge_curriculum(difficulty_tiers(synthetic(), LADDER), {}, ladder=LADDER)
    tasks = tuple(
        FamilyTask(
            task_id=name,
            lineage_id=name,
            role="world_learning",
            source_trace_ids=(f"tr-{name}",),
            start=StartingState(task_id=name, initial_context_sha256="0" * 64),
        )
        for name in ("easy", "middle", "hard")
    )
    record = TaskTypes(
        families="environment-families-x",
        model={"model_id": "namer"},
        spec=TaskTypeSpec(),
        evidence_sha256="0" * 64,
        by_family=(
            FamilyTaskTypes(
                family_id="f",
                clustering="c",
                names="n",
                reason="r",
                task_types=(
                    TaskType(
                        task_type_id="tt",
                        family_id="f",
                        cluster_id="c00",
                        members=("tr-easy",),
                        tasks=tasks,
                    ),
                ),
            ),
        ),
    )
    filled = with_task_type_difficulty(record, curriculum).all()[0].difficulty
    assert (filled.level, filled.lowest, filled.highest) == (2, 0, 5)

    store = ProjectStore(tmp_path, "case").artifacts
    reference = persist_difficulty_curriculum(store, curriculum, code_revision="t")
    assert store.read(reference.artifact_id).manifest.artifact_type == DIFFICULTY_TYPE
    assert persist_difficulty_curriculum(store, curriculum, code_revision="t") == reference
    assert load_difficulty_curriculum(store, reference) == curriculum
