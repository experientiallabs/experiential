"""Difficulty tiers from judged attempts of an actor ladder, merged with nudge rungs.

Every task is attempted ``G`` times by each actor of an ordered ladder (weakest first; five
open-weight actors in the reference setup). Each attempt is labelled only by the frozen episode
judge (:func:`~exp.optimize.evaluation.variance.judged.slot_rewards`), so an attempt is a pass, a
fail, or unscored, and unscored attempts are never counted as either.

For each task and actor the judged pass rate carries a Wilson score interval over the scored
attempts. An actor **passes** a task when its rate reaches ``pass_rate``. The task's difficulty
level is the position of the weakest actor that passes, from 0 ("weak models pass") up to the
ladder length ("no model passes"). Uncertainty is kept rather than hidden: the easiest plausible
level reads an actor as passing when the interval's upper bound reaches ``pass_rate``, the
hardest plausible level only when its lower bound does, and an actor with fewer than
``min_scored`` scored attempts is plausible both ways. A stronger actor failing where a weaker one
passes, both confidently, is recorded as an inversion. The result is the simulation-side
:class:`~exp.simulation.mining.task_types.TaskDifficulty`, so task types can carry it
(:func:`with_task_type_difficulty`).

:func:`merge_curriculum` combines difficulty with nudge-ladder rungs into one ordered curriculum:
tasks from the easiest level to the hardest (by mean failure inside a level), and inside one task
its admitted nudged rungs from strong to weak and then its un-nudged target, or the base task when
it has no ladder. :func:`curriculum_plan` hands the bound rungs, in that order, to the training
schedule's unchanged :class:`~exp.optimize.evaluation.variance.nudge_ladder.CurriculumPlan`, whose
strength tiers then run easier tasks' rungs before harder ones; :func:`difficulty_band` gives the
difficulty level a number of completed rollouts has reached, for a schedule that advances by
difficulty. The curriculum is saved as a content-addressed ``difficulty-curriculum`` artifact and
replays with no call.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.variance.judged import slot_rewards
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumPlan, CurriculumStage
from exp.optimize.evaluation.variance.nudges import STRENGTH_RANK, RungStrength
from exp.optimize.evaluation.variance.records import load_judged_study
from exp.simulation.mining.tag_clusters import read_record, write_record
from exp.simulation.mining.task_types import TaskDifficulty, TaskTypes, with_difficulty

DIFFICULTY_CURRICULUM_PROTOCOL = "difficulty-curriculum-v1"
"""Version of the pass-rate intervals, tier rules and curriculum merge; bump when any changes."""

DIFFICULTY_TYPE = "difficulty-curriculum"
"""Artifact type of a saved :class:`DifficultyCurriculum` (``curriculum.json``)."""

Verdict = Literal["pass", "fail", "uncertain"]
"""Whether an actor passes a task: confidently, confidently not, or not settled."""


class ActorLadder(ContractModel):
    """Actor tiers ordered from the weakest to the strongest.

    Attributes:
        actors: Tier labels (actor aliases), weakest first.
    """

    actors: tuple[str, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        """Require distinct actors."""
        if len(set(self.actors)) != len(self.actors):
            raise ValueError("an actor ladder names each actor once")
        return self


class DifficultySpec(ContractModel):
    """How judged attempts become difficulty tiers.

    Attributes:
        pass_rate: Judged pass rate at which an actor passes a task.
        z: Normal quantile of the Wilson interval (1.96 for 95 percent).
        min_scored: Fewest scored attempts before an actor's verdict is read; fewer leaves it
            uncertain.
    """

    pass_rate: float = Field(default=0.5, gt=0.0, le=1.0)
    z: float = Field(default=1.96, gt=0.0, le=5.0)
    min_scored: int = Field(default=4, ge=1, le=10_000)


class TaskAttempts(ContractModel):
    """One actor's judged attempts at one task.

    Attributes:
        task_id: The task.
        actor: The actor tier.
        outcomes: One label per planned attempt: 1.0 pass, 0.0 fail, None unscored.
    """

    task_id: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    outcomes: tuple[float | None, ...]

    @model_validator(mode="after")
    def _labels(self) -> Self:
        """Require every scored label to be a pass or a fail."""
        if any(value not in (0.0, 1.0) for value in self.outcomes if value is not None):
            raise ValueError("a judged attempt is 1.0, 0.0 or unscored")
        return self


class ActorPassRate(ContractModel):
    """One actor's judged pass rate on one task, with its interval.

    Attributes:
        actor: The actor tier.
        attempts: Planned attempts.
        scored: Attempts with a label.
        passed: Passing attempts.
        rate: Pass rate over scored attempts, None when none was scored.
        low: Wilson lower bound.
        high: Wilson upper bound.
        verdict: Whether it passes the task.
    """

    actor: str
    attempts: int = Field(ge=0)
    scored: int = Field(ge=0)
    passed: int = Field(ge=0)
    rate: float | None = None
    low: float = Field(ge=0.0, le=1.0)
    high: float = Field(ge=0.0, le=1.0)
    verdict: Verdict


class TaskDifficultyRecord(ContractModel):
    """One task's per-actor pass rates and difficulty.

    Attributes:
        task_id: The task.
        actors: Each ladder actor's pass rate, weakest first.
        difficulty: The difficulty tier.
        inversions: ``weaker>stronger`` pairs where the weaker confidently passes and the
            stronger confidently fails.
    """

    task_id: str
    actors: tuple[ActorPassRate, ...]
    difficulty: TaskDifficulty
    inversions: tuple[str, ...] = ()


class CurriculumEntry(ContractModel):
    """One step of the merged curriculum.

    Attributes:
        order: Position in the curriculum, from 0.
        task_id: The task.
        level: Its difficulty level.
        label: Its difficulty label.
        strength: The rung's nudge strength; ``none`` for the un-nudged target or base task.
        ladder: The rung's ladder, when the step is a rung.
        scenario_id: The rung's scenario, when the step is a rung.
        stage: The rung as a training stage (bound when its task qualified), when it is one.
    """

    order: int = Field(ge=0)
    task_id: str
    level: int = Field(ge=0)
    label: str
    strength: RungStrength = "none"
    ladder: str | None = None
    scenario_id: str | None = None
    stage: CurriculumStage | None = Field(default=None, exclude_if=lambda value: value is None)


class DifficultyCurriculum(ContractModel):
    """The ``difficulty-curriculum`` record.

    Attributes:
        protocol: Curriculum protocol.
        ladder: The actor ladder.
        spec: The tier registration.
        tasks: Every task's difficulty, easiest first.
        entries: The merged curriculum, in order.
    """

    protocol: str = DIFFICULTY_CURRICULUM_PROTOCOL
    ladder: ActorLadder
    spec: DifficultySpec
    tasks: tuple[TaskDifficultyRecord, ...]
    entries: tuple[CurriculumEntry, ...]

    def identity(self) -> str:
        """Return the content-addressed identity of this record."""
        return stable_id(DIFFICULTY_TYPE, {"protocol": self.protocol, "record": sha256_json(self)})

    def difficulty(self) -> dict[str, TaskDifficulty]:
        """Return each task's difficulty, by task."""
        return {row.task_id: row.difficulty for row in self.tasks}


def wilson(passed: int, scored: int, z: float) -> tuple[float, float]:
    """Return the Wilson score interval of a pass rate; ``(0, 1)`` with nothing scored."""
    if scored == 0:
        return 0.0, 1.0
    rate = passed / scored
    denominator = 1.0 + z * z / scored
    centre = (rate + z * z / (2 * scored)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / scored + z * z / (4 * scored * scored))
    margin /= denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def tier_label(level: int, ladder: ActorLadder) -> str:
    """Return the readable tier of one level on a ladder."""
    count = len(ladder.actors)
    if level >= count:
        return "no model passes"
    if level == 0:
        return "weak models pass"
    if level == count - 1:
        return f"only the strongest ({ladder.actors[-1]}) passes"
    return f"{ladder.actors[level]} and stronger pass"


def actor_pass_rate(
    actor: str, outcomes: Sequence[float | None], spec: DifficultySpec
) -> ActorPassRate:
    """Return one actor's pass rate, interval and verdict over its judged attempts."""
    scored = [value for value in outcomes if value is not None]
    passed = sum(1 for value in scored if value == 1.0)
    low, high = wilson(passed, len(scored), spec.z)
    verdict: Verdict = "uncertain"
    if len(scored) >= spec.min_scored:
        if low >= spec.pass_rate:
            verdict = "pass"
        elif high < spec.pass_rate:
            verdict = "fail"
    return ActorPassRate(
        actor=actor,
        attempts=len(outcomes),
        scored=len(scored),
        passed=passed,
        rate=round(passed / len(scored), 6) if scored else None,
        low=round(low, 6),
        high=round(high, 6),
        verdict=verdict,
    )


def task_difficulty(
    task_id: str,
    attempts: Sequence[TaskAttempts],
    ladder: ActorLadder,
    spec: DifficultySpec | None = None,
) -> TaskDifficultyRecord:
    """Return one task's difficulty tier from every ladder actor's judged attempts.

    An actor with no attempts counts as having none scored, so it is uncertain both ways.

    Raises:
        ValueError: An attempt row is for another task, an actor off the ladder, or repeats
            an actor.
    """
    spec = spec or DifficultySpec()
    by_actor: dict[str, TaskAttempts] = {}
    for row in attempts:
        if row.task_id != task_id:
            raise ValueError(f"attempts for task {row.task_id} given for task {task_id}")
        if row.actor not in ladder.actors:
            raise ValueError(f"actor {row.actor} is not on the ladder {list(ladder.actors)}")
        if row.actor in by_actor:
            raise ValueError(f"actor {row.actor} has two attempt groups for task {task_id}")
        by_actor[row.actor] = row
    rates = tuple(
        actor_pass_rate(actor, by_actor[actor].outcomes if actor in by_actor else (), spec)
        for actor in ladder.actors
    )
    count = len(ladder.actors)

    def first(passes: Sequence[bool]) -> int:
        """Return the position of the first passing actor, or the ladder length."""
        return next((index for index, ok in enumerate(passes) if ok), count)

    readable = [r.scored >= spec.min_scored for r in rates]
    point = first(
        [
            ok and r.rate is not None and r.rate >= spec.pass_rate
            for r, ok in zip(rates, readable, strict=True)
        ]
    )
    low = first([not ok or r.high >= spec.pass_rate for r, ok in zip(rates, readable, strict=True)])
    high = first([r.verdict == "pass" for r in rates])
    low, high = min(low, point), max(high, point)
    rated = [r.rate for r in rates if r.rate is not None]
    score = 1.0 - sum(rated) / len(rated) if rated else 1.0
    inversions = tuple(
        f"{weak.actor}>{strong.actor}"
        for index, weak in enumerate(rates)
        for strong in rates[index + 1 :]
        if weak.verdict == "pass" and strong.verdict == "fail"
    )
    return TaskDifficultyRecord(
        task_id=task_id,
        actors=rates,
        difficulty=TaskDifficulty(
            level=point,
            label=tier_label(point, ladder),
            low=low,
            high=high,
            actors=count,
            score=round(min(1.0, max(0.0, score)), 6),
            confident=low == high,
        ),
        inversions=inversions,
    )


def difficulty_tiers(
    attempts: Sequence[TaskAttempts], ladder: ActorLadder, spec: DifficultySpec | None = None
) -> tuple[TaskDifficultyRecord, ...]:
    """Return every attempted task's difficulty, easiest first (level, mean failure, task)."""
    spec = spec or DifficultySpec()
    grouped: dict[str, list[TaskAttempts]] = {}
    for row in attempts:
        grouped.setdefault(row.task_id, []).append(row)
    records = [task_difficulty(task, rows, ladder, spec) for task, rows in grouped.items()]
    return tuple(sorted(records, key=_order_key))


def merge_curriculum(
    tasks: Sequence[TaskDifficultyRecord],
    stages_by_task: Mapping[str, Sequence[CurriculumStage]],
    *,
    ladder: ActorLadder,
    spec: DifficultySpec | None = None,
) -> DifficultyCurriculum:
    """Merge difficulty tiers with nudge-ladder rungs into one ordered curriculum.

    Args:
        tasks: Every task's difficulty.
        stages_by_task: Each task's nudge curriculum stages (from
            :func:`~exp.optimize.evaluation.variance.nudge_ladder.nudge_curriculum` over that
            task's ladders, bound or not); a task it omits enters as its base task.
        ladder: The actor ladder the tiers were read on.
        spec: The tier registration the tiers were read under.

    Raises:
        ValueError: A stage is given for a task with no difficulty, or a task repeats.
    """
    spec = spec or DifficultySpec()
    known = [row.task_id for row in tasks]
    if len(set(known)) != len(known):
        raise ValueError("each task has one difficulty in a curriculum")
    stray = sorted(set(stages_by_task) - set(known))
    if stray:
        raise ValueError(f"nudge rungs given for tasks with no difficulty {stray[:5]}")
    ordered = sorted(tasks, key=_order_key)
    entries: list[CurriculumEntry] = []
    for row in ordered:
        level, label = row.difficulty.level, row.difficulty.label
        stages = sorted(
            stages_by_task.get(row.task_id, ()),
            key=lambda stage: (STRENGTH_RANK[stage.strength], stage.ladder, stage.position),
        )
        if not stages:
            entries.append(
                CurriculumEntry(order=len(entries), task_id=row.task_id, level=level, label=label)
            )
        for stage in stages:
            entries.append(
                CurriculumEntry(
                    order=len(entries),
                    task_id=row.task_id,
                    level=level,
                    label=label,
                    strength=stage.strength,
                    ladder=stage.ladder,
                    scenario_id=stage.scenario_id,
                    stage=stage,
                )
            )
    return DifficultyCurriculum(
        ladder=ladder, spec=spec, tasks=tuple(ordered), entries=tuple(entries)
    )


def curriculum_plan(
    curriculum: DifficultyCurriculum, *, rollouts_per_tier: int, bound_only: bool = True
) -> CurriculumPlan | None:
    """Return the schedule's curriculum plan over the merged rungs, easier tasks first.

    The schedule advances by nudge strength (:func:`~exp.optimize.evaluation.variance.
    nudge_ladder.curriculum_tier`) and keeps stage order inside a tier, so inside every strength
    tier the easier tasks' rungs come first.

    Args:
        curriculum: The merged curriculum.
        rollouts_per_tier: Rollouts each nudged strength tier lasts.
        bound_only: Keep only stages bound to a qualified task, as a runnable schedule needs.

    Returns:
        The plan, or None when no stage qualifies.
    """
    stages = tuple(
        entry.stage
        for entry in curriculum.entries
        if entry.stage is not None and (entry.stage.job is not None or not bound_only)
    )
    if not stages:
        return None
    return CurriculumPlan(stages=stages, rollouts_per_tier=rollouts_per_tier)


def difficulty_band(
    curriculum: DifficultyCurriculum, completed: int, per_band: int
) -> tuple[CurriculumEntry, ...]:
    """Return the entries of the difficulty level a number of completed rollouts has reached.

    Levels run from the easiest present to the hardest, each for ``per_band`` rollouts; once
    every level is spent the hardest stays current.

    Raises:
        ValueError: ``completed`` is negative or ``per_band`` is not positive.
    """
    if completed < 0 or per_band < 1:
        raise ValueError("completed must be non-negative and per_band positive")
    levels = sorted({entry.level for entry in curriculum.entries})
    if not levels:
        return ()
    current = levels[min(completed // per_band, len(levels) - 1)]
    return tuple(entry for entry in curriculum.entries if entry.level == current)


def judged_attempts(
    project: ProjectStore, *, task_id: str, actor: str, reference: ArtifactInput, judge: str
) -> TaskAttempts:
    """Read one actor's judged attempts at one task from its saved judged study.

    Args:
        project: Project holding the study and its assessments.
        task_id: The task attempted.
        actor: The actor tier that attempted it.
        reference: The saved ``judged-continuation-study`` of the actor's group of attempts.
        judge: The frozen episode judge's definition identity.
    """
    study = load_judged_study(project, reference)
    return TaskAttempts(
        task_id=task_id, actor=actor, outcomes=slot_rewards(project, study.branches, judge)
    )


def with_task_type_difficulty(record: TaskTypes, curriculum: DifficultyCurriculum) -> TaskTypes:
    """Fill every task type's difficulty from the curriculum's task tiers."""
    return with_difficulty(record, curriculum.difficulty())


def curriculum_summary(curriculum: DifficultyCurriculum) -> JsonObject:
    """Return a short JSON summary for a stage report: tasks per level and the step order."""
    levels: dict[str, int] = {}
    for row in curriculum.tasks:
        levels[row.difficulty.label] = levels.get(row.difficulty.label, 0) + 1
    return {
        "ladder": list(curriculum.ladder.actors),
        "levels": dict(levels),
        "uncertain": [row.task_id for row in curriculum.tasks if not row.difficulty.confident],
        "inversions": {
            row.task_id: list(row.inversions) for row in curriculum.tasks if row.inversions
        },
        "entries": [
            {"task_id": e.task_id, "level": e.level, "strength": e.strength}
            for e in curriculum.entries
        ],
    }


def persist_difficulty_curriculum(
    store: ArtifactStore,
    curriculum: DifficultyCurriculum,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> ArtifactInput:
    """Save a curriculum under its content identity, or adopt the identical saved one.

    Raises:
        ValueError: A saved artifact under that identity holds a different record.
    """
    identity = curriculum.identity()
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        if load_difficulty_curriculum(store, reference) != curriculum:
            raise ValueError(f"saved curriculum {identity} holds a different record")
        return reference
    return write_record(
        store, identity, DIFFICULTY_TYPE, "curriculum.json", curriculum, code_revision, inputs
    )


def load_difficulty_curriculum(
    store: ArtifactStore, reference: ArtifactInput
) -> DifficultyCurriculum:
    """Read a saved curriculum, refusing a changed or mistyped artifact."""
    payload = read_record(store, reference, DIFFICULTY_TYPE, "curriculum.json")
    return DifficultyCurriculum.model_validate_json(payload)


def _order_key(row: TaskDifficultyRecord) -> tuple[int, float, str]:
    """Order tasks from the easiest level, by mean failure inside a level, then by task."""
    return row.difficulty.level, row.difficulty.score, row.task_id
