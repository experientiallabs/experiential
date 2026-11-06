"""Hybrid dense collaborators and the difficulty curriculum, wired into the records stage.

- **Hybrid dense**: under ``records.dense.hybrid`` every confirmation episode's actor steps are
  answered by the decision model (Clef) over the tags' step bank through the admitted transport
  and the decision cache (:func:`hybrid_inputs`); only fired, uncertain, untagged or
  linter-proposed tags reach the dense judge. Unscored stays unscored.
- **Difficulty curriculum**: each tier's judged group of attempts gives that actor's judged pass
  rate on the start's task (:func:`~.difficulty_curriculum.judged_attempts`); the declared actor
  order is the ladder, weakest first (:func:`tier_difficulty`). Per tier, the merged curriculum
  orders the task's qualified nudge rungs by difficulty and strength and becomes the schedule's
  curriculum plan (:func:`difficulty_plan`), saved and summarized.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import load_model_catalog
from exp.optimize.evaluation.variance.difficulty_curriculum import (
    ActorLadder,
    TaskDifficultyRecord,
    curriculum_plan,
    curriculum_summary,
    difficulty_tiers,
    judged_attempts,
    merge_curriculum,
    persist_difficulty_curriculum,
)
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumPlan, CurriculumStage
from exp.optimize.evaluation.variance.pipeline_inputs import admitted_transport
from exp.optimize.evaluation.variance.pipeline_spec import MiningOutputs, StageContext
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.questions import QuestionBank, resolve_question_bank
from exp.simulation.mining.tagging import DecisionCache, TraceTagger


def hybrid_inputs(context: StageContext) -> tuple[TraceTagger | None, QuestionBank | None]:
    """Return the decision tagger and step bank of hybrid dense routing, when registered."""
    dense = context.spec.records.dense
    if dense is None or dense.hybrid is None:
        return None, None
    hybrid = dense.hybrid
    bank = resolve_question_bank(hybrid.bank_id, root=context.spec.traceml.bank_root)
    injected = context.dependencies.traceml_tagger
    if injected is not None:
        return injected, bank
    project = context.project
    resolved = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path)).resolve(
        hybrid.decision_alias
    )
    deadline = datetime.now(UTC) + timedelta(hours=context.spec.traceml.deadline_hours)
    transport = admitted_transport(context, "dense-hybrid", 100_000, deadline).priced_as(
        hybrid.decision_alias
    )
    tagger = DecisionsTraceTagger.for_alias(
        resolved, transport=transport, cache=DecisionCache(project.paths)
    )
    return tagger, bank


def tier_difficulty(
    context: StageContext, mining: MiningOutputs
) -> tuple[ActorLadder, tuple[TaskDifficultyRecord, ...]]:
    """Read every tier's judged pass rate on the start's task, the declared order the ladder.

    Returns:
        The ladder (weakest first, as the actors are declared) and the task's difficulty.
    """
    project = context.project
    ladder = ActorLadder(actors=tuple(tier.tier for tier in mining.tiers))
    attempts = []
    for tier in mining.tiers:
        start = load_continuation_checkpoint(project.artifacts, tier.job.study.checkpoint)
        attempts.append(
            judged_attempts(
                project,
                task_id=start.task.task_id,
                actor=tier.tier,
                reference=tier.judged_group,
                judge=tier.judge.identity,
            )
        )
    return ladder, difficulty_tiers(attempts, ladder)


def difficulty_plan(
    context: StageContext,
    stages: Sequence[CurriculumStage],
    ladder: ActorLadder,
    difficulty: Sequence[TaskDifficultyRecord],
    *,
    inputs: tuple[ArtifactInput, ...],
) -> tuple[CurriculumPlan | None, ArtifactInput, JsonObject]:
    """Merge the task's difficulty with its bound rungs into the tier's curriculum plan.

    Returns:
        The schedule's curriculum plan (None without bound rungs), the saved curriculum and
        its summary.
    """
    task_id = difficulty[0].task_id if difficulty else None
    by_task = {} if task_id is None else {task_id: tuple(stages)}
    curriculum = merge_curriculum(difficulty, by_task, ladder=ladder)
    saved = persist_difficulty_curriculum(
        context.project.artifacts, curriculum, code_revision=context.code_revision, inputs=inputs
    )
    plan = curriculum_plan(curriculum, rollouts_per_tier=context.spec.records.rollouts_per_tier)
    return plan, saved, curriculum_summary(curriculum)
