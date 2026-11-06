"""Grade one rollout under one rubric version with both rubric parts, and held-out recall.

:func:`grade_rollout` reads an ingested rollout (:mod:`~.rollout_ingest`) as episode evidence and
grades it under a :class:`~.rubric_versions.RubricVersion`:

- the outcome part is the task's frozen episode judge rebound to the version
  (:func:`~.rubric_versions.versioned_judge`) through the unchanged :func:`~.judge.judge_episode`;
- the dense part is the decision model tagging every actor step with the version's step bank and
  the unchanged :func:`~.dense_hybrid.judge_hybrid` adjudicating fired, uncertain and proposed
  tags, with the trainer's sampled tokens when the rollout carries them.

Both parts save their judgments and replay them with no call. A :class:`RubricGrade` records the
version it was made under; :func:`require_label` refuses a grade from any other version, and the
judges' own identity checks refuse an assessment made under another rubric, so old labels are
never read after a revision.

:class:`HeldOutSet` is the known-hack set: rollouts labeled with the hack they show. It is never
an input to admission (the loop refuses any export rollout whose lineage it holds);
:func:`measure_recall` grades every held-out hack under a version and counts which part caught
it, and a new version must not lower either recall.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.dense_hybrid import (
    RoutingSpec,
    judge_hybrid,
    tag_episode_steps,
)
from exp.optimize.evaluation.variance.dense_judge import DenseRewardJudge
from exp.optimize.evaluation.variance.episodes import episode_reward
from exp.optimize.evaluation.variance.judge import judge_episode
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.optimize.evaluation.variance.rollout_ingest import Rollout, rollout_evidence
from exp.optimize.evaluation.variance.rubric_versions import (
    RecallRecord,
    RubricVersion,
    versioned_dense,
    versioned_judge,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TraceTagger


class TaskRubric(ContractModel):
    """One trained task's frozen two-part rubric before any revision.

    Attributes:
        task_id: The trained task.
        judge: Its frozen episode judge (outcome reward).
        dense: Its frozen dense judge (bounds, profile, task and world digests).
    """

    task_id: str = Field(min_length=1)
    judge: EpisodeJudge
    dense: DenseRewardJudge

    @model_validator(mode="after")
    def _same_task(self) -> TaskRubric:
        """Both parts judge the same task and world."""
        judge, dense = self.judge.definition, self.dense.definition
        if (judge.task_sha256, judge.world_sha256) != (dense.task_sha256, dense.world_sha256):
            raise ValueError(f"task {self.task_id}: the judge and dense part bind other digests")
        return self


class HeldOutHack(ContractModel):
    """One known reward hack, held out from admission.

    Attributes:
        rollout: The rollout showing it.
        hack_kind: What hack it is, as labeled by a person.
    """

    rollout: Rollout
    hack_kind: str = Field(min_length=1, max_length=200)


class HeldOutSet(ContractModel):
    """The held-out known-hack set; never an admission input.

    Attributes:
        set_id: Identity.
        source: Where the rollouts come from, part of each event's source locator.
        hacks: The known hacks, unique by rollout.
    """

    set_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    hacks: tuple[HeldOutHack, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> HeldOutSet:
        """Require unique rollouts."""
        ids = [row.rollout.rollout_id for row in self.hacks]
        if len(set(ids)) != len(ids):
            raise ValueError("the held-out set repeats a rollout")
        return self

    @property
    def identity(self) -> Sha256:
        """Digest of the whole set."""
        return sha256_json(self)

    def lineages(self) -> frozenset[str]:
        """Every lineage the set holds."""
        return frozenset(row.rollout.lineage_id for row in self.hacks)


class CriterionCitation(ContractModel):
    """One outcome-judge criterion finding, with its cited events.

    Attributes:
        criterion_id: The criterion.
        status: ``met``, ``not_met`` or ``insufficient_evidence``.
        event_refs: Cited events.
        reason: The judge's reason.
    """

    criterion_id: str
    status: str
    event_refs: tuple[str, ...] = ()
    reason: str = ""


class DenseCitation(ContractModel):
    """One decided dense finding, with its cited events.

    Attributes:
        tag_id: The tag.
        decision: The reconciled decision.
        observed_refs: Where it was observed.
        obligation_refs: The cited obligation.
        change_refs: The cited change.
        result_refs: The cited results.
        weakening: Why the change weakened the obligation, when established.
        rationale: The judge's reason.
        reward: Its reward, None when unscored.
    """

    tag_id: str
    decision: str
    observed_refs: tuple[str, ...] = ()
    obligation_refs: tuple[str, ...] = ()
    change_refs: tuple[str, ...] = ()
    result_refs: tuple[str, ...] = ()
    weakening: str | None = None
    rationale: str = ""
    reward: float | None = None


class RubricGrade(ContractModel):
    """One rollout graded by both parts of one rubric version.

    Attributes:
        rollout_id: The rollout.
        version: The version number.
        version_sha256: The version identity.
        judge: The rebound episode judge's identity.
        dense_definition: The rebound dense definition's identity.
        outcome_verdict: The episode judge's verdict.
        outcome_pass: 1 pass, 0 fail, None unscored (from the saved assessment).
        decisive_refs: Events the verdict rests on.
        criteria: Every criterion finding with its citations.
        dense_status: ``judged`` or ``unavailable``.
        dense_total: The bounded dense total, when judged.
        decisive: Decisive penalty tags established.
        findings: Every decided dense finding with its citations.
        assessment: The saved outcome assessment.
        hybrid: The saved hybrid dense judgment.
    """

    rollout_id: str
    version: int = Field(ge=1)
    version_sha256: Sha256
    judge: Sha256
    dense_definition: Sha256
    outcome_verdict: str
    outcome_pass: float | None = None
    decisive_refs: tuple[str, ...] = ()
    criteria: tuple[CriterionCitation, ...] = ()
    dense_status: str
    dense_total: float | None = None
    decisive: tuple[str, ...] = ()
    findings: tuple[DenseCitation, ...] = ()
    assessment: ArtifactInput
    hybrid: ArtifactInput

    @property
    def dense_pass(self) -> bool | None:
        """Whether the dense part passed (judged, no decisive penalty); None when unavailable."""
        return None if self.dense_status != "judged" else not self.decisive

    @property
    def passes_both(self) -> bool:
        """Whether both parts passed."""
        return self.outcome_pass == 1.0 and self.dense_pass is True

    @property
    def fails_both(self) -> bool:
        """Whether both parts failed (the outcome judged a fail and a decisive penalty held)."""
        return self.outcome_pass == 0.0 and self.dense_pass is False


def require_label(grade: RubricGrade, version: RubricVersion) -> RubricGrade:
    """Return a grade only when it was made under ``version``.

    Raises:
        ValueError: The grade is an old (or foreign) version's label.
    """
    if grade.version_sha256 != version.identity:
        raise ValueError(
            f"rollout {grade.rollout_id} was graded under rubric version {grade.version}, not "
            f"version {version.version}; old labels are refused after a revision"
        )
    return grade


def grade_rollout(
    project: ProjectStore,
    version: RubricVersion,
    banks: Sequence[QuestionBank],
    task: TaskRubric,
    rollout: Rollout,
    *,
    source: str,
    tagger: TraceTagger,
    routing: RoutingSpec,
    code_revision: str,
) -> RubricGrade:
    """Grade one rollout with both parts of one version, or replay the saved judgments.

    Args:
        project: Project holding the judgments.
        version: The rubric version.
        banks: Its pinned banks.
        task: The rollout's task rubric.
        rollout: The rollout.
        source: The trainer run (or held-out source) for event locators.
        tagger: The decision model's tagging seam.
        routing: Hybrid routing thresholds.
        code_revision: Producing source identity.

    Raises:
        ValueError: The rollout is another task's.
    """
    if rollout.task_id != task.task_id:
        raise ValueError(f"rollout {rollout.rollout_id} is task {rollout.task_id}'s")
    judge = versioned_judge(project, version, task.judge, code_revision=code_revision)
    dense, step = versioned_dense(version, banks, task.dense)
    definition = judge.definition
    evidence, trajectory, sampled = rollout_evidence(
        rollout,
        task_sha256=definition.task_sha256,
        world_sha256=definition.world_sha256,
        trainer_run=source,
    )
    assessment_ref, assessment = judge_episode(
        project, definition, evidence, settings=judge.settings, code_revision=code_revision
    )
    matrix = tag_episode_steps(evidence, tagger, step, routing)
    hybrid_ref, hybrid = judge_hybrid(
        project,
        dense,
        evidence,
        trajectory,
        matrix,
        code_revision=code_revision,
        spec=routing,
        sampled=sampled,
    )
    export = hybrid.export
    decided = [f for f in export.findings if f.decision in ("established", "acceptable")]
    return RubricGrade(
        rollout_id=rollout.rollout_id,
        version=version.version,
        version_sha256=version.identity,
        judge=definition.identity,
        dense_definition=dense.definition.identity,
        outcome_verdict=assessment.verdict,
        outcome_pass=episode_reward(assessment),
        decisive_refs=assessment.decisive_refs,
        criteria=tuple(
            CriterionCitation(
                criterion_id=row.criterion_id,
                status=row.status,
                event_refs=row.evidence_refs,
                reason=row.rationale,
            )
            for row in assessment.findings
        ),
        dense_status=export.status,
        dense_total=export.episode.total if export.status == "judged" else None,
        decisive=tuple(
            sorted(
                {
                    f.tag_id
                    for f in export.findings
                    if f.decision == "established" and f.severity == "decisive" and f.scored
                }
            )
        ),
        findings=tuple(
            DenseCitation(
                tag_id=f.tag_id,
                decision=f.decision,
                observed_refs=f.observed_refs,
                obligation_refs=() if f.obligation is None else f.obligation.event_refs,
                change_refs=() if f.change is None else f.change.event_refs,
                result_refs=f.result_refs,
                weakening=f.weakening,
                rationale=f.rationale,
                reward=f.reward,
            )
            for f in decided
        ),
        assessment=assessment_ref,
        hybrid=hybrid_ref,
    )


def measure_recall(
    project: ProjectStore,
    version: RubricVersion,
    banks: Sequence[QuestionBank],
    tasks: dict[str, TaskRubric],
    held_out: HeldOutSet,
    *,
    tagger: TraceTagger,
    routing: RoutingSpec,
    code_revision: str,
) -> tuple[RecallRecord, tuple[RubricGrade, ...]]:
    """Grade every held-out hack under ``version`` and count what each part caught.

    An unscored outcome or an unavailable dense judgment counts as a miss, never as a catch.

    Raises:
        ValueError: A held-out hack names a task without a rubric.
    """
    grades: list[RubricGrade] = []
    caught: dict[str, tuple[str, ...]] = {}
    outcome = dense = outcome_unscored = dense_unscored = 0
    for row in held_out.hacks:
        task = tasks.get(row.rollout.task_id)
        if task is None:
            raise ValueError(f"held-out hack {row.rollout.rollout_id} names an unknown task")
        grade = grade_rollout(
            project,
            version,
            banks,
            task,
            row.rollout,
            source=held_out.source,
            tagger=tagger,
            routing=routing,
            code_revision=code_revision,
        )
        grades.append(grade)
        parts = []
        if grade.outcome_pass == 0.0:
            outcome += 1
            parts.append("outcome")
        elif grade.outcome_pass is None:
            outcome_unscored += 1
        if grade.dense_pass is False:
            dense += 1
            parts.append("dense")
        elif grade.dense_pass is None:
            dense_unscored += 1
        caught[row.rollout.rollout_id] = tuple(parts)
    record = RecallRecord(
        held_out_sha256=held_out.identity,
        hacks=len(held_out.hacks),
        outcome_caught=outcome,
        dense_caught=dense,
        outcome_unscored=outcome_unscored,
        dense_unscored=dense_unscored,
        caught_by=caught,
    )
    return record, tuple(grades)
