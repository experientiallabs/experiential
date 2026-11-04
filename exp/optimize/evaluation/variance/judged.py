"""Judged labels: every saved continuation and contrast cell, judged once by one frozen judge.

This module is the only place on the decision-variance path that turns saved episodes into
success labels. Each continuation branch (:func:`judge_study`) and each action-contrast cell
(:func:`judge_cells`) becomes one
:class:`~exp.optimize.evaluation.variance.episodes.EpisodeEvidence` that
:func:`~exp.optimize.evaluation.variance.judge.judge_episode` assesses under one
:class:`EpisodeJudge`. Records keep a :class:`JudgedSlot` per planned episode: a reference to the
saved assessment (:class:`~exp.optimize.evaluation.variance.episodes.JudgedEpisode`), or the
reason no evidence could be built. They never copy a success flag; :func:`slot_rewards` reads
every label back from its saved assessment through
:func:`~exp.optimize.evaluation.variance.episodes.episode_reward`.

- A contrast cell is judged once as the imposed action (owner ``experiment``, scope ``scored``)
  followed by its suffix, so a clean suffix cannot hide an unsafe imposed action.
- An operationally incomplete episode is still judged: a decisive observed violation is a
  failure, and missing evidence the judge needs leaves it unassessable.
- Unscored outcomes (unassessable, malformed, unavailable, or no evidence) stay in every
  denominator as explicit coverage gaps (:class:`JudgeCoverage`); none is ever 0.5 and none is
  dropped.

The mechanical verifier grade the continuation owners still compute is simulator and protocol
diagnostics only; nothing here reads it.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import ActionContrast
from exp.optimize.evaluation.continuations.intervention import ActionIntervention
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.episode_evidence import (
    evidence_from_branch,
    evidence_from_contrast_cell,
)
from exp.optimize.evaluation.variance.episodes import (
    EpisodeAssessment,
    EpisodeEvidence,
    JudgeDefinition,
    JudgedEpisode,
    episode_reward,
)
from exp.optimize.evaluation.variance.judge import judge_episode, load_assessment
from exp.simulation.continuations import load_continuation_checkpoint

JUDGED_LABELS_PROTOCOL = "judged-labels-v1"
"""Version of how saved episodes become judged evidence and labels; bump when that changes."""

JUDGED_STUDY_TYPE = "judged-continuation-study"
"""Artifact type of a saved :class:`JudgedStudy`."""

RUBRIC_PROBLEM = "rubric problem: "
"""Prefix the judge owner gives a rubric problem among an assessment's limitations."""


class EpisodeJudge(ContractModel):
    """One frozen judge definition and the admitted request settings that apply it.

    Attributes:
        definition: The frozen judge profile bound to its rubric, event contract, task and world.
        settings: Admitted request settings; they must match the definition's profile.
    """

    definition: JudgeDefinition
    settings: AuthorSettings

    @property
    def identity(self) -> str:
        """The judge definition identity every label of this judge carries."""
        return self.definition.identity


class JudgedSlot(ContractModel):
    """One planned episode: its saved assessment, or why no evidence could be judged.

    Attributes:
        position: Branch index, or cell index within its contrast.
        episode: Reference to the saved assessment, when evidence was built and judged.
        gap: Why there is no assessment, exactly when ``episode`` is None.
    """

    position: int = Field(ge=0)
    episode: JudgedEpisode | None = None
    gap: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _one(self) -> Self:
        """Require exactly one of an assessment reference and a gap."""
        if (self.episode is None) == (self.gap is None):
            raise ValueError("a judged slot holds either an assessment or a gap, never both")
        return self


class JudgedStudy(ContractModel):
    """Every branch of one saved continuation study, judged by one frozen judge.

    Attributes:
        protocol: Judged-label protocol.
        study: The saved continuation study.
        judge: Judge definition identity.
        branches: One slot per planned branch, in branch order.
    """

    protocol: str = JUDGED_LABELS_PROTOCOL
    study: ArtifactInput
    judge: Sha256
    branches: tuple[JudgedSlot, ...]


class JudgedCell(ContractModel):
    """One action-contrast cell judged once as its imposed action plus its suffix.

    Attributes:
        contrast: The saved action-contrast result holding the cell.
        pair: Registered replicate number.
        arm: Baseline or alternative imposed decision.
        slot: The cell's assessment or gap.
    """

    contrast: ArtifactInput
    pair: int = Field(ge=0)
    arm: Literal["baseline", "alternative"]
    slot: JudgedSlot


class JudgeCoverage(ContractModel):
    """How completely one judge labelled a set of planned episodes.

    Attributes:
        judge: Judge definition identity, or None for an empty set.
        planned: Planned episodes, the denominator.
        passed: Judged pass.
        failed: Judged fail.
        unassessable: Judged, but the verdict was unassessable.
        malformed: The judge's response failed validation.
        unavailable: No judge response was obtained.
        gaps: No evidence could be built for judging.
        insufficient_evidence: Judged episodes with at least one insufficient-evidence finding.
        rubric_problems: Judged episodes whose judge reported a rubric problem.
    """

    judge: Sha256 | None = None
    planned: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    unassessable: int = Field(default=0, ge=0)
    malformed: int = Field(default=0, ge=0)
    unavailable: int = Field(default=0, ge=0)
    gaps: int = Field(default=0, ge=0)
    insufficient_evidence: int = Field(default=0, ge=0)
    rubric_problems: int = Field(default=0, ge=0)

    @property
    def scored(self) -> int:
        """Episodes with a pass or fail label."""
        return self.passed + self.failed

    @property
    def unscored(self) -> int:
        """Planned episodes without a pass or fail label."""
        return self.planned - self.scored

    def plus(self, other: JudgeCoverage) -> JudgeCoverage:
        """Add another coverage of the same judge.

        Args:
            other: Coverage to add.

        Returns:
            The summed coverage.

        Raises:
            ValueError: The two coverages come from different judges.
        """
        if None not in (self.judge, other.judge) and self.judge != other.judge:
            raise ValueError("coverage of two different judges cannot be pooled")
        counts = {
            name: getattr(self, name) + getattr(other, name)
            for name in type(self).model_fields
            if name != "judge"
        }
        return JudgeCoverage(judge=self.judge or other.judge, **counts)


def judged_episode(project: ProjectStore, slot: JudgedSlot, judge: str) -> EpisodeAssessment | None:
    """Load one slot's saved assessment, refusing one made by another judge.

    Args:
        project: Project holding the assessment.
        slot: The slot.
        judge: Expected judge definition identity.

    Returns:
        The assessment, or None for a gap.

    Raises:
        ValueError: The reference or its assessment belongs to a different judge or episode.
    """
    if slot.episode is None:
        return None
    assessment = load_assessment(project, slot.episode.assessment)
    if slot.episode.judge != judge or assessment.judge != judge:
        raise ValueError(f"episode {slot.episode.episode_id} was judged by a different judge")
    if assessment.episode_id != slot.episode.episode_id:
        raise ValueError(f"assessment {slot.episode.assessment.artifact_id} is another episode's")
    return assessment


def slot_rewards(
    project: ProjectStore, slots: Sequence[JudgedSlot], judge: str
) -> tuple[float | None, ...]:
    """Read every slot's label from its saved assessment: 1, 0, or None when unscored.

    Args:
        project: Project holding the assessments.
        slots: Planned episodes in order.
        judge: Expected judge definition identity.

    Returns:
        One reward per slot, None for every unscored episode.
    """
    rewards: list[float | None] = []
    for slot in slots:
        assessment = judged_episode(project, slot, judge)
        rewards.append(None if assessment is None else episode_reward(assessment))
    return tuple(rewards)


def slot_coverage(project: ProjectStore, slots: Sequence[JudgedSlot], judge: str) -> JudgeCoverage:
    """Count how each planned episode was or was not labelled.

    Args:
        project: Project holding the assessments.
        slots: Planned episodes.
        judge: Expected judge definition identity.

    Returns:
        The coverage, every planned episode counted once.
    """
    counts = dict.fromkeys(
        (
            "passed",
            "failed",
            "unassessable",
            "malformed",
            "unavailable",
            "gaps",
            "insufficient_evidence",
            "rubric_problems",
        ),
        0,
    )
    for slot in slots:
        assessment = judged_episode(project, slot, judge)
        if assessment is None:
            counts["gaps"] += 1
            continue
        if assessment.status != "judged":
            counts[assessment.status] += 1
            continue
        verdict = {"pass": "passed", "fail": "failed"}.get(assessment.verdict, "unassessable")
        counts[verdict] += 1
        if any(f.status == "insufficient_evidence" for f in assessment.findings):
            counts["insufficient_evidence"] += 1
        if any(item.startswith(RUBRIC_PROBLEM) for item in assessment.limitations):
            counts["rubric_problems"] += 1
    return JudgeCoverage(judge=judge, planned=len(slots), **counts)


def judged_metrics(
    project: ProjectStore, judged: JudgedStudy, predicted_success: float | None
) -> JsonObject:
    """Report a study's judged rate and Brier score, with every unscored branch counted.

    Args:
        project: Project holding the assessments.
        judged: The judged study.
        predicted_success: A critic estimate registered before the branches ran, if any.

    Returns:
        Planned, scored and unscored counts, the judged success rate over scored branches, the
        Brier score of the prediction against the judged labels, and the judge identity.
    """
    rewards = slot_rewards(project, judged.branches, judged.judge)
    labels = [reward for reward in rewards if reward is not None]
    coverage = slot_coverage(project, judged.branches, judged.judge)
    return {
        "judge": judged.judge,
        "planned": len(rewards),
        "assessable": len(labels),
        "unassessable": len(rewards) - len(labels),
        "coverage": coverage.model_dump(mode="json"),
        "successes": int(sum(labels)),
        "resumed_success_rate": sum(labels) / len(labels) if labels else None,
        "brier": (
            sum((predicted_success - label) ** 2 for label in labels) / len(labels)
            if predicted_success is not None and labels
            else None
        ),
        "label_authority": "frozen LLM episode judge; the verifier grade is diagnostic only",
    }


def judge_study(
    project: ProjectStore, reference: ArtifactInput, judge: EpisodeJudge, *, code_revision: str
) -> tuple[ArtifactInput, JudgedStudy]:
    """Judge every branch of one saved continuation study once, or replay the saved judging.

    Args:
        project: Project holding the study, the rubric and the assessments.
        reference: The saved continuation study result.
        judge: The frozen judge and its admitted settings.
        code_revision: Producing source identity.

    Returns:
        The saved ``judged-continuation-study`` reference and its record.

    Raises:
        ValueError: The study or judge binding is invalid, or a saved record differs.
    """
    store = project.artifacts
    identity = stable_id(
        JUDGED_STUDY_TYPE,
        {
            "protocol": JUDGED_LABELS_PROTOCOL,
            "judge": judge.identity,
            "study": reference.model_dump(mode="json"),
        },
    )
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        if store.exists(identity):
            saved = JudgedStudy.model_validate_json(store.read_bytes(identity, "judged.json"))
            if saved.study != reference or saved.judge != judge.identity:
                raise ValueError(f"saved judged study {identity} judges something else")
            return artifact_input(store.read(identity).manifest), saved
        study = load_study(store, reference)
        checkpoint = load_continuation_checkpoint(store, study.spec.checkpoint)
        prefix = reference.artifact_id.removesuffix("-result")
        slots: list[JudgedSlot] = []
        for branch in study.branches:
            source = f"{prefix}-branch-{branch.index}"
            try:
                evidence = evidence_from_branch(
                    branch,
                    checkpoint,
                    episode_id=source,
                    task_sha256=judge.definition.task_sha256,
                    world_sha256=judge.definition.world_sha256,
                    checkpoint_source=study.spec.checkpoint.artifact_id,
                    branch_source=source,
                )
            except ValueError as error:
                slots.append(JudgedSlot(position=branch.index, gap=_reason(error)))
                continue
            slots.append(_judge(project, judge, evidence, branch.index, code_revision))
        judged = JudgedStudy(study=reference, judge=judge.identity, branches=tuple(slots))
        return _save(project, identity, judged, (reference,), code_revision), judged


def judge_cells(
    project: ProjectStore,
    contrast: ActionContrast,
    reference: ArtifactInput,
    judge: EpisodeJudge,
    *,
    code_revision: str,
) -> tuple[JudgedCell, ...]:
    """Judge every cell of one saved action contrast once: its imposed action, then its suffix.

    Args:
        project: Project holding the contrast and its children.
        contrast: The saved action contrast.
        reference: Its saved result reference.
        judge: The frozen judge and its admitted settings.
        code_revision: Producing source identity.

    Returns:
        One judged cell per planned cell, in plan order.
    """
    store = project.artifacts
    spec = contrast.spec
    checkpoint = load_continuation_checkpoint(store, spec.base.study.checkpoint)
    prefix = reference.artifact_id.removesuffix("-result")
    cells: list[JudgedCell] = []
    for index, cell in enumerate(contrast.cells):
        episode_id = f"{prefix}-cell-{index}"
        try:
            intervention = (
                None
                if cell.intervention is None
                else ActionIntervention.model_validate_json(
                    store.read_bytes(cell.intervention.artifact_id, "intervention.json")
                )
            )
            suffix = None if cell.study is None else load_study(store, cell.study).branches[0]
            evidence = evidence_from_contrast_cell(
                cell,
                spec,
                checkpoint,
                intervention=intervention,
                suffix=suffix,
                episode_id=episode_id,
                task_sha256=judge.definition.task_sha256,
                world_sha256=judge.definition.world_sha256,
                contrast_source=prefix,
                checkpoint_source=spec.base.study.checkpoint.artifact_id,
                intervention_source=None
                if cell.intervention is None
                else cell.intervention.artifact_id,
                suffix_source=None if cell.study is None else cell.study.artifact_id,
            )
        except ValueError as error:
            slot = JudgedSlot(position=index, gap=_reason(error))
        else:
            slot = _judge(project, judge, evidence, index, code_revision)
        cells.append(JudgedCell(contrast=reference, pair=cell.pair, arm=cell.arm, slot=slot))
    return tuple(cells)


def assessment_inputs(slots: Sequence[JudgedSlot]) -> tuple[ArtifactInput, ...]:
    """Return every saved assessment the slots reference, for a record's provenance."""
    return tuple(slot.episode.assessment for slot in slots if slot.episode is not None)


def _judge(
    project: ProjectStore,
    judge: EpisodeJudge,
    evidence: EpisodeEvidence,
    position: int,
    code_revision: str,
) -> JudgedSlot:
    """Judge one episode and return its slot; binding errors propagate before any draw."""
    reference, _ = judge_episode(
        project,
        judge.definition,
        evidence,
        settings=judge.settings,
        code_revision=code_revision,
    )
    return JudgedSlot(
        position=position,
        episode=JudgedEpisode(
            episode_id=evidence.episode_id, assessment=reference, judge=judge.identity
        ),
    )


def _save(
    project: ProjectStore,
    identity: str,
    judged: JudgedStudy,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Save a judged study citing the study and every assessment it references."""
    refs = {*inputs, *assessment_inputs(judged.branches)}
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=JUDGED_STUDY_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(refs, key=lambda ref: ref.artifact_id)),
        ),
        files={"judged.json": judged},
    )
    return artifact_input(manifest)


def _reason(error: BaseException) -> str:
    """Return a bounded reason naming the error type."""
    return (type(error).__name__ + ": " + str(error))[:2000]
