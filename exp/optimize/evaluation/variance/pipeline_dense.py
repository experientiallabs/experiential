"""Frozen dense task bindings and confirmation records for the decision-variance pipeline.

Settings freeze the training vocabulary before records or executable qualification. Each task
gets a definition bound to its exact outcome judge's task and world. Confirmation assessments
reuse the canonical contrast evidence and the suffix's recorded model turns; imposed actions
stay outside the actor trajectory, and no sampled-token evidence is manufactured.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.action_contrast import ActionContrast
from exp.optimize.evaluation.continuations.intervention import ActionIntervention
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.executable.task import ExecutableTask
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.behavior_profiles import BehaviorGuidance
from exp.optimize.evaluation.variance.dense_candidates import (
    DenseCandidate,
    definition_candidates,
    error_tag_candidates,
    linter_candidates,
    merge_candidates,
)
from exp.optimize.evaluation.variance.dense_judge import DenseRewardJudge, judge_dense
from exp.optimize.evaluation.variance.dense_prompt import (
    dense_model_calls,
    dense_prompt_sha256,
    dense_schema_sha256,
)
from exp.optimize.evaluation.variance.dense_records import attach_dense
from exp.optimize.evaluation.variance.dense_reward import (
    DenseBounds,
    DenseJudgeProfile,
    DenseRewardDefinition,
    DenseTag,
    OutcomeReference,
)
from exp.optimize.evaluation.variance.dense_spans import ActorTrajectory, trajectory_from_branch
from exp.optimize.evaluation.variance.episode_evidence import evidence_from_contrast_cell
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, episode_reward
from exp.optimize.evaluation.variance.judge import _load_rubric, load_assessment
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.optimize.evaluation.variance.records import ContinuationGroup, TrainingRecords
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.error_tags import ErrorTraceTags

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.family import FamilyReport


class DenseSettings(ContractModel):
    """Frozen settings shared by each task's separately bound dense definition.

    Attributes:
        tags: Immutable behavior vocabulary and semantic grading instructions.
        bounds: Bounded credit and episode aggregation settings.
        profile: Exact dense prompts, model and request limits.
        settings: Admitted per-call settings matching the profile.
    """

    tags: tuple[DenseTag, ...] = Field(min_length=1, max_length=64)
    bounds: DenseBounds = DenseBounds()
    profile: DenseJudgeProfile
    settings: AuthorSettings

    @model_validator(mode="after")
    def _bound(self) -> Self:
        """Refuse inconsistent settings and invalid tag bounds before any provider dispatch."""
        profile, settings = self.profile, self.settings
        if (
            profile.model_alias != settings.model_alias
            or profile.model != settings.model
            or profile.maximum_output_tokens != settings.output_tokens
            or profile.reasoning_effort != settings.reasoning_effort
            or profile.maximum_input_bytes != settings.maximum_input_bytes
        ):
            raise ValueError("dense settings must match the frozen profile's model and limits")
        if profile.prompt_sha256 != dense_prompt_sha256(profile.prompt_variants) or (
            profile.output_schema_sha256 != dense_schema_sha256()
        ):
            raise ValueError("dense profile must bind the current prompts and schemas")
        self.bind(task_sha256="0" * 64, world_sha256="0" * 64)
        return self

    def bind(self, *, task_sha256: Sha256, world_sha256: Sha256) -> DenseRewardJudge:
        """Return one exact task/world definition without changing the shared vocabulary."""
        return DenseRewardJudge(
            definition=DenseRewardDefinition(
                tags=self.tags,
                bounds=self.bounds,
                profile=self.profile,
                task_sha256=task_sha256,
                world_sha256=world_sha256,
            ),
            settings=self.settings,
        )


def bind_dense_task(task: ExecutableTask, settings: DenseSettings) -> ExecutableTask:
    """Bind dense grading before persisting or qualifying this exact executable task.

    Raises:
        ValueError: The task lacks an outcome judge or already freezes another dense reward.
    """
    if task.judge is None:
        raise ValueError("dense task binding needs the task's frozen outcome judge")
    judge = task.judge.definition
    dense = settings.bind(task_sha256=judge.task_sha256, world_sha256=judge.world_sha256)
    if task.dense is not None and task.dense != dense:
        raise ValueError("the task already binds another dense reward; register a fresh task")
    return task.model_copy(update={"dense": dense})


def dense_costs(
    project: ProjectStore, settings: DenseSettings, episodes: int
) -> tuple[CostComponent, ...]:
    """Price every variant, evidence read and repair draw for a finite episode count."""
    if episodes < 0:
        raise ValueError("dense episode count cannot be negative")
    if not episodes:
        return ()
    calls = episodes * dense_model_calls(settings.profile)
    return tuple(row.scaled(calls) for row in evaluation_costs(project, settings.settings))


@dataclass(frozen=True)
class _Confirmation:
    """Verified inputs of one confirmation episode's dense assessment.

    Attributes:
        scenario: The variant whose frozen definition applies.
        evidence: Canonical imposed-action plus suffix evidence, identical to outcome evidence.
        trajectory: Only the suffix's actual actor model turns.
        outcome: Unchanged outcome assessment and its reward.
    """

    scenario: str
    evidence: EpisodeEvidence
    trajectory: ActorTrajectory
    outcome: OutcomeReference


def attach_family_dense(
    project: ProjectStore,
    family: FamilyReport,
    settings: DenseSettings,
    *,
    code_revision: str,
    error_tags: Mapping[str, ErrorTraceTags] | None = None,
) -> tuple[TrainingRecords, tuple[ArtifactInput, ...]]:
    """Judge confirmation episodes and attach their dense signal beside unchanged advantages.

    Args:
        project: Project holding the family's exact confirmation and outcome artifacts.
        family: Eligible family with actual confirmation training records.
        settings: Vocabulary and admitted dense judge settings fixed before this records stage.
        code_revision: Producing revision used by the dense assessment owner.
        error_tags: Optional saved error-tag summaries keyed by the exact episode identity.

    Returns:
        Updated training records and every dense assessment reference they cite.

    Raises:
        ValueError: Records, frozen judges, confirmation ancestry or outcome evidence disagree.
    """
    if family.records is None or not family.eligibility.eligible:
        raise ValueError("dense training records require an eligible family's confirmation records")
    records = family.records
    scenarios = {group.scenario_id for groups in records.groups.values() for group in groups}
    judges = _judges(family, settings, scenarios)
    confirmations = _confirmations(project, family, records, judges)
    refs: dict[str, ArtifactInput] = {}
    for item in confirmations:
        judge = judges[item.scenario]
        outcome = (family.judging or {})[item.scenario].judge
        candidates = _candidates(
            project, judge, outcome, (error_tags or {}).get(item.evidence.episode_id)
        )
        reference, _ = judge_dense(
            project,
            judge,
            item.evidence,
            candidates,
            item.trajectory,
            code_revision=code_revision,
            outcome=item.outcome,
        )
        refs[item.evidence.episode_id] = reference
    groups: dict[str, tuple[ContinuationGroup, ...]] = {}
    for tier, rows in records.groups.items():
        updated: list[ContinuationGroup] = []
        for group in rows:
            one = records.model_copy(update={"groups": {tier: (group,)}})
            attached = attach_dense(
                project, one, refs, definition_sha256=judges[group.scenario_id].identity
            )
            updated.append(attached.groups[tier][0])
        groups[tier] = tuple(updated)
    return records.model_copy(update={"groups": groups}), tuple(
        sorted(set(refs.values()), key=lambda ref: ref.artifact_id)
    )


def _judges(
    family: FamilyReport, settings: DenseSettings, scenarios: set[str]
) -> dict[str, DenseRewardJudge]:
    """Bind one dense definition per recorded variant's frozen outcome task and world."""
    result: dict[str, DenseRewardJudge] = {}
    for scenario in scenarios:
        binding = (family.judging or {}).get(scenario)
        if binding is None:
            raise ValueError(f"recorded scenario {scenario} has no frozen outcome judge")
        definition = binding.judge.definition
        result[scenario] = settings.bind(
            task_sha256=definition.task_sha256, world_sha256=definition.world_sha256
        )
    return result


def _confirmations(
    project: ProjectStore,
    family: FamilyReport,
    records: TrainingRecords,
    judges: Mapping[str, DenseRewardJudge],
) -> tuple[_Confirmation, ...]:
    """Load and verify every recorded confirmation before the first dense provider call."""
    outcomes = {
        slot.episode.episode_id: (group.scenario_id, group.judge, slot.episode.assessment)
        for groups in records.groups.values()
        for group in groups
        for slot in group.episodes
        if slot.episode is not None
    }
    found: dict[str, _Confirmation] = {}
    store = project.artifacts
    for variant in family.variants:
        if variant.scenario_id not in judges or variant.confirmation is None:
            continue
        for reference in variant.confirmation.contrasts:
            if artifact_input(store.read(reference.artifact_id).manifest) != reference:
                raise ValueError("confirmation contrast differs from its frozen reference")
            contrast = ActionContrast.model_validate_json(
                store.read_bytes(reference.artifact_id, "report.json")
            )
            checkpoint = load_continuation_checkpoint(store, contrast.spec.base.study.checkpoint)
            prefix = reference.artifact_id.removesuffix("-result")
            for index, cell in enumerate(contrast.cells):
                episode_id = f"{prefix}-cell-{index}"
                expected = outcomes.get(episode_id)
                if expected is None:
                    continue
                scenario, outcome_judge, assessment_ref = expected
                if scenario != variant.scenario_id:
                    raise ValueError("a confirmation episode is assigned to another scenario")
                binding = (family.judging or {})[scenario].judge
                assessment = load_assessment(project, assessment_ref)
                if assessment.judge != binding.identity or outcome_judge != binding.identity:
                    raise ValueError("confirmation records name another outcome judge")
                if cell.intervention is not None and (
                    artifact_input(store.read(cell.intervention.artifact_id).manifest)
                    != cell.intervention
                ):
                    raise ValueError("confirmation intervention differs from its frozen reference")
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
                    contrast.spec,
                    checkpoint,
                    intervention=intervention,
                    suffix=suffix,
                    episode_id=episode_id,
                    contrast_source=prefix,
                    checkpoint_source=contrast.spec.base.study.checkpoint.artifact_id,
                    intervention_source=None
                    if cell.intervention is None
                    else cell.intervention.artifact_id,
                    suffix_source=None if cell.study is None else cell.study.artifact_id,
                )
                assessment.check_citations(evidence)
                definition = judges[scenario].definition
                if (evidence.task_sha256, evidence.world_sha256) != (
                    definition.task_sha256,
                    definition.world_sha256,
                ):
                    raise ValueError("confirmation ran another task or world than its frozen judge")
                found[episode_id] = _Confirmation(
                    scenario=scenario,
                    evidence=evidence,
                    trajectory=ActorTrajectory(kind="continuation-branch", turns=())
                    if suffix is None
                    else trajectory_from_branch(suffix),
                    outcome=OutcomeReference(
                        judge=binding.identity,
                        assessment=assessment_ref,
                        verdict=assessment.verdict,
                        reward=episode_reward(assessment),
                    ),
                )
    missing = sorted(set(outcomes) - set(found))
    if missing:
        raise ValueError(f"training episodes lack exact confirmation evidence: {missing}")
    return tuple(found.values())


def _candidates(
    project: ProjectStore,
    judge: DenseRewardJudge,
    outcome: EpisodeJudge,
    error_tags: ErrorTraceTags | None,
) -> tuple[DenseCandidate, ...]:
    """Gather hypotheses from frozen tags, saved episode tags and authored rubric linters."""
    definition = judge.definition
    rubric = _load_rubric(project, outcome.definition)
    raw = rubric.content.get("behavior_guidance", [])
    if not isinstance(raw, list):
        raise ValueError("frozen rubric behavior_guidance must be a list")
    guidance = tuple(BehaviorGuidance.model_validate(row) for row in raw)
    return merge_candidates(
        definition_candidates(definition),
        () if error_tags is None else error_tag_candidates(definition, error_tags),
        linter_candidates(definition, guidance),
    )
