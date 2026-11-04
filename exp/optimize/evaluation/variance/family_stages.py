"""The executed per-variant stages of a decision family, and rubric grading over its episodes.

Each scenario's variant goes through the same stages in order, stopping at the first gap with the
reason recorded on its :class:`FamilyVariant`:

1. **Variant.** A coherent variant checkpoint is built
   (:func:`~exp.optimize.evaluation.variance.variants.build_variant_checkpoint`) and proven to
   restore to its edited state.
2. **Replay.** A variant that forks before the decision is replayed to it by imposing the original
   trajectory's recorded calls
   (:func:`~exp.optimize.evaluation.variance.replay.replay_to_decision`); a replay that diverges
   stops the variant as ``replay_diverged``.
3. **Witness.** The scenario's witness calls execute in the restored decision world.
4. **Selection.** Every option is forced from the decision checkpoint, citing the saved witness as
   evidence.

Every contrast cell is judged by the variant's own frozen episode judge, fixed before the variant's
first contrast runs. When the variant's judge binding names a world freeze, the frozen variant is
what runs: the witness and every contrast start from the freeze's saved checkpoint under the job
rebound to the frozen runtime definition, and the edited decision checkpoint is coherence evidence
only.

A variant whose selection winner qualifies is contrasted again on fresh samples by
:func:`confirm_variant`, and :func:`grade_variants` reports the judge's coverage of the
confirmation episodes, with optional probabilistic criterion grades kept as calibration evidence
only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.variance.contracts import DecisionState, Scenario, StatePart
from exp.optimize.evaluation.variance.contrast import (
    OptionContrast,
    WitnessResult,
    run_observation_witness,
    run_option_contrast,
    witness_reference,
)
from exp.optimize.evaluation.variance.exposure import EstimateMeaning, estimate_meaning
from exp.optimize.evaluation.variance.grader import (
    GradedCriterion,
    GradeReport,
    GraderSettings,
    grade_episodes,
)
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgeCoverage
from exp.optimize.evaluation.variance.records import (
    bound_job,
    confirmation_episodes,
    confirmation_job,
)
from exp.optimize.evaluation.variance.replay import DecisionReplay, replay_to_decision
from exp.optimize.evaluation.variance.variants import (
    VariantCheckpoint,
    build_variant_checkpoint,
    verify_restoration,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger


class FamilyVariant(ContractModel):
    """One scenario's variant and how far it got.

    Attributes:
        scenario_id: The scenario.
        variant: The built variant checkpoint with every coherence check.
        restored: Whether the saved variant restores to its edited state, or None when unbuilt.
        replay: The replay to the decision, when the variant forks before it.
        witness: The executed observation witness, when one ran.
        witness_artifact: The saved witness the contrasts cite as evidence, when one ran.
        selection: The selection option contrast, when one ran.
        confirmation: The fresh confirmation option contrast, when one ran.
        meaning: What the variant's contrast outcomes are conditional on, when contrasted.
        frozen: The world freeze's saved start the variant's witness and contrasts ran from, when
            a freeze was bound.
        stopped: Why the variant went no further, when it stopped before confirmation; a replay
            that diverged stops it with a reason starting ``replay_diverged``.
    """

    scenario_id: str
    variant: VariantCheckpoint
    restored: bool | None = None
    replay: DecisionReplay | None = None
    witness: WitnessResult | None = None
    witness_artifact: ArtifactInput | None = None
    selection: OptionContrast | None = None
    confirmation: OptionContrast | None = None
    meaning: EstimateMeaning | None = None
    frozen: ArtifactInput | None = None
    stopped: str | None = None

    @property
    def decision_checkpoint(self) -> ArtifactInput | None:
        """The checkpoint contrasts start from: the frozen start, the replayed one, or the variant.

        None when the variant was not persisted or its replay diverged.
        """
        if self.frozen is not None:
            return self.frozen
        if self.replay is not None:
            return self.replay.checkpoint
        return self.variant.reference


def run_variant(
    project: ProjectStore,
    *,
    job: ContinuationJob,
    base: ArtifactInput,
    boundaries: tuple[ArtifactInput, ...],
    decision: DecisionState,
    scenario: Scenario,
    actions: Mapping[str, ToolCall],
    facts: InstanceFacts | None,
    repeats: int,
    judge: EpisodeJudge,
    code_revision: str,
    required_state: frozenset[StatePart] = frozenset(),
    frozen: ArtifactInput | None = None,
) -> FamilyVariant:
    """Build, prove, replay, witness and contrast one scenario's variant, stopping at a gap.

    Args:
        project: Project holding every saved step.
        job: Frozen job every witness and contrast runs under.
        base: The decision's checkpoint in the picked trajectory.
        boundaries: Every saved boundary of the picked attempt, start first.
        decision: The decision state.
        scenario: The scenario.
        actions: Concrete forced first action of every option.
        facts: Instance facts the world model was given, if any.
        repeats: Pairs per selection contrast.
        judge: The variant's frozen episode judge.
        code_revision: Producing source identity.
        required_state: State parts the variant must hold beyond the baseline, such as
            ``files``.
        frozen: The world freeze's saved start for this variant; ``job`` must already run its
            frozen runtime definition. The witness and contrasts then start from it, and the
            edited decision checkpoint is not replayed.

    Returns:
        The variant with every stage it reached.
    """
    variant = build_variant_checkpoint(
        project,
        base=base,
        branch_checkpoints=boundaries,
        decision=decision,
        scenario=scenario,
        facts=facts,
        code_revision=code_revision,
        maximum_model_calls=job.maximum_model_calls,
        required_state=required_state,
    )
    row = FamilyVariant(scenario_id=scenario.scenario_id, variant=variant)
    reference = variant.reference
    if reference is None:
        return row.model_copy(update={"stopped": "variant: " + "; ".join(variant.problems)})
    restored = verify_restoration(
        project, reference, expected_state_sha256=variant.edited_state_sha256
    )
    row = row.model_copy(update={"restored": restored})
    if not restored:
        return row.model_copy(update={"stopped": "variant does not restore to its edited state"})
    if frozen is not None:
        row = row.model_copy(update={"frozen": frozen})
        reference = frozen
        if not verify_restoration(project, reference):
            return row.model_copy(update={"stopped": "the frozen start does not restore"})
    elif variant.replay_turns:
        try:
            replay = replay_to_decision(
                project,
                variant=variant,
                branch_checkpoints=boundaries,
                scenario=scenario,
                job=job,
                code_revision=code_revision,
            )
        except (ValueError, RuntimeError, OSError) as error:
            return row.model_copy(update={"stopped": f"replay: {failure_reason(error)}"})
        row = row.model_copy(update={"replay": replay})
        if replay.checkpoint is None:
            return row.model_copy(
                update={"stopped": "replay_diverged: " + "; ".join(replay.problems)}
            )
        reference = replay.checkpoint
        if not verify_restoration(project, reference):
            return row.model_copy(update={"stopped": "replayed decision does not restore"})
    try:
        witness = run_observation_witness(
            project,
            checkpoint=reference,
            scenario=scenario,
            decision=decision,
            job=job,
            code_revision=code_revision,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"stopped": f"witness: {failure_reason(error)}"})
    row = row.model_copy(update={"witness": witness})
    if not witness.holds:
        return row.model_copy(update={"stopped": "witness: " + "; ".join(witness.problems)})
    cited = witness_reference(
        project, checkpoint=reference, scenario=scenario, job=job, code_revision=code_revision
    )
    row = row.model_copy(update={"witness_artifact": cited})
    try:
        selection = run_option_contrast(
            project,
            checkpoint=reference,
            decision=decision,
            actions=actions,
            job=job,
            repeats=repeats,
            judge=judge,
            code_revision=code_revision,
            evidence=(cited,),
        )
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"stopped": f"selection contrast: {failure_reason(error)}"})
    meaning = estimate_meaning(
        bound_job(job, reference),
        load_continuation_checkpoint(project.artifacts, reference),
        judge=judge.definition,
    )
    return row.model_copy(update={"selection": selection, "meaning": meaning})


def confirm_variant(
    project: ProjectStore,
    *,
    job: ContinuationJob,
    decision: DecisionState,
    row: FamilyVariant,
    actions: Mapping[str, ToolCall],
    seed: int,
    repeats: int,
    judge: EpisodeJudge,
    code_revision: str,
) -> FamilyVariant:
    """Contrast a qualifying variant again on fresh samples, citing the selection as evidence.

    Args:
        project: Project holding the saved contrasts.
        job: The frozen job selection ran under.
        decision: The decision state.
        row: A variant whose selection winner separated and matched.
        actions: Concrete forced first action of every option.
        seed: Confirmation seed named in every confirmation draw's selector.
        repeats: Pairs per confirmation contrast.
        judge: The variant's frozen episode judge, the same one selection used.
        code_revision: Producing source identity.

    Returns:
        The variant with its confirmation, or why it has none.
    """
    checkpoint = row.decision_checkpoint
    if checkpoint is None or row.selection is None:
        return row
    cited = (
        *row.selection.contrasts,
        *(() if row.witness_artifact is None else (row.witness_artifact,)),
    )
    try:
        confirmation = run_option_contrast(
            project,
            checkpoint=checkpoint,
            decision=decision,
            actions=actions,
            job=confirmation_job(job, seed),
            repeats=repeats,
            judge=judge,
            code_revision=code_revision,
            evidence=cited,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"stopped": f"confirmation contrast: {failure_reason(error)}"})
    return row.model_copy(update={"confirmation": confirmation})


def grade_variants(
    project: ProjectStore,
    decision: DecisionState,
    scenarios: Sequence[Scenario],
    variants: Sequence[FamilyVariant],
    grader: DecisionsTraceTagger | None,
    settings: GraderSettings,
) -> tuple[dict[str, JudgeCoverage], dict[str, GradeReport], tuple[str, ...], str | None]:
    """Report the judge's coverage of each confirmed variant, and optional calibration grades.

    The episode judge already labelled every confirmation cell; this counts, per variant, how
    many were judged pass or fail, unassessable, malformed, unavailable or without evidence, and
    how many carried insufficient evidence or a rubric problem. When a criterion grader is
    supplied, its probabilistic criterion grades are calibration evidence only: an uncertain
    criterion is never treated as proof of an ambiguous rubric and never decides eligibility.

    Args:
        project: Project holding the saved contrasts.
        decision: The decision state declaring the criteria.
        scenarios: Every scenario, for its applicable criteria.
        variants: Every variant.
        grader: Calibration criterion grader, or None to skip it.
        settings: Calibration grading controls.

    Returns:
        Judge coverage by confirmed scenario, calibration reports by scenario, criteria the
        calibration grader found uncertain, and a calibration grading failure if any.
    """
    coverage = {
        row.scenario_id: row.confirmation.coverage
        for row in variants
        if row.confirmation is not None
    }
    if grader is None:
        return coverage, {}, (), None
    criteria = {criterion.criterion_id: criterion for criterion in decision.criteria}
    applies = {scenario.scenario_id: scenario.criteria for scenario in scenarios}
    reports: dict[str, GradeReport] = {}
    failure: str | None = None
    try:
        for row in variants:
            if row.confirmation is None:
                continue
            episodes = confirmation_episodes(project, row.scenario_id, row.confirmation)
            if not episodes:
                continue
            reports[row.scenario_id] = grade_episodes(
                grader,
                episodes,
                [GradedCriterion(criterion=criteria[c]) for c in applies[row.scenario_id]],
                settings=settings,
            )
    except ValueError as error:
        failure = f"calibration grading: {failure_reason(error)}"
    uncertain = {s.criterion_id for r in reports.values() for s in r.summaries if s.ambiguous}
    return coverage, reports, tuple(sorted(uncertain)), failure


def failure_reason(error: BaseException) -> str:
    """Return a bounded failure reason naming the error type."""
    return (type(error).__name__ + ": " + str(error))[:2000]
