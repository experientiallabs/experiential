"""One decision family, from a judged point to computed training eligibility.

A run follows a fixed sequence over saved, replayable owners, starting from one chosen point of a
saved actor-critic run over a mining-probe task:

1. **Context.** The point's checkpoint, the public conversation the actor had seen there, the
   evaluator state of the restored world before the step, and the rules and open questions about
   the actor's tools are saved as one ``decision-context`` artifact. No model is called.
2. **Decision state.** One forced decision-state draw turns the point into options, axes,
   invariants and criteria
   (:func:`~exp.optimize.evaluation.variance.authors.author_decision_state`).
3. **Scenarios.** One forced scenario draw varies it; a set with any structural problem stops the
   family.
4. **Variants.** Each scenario becomes a coherent variant checkpoint, proven to restore to its
   edited state. A variant forked before the decision is kept but not contrasted, because forcing
   an option there would not force this decision.
5. **Witnesses.** Each variant's witness calls execute in its restored world; a variant whose
   witness fails is not contrasted.
6. **Selection.** Every option is forced from each variant's checkpoint under the frozen job, and
   :func:`~exp.optimize.evaluation.variance.contrast.family_flip` decides whether the better option
   flips.
7. **Fresh confirmation.** Each variant that qualified is contrasted again with independently drawn
   samples: the job's selector names the confirmation seed and the selection contrasts are cited as
   evidence, so every confirmation draw has a new identity and none of its samples took part in
   selection. Qualification stands only when every qualifying variant's confirmation reproduces
   its separated, matching winner.
8. **Rubric.** When a grader is supplied, the decision state's criteria are graded over the
   confirmation episodes; ambiguous criteria are listed for rubric authoring, not trained on.
9. **Eligibility.** Every condition in :mod:`~exp.optimize.evaluation.variance.records` is computed,
   and records are built only when all hold.

Option identities are authored, so the concrete forced action of each option comes either from the
spec's mapping by option identity or from a caller-supplied function of the decision state; the
mapping used is saved in the report. Every stage result is saved by its owner, the family report
is saved as a ``decision-family-report`` artifact and replayed exactly, failures are kept and never
redrawn, and adding audit dispositions makes a new report while every draw replays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    load_continuation_runtime,
    preflight_continuation_job,
)
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import restore_continuation_world
from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.task import ExecutableTask, persist_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageReport, load_rulebook
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.authors import (
    DecisionContext,
    DecisionStateDraft,
    ScenarioDraft,
    author_decision_state,
    author_scenarios,
    authors_protocol_sha256,
    relevant_questions,
    relevant_rules,
)
from exp.optimize.evaluation.variance.contracts import (
    DecisionState,
    EstimationPoint,
    Scenario,
    ScenarioLimits,
)
from exp.optimize.evaluation.variance.contrast import (
    FlipVerdict,
    OptionContrast,
    WitnessResult,
    family_flip,
    run_observation_witness,
    run_option_contrast,
)
from exp.optimize.evaluation.variance.coverage import uncovered_tools
from exp.optimize.evaluation.variance.exposure import (
    EstimateMeaning,
    ExposurePlan,
    estimate_meaning,
)
from exp.optimize.evaluation.variance.grader import (
    GradedCriterion,
    GradeReport,
    GraderSettings,
    grade_episodes,
)
from exp.optimize.evaluation.variance.records import (
    AuditDisposition,
    AuditFinding,
    TrainingEligibility,
    TrainingRecords,
    audit_finding,
    confirmation_episodes,
    confirmation_job,
    relied_rules,
    training_eligibility,
    training_records,
)
from exp.optimize.evaluation.variance.variants import (
    VariantCheckpoint,
    build_variant_checkpoint,
    verify_restoration,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger

FAMILY_REPORT_TYPE = "decision-family-report"
"""Artifact type of a saved family report."""

OptionActions = Callable[[DecisionState], Mapping[str, ToolCall]]
"""Maps an authored decision state to the concrete forced first action of every option."""


class DecisionFamilySpec(ContractModel):
    """One registered decision family over one point of an actor-critic run.

    Attributes:
        decision_author: Frozen decision-state author model and allowance.
        scenario_author: Frozen scenario author model and allowance.
        rules_stage: Saved ``rules-stage-report`` whose rule book and challenges apply.
        job: Frozen job every witness and contrast runs under; None uses the run's own group job,
            which keeps every estimate under the screening policy.
        actions: Concrete forced first action per option identity, when known in advance; else
            the caller supplies a function of the decision state.
        repeats: Pairs per selection contrast.
        confirmation_repeats: Pairs per fresh confirmation contrast.
        confirmation_seed: Seed named in every confirmation draw's selector; a new seed draws a
            new confirmation.
        facts: Instance facts the world model was given for the task, if any.
        instruction_variation: Whether scenarios may replace the task instruction.
        grading: Rubric grading controls.
        tiers: Tier label per actor model identity, when tiers are recorded.
        dispositions: Recorded dispositions of audit findings. They never re-key a draw.
    """

    decision_author: AuthorSettings
    scenario_author: AuthorSettings
    rules_stage: ArtifactInput
    job: ContinuationJob | None = None
    actions: dict[str, ToolCall] | None = None
    repeats: int = Field(default=8, ge=1, le=100, strict=True)
    confirmation_repeats: int = Field(default=8, ge=1, le=100, strict=True)
    confirmation_seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    facts: InstanceFacts | None = None
    instruction_variation: bool = False
    grading: GraderSettings = GraderSettings()
    tiers: dict[str, str] = Field(default_factory=dict)
    dispositions: tuple[AuditDisposition, ...] = ()

    def draws_identity(self, report: ActorCriticReport, point: int, code_revision: str) -> str:
        """Bind everything the draws read, leaving dispositions out so they never re-key one.

        Args:
            report: The actor-critic run.
            point: The chosen point.
            code_revision: Producing source identity.

        Returns:
            Identity every stage of this family derives from.
        """
        return stable_id(
            "decision-family-draws",
            {
                "spec": sha256_json(self.model_dump(mode="json", exclude={"dispositions"})),
                "report": sha256_json(report),
                "point": point,
                "authors": authors_protocol_sha256(),
                "code": code_revision,
            },
        )

    def identity(self, report: ActorCriticReport, point: int, code_revision: str) -> str:
        """Bind the whole registration, dispositions included, and the producing code.

        Args:
            report: The actor-critic run.
            point: The chosen point.
            code_revision: Producing source identity.

        Returns:
            Identity naming the saved report.
        """
        return stable_id(
            "decision-family",
            {
                "draws": self.draws_identity(report, point, code_revision),
                "dispositions": sha256_json(
                    [item.model_dump(mode="json") for item in self.dispositions]
                ),
            },
        )


class FamilyVariant(ContractModel):
    """One scenario's variant and how far it got.

    Attributes:
        scenario_id: The scenario.
        variant: The built variant checkpoint with every coherence check.
        restored: Whether the saved variant restores to its edited state, or None when unbuilt.
        witness: The executed observation witness, when one ran.
        selection: The selection option contrast, when one ran.
        confirmation: The fresh confirmation option contrast, when one ran.
        meaning: What the variant's contrast outcomes are conditional on, when contrasted.
        stopped: Why the variant went no further, when it stopped before confirmation.
    """

    scenario_id: str
    variant: VariantCheckpoint
    restored: bool | None = None
    witness: WitnessResult | None = None
    selection: OptionContrast | None = None
    confirmation: OptionContrast | None = None
    meaning: EstimateMeaning | None = None
    stopped: str | None = None


class FamilyReport(ContractModel):
    """The complete record of one decision family, failed and unconfirmed parts included.

    Attributes:
        spec: The registration.
        identity: Identity naming this report.
        point: The chosen point.
        checkpoint: The point's checkpoint.
        estimation_point: The judge's estimation point the family was authored from.
        context: The saved decision context.
        decision: The decision-state draw.
        scenarios: The scenario draw, when the decision state was usable.
        job: The frozen job witnesses and contrasts ran under.
        actions: Concrete forced first action per option, when resolved.
        variants: One entry per scenario, in scenario order.
        selection: Verdict over the selection contrasts, when any ran.
        confirmation: Verdict over the fresh confirmation contrasts, when any ran.
        grades: Rubric grades over each confirmed variant's episodes, by scenario.
        ambiguous_criteria: Criteria routed back to rubric authoring.
        findings: Every audit finding.
        eligibility: Every computed eligibility condition.
        records: Training records, only when every condition holds.
        failure: Why the family stopped early, when it did.
    """

    spec: DecisionFamilySpec
    identity: str
    point: int
    checkpoint: ArtifactInput
    estimation_point: EstimationPoint
    context: ArtifactInput
    decision: DecisionStateDraft
    scenarios: ScenarioDraft | None = None
    job: ContinuationJob
    actions: dict[str, ToolCall] | None = None
    variants: tuple[FamilyVariant, ...] = ()
    selection: FlipVerdict | None = None
    confirmation: FlipVerdict | None = None
    grades: dict[str, GradeReport] = Field(default_factory=dict)
    ambiguous_criteria: tuple[str, ...] = ()
    findings: tuple[AuditFinding, ...] = ()
    eligibility: TrainingEligibility
    records: TrainingRecords | None = None
    failure: str | None = None


def run_decision_family(
    project: ProjectStore,
    spec: DecisionFamilySpec,
    *,
    exposure: ExposurePlan,
    report: ActorCriticReport,
    point: int,
    code_revision: str,
    actions: OptionActions | None = None,
    grader: DecisionsTraceTagger | None = None,
) -> FamilyReport:
    """Run, or replay, one decision family from a chosen point of an actor-critic run.

    Args:
        project: Project holding every saved step.
        spec: The registration.
        exposure: The frozen data roles; the task must come from a mining-probe lineage.
        report: The saved actor-critic run.
        point: A point the run chose and continued from.
        code_revision: Producing source identity.
        actions: Function giving every option's concrete forced action, used when the spec has
            no mapping. Its result is saved, so a replay never calls it.
        grader: Decision client grading rubric criteria over confirmation episodes; without it
            criteria are not graded.

    Returns:
        The report, also saved as a ``decision-family-report`` artifact.

    Raises:
        ValueError: The task is not a mining probe, the point was not chosen and continued in the
            run, the rules stage reference is not a saved stage report, or the action mapping is
            missing or given twice.
    """
    spec = DecisionFamilySpec.model_validate_json(spec.model_dump_json())
    if (spec.actions is None) == (actions is None):
        raise ValueError("give option actions either as spec.actions or as actions=, exactly once")
    store = project.artifacts
    rows = {row.point: row for row in report.points}
    if point not in rows:
        raise ValueError(f"point {point} was not chosen in this run; choose from {sorted(rows)}")
    checkpoint_ref = rows[point].checkpoint
    checkpoint = load_continuation_checkpoint(store, checkpoint_ref)
    exposure.require_lineage(checkpoint.task.lineage_group_id, "mining_probes")
    estimation = _estimation_point(report, point)
    stage = _rules_stage(project, spec.rules_stage)
    identity = spec.identity(report, point, code_revision)
    draws = spec.draws_identity(report, point, code_revision)
    job = spec.job or report.spec.initial
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return FamilyReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        book = load_rulebook(project, stage.rulebook)
        tools = frozenset(tool.name for tool in checkpoint.task.tools)
        context = DecisionContext(
            point=estimation,
            checkpoint=checkpoint_ref,
            prefix=checkpoint.actor.messages,
            before=_saved_before(project, draws, job, checkpoint_ref),
            rules=relevant_rules(book, tools),
            questions=relevant_questions(book, tools),
            tools=checkpoint.task.tools,
            instruction_variation=spec.instruction_variation,
        )
        context_ref = _save_context(
            project, draws, context, (stage.rulebook, spec.rules_stage), code_revision
        )
        sources = (context_ref, spec.rules_stage, stage.rulebook, report.group)
        decision = author_decision_state(
            project,
            spec.decision_author,
            context,
            identity=draws + "-decision-state",
            code_revision=code_revision,
            sources=sources,
        )
        family = _Family(
            project=project,
            spec=spec,
            report=report,
            stage=stage,
            uncovered=uncovered_tools(book, (checkpoint.task,)).get(checkpoint.task.task_id, ()),
        )
        base = FamilyReport(
            spec=spec,
            identity=identity,
            point=point,
            checkpoint=checkpoint_ref,
            estimation_point=estimation,
            context=context_ref,
            decision=decision,
            job=job,
            eligibility=family.eligibility(None, None, (), ()),
        )
        result = _continue(family, base, context, decision, draws, actions, grader, code_revision)
        store.write_json(
            artifact_id=result_id,
            artifact_type=FAMILY_REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=_dependencies(result, report),
            ),
            files={"report.json": result},
        )
        return result


def export_family_harbor(
    project: ProjectStore,
    family_report: FamilyReport,
    scenario_id: str,
    destination: Path,
    *,
    name: str,
    intent: str,
    code_revision: str,
    qualification: ArtifactInput | None = None,
) -> Path:
    """Package one variant's checkpoint as a Harbor environment that starts there.

    The package is diagnostic, awarding no reward, unless the family is eligible, the variant's
    winner was confirmed on fresh samples, and an executable qualification for the exact task is
    supplied.

    Args:
        project: Project holding the family.
        family_report: The saved family.
        scenario_id: The variant to package.
        destination: New package directory.
        name: Harbor ``organization/name``.
        intent: The capability the environment measures.
        code_revision: Producing source identity.
        qualification: Qualification report for the exact executable task, when it has one.

    Returns:
        The published package directory.

    Raises:
        ValueError: The scenario has no persisted variant checkpoint in this family.
    """
    rows = {row.scenario_id: row for row in family_report.variants}
    row = rows.get(scenario_id)
    if row is None or row.variant.reference is None:
        built = sorted(key for key, item in rows.items() if item.variant.reference is not None)
        raise ValueError(f"scenario {scenario_id} has no persisted variant; choose from {built}")
    confirmed = family_report.confirmation is not None and any(
        verdict.scenario_id == scenario_id and verdict.status == "separated_matches"
        for verdict in family_report.confirmation.variants
    )
    qualified = family_report.eligibility.eligible and confirmed and qualification is not None
    task = ExecutableTask(intent=intent, job=_bound(family_report.job, row.variant.reference))
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision=code_revision
    )
    return export_executable_harbor(
        project,
        reference,
        destination,
        name=name,
        qualification=qualification if qualified else None,
        diagnostic=not qualified,
    )


@dataclass(frozen=True)
class _Family:
    """What every stage of one family shares, and the eligibility computed from it.

    Attributes:
        project: Project holding every saved step.
        spec: The registration.
        report: The actor-critic run.
        stage: The rules stage report.
        uncovered: The task's tools without a learned contract.
    """

    project: ProjectStore
    spec: DecisionFamilySpec
    report: ActorCriticReport
    stage: RulesStageReport
    uncovered: tuple[str, ...]

    def eligibility(
        self,
        selection: FlipVerdict | None,
        confirmation: FlipVerdict | None,
        findings: tuple[AuditFinding, ...],
        relied: tuple[str, ...],
        contrasted: tuple[EstimateMeaning, ...] = (),
    ) -> TrainingEligibility:
        """Compute eligibility from the family's evidence so far.

        Args:
            selection: Selection verdict, when contrasts ran.
            confirmation: Confirmation verdict, when it ran.
            findings: Every audit finding.
            relied: Rule identities the decision state cites.
            contrasted: Meanings of every variant contrast.

        Returns:
            The computed eligibility.
        """
        meanings = tuple(row.meaning for row in self.report.points if row.failure is None)
        return training_eligibility(
            selection=selection,
            confirmation=confirmation,
            findings=findings,
            dispositions=self.spec.dispositions,
            critics=self.report.critics,
            uncovered_tools=self.uncovered,
            relied_rules=relied,
            contradicted_rules=self.stage.families_must_stay_diagnostic,
            unwitnessed_rules=self.stage.diagnostic,
            meanings=(*meanings, *contrasted),
        )


def _continue(
    family: _Family,
    base: FamilyReport,
    context: DecisionContext,
    decision: DecisionStateDraft,
    draws: str,
    actions: OptionActions | None,
    grader: DecisionsTraceTagger | None,
    code_revision: str,
) -> FamilyReport:
    """Run every stage after the decision-state draw, stopping explicitly at the first gap.

    Args:
        family: Shared inputs.
        base: The report so far.
        context: The decision context.
        decision: The decision-state draw.
        draws: Identity every stage derives from.
        actions: Caller-supplied option action function, when the spec has no mapping.
        grader: Rubric grader, when supplied.
        code_revision: Producing source identity.

    Returns:
        The complete report.
    """
    project, spec, report = family.project, family.spec, family.report
    findings = [audit_finding("actor_critic", flag) for flag in report.audit]
    if decision.answer is None:
        return _finish(family, base, findings, (), failure=f"decision state: {decision.failure}")
    state = decision.answer.state
    relied = relied_rules(decision.answer)
    findings.extend(
        audit_finding("decision_state", flag.model_dump(mode="json")) for flag in state.audit
    )
    checkpoint = load_continuation_checkpoint(project.artifacts, base.checkpoint)
    remaining = ChatAgentRuntime(
        maximum_model_calls=base.job.maximum_model_calls, system_prompt=base.job.system_prompt
    ).remaining_calls(checkpoint.actor)
    limits = ScenarioLimits(
        tools=frozenset(tool.name for tool in context.tools), remaining_calls=remaining
    )
    scenarios = author_scenarios(
        project,
        spec.scenario_author,
        state,
        context.before,
        limits,
        identity=draws + "-scenarios",
        code_revision=code_revision,
        sources=(base.context, *(() if decision.response is None else (decision.response,))),
    )
    base = base.model_copy(update={"scenarios": scenarios})
    findings.extend(
        audit_finding("scenario", problem.model_dump(mode="json")) for problem in scenarios.problems
    )
    if not scenarios.accepted or scenarios.scenarios is None:
        reason = scenarios.failure or f"{len(scenarios.problems)} structural problems"
        return _finish(family, base, findings, relied, failure=f"scenarios: {reason}")
    try:
        mapping = dict(spec.actions or (actions(state) if actions is not None else {}))
    except (ValueError, KeyError, TypeError) as error:
        return _finish(family, base, findings, relied, failure=f"option actions: {error}")
    base = base.model_copy(update={"actions": mapping})
    study = load_study(project.artifacts, report.group)
    picked = next(row for row in study.branches if row.index == report.picked)
    boundaries = (study.spec.checkpoint, *picked.checkpoints)
    variants = [
        _variant(family, base, state, scenario, boundaries, code_revision)
        for scenario in scenarios.scenarios.scenarios
    ]
    for row in variants:
        findings.extend(
            audit_finding(
                "variant_review",
                {"scenario_id": row.scenario_id, **check.model_dump(mode="json")},
            )
            for check in row.variant.needs_review
        )
    measured = {row.scenario_id: row.selection for row in variants if row.selection is not None}
    if not measured:
        base = base.model_copy(update={"variants": tuple(variants)})
        return _finish(family, base, findings, relied, failure="no variant was contrasted")
    selection = family_flip(measured, scenarios.scenarios.scenarios)
    winners = {v.scenario_id for v in selection.variants if v.status == "separated_matches"}
    confirmation: FlipVerdict | None = None
    if selection.qualifies:
        variants = [
            _confirm(family, base, state, row, code_revision) if row.scenario_id in winners else row
            for row in variants
        ]
        confirmation = family_flip(
            {r.scenario_id: r.confirmation for r in variants if r.confirmation is not None},
            [s for s in scenarios.scenarios.scenarios if s.scenario_id in winners],
        )
    grades, ambiguous, grading_failure = _grade(
        project, state, scenarios.scenarios.scenarios, variants, grader, spec.grading
    )
    base = base.model_copy(
        update={
            "variants": tuple(variants),
            "selection": selection,
            "confirmation": confirmation,
            "grades": grades,
            "ambiguous_criteria": ambiguous,
        }
    )
    contrasted = tuple(row.meaning for row in variants if row.meaning is not None)
    return _finish(
        family,
        base,
        findings,
        relied,
        selection=selection,
        confirmation=confirmation,
        contrasted=contrasted,
        failure=grading_failure,
    )


def _variant(
    family: _Family,
    base: FamilyReport,
    decision: DecisionState,
    scenario: Scenario,
    boundaries: tuple[ArtifactInput, ...],
    code_revision: str,
) -> FamilyVariant:
    """Build, prove, witness and contrast one scenario's variant, stopping at the first gap.

    Args:
        family: Shared inputs.
        base: The report so far, with the resolved actions.
        decision: The decision state.
        scenario: The scenario.
        boundaries: Every saved boundary of the picked attempt, start first.
        code_revision: Producing source identity.

    Returns:
        The variant with every stage it reached.
    """
    project, job = family.project, base.job
    variant = build_variant_checkpoint(
        project,
        base=base.checkpoint,
        branch_checkpoints=boundaries,
        decision=decision,
        scenario=scenario,
        facts=family.spec.facts,
        code_revision=code_revision,
        maximum_model_calls=job.maximum_model_calls,
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
    if variant.replay_turns:
        return row.model_copy(
            update={
                "stopped": f"variant forks {variant.replay_turns} turns before the decision; "
                "forcing an option there would not force this decision"
            }
        )
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
        return row.model_copy(update={"stopped": f"witness: {_reason(error)}"})
    row = row.model_copy(update={"witness": witness})
    if not witness.holds:
        return row.model_copy(update={"stopped": "witness: " + "; ".join(witness.problems)})
    try:
        selection = run_option_contrast(
            project,
            checkpoint=reference,
            decision=decision,
            actions=base.actions or {},
            job=job,
            repeats=family.spec.repeats,
            code_revision=code_revision,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"stopped": f"selection contrast: {_reason(error)}"})
    meaning = estimate_meaning(
        _bound(job, reference), load_continuation_checkpoint(project.artifacts, reference)
    )
    return row.model_copy(update={"selection": selection, "meaning": meaning})


def _confirm(
    family: _Family,
    base: FamilyReport,
    decision: DecisionState,
    row: FamilyVariant,
    code_revision: str,
) -> FamilyVariant:
    """Contrast a qualifying variant again on fresh samples, citing the selection as evidence.

    Args:
        family: Shared inputs.
        base: The report so far.
        decision: The decision state.
        row: A variant whose selection winner separated and matched.
        code_revision: Producing source identity.

    Returns:
        The variant with its confirmation, or why it has none.
    """
    if row.variant.reference is None or row.selection is None:
        return row
    try:
        confirmation = run_option_contrast(
            family.project,
            checkpoint=row.variant.reference,
            decision=decision,
            actions=base.actions or {},
            job=confirmation_job(base.job, family.spec.confirmation_seed),
            repeats=family.spec.confirmation_repeats,
            code_revision=code_revision,
            evidence=row.selection.contrasts,
        )
    except (ValueError, RuntimeError, OSError) as error:
        return row.model_copy(update={"stopped": f"confirmation contrast: {_reason(error)}"})
    return row.model_copy(update={"confirmation": confirmation})


def _grade(
    project: ProjectStore,
    decision: DecisionState,
    scenarios: tuple[Scenario, ...],
    variants: list[FamilyVariant],
    grader: DecisionsTraceTagger | None,
    settings: GraderSettings,
) -> tuple[dict[str, GradeReport], tuple[str, ...], str | None]:
    """Grade each confirmed variant's episodes on the criteria that apply to it.

    Args:
        project: Project holding the saved contrasts.
        decision: The decision state declaring the criteria.
        scenarios: Every scenario, for its applicable criteria.
        variants: Every variant.
        grader: Rubric grader, or None to skip grading.
        settings: Grading controls.

    Returns:
        Reports by scenario, ambiguous criterion identities, and a grading failure if any.
    """
    if grader is None:
        return {}, (), None
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
        failure = f"rubric grading: {_reason(error)}"
    ambiguous = {s.criterion_id for r in reports.values() for s in r.summaries if s.ambiguous}
    return reports, tuple(sorted(ambiguous)), failure


def _finish(
    family: _Family,
    base: FamilyReport,
    findings: list[AuditFinding],
    relied: tuple[str, ...],
    *,
    selection: FlipVerdict | None = None,
    confirmation: FlipVerdict | None = None,
    contrasted: tuple[EstimateMeaning, ...] = (),
    failure: str | None = None,
) -> FamilyReport:
    """Compute eligibility and, when every condition holds, the training records.

    Args:
        family: Shared inputs.
        base: The report so far.
        findings: Every audit finding.
        relied: Rule identities the decision state cites.
        selection: Selection verdict, when contrasts ran.
        confirmation: Confirmation verdict, when it ran.
        contrasted: Meanings of every variant contrast.
        failure: Why the family stopped early, when it did.

    Returns:
        The complete report.
    """
    unique = tuple({finding.finding_id: finding for finding in findings}.values())
    eligibility = family.eligibility(selection, confirmation, unique, relied, contrasted)
    built = training_records(eligibility, family.report, tiers=family.spec.tiers)
    return base.model_copy(
        update={
            "findings": unique,
            "eligibility": eligibility,
            "records": built if isinstance(built, TrainingRecords) else None,
            "failure": failure,
        }
    )


def _estimation_point(report: ActorCriticReport, point: int) -> EstimationPoint:
    """Return the first judge's estimation point at the chosen step.

    Args:
        report: The actor-critic run.
        point: The chosen point.

    Returns:
        The judge's point.

    Raises:
        ValueError: No usable judge selection chose the point.
    """
    for draw in report.judges:
        if draw.selection is None:
            continue
        for chosen in draw.selection.points:
            if chosen.step == point:
                return chosen
    raise ValueError(f"no judge selection chose point {point}")


def _rules_stage(project: ProjectStore, reference: ArtifactInput) -> RulesStageReport:
    """Read the pinned rules stage report.

    Args:
        project: Project holding it.
        reference: The ``rules-stage-report`` artifact.

    Returns:
        The report.

    Raises:
        ValueError: The reference is not the saved stage report it pins.
    """
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != (
        REPORT_TYPE
    ):
        raise ValueError(
            f"artifact {reference.artifact_id} is not the pinned rules stage report; pass the "
            "reference of a saved run_rules_stage result"
        )
    return RulesStageReport.model_validate_json(stored.payloads["report.json"])


def _saved_before(
    project: ProjectStore, draws: str, job: ContinuationJob, checkpoint: ArtifactInput
) -> JsonObject:
    """Return the evaluator state before the step, from the saved context or by probing once.

    Probing restores the point's world through the job's explicitly selected runtime and reads
    its private evaluator projection; no model is called.

    Args:
        project: Project holding the checkpoint.
        draws: Identity the saved context derives from.
        job: Frozen job selecting the runtime.
        checkpoint: The point's checkpoint.

    Returns:
        The before-state.
    """
    store = project.artifacts
    if store.exists(draws + "-context"):
        saved = DecisionContext.model_validate_json(
            store.read_bytes(draws + "-context", "context.json")
        )
        return dict(saved.before)
    bound = _bound(job, checkpoint)
    preflight_continuation_job(project, bound)
    original = load_continuation_checkpoint(store, checkpoint)
    with (
        load_continuation_runtime(project, bound) as runtime,
        restore_continuation_world(runtime.world, original.world_task, original.world) as world,
    ):
        return dict(runtime.probe(world.checkpoint()))


def _save_context(
    project: ProjectStore,
    draws: str,
    context: DecisionContext,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> ArtifactInput:
    """Save the decision context once, refusing a different one under the same identity.

    Args:
        project: Project receiving it.
        draws: Identity the context derives from.
        context: What the decision-state author is shown.
        inputs: Saved artifacts it was rendered from, besides the checkpoint.
        code_revision: Producing source identity.

    Returns:
        The context artifact.

    Raises:
        ValueError: A different context is saved under the same identity.
    """
    store, artifact_id = project.artifacts, draws + "-context"
    if store.exists(artifact_id):
        saved = DecisionContext.model_validate_json(store.read_bytes(artifact_id, "context.json"))
        if saved != context:
            raise ValueError("saved decision context differs from the one rendered now")
        return artifact_input(store.read(artifact_id).manifest)
    return artifact_input(
        store.write_json(
            artifact_id=artifact_id,
            artifact_type="decision-context",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted({context.checkpoint, *inputs}, key=lambda ref: ref.artifact_id)
                ),
            ),
            files={"context.json": context},
        )
    )


def _bound(job: ContinuationJob, checkpoint: ArtifactInput) -> ContinuationJob:
    """Start the frozen job at a checkpoint, with no carried prediction or selection plan."""
    study = job.study.model_copy(
        update={
            "checkpoint": checkpoint,
            "critic_prediction": None,
            "predicted_success": None,
            "selection_plan": None,
        }
    )
    return ContinuationJob.model_validate_json(
        job.model_copy(update={"study": study}).model_dump_json()
    )


def _dependencies(result: FamilyReport, report: ActorCriticReport) -> tuple[ArtifactInput, ...]:
    """Collect every saved artifact the family report rests on."""
    refs = {result.checkpoint, result.context, report.group, result.spec.rules_stage}
    for draft in (result.decision, result.scenarios):
        if draft is not None and draft.response is not None:
            refs.add(draft.response)
    for row in result.variants:
        if row.variant.reference is not None:
            refs.add(row.variant.reference)
        for contrast in (row.selection, row.confirmation):
            if contrast is not None:
                refs |= set(contrast.contrasts)
    return tuple(sorted(refs, key=lambda ref: ref.artifact_id))


def _reason(error: BaseException) -> str:
    """Return a bounded failure reason naming the error type."""
    return (type(error).__name__ + ": " + str(error))[:2000]
