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
   family. With ``search`` set, a measured variant search
   (:func:`~exp.optimize.evaluation.variance.family_search.searched_scenarios`) replaces the draw
   and only a set the joint measure admits goes on. With ``reuse`` set, an earlier
   registration's decision and scenario set replace both draws, to be measured again.
4. **Variants.** Each scenario becomes a coherent variant checkpoint, proven to restore to its
   edited state. A variant forked before the decision is replayed to it by imposing the original
   trajectory's recorded calls in the edited world; a replay that diverges is kept as
   ``replay_diverged`` and not contrasted.
5. **Witnesses.** Each variant's witness calls execute in its restored decision world; a variant
   whose witness fails is not contrasted.
6. **Judges.** Each scenario's :class:`~exp.optimize.evaluation.variance.admission.VariantJudging`
   (its frozen episode judge, world freeze and rubric compatibility review) is fixed from the spec
   or a caller function and saved in the report before any of the variant's contrasts run.
   Every contrast cell is judged by it; those saved verdicts are the only labels. When the binding
   names a world freeze, the frozen variant is what runs: its witness and contrasts start from the
   freeze's saved checkpoint under the family job rebound to the frozen runtime definition
   (:func:`~exp.optimize.evaluation.variance.admission.frozen_start`), and admission records
   ``world_provenance`` for any contrast that did not.
7. **Selection.** Every option is forced from each variant's decision checkpoint under the frozen
   job, citing the saved witness as evidence, and
   :func:`~exp.optimize.evaluation.variance.contrast.family_flip` decides whether the better option
   flips. The per-variant stages live in :mod:`~exp.optimize.evaluation.variance.family_stages`.
8. **Fresh confirmation.** Each variant that qualified is contrasted again with independently drawn
   samples: the job's selector names the confirmation seed and the selection contrasts are cited as
   evidence, so every confirmation draw has a new identity and none of its samples took part in
   selection. Qualification stands only when every qualifying variant's confirmation reproduces
   its separated, matching winner. Confirmation runs only after
   :func:`~exp.optimize.evaluation.variance.decision_measure.measure_family` admits the family
   on choice shift, consequence and clarity; the measure is on by default, under the spec's
   ``decision_variance`` settings or else the family's own open-weight authors, and a family it
   blocks names the ``decision_variance`` eligibility condition. Its joint score is the version 2
   selector value of the family's point. With ``ladder`` set, an admitted family then builds a
   nudge ladder over its hardest state
   (:func:`~exp.optimize.evaluation.variance.family_search.family_ladder`); every curriculum rung
   becomes its own task recording its strength and position, the last always un-nudged.
9. **Coverage.** The judge's coverage of each confirmed variant is recorded. When a criterion
   grader is supplied, its probabilistic grades are calibration evidence only.
10. **Eligibility.** Every condition in :mod:`~exp.optimize.evaluation.variance.records`, including
    the frozen admission policy (:mod:`~exp.optimize.evaluation.variance.admission`), is computed,
    and records are built from the judged confirmations only when all hold.

Option identities are authored, so the concrete forced action of each option comes either from the
spec's mapping by option identity or from a caller-supplied function of the decision state; the
mapping used is saved in the report. Every stage result is saved by its owner, the family report
is saved as a ``decision-family-report`` artifact and replayed exactly, failures are kept and never
redrawn, and adding audit dispositions makes a new report while every draw replays.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.tasks.facts import InstanceFacts
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.stage import RulesStageReport, load_rulebook
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.admission import (
    FamilyAdmissionPolicy,
    VariantJudging,
    frozen_start,
    judging_problems,
)
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
from exp.optimize.evaluation.variance.bounded import run_keyed
from exp.optimize.evaluation.variance.coalition import CoalitionEvidence
from exp.optimize.evaluation.variance.contracts import (
    ActorView,
    DecisionState,
    EstimationPoint,
    Scenario,
    ScenarioLimits,
    StatePart,
    message_texts,
)
from exp.optimize.evaluation.variance.contrast import FlipVerdict, family_flip
from exp.optimize.evaluation.variance.coverage import uncovered_tools
from exp.optimize.evaluation.variance.decision_measure import (
    DecisionMeasure,
    DecisionMeasureSpec,
    load_decision_measure,
    measure_family,
)
from exp.optimize.evaluation.variance.exposure import EstimateMeaning, ExposurePlan
from exp.optimize.evaluation.variance.family_context import (
    estimation_point,
    rules_stage_report,
    save_context,
    saved_before,
)
from exp.optimize.evaluation.variance.family_export import (
    export_family_harbor as export_family_harbor,
)
from exp.optimize.evaluation.variance.family_export import (
    family_executable_task as family_executable_task,
)
from exp.optimize.evaluation.variance.family_export import variant_job as variant_job
from exp.optimize.evaluation.variance.family_search import (
    FamilyBinder,
    FamilyInputs,
    RegisteredSet,
    RungTask,
    SetJudges,
    family_ladder,
    ladder_base,
    searched_scenarios,
)
from exp.optimize.evaluation.variance.family_stages import (
    FamilyVariant,
    confirm_variant,
    grade_variants,
    run_variant,
)
from exp.optimize.evaluation.variance.grader import GradeReport, GraderSettings
from exp.optimize.evaluation.variance.judged import JudgeCoverage
from exp.optimize.evaluation.variance.nudge_ladder import NudgeLadderSpec
from exp.optimize.evaluation.variance.records import (
    AuditDisposition,
    AuditFinding,
    TrainingEligibility,
    TrainingRecords,
    audit_finding,
    relied_rules,
    training_eligibility,
    training_records,
)
from exp.optimize.evaluation.variance.rewards import PointDecision
from exp.optimize.evaluation.variance.variant_search import VariantSearchSpec
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger

FAMILY_REPORT_TYPE = "decision-family-report"
"""Artifact type of a saved family report."""

OptionActions = Callable[[DecisionState], Mapping[str, ToolCall]]
"""Maps an authored decision state to the concrete forced first action of every option."""

VariantJudges = SetJudges
"""Gives every scenario of one set its frozen judge, world freeze and compatibility review; the
family asks once per scenario set (its authored set, each search round's set and each measured
ladder round), before any of the set's contrasts."""


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
        instruction_variation: Whether scenarios may replace the task instruction. A changed
            instruction must start by ``fresh_briefing``, so it also needs ``fresh_briefing``.
        fresh_briefing: Whether scenarios may start the actor on an authored fresh briefing.
            The scenario author and its checks then receive the actor's view at the point, so
            briefings are leak-checked against what the actor had seen.
        required_state: State parts every variant must hold beyond the baseline, such as
            ``files``; a variant that cannot restore one is refused.
        judging: Frozen judge binding per scenario identity, when known in advance; else the
            caller supplies a function of the decision state and scenario.
        admission: Admission policy every confirmed variant must meet, frozen here.
        grading: Calibration criterion-grading controls; never an eligibility input.
        tiers: Tier label per actor model identity, when tiers are recorded.
        dispositions: Recorded dispositions of audit findings. They never re-key a draw.
        decision_variance: Joint decision-variance measurement settings (choice shift,
            consequence, clarity). Unkeyed while None.
        measure_decisions: Whether the joint measure gates confirmation (the default). With no
            ``decision_variance`` settings it runs under the family's own authors: the scenario
            author classifies choices and the decision author reviews clarity. False skips it
            and keeps the identity families had before the measure existed.
        search: Variant search settings: the scenarios come from measured search rounds instead
            of one draw, and only an admitted set goes on. Needs the measure; unkeyed while None.
        coalition: Optional frozen behavioral evidence for the early decision-state author.
        ladder: Nudge ladder settings: an admitted family builds a ladder over its hardest
            state and records every curriculum rung as its own task. Needs the measure; unkeyed
            while None.
        reuse: An earlier registration's decision draw, scenario set and ladder, used instead of
            fresh draws when the family is registered again under revised judges; the set is
            contrasted and measured again and the ladder is carried over. Unkeyed while None.
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
    fresh_briefing: bool = False
    required_state: tuple[StatePart, ...] = Field(default=(), max_length=8)
    judging: dict[str, VariantJudging] | None = None
    admission: FamilyAdmissionPolicy = FamilyAdmissionPolicy()
    grading: GraderSettings = GraderSettings()
    tiers: dict[str, str] = Field(default_factory=dict)
    dispositions: tuple[AuditDisposition, ...] = ()
    decision_variance: DecisionMeasureSpec | None = None
    measure_decisions: bool = True
    search: VariantSearchSpec | None = None
    ladder: NudgeLadderSpec | None = None
    coalition: CoalitionEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reuse: RegisteredSet | None = None

    @model_validator(mode="after")
    def _measured(self) -> DecisionFamilySpec:
        """Require the joint measure for a search or a ladder, and neither beside a reused set."""
        if (self.search is not None or self.ladder is not None) and not self.measure_decisions:
            raise ValueError("a variant search or a nudge ladder needs measure_decisions")
        if self.reuse is not None and (self.search is not None or self.ladder is not None):
            raise ValueError("a reused scenario set replaces the variant search and the ladder")
        return self

    @property
    def measure_spec(self) -> DecisionMeasureSpec | None:
        """The joint measure's settings, or None when the measure is switched off."""
        if not self.measure_decisions:
            return None
        return self.decision_variance or DecisionMeasureSpec(
            classifier=self.scenario_author, reviewer=self.decision_author
        )

    def _unkeyed(self) -> set[str]:
        """Fields left out of the draws identity: dispositions, the measure switch, and new fields
        at their defaults.

        Leaving ``fresh_briefing`` and ``required_state`` out while unset keeps every family saved
        before they existed replaying under its identity. The measure switch gates confirmation
        only, so it keys the report (:meth:`identity`) and never re-buys an authored draw.
        """
        unset = {"fresh_briefing"} if not self.fresh_briefing else set()
        unset |= {"decision_variance"} if self.decision_variance is None else set()
        unset |= {name for name in ("search", "ladder", "reuse") if getattr(self, name) is None}
        return {
            "dispositions",
            "measure_decisions",
            *unset,
            *(() if self.required_state else ("required_state",)),
        }

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
                "spec": sha256_json(self.model_dump(mode="json", exclude=self._unkeyed())),
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
                **({"measure_decisions": True} if self.measure_spec is not None else {}),
                "world_agent": _world_agent_protocol(),
            },
        )


def _world_agent_protocol() -> str:
    """The world author's protocol: a new one re-keys the family, never its draws."""
    from exp.optimize.evaluation.variance.world_agent import WORLD_AGENT_PROTOCOL
    from exp.optimize.evaluation.variance.world_freeze import WORLD_FREEZE_PROTOCOL

    return WORLD_AGENT_PROTOCOL + "/" + WORLD_FREEZE_PROTOCOL


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
        judging: Frozen judge binding per scenario, fixed before any contrast ran.
        variants: One entry per scenario, in scenario order.
        selection: Verdict over the selection contrasts, when any ran.
        confirmation: Verdict over the fresh confirmation contrasts, when any ran.
        coverage: The judge's coverage of each confirmed variant's confirmation episodes.
        grades: Calibration criterion grades over confirmed variants' episodes, by scenario.
        ambiguous_criteria: Criteria the calibration grader found uncertain; diagnostic only.
        findings: Every audit finding.
        eligibility: Every computed eligibility condition.
        records: Training records, only when every condition holds.
        failure: Why the family stopped early, when it did.
        decision_variance: The saved joint decision-variance measure, or why it could not run.
        search: The saved variant search the scenarios came from, when one ran.
        ladder: The saved nudge ladder, or why it could not be built, when one was attempted.
        rungs: One task per curriculum rung of the ladder, strongest first, the un-nudged last.
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
    judging: dict[str, VariantJudging] | None = None
    variants: tuple[FamilyVariant, ...] = ()
    selection: FlipVerdict | None = None
    confirmation: FlipVerdict | None = None
    coverage: dict[str, JudgeCoverage] = Field(default_factory=dict)
    grades: dict[str, GradeReport] = Field(default_factory=dict)
    ambiguous_criteria: tuple[str, ...] = ()
    findings: tuple[AuditFinding, ...] = ()
    eligibility: TrainingEligibility
    records: TrainingRecords | None = None
    failure: str | None = None
    decision_variance: ArtifactInput | str | None = None
    search: ArtifactInput | None = None
    ladder: ArtifactInput | str | None = None
    rungs: tuple[RungTask, ...] = ()


def run_decision_family(
    project: ProjectStore,
    spec: DecisionFamilySpec,
    *,
    exposure: ExposurePlan,
    report: ActorCriticReport,
    point: int,
    code_revision: str,
    actions: OptionActions | None = None,
    judging: VariantJudges | None = None,
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
        judging: Function giving every scenario of a set its frozen judge binding, used when the
            spec has no mapping. It is called before the set's first contrast and its result is
            saved, so a replay of the family report never calls it.
        grader: Calibration criterion grader over confirmation episodes; never an eligibility
            input.

    Returns:
        The report, also saved as a ``decision-family-report`` artifact.

    Raises:
        ValueError: The task is not a mining probe, the point was not chosen and continued in the
            run, the rules stage reference is not a saved stage report, or the action mapping or
            judge binding is missing or given twice.
    """
    spec = DecisionFamilySpec.model_validate_json(spec.model_dump_json())
    if (spec.actions is None) == (actions is None):
        raise ValueError("give option actions either as spec.actions or as actions=, exactly once")
    if (spec.judging is None) == (judging is None):
        raise ValueError("give variant judges either as spec.judging or as judging=, exactly once")
    store = project.artifacts
    rows = {row.point: row for row in report.points}
    if point not in rows:
        raise ValueError(f"point {point} was not chosen in this run; choose from {sorted(rows)}")
    checkpoint_ref = rows[point].checkpoint
    checkpoint = load_continuation_checkpoint(store, checkpoint_ref)
    exposure.require_lineage(checkpoint.task.lineage_group_id, "mining_probes")
    estimation = estimation_point(report, point)
    stage = rules_stage_report(project, spec.rules_stage)
    identity = spec.identity(report, point, code_revision)
    draws = spec.draws_identity(report, point, code_revision)
    job = spec.job or report.spec.initial
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        result_id = identity + "-result"
        if store.exists(result_id):
            return FamilyReport.model_validate_json(store.read_bytes(result_id, "report.json"))
        if spec.coalition is not None:
            spec.coalition.check_exposure(exposure)
        book = load_rulebook(project, stage.rulebook)
        tools = frozenset(tool.name for tool in checkpoint.task.tools)
        context = DecisionContext(
            point=estimation,
            checkpoint=checkpoint_ref,
            prefix=checkpoint.actor.messages,
            before=saved_before(project, draws, job, checkpoint_ref),
            rules=relevant_rules(book, tools),
            questions=relevant_questions(book, tools),
            tools=checkpoint.task.tools,
            instruction_variation=spec.instruction_variation,
            task_instruction=None if spec.coalition is None else checkpoint.task.instruction,
            coalition=spec.coalition or CoalitionEvidence(),
        )
        context_ref = save_context(
            project,
            draws,
            context,
            (stage.rulebook, spec.rules_stage, *context.coalition.references),
            code_revision,
        )
        sources = (context_ref, spec.rules_stage, stage.rulebook, report.group)
        decision = (
            spec.reuse.decision
            if spec.reuse is not None
            else author_decision_state(
                project,
                spec.decision_author,
                context,
                identity=draws + "-decision-state",
                code_revision=code_revision,
                sources=sources,
            )
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
        result = _continue(
            family, base, context, decision, draws, actions, judging, grader, code_revision
        )
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
        judging: Mapping[str, VariantJudging] | None = None,
        coverage: Mapping[str, JudgeCoverage] | None = None,
        starts: Mapping[str, tuple[ArtifactInput, ...]] | None = None,
        measured: tuple[bool, str] | None = None,
    ) -> TrainingEligibility:
        """Compute eligibility from the family's evidence so far.

        Args:
            selection: Selection verdict, when contrasts ran.
            confirmation: Confirmation verdict, when it ran.
            findings: Every audit finding.
            relied: Rule identities the decision state cites.
            contrasted: Meanings of every variant contrast.
            judging: Frozen judge binding per scenario, once fixed.
            coverage: Judge coverage of each confirmed variant, once confirmed.
            starts: Starting checkpoint of each variant's selection and confirmation contrasts.
            measured: Whether the joint measure admitted the family and what it found, once run.

        Returns:
            The computed eligibility.
        """
        if measured is None and self.spec.measure_spec is not None:
            measured = (False, "the family stopped before the joint decision-variance measure")
        meanings = tuple(row.meaning for row in self.report.points if row.failure is None)
        problems = (
            judging_problems(self.project, judging, coverage, self.spec.admission, starts)
            if judging is not None and coverage
            else {}
        )
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
            judging=problems,
            decision_variance=measured,
        )


def _continue(
    family: _Family,
    base: FamilyReport,
    context: DecisionContext,
    decision: DecisionStateDraft,
    draws: str,
    actions: OptionActions | None,
    judges: VariantJudges | None,
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
        judges: Caller-supplied judge binding function, when the spec has no mapping.
        grader: Calibration criterion grader, when supplied.
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
    seen = ActorView(
        instruction=checkpoint.task.instruction, observed=message_texts(checkpoint.actor.messages)
    )
    view = seen if spec.fresh_briefing else None
    sources = (base.context, *(() if decision.response is None else (decision.response,)))
    study = load_study(project.artifacts, report.group)
    picked = next(row for row in study.branches if row.index == report.picked)
    boundaries = (study.spec.checkpoint, *picked.checkpoints)
    binder = FamilyBinder(project, base.job, spec.judging, judges)
    inputs: FamilyInputs | None = None
    if spec.reuse is not None:
        scenarios = spec.reuse.scenarios
        reused = {"search": spec.reuse.search, "ladder": spec.reuse.ladder}
        base = base.model_copy(update={**reused, "rungs": spec.reuse.rungs})
    elif spec.search is not None and spec.measure_spec is not None:
        try:
            resolved = dict(spec.actions or (actions(state) if actions is not None else {}))
        except (ValueError, KeyError, TypeError) as error:
            return _finish(family, base, findings, relied, failure=f"option actions: {error}")
        inputs = _inputs(spec, base, boundaries, resolved)
        searched, _, scenarios = searched_scenarios(
            project,
            spec.search,
            spec.measure_spec,
            decision=state,
            before=context.before,
            limits=limits,
            view=view,
            binder=binder,
            inputs=inputs,
            identity=draws + "-search",
            sources=sources,
            code_revision=code_revision,
        )
        base = base.model_copy(update={"search": searched})
    else:
        scenarios = author_scenarios(
            project,
            spec.scenario_author,
            state,
            context.before,
            limits,
            identity=draws + "-scenarios",
            code_revision=code_revision,
            sources=sources,
            view=view,
        )
    base = base.model_copy(update={"scenarios": scenarios})
    findings.extend(
        audit_finding("scenario", problem.model_dump(mode="json")) for problem in scenarios.problems
    )
    if not scenarios.accepted or scenarios.scenarios is None:
        reason = scenarios.failure or f"{len(scenarios.problems)} structural problems"
        return _finish(family, base, findings, relied, failure=f"scenarios: {reason}")
    try:
        mapping = (
            dict(inputs.actions)
            if inputs is not None
            else dict(spec.actions or (actions(state) if actions is not None else {}))
        )
    except (ValueError, KeyError, TypeError) as error:
        return _finish(family, base, findings, relied, failure=f"option actions: {error}")
    inputs = inputs or _inputs(spec, base, boundaries, mapping)
    base = base.model_copy(update={"actions": mapping})
    try:
        judging = dict(
            spec.judging
            or (judges(state, scenarios.scenarios.scenarios) if judges is not None else {})
        )
    except (ValueError, KeyError, TypeError) as error:
        return _finish(family, base, findings, relied, failure=f"variant judges: {error}")
    unbound = sorted({s.scenario_id for s in scenarios.scenarios.scenarios} - set(judging))
    if unbound:
        return _finish(
            family, base, findings, relied, failure=f"variant judges: none for {unbound}"
        )
    base = base.model_copy(update={"judging": judging})
    try:
        starts = {sid: frozen_start(project, binding) for sid, binding in judging.items()}
    except ValueError as error:
        return _finish(family, base, findings, relied, failure=f"frozen start: {error}")
    jobs = {sid: variant_job(base.job, start) for sid, start in starts.items()}
    frozen = {sid: start[0] for sid, start in starts.items() if start is not None}
    # Scenarios are independent variants (own checkpoint, witness and contrast), so they run at
    # once; ones that could share a saved identity (same frozen start, or the same edits without
    # one) run in order on one worker. Results keep scenario order.
    ordered = scenarios.scenarios.scenarios
    variants = list(
        run_keyed(
            [
                lambda scenario=scenario: run_variant(
                    project,
                    job=jobs[scenario.scenario_id],
                    base=base.checkpoint,
                    boundaries=boundaries,
                    decision=state,
                    scenario=scenario,
                    actions=mapping,
                    facts=spec.facts,
                    repeats=spec.repeats,
                    judge=judging[scenario.scenario_id].judge,
                    code_revision=code_revision,
                    required_state=frozenset(spec.required_state),
                    frozen=frozen.get(scenario.scenario_id),
                )
                for scenario in ordered
            ],
            [_variant_key(scenario, frozen.get(scenario.scenario_id)) for scenario in ordered],
            max(1, len(ordered)),
        )
    )
    for row in variants:
        findings.extend(
            audit_finding(
                "variant_review",
                {"scenario_id": row.scenario_id, **check.model_dump(mode="json")},
            )
            for check in row.variant.needs_review
        )
        if row.stopped is not None and row.stopped.startswith("replay_diverged"):
            findings.append(
                audit_finding(
                    "replay_divergence", {"scenario_id": row.scenario_id, "reason": row.stopped}
                )
            )
    measured = {row.scenario_id: row.selection for row in variants if row.selection is not None}
    if not measured:
        base = base.model_copy(update={"variants": tuple(variants)})
        return _finish(family, base, findings, relied, failure="no variant was contrasted")
    selection = family_flip(measured, scenarios.scenarios.scenarios)
    winners = {v.scenario_id for v in selection.variants if v.status == "separated_matches"}
    confirmation: FlipVerdict | None = None
    measure_spec, measure = spec.measure_spec, None
    admitted = measure_spec is None
    if measure_spec is not None and selection.qualifies:
        frozen_judges = {sid: binding.judge for sid, binding in judging.items()}
        measured_ref, measure, why = measure_family(
            project,
            measure_spec,
            decision=state,
            scenarios=scenarios.scenarios.scenarios,
            variants=variants,
            jobs=jobs,
            judges=frozen_judges,
            code_revision=code_revision,
        )
        admitted = measure is not None and measure.verdict.admitted
        base = base.model_copy(update={"decision_variance": measured_ref or why})
        if admitted and measure is not None and spec.ladder is not None:
            target = ladder_base(measure, scenarios.scenarios.scenarios)
            built = (
                "no measured state to ladder"
                if target is None
                else family_ladder(
                    project,
                    spec.ladder,
                    measure_spec,
                    decision=state,
                    base=target,
                    instruction=checkpoint.task.instruction,
                    view=seen,
                    before=context.before,
                    binder=binder,
                    inputs=inputs,
                    identity=draws + "-ladder",
                    sources=(*sources, *(() if measured_ref is None else (measured_ref,))),
                    code_revision=code_revision,
                )
            )
            ladder = built if isinstance(built, str) else built[0]
            rungs = () if isinstance(built, str) else built[2]
            base = base.model_copy(update={"ladder": ladder, "rungs": rungs})
    if selection.qualifies and admitted:
        # Each winner confirms from its own decision checkpoint, so winners confirm at once;
        # winners sharing a checkpoint confirm in order on one worker.
        confirmed = run_keyed(
            [
                lambda row=row: (
                    confirm_variant(
                        project,
                        job=jobs[row.scenario_id],
                        decision=state,
                        row=row,
                        actions=mapping,
                        seed=spec.confirmation_seed,
                        repeats=spec.confirmation_repeats,
                        judge=judging[row.scenario_id].judge,
                        code_revision=code_revision,
                    )
                    if row.scenario_id in winners
                    else row
                )
                for row in variants
            ],
            [
                str(
                    None if row.decision_checkpoint is None else row.decision_checkpoint.artifact_id
                )
                + ("" if row.scenario_id in winners else "|" + row.scenario_id)
                for row in variants
            ],
            max(1, len(variants)),
        )
        variants = list(confirmed)
        confirmation = family_flip(
            {r.scenario_id: r.confirmation for r in variants if r.confirmation is not None},
            [s for s in scenarios.scenarios.scenarios if s.scenario_id in winners],
        )
    coverage, grades, uncertain, grading_failure = grade_variants(
        project, state, scenarios.scenarios.scenarios, variants, grader, spec.grading
    )
    base = base.model_copy(
        update={
            "variants": tuple(variants),
            "selection": selection,
            "confirmation": confirmation,
            "coverage": coverage,
            "grades": grades,
            "ambiguous_criteria": uncertain,
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
        measure=measure if measure is not None else base.decision_variance,
    )


def _variant_key(scenario: Scenario, frozen: ArtifactInput | None) -> str:
    """Return what two scenarios must share to possibly work one saved identity."""
    if frozen is not None:
        return "frozen:" + frozen.artifact_id
    return "edits:" + sha256_json(
        {
            "edits": [edit.model_dump(mode="json") for edit in scenario.edits],
            "initialization": scenario.initialization,
        }
    )


def _inputs(
    spec: DecisionFamilySpec,
    base: FamilyReport,
    boundaries: tuple[ArtifactInput, ...],
    actions: Mapping[str, ToolCall],
) -> FamilyInputs:
    """Return what the measured search and ladder share with the family's own variants."""
    return FamilyInputs(
        base=base.checkpoint,
        boundaries=boundaries,
        actions=dict(actions),
        repeats=spec.repeats,
        facts=spec.facts,
        required_state=frozenset(spec.required_state),
    )


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
    measure: DecisionMeasure | ArtifactInput | str | None = None,
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
        measure: The joint measure, or why it could not run, once selection was measured.

    Returns:
        The complete report.
    """
    unique = tuple({finding.finding_id: finding for finding in findings}.values())
    measured, decisions = None, None
    if isinstance(measure, DecisionMeasure):
        verdict = measure.verdict
        detail = f"admitted, joint score {verdict.score:.3f}" if verdict.admitted else ""
        measured = (verdict.admitted, detail or "; ".join(verdict.deficits))
        point = f"point-{base.point}"
        decisions = {point: PointDecision(point_id=point, joint_score=verdict.score)}
    elif isinstance(measure, str):
        measured = (False, f"the joint measure could not run: {measure}")
    elif selection is not None and not selection.qualifies:
        measured = (False, "selection did not qualify, so decisions were not measured")
    starts = {
        row.scenario_id: tuple(
            contrast.checkpoint
            for contrast in (row.selection, row.confirmation)
            if contrast is not None
        )
        for row in base.variants
    }
    eligibility = family.eligibility(
        selection,
        confirmation,
        unique,
        relied,
        contrasted,
        base.judging,
        base.coverage,
        starts,
        measured if family.spec.measure_spec is not None else None,
    )
    confirmed = {
        row.scenario_id: (row.confirmation, row.meaning)
        for row in base.variants
        if row.confirmation is not None and row.meaning is not None
    }
    built = training_records(
        family.project,
        eligibility,
        family.report,
        confirmed,
        tiers=family.spec.tiers,
        decisions=decisions,
    )
    return base.model_copy(
        update={
            "findings": unique,
            "eligibility": eligibility,
            "records": built if isinstance(built, TrainingRecords) else None,
            "failure": failure,
        }
    )


def family_decisions(
    project: ProjectStore, reports: Sequence[FamilyReport]
) -> dict[str, PointDecision]:
    """Return the version 2 selector evidence of every measured family's point.

    Args:
        project: Project holding the saved joint measures.
        reports: Families of one error-mining run.

    Returns:
        Each measured point's joint score by ``point-<step>`` (0 when not admitted), for
        :func:`~exp.optimize.evaluation.variance.records.selector_records`.
    """
    found: dict[str, PointDecision] = {}
    for report in reports:
        if isinstance(report.decision_variance, ArtifactInput):
            score = load_decision_measure(project, report.decision_variance).verdict.score
            point = f"point-{report.point}"
            found[point] = PointDecision(point_id=point, joint_score=score)
    return found


def _dependencies(result: FamilyReport, report: ActorCriticReport) -> tuple[ArtifactInput, ...]:
    """Collect every saved artifact the family report rests on."""
    refs = {result.checkpoint, result.context, report.group, result.spec.rules_stage}
    refs |= {ref for ref in (result.search, result.ladder) if isinstance(ref, ArtifactInput)}
    refs |= set() if result.spec.reuse is None else {result.spec.reuse.source}
    refs |= {ref for row in result.rungs if (ref := row.variant.decision_checkpoint) is not None}
    refs |= {row.judge.definition.rubric for row in (result.judging or {}).values()}
    refs |= {
        row.world_freeze for row in (result.judging or {}).values() if row.world_freeze is not None
    }
    for draft in (result.decision, result.scenarios):
        if draft is not None and draft.response is not None:
            refs.add(draft.response)
    for row in result.variants:
        refs |= {
            ref
            for ref in (row.variant.reference, row.decision_checkpoint, row.witness_artifact)
            if ref is not None
        }
        for contrast in (row.selection, row.confirmation):
            if contrast is not None:
                refs |= set(contrast.contrasts)
    return tuple(sorted(refs, key=lambda ref: ref.artifact_id))
