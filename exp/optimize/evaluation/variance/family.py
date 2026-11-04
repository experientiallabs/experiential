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
   edited state. A variant forked before the decision is replayed to it by imposing the original
   trajectory's recorded calls in the edited world; a replay that diverges is kept as
   ``replay_diverged`` and not contrasted.
5. **Witnesses.** Each variant's witness calls execute in its restored decision world; a variant
   whose witness fails is not contrasted.
6. **Judges.** Each scenario's :class:`~exp.optimize.evaluation.variance.admission.VariantJudging`
   (its frozen episode judge, world freeze and rubric compatibility review) is fixed from the spec
   or a caller function and saved in the report before any of the variant's contrasts run.
   Every contrast cell is judged by it; those saved verdicts are the only labels.
7. **Selection.** Every option is forced from each variant's decision checkpoint under the frozen
   job, citing the saved witness as evidence, and
   :func:`~exp.optimize.evaluation.variance.contrast.family_flip` decides whether the better option
   flips. The per-variant stages live in :mod:`~exp.optimize.evaluation.variance.family_stages`.
8. **Fresh confirmation.** Each variant that qualified is contrasted again with independently drawn
   samples: the job's selector names the confirmation seed and the selection contrasts are cited as
   evidence, so every confirmation draw has a new identity and none of its samples took part in
   selection. Qualification stands only when every qualifying variant's confirmation reproduces
   its separated, matching winner.
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
from exp.optimize.evaluation.variance.admission import (
    FamilyAdmissionPolicy,
    VariantJudging,
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
from exp.optimize.evaluation.variance.exposure import EstimateMeaning, ExposurePlan
from exp.optimize.evaluation.variance.family_stages import (
    FamilyVariant,
    confirm_variant,
    grade_variants,
    run_variant,
)
from exp.optimize.evaluation.variance.grader import GradeReport, GraderSettings
from exp.optimize.evaluation.variance.judged import JudgeCoverage
from exp.optimize.evaluation.variance.records import (
    AuditDisposition,
    AuditFinding,
    TrainingEligibility,
    TrainingRecords,
    audit_finding,
    bound_job,
    relied_rules,
    training_eligibility,
    training_records,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.mining.decisions import DecisionsTraceTagger

FAMILY_REPORT_TYPE = "decision-family-report"
"""Artifact type of a saved family report."""

OptionActions = Callable[[DecisionState], Mapping[str, ToolCall]]
"""Maps an authored decision state to the concrete forced first action of every option."""

VariantJudges = Callable[[DecisionState, Scenario], VariantJudging]
"""Gives one authored scenario its frozen judge, world freeze and compatibility review."""


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

    def _unkeyed(self) -> set[str]:
        """Fields left out of the draws identity: dispositions, and new fields at their defaults.

        Leaving ``fresh_briefing`` and ``required_state`` out while unset keeps every family saved
        before they existed replaying under its identity.
        """
        unset = {"fresh_briefing"} if not self.fresh_briefing else set()
        return {"dispositions", *unset, *(() if self.required_state else ("required_state",))}

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
            },
        )


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
        judging: Function giving every scenario its frozen judge binding, used when the spec has
            no mapping. It is called before the scenario's first contrast and its result is
            saved, so a replay never calls it.
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

    The package starts at the variant's decision checkpoint, the replayed one when the variant
    forks before the decision. It is diagnostic, awarding no reward, unless the family is
    eligible, the variant's winner was confirmed on fresh samples, and an executable
    qualification for the exact task (:func:`family_executable_task`) is supplied.

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
        ValueError: The scenario has no decision checkpoint in this family.
    """
    task = family_executable_task(family_report, scenario_id, intent=intent)
    confirmed = family_report.confirmation is not None and any(
        verdict.scenario_id == scenario_id and verdict.status == "separated_matches"
        for verdict in family_report.confirmation.variants
    )
    qualified = family_report.eligibility.eligible and confirmed and qualification is not None
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


def family_executable_task(
    family_report: FamilyReport, scenario_id: str, *, intent: str
) -> ExecutableTask:
    """Return the executable task that starts at one variant's decision checkpoint.

    It is the exact task :func:`export_family_harbor` packages, so a qualification for it can be
    built before export.

    Args:
        family_report: The saved family.
        scenario_id: The variant.
        intent: The capability the environment measures.

    Returns:
        The task under the family's frozen job, rewarded only by the variant's frozen judge.

    Raises:
        ValueError: The scenario has no decision checkpoint or judge binding in this family.
    """
    rows = {row.scenario_id: row for row in family_report.variants}
    row = rows.get(scenario_id)
    checkpoint = None if row is None else row.decision_checkpoint
    if checkpoint is None:
        built = sorted(key for key, item in rows.items() if item.decision_checkpoint is not None)
        raise ValueError(
            f"scenario {scenario_id} has no persisted decision checkpoint; choose from {built}"
        )
    judging = (family_report.judging or {}).get(scenario_id)
    if judging is None:
        raise ValueError(f"scenario {scenario_id} has no frozen judge binding in this family")
    return ExecutableTask(
        intent=intent, job=bound_job(family_report.job, checkpoint), judge=judging.judge
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
        judging: Mapping[str, VariantJudging] | None = None,
        coverage: Mapping[str, JudgeCoverage] | None = None,
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

        Returns:
            The computed eligibility.
        """
        meanings = tuple(row.meaning for row in self.report.points if row.failure is None)
        problems = (
            judging_problems(self.project, judging, coverage, self.spec.admission)
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
    view = (
        ActorView(
            instruction=checkpoint.task.instruction,
            observed=message_texts(checkpoint.actor.messages),
        )
        if spec.fresh_briefing
        else None
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
        mapping = dict(spec.actions or (actions(state) if actions is not None else {}))
    except (ValueError, KeyError, TypeError) as error:
        return _finish(family, base, findings, relied, failure=f"option actions: {error}")
    base = base.model_copy(update={"actions": mapping})
    try:
        judging = dict(
            spec.judging
            or (
                {s.scenario_id: judges(state, s) for s in scenarios.scenarios.scenarios}
                if judges is not None
                else {}
            )
        )
    except (ValueError, KeyError, TypeError) as error:
        return _finish(family, base, findings, relied, failure=f"variant judges: {error}")
    unbound = sorted({s.scenario_id for s in scenarios.scenarios.scenarios} - set(judging))
    if unbound:
        return _finish(
            family, base, findings, relied, failure=f"variant judges: none for {unbound}"
        )
    base = base.model_copy(update={"judging": judging})
    study = load_study(project.artifacts, report.group)
    picked = next(row for row in study.branches if row.index == report.picked)
    boundaries = (study.spec.checkpoint, *picked.checkpoints)
    variants = [
        run_variant(
            project,
            job=base.job,
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
        )
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
    if selection.qualifies:
        variants = [
            confirm_variant(
                project,
                job=base.job,
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
            for row in variants
        ]
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
    eligibility = family.eligibility(
        selection, confirmation, unique, relied, contrasted, base.judging, base.coverage
    )
    confirmed = {
        row.scenario_id: (row.confirmation, row.meaning)
        for row in base.variants
        if row.confirmation is not None and row.meaning is not None
    }
    built = training_records(
        family.project, eligibility, family.report, confirmed, tiers=family.spec.tiers
    )
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
    bound = bound_job(job, checkpoint)
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


def _dependencies(result: FamilyReport, report: ActorCriticReport) -> tuple[ArtifactInput, ...]:
    """Collect every saved artifact the family report rests on."""
    refs = {result.checkpoint, result.context, report.group, result.spec.rules_stage}
    refs |= {row.judge.definition.rubric for row in (result.judging or {}).values()}
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
