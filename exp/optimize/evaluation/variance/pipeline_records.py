"""The ``records`` stage: training records, selector records, Harbor exports and cross-talk.

Only admitted families contribute. Each admitted family's training records (judged continuation
groups per tier and their critics, read back from saved assessments by the family owner) are
saved beside the family report, and each tier's point-selector records are saved from its
error-mining report under its episode judge. Every confirmed variant of an admitted family is
exported for Harbor at its frozen start. With ``records.qualification`` each variant's exact
executable task (frozen start, frozen runtime and the variant's own frozen judge) is first
executed once for feasibility and qualified by the executable qualification owner against
independently witnessed controls; a variant that qualifies is exported qualified, and one that
does not is exported as diagnostic, awarding no reward, with its refusals or failure recorded.
Without qualification every export is diagnostic. When enabled and two families are admitted,
the cross-talk probe measures whether exemplars from the first family's best judged attempts
move the second family's judged scores.

Version 2 selector records score every selector draw: with ``records.point_decisions`` (the
default) every continued point of each tier's run gets options derived from its judged
continuations by the agentic option proposer and each continuation's choice classified
(:mod:`~exp.optimize.evaluation.variance.point_options`); a point a family was measured from
also carries that family's joint score, which decides its value. The nudge ladders of admitted
families become one curriculum (:func:`~exp.optimize.evaluation.variance.nudge_ladder.
nudge_curriculum`): admitted rungs from strong to weak, every un-nudged target last, which a
training schedule reads (``TrainingScheduleSpec.curriculum``).

Every curriculum rung of an admitted family's ladder (each admitted nudged rung and always the
un-nudged target) is a task of its own and goes through the same qualification and export as a
confirmed variant: its exact executable task starts at the rung's replayed decision checkpoint
under the rung's own frozen judge, and its package and ``HarborExport`` row record its ladder,
strength and position. With ``records.schedule`` (the default) the stage also emits one
:class:`~exp.optimize.evaluation.variance.schedule.TrainingScheduleSpec` per tier: the tier's
error-mining registration (or the mining stage's own schedule for that tier) with a
:class:`~exp.optimize.evaluation.variance.schedule.CurriculumPlan` over the tier's admitted
ladders, saved as a ``variance-schedule`` artifact and written as the spec file the
``variance-schedule`` action starts from. Building it calls nothing.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.controls import ControlWitnessSpec, witness_controls
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.executable.qualification import (
    load_executable_qualification,
    qualify_executable_task,
)
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    load_executable_task,
    persist_executable_task,
)
from exp.optimize.evaluation.variance.admission import VariantJudging, frozen_start
from exp.optimize.evaluation.variance.crosstalk import (
    CrosstalkProbeSpec,
    ProbeTarget,
    run_crosstalk_probe,
)
from exp.optimize.evaluation.variance.family import (
    FamilyReport,
    export_family_harbor,
    family_decisions,
    family_executable_task,
)
from exp.optimize.evaluation.variance.family_export import export_rung_harbor, rung_executable_task
from exp.optimize.evaluation.variance.family_search import RungTask
from exp.optimize.evaluation.variance.nudge_ladder import (
    CurriculumPlan,
    load_ladder,
    nudge_curriculum,
)
from exp.optimize.evaluation.variance.pipeline_dense import attach_family_dense, dense_costs
from exp.optimize.evaluation.variance.pipeline_spec import (
    CostLine,
    DecisionVarianceSpec,
    FamiliesOutputs,
    FamilyOutcome,
    HarborExport,
    MiningOutputs,
    RecordsOutputs,
    StageContext,
    StageRun,
    TierMining,
    TierSchedule,
    cost_lines,
    family_key,
)
from exp.optimize.evaluation.variance.pipeline_stages import (
    group_job,
    judge_costs,
    load_report,
    tier_job,
)
from exp.optimize.evaluation.variance.point_options import point_decisions
from exp.optimize.evaluation.variance.records import selector_records
from exp.optimize.evaluation.variance.rewards import SelectorObjectiveV2
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec, save_schedule

__all__ = ["RECORDS_STAGE"]

RECORDS_TYPE = "decision-variance-records"
"""Artifact type of one saved set of training or selector records."""


class _RecordsStage:
    """The ``records`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The stage reads only its spec and the earlier stages."""
        del project, spec
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price the feasibility runs of qualification and the optional cross-talk probe."""
        del code_revision
        records = spec.records
        qualify = records.harbor and records.qualification is not None
        exports = (
            "disabled"
            if not records.harbor
            else "qualified first, diagnostic when they do not qualify"
            if qualify
            else "diagnostic without a qualification"
        )
        lines = [
            "Records: training and selector records of admitted families; Harbor exports of "
            f"their confirmed variants ({exports}).",
        ]
        costs: list[CostComponent] = []
        if records.dense is not None:
            authoring = spec.authoring
            # Decision authors may emit at most six executable options.
            episodes = (
                len(spec.mining.actors)
                * authoring.families_per_tier
                * (authoring.maximum_variants + 1)
                * (len(authoring.actions) or 6)
                * authoring.confirmation_repeats
            )
            costs.extend(dense_costs(project, records.dense, episodes))
            lines.append(
                f"Dense training signal: up to {episodes} confirmation episodes, bounded "
                "separately from outcome rewards; recorded actor actions only."
            )
        options = spec.point_options()
        if options is not None:
            populations = []
            for alias in spec.mining.actors:
                points = spec.mining.selector_draws * spec.mining.points
                continuations = spec.mining.continuations
                schedule = spec.mining.schedule
                if schedule is not None and schedule.base.initial.actor_alias == alias:
                    points = (
                        max(len(schedule.base.judges), schedule.steady_draws) * schedule.base.points
                    )
                    continuations = schedule.base.continuations
                per_point = options.profile.budget.model_calls * options.draws_per_point(
                    continuations
                )
                for settings in (options.proposer, options.classifier):
                    costs.extend(
                        row.scaled(points * per_point)
                        for row in evaluation_costs(project, settings)
                    )
                populations.append(
                    f"{alias}: up to {points} points x {continuations} continuations"
                )
            lines.append(
                "Point decisions: " + "; ".join(populations) + ". Each point gets options from "
                f"{len(options.stances)} proposer stances and a reconcile on "
                f"{options.proposer.model_alias}; judged continuations are classified on "
                f"{options.classifier.model_alias} for version 2 selector records."
            )
        if qualify:
            ladder = spec.authoring.ladder_spec()
            rungs = 0 if ladder is None else len(ladder.strengths) + 1
            per_family = spec.authoring.maximum_variants + 1 + rungs
            variants = spec.authoring.families_per_tier * per_family
            for alias in spec.mining.actors:
                feasibility = group_job(project, spec, spec.mining.registered, alias)
                costs.extend(row.scaled(variants) for row in job_costs(project, feasibility))
            lines.append(
                f"Qualification: one feasibility run per exported variant and curriculum rung "
                f"(up to {rungs} rungs per family), up to {variants} per tier, and "
                f"{len(records.qualification.controls)} witnessed controls each; qualification "
                "itself makes no model call."
            )
        if records.schedule:
            lines.append(
                "Training schedules: one per tier from its error-mining registration and the "
                f"nudge curriculum of its admitted families ({records.rollouts_per_tier} rollouts "
                "per strength tier), saved for the variance-schedule action; no model call."
            )
        if not records.crosstalk:
            return cost_lines(tuple(costs)), tuple(lines)
        alias = spec.mining.actors[0]
        episode = tier_job(project, spec.mining.registered, alias, 1)
        jobs = 1 + 3 * 1
        episodes = records.crosstalk_attempts + 3 * records.crosstalk_repeats
        costs.extend(row.scaled(jobs) for row in job_costs(project, episode))
        costs.extend(judge_costs(project, spec, episodes))
        lines.append(
            f"Cross-talk: best-of-{records.crosstalk_attempts} source, top "
            f"{records.crosstalk_top_k} exemplars, three arms of {records.crosstalk_repeats} "
            "attempts on one target."
        )
        return cost_lines(tuple(costs)), tuple(lines)

    def run(self, context: StageContext) -> StageRun:
        """Save records, export admitted variants and run the cross-talk probe."""
        project, spec = context.project, context.spec
        families = FamiliesOutputs.model_validate(context.upstream["families"].outputs)
        mining = MiningOutputs.model_validate(context.upstream["error-mining"].outputs)
        admitted = [row for row in families.families if row.admitted and row.report is not None]
        reports = {family_key(row.tier, row.point): _family(project, row) for row in admitted}
        training: dict[str, ArtifactInput] = {}
        for row in admitted:
            key = family_key(row.tier, row.point)
            report = reports[key]
            assert row.report is not None
            if report.records is not None:
                records = report.records
                refs = ()
                if spec.records.dense is not None:
                    records, refs = attach_family_dense(
                        project, report, spec.records.dense, code_revision=context.code_revision
                    )
                training[key] = _save(context, "training", key, records, (row.report, *refs))
        selectors: dict[str, ArtifactInput] = {}
        derived: dict[str, tuple[ArtifactInput, ...]] = {}
        options = spec.point_options()
        for tier in mining.tiers:
            # Version 2 selector records score each pick by its family's joint decision measure;
            # every family of the tier counts, admitted or not, so a refused point scores zero.
            measured = [
                _family(project, row)
                for row in families.families
                if row.tier == tier.tier and row.report is not None
            ]
            run = load_report(project, tier.report)
            decisions = family_decisions(project, measured)
            if options is not None:
                decisions, derived[tier.tier] = point_decisions(
                    project, options, run, code_revision=context.code_revision, families=decisions
                )
            records = selector_records(project, run, judge=tier.judge.identity, decisions=decisions)
            selectors[tier.tier] = _save(context, "selectors", tier.tier, records, (tier.report,))
        exports: list[HarborExport] = []
        if spec.records.harbor:
            for row in admitted:
                report = reports[family_key(row.tier, row.point)]
                exports.extend(_export(context, row, report))
                exports.extend(_export_rungs(context, row, report))
        crosstalk, skipped = None, None
        if spec.records.crosstalk:
            crosstalk, skipped = _crosstalk(context, admitted, reports)
        curriculum = nudge_curriculum(
            [load_ladder(project, row.ladder) for row in admitted if row.ladder is not None]
        )
        schedules = (
            {tier.tier: _schedule(context, tier, admitted, exports) for tier in mining.tiers}
            if spec.records.schedule
            else {}
        )
        outputs = RecordsOutputs(
            training=training,
            selectors=selectors,
            harbor=tuple(exports),
            crosstalk=crosstalk,
            crosstalk_skipped=skipped,
            curriculum=curriculum,
            point_options=derived,
            schedules=schedules,
        )
        summary: JsonObject = {
            "admitted": len(admitted),
            "training": sorted(training),
            "harbor": [row.path for row in exports],
            "qualified": [row.path for row in exports if not row.diagnostic],
            "crosstalk": None if crosstalk is None else crosstalk.artifact_id,
            "crosstalk_skipped": skipped,
            "curriculum": [
                {"scenario_id": stage.scenario_id, "strength": stage.strength}
                for stage in curriculum
            ],
            "point_options": {tier: len(refs) for tier, refs in derived.items()},
            "rungs": [row.path for row in exports if row.strength is not None],
            "schedules": {tier: row.path or row.failure for tier, row in schedules.items()},
        }
        refs = (
            *training.values(),
            *selectors.values(),
            crosstalk,
            *(row.task for row in exports),
            *(row.qualification for row in exports),
            *(ref for refs in derived.values() for ref in refs),
            *(row.ladder for row in admitted),
            *(row.schedule for row in schedules.values()),
        )
        return StageRun(outputs, refs, summary)


def _family(project: ProjectStore, row: FamilyOutcome) -> FamilyReport:
    """Read one family's saved report."""
    assert row.report is not None
    return FamilyReport.model_validate_json(
        project.artifacts.read_bytes(row.report.artifact_id, "report.json")
    )


def _save(
    context: StageContext,
    kind: str,
    key: str,
    records: ContractModel,
    inputs: tuple[ArtifactInput, ...],
) -> ArtifactInput:
    """Save one set of records under this stage, or adopt the identical saved one."""
    store = context.project.artifacts
    artifact_id = stable_id("decision-variance-records", {"stage": context.identity, kind: key})
    if not store.exists(artifact_id):
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=RECORDS_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=context.code_revision,
                inputs=inputs,
            ),
            files={"records.json": records},
        )
    return artifact_input(store.read(artifact_id).manifest)


def _confirmed(report: FamilyReport) -> tuple[str, ...]:
    """Variants whose winner was confirmed on fresh samples, in scenario order."""
    if report.confirmation is None:
        return ()
    winners = {
        verdict.scenario_id
        for verdict in report.confirmation.variants
        if verdict.status == "separated_matches"
    }
    return tuple(row.scenario_id for row in report.variants if row.scenario_id in winners)


def _export(context: StageContext, row: FamilyOutcome, report: FamilyReport) -> list[HarborExport]:
    """Qualify and export every confirmed variant of one admitted family, reusing earlier work.

    Args:
        context: The running stage.
        row: The admitted family.
        report: Its saved report.

    Returns:
        One export per confirmed variant, qualified or diagnostic.
    """
    root = context.dependencies.harbor_root
    if root is None:
        root = context.project.paths.runtime_directory / "decision-variance"
    exports: list[HarborExport] = []
    for scenario in _confirmed(report):
        name = _slug(f"{row.tier}-{row.point}-{scenario}")
        destination = root / context.identity / name
        export = HarborExport(
            tier=row.tier, point=row.point, scenario_id=scenario, path=str(destination)
        )
        if context.spec.records.qualification is not None:
            task = family_executable_task(
                report,
                scenario,
                intent=context.spec.records.intent,
                dense=context.spec.records.dense,
            )
            export = _qualify(context, task, export)
        if not destination.exists():
            export_family_harbor(
                context.project,
                report,
                scenario,
                Path(destination),
                name=f"decision-variance/{name}",
                intent=context.spec.records.intent,
                code_revision=context.code_revision,
                qualification=None if export.diagnostic else export.qualification,
                dense=context.spec.records.dense,
            )
        exports.append(export)
    return exports


def _export_rungs(
    context: StageContext, row: FamilyOutcome, report: FamilyReport
) -> list[HarborExport]:
    """Qualify and export every curriculum rung of one admitted family's ladder.

    Args:
        context: The running stage.
        row: The admitted family.
        report: Its saved report, whose rungs are the ladder's curriculum.

    Returns:
        One export per rung, qualified or diagnostic, recording its strength and position.
    """
    if not isinstance(report.ladder, ArtifactInput) or not report.rungs:
        return []
    root = context.dependencies.harbor_root
    if root is None:
        root = context.project.paths.runtime_directory / "decision-variance"
    ladder, intent = report.ladder.artifact_id, context.spec.records.intent
    exports: list[HarborExport] = []
    for rung in report.rungs:
        name = _slug(f"{row.tier}-{row.point}-rung-{rung.position}-{rung.scenario_id}")
        destination = root / context.identity / name
        export = HarborExport(
            tier=row.tier,
            point=row.point,
            scenario_id=rung.scenario_id,
            path=str(destination),
            ladder=ladder,
            strength=rung.strength,
            position=rung.position,
        )
        try:
            task = rung_executable_task(
                report, rung, intent=intent, dense=context.spec.records.dense
            )
        except ValueError as error:
            exports.append(export.model_copy(update={"failure": f"not exported: {error}"}))
            continue
        if context.spec.records.qualification is not None:
            export = _qualify(context, task, export)
        if not destination.exists():
            _package_rung(context, report, rung, export, name)
        exports.append(export)
    return exports


def _package_rung(
    context: StageContext, report: FamilyReport, rung: RungTask, export: HarborExport, name: str
) -> None:
    """Write one rung's Harbor package, qualified only when its qualification passed."""
    assert export.ladder is not None
    export_rung_harbor(
        context.project,
        report,
        rung,
        Path(export.path),
        name=f"decision-variance/{name}",
        intent=context.spec.records.intent,
        code_revision=context.code_revision,
        ladder=export.ladder,
        dense=context.spec.records.dense,
        qualification=None if export.diagnostic else export.qualification,
    )


def _schedule(
    context: StageContext,
    tier: TierMining,
    admitted: list[FamilyOutcome],
    exports: list[HarborExport],
) -> TierSchedule:
    """Emit one tier's training schedule over its qualified curriculum rungs.

    The base is the mining stage's own schedule for the tier when it has one, else the tier's
    saved error-mining registration. Each curriculum stage whose rung export qualified is bound
    to that exact task's job and frozen judge, which the stage's rollouts then run; diagnostic
    rungs stay out. The objective is version 2, scored by point decisions the records stage's
    option settings derive. The spec is saved and written as a file the ``variance-schedule``
    action reads; nothing is called.

    Args:
        context: The running stage.
        tier: The tier's error mining.
        admitted: Every admitted family.
        exports: Every Harbor export of this stage, curriculum rungs included.

    Returns:
        The saved schedule and its spec file, or why none could be built.
    """
    project, records = context.project, context.spec.records
    ladders = [row.ladder for row in admitted if row.tier == tier.tier and row.ladder is not None]
    qualified = {
        (row.ladder, row.scenario_id): row
        for row in exports
        if row.tier == tier.tier and row.strength is not None and not row.diagnostic
    }
    stages = []
    for stage in nudge_curriculum([load_ladder(project, ref) for ref in ladders]):
        export = qualified.get((stage.ladder, stage.scenario_id))
        if export is not None and export.task is not None:
            bound = load_executable_task(project, export.task)
            stages.append(
                stage.model_copy(
                    update={
                        "job": bound.job,
                        "judge": bound.judge,
                        "task": export.task,
                        "qualification": export.qualification,
                    }
                )
            )
    plan = (
        CurriculumPlan(stages=tuple(stages), rollouts_per_tier=records.rollouts_per_tier)
        if stages
        else None
    )
    scored: JsonObject = {
        "curriculum": None if plan is None else plan.model_dump(mode="json"),
        "objective": SelectorObjectiveV2().model_dump(mode="json"),
        "point_options": (
            None
            if (options := context.spec.point_options()) is None
            else options.model_dump(mode="json")
        ),
    }
    mined = context.spec.mining.schedule
    try:
        if mined is not None and mined.base.initial.actor_alias == tier.tier:
            spec = TrainingScheduleSpec.model_validate({**mined.model_dump(mode="json"), **scored})
        else:
            spec = TrainingScheduleSpec.model_validate(
                {
                    "name": _slug(f"dv-{tier.tier}")[:64],
                    "base": load_report(project, tier.report).spec.model_dump(mode="json"),
                    **scored,
                }
            )
    except ValueError as error:
        return TierSchedule(tier=tier.tier, failure=f"{type(error).__name__}: {error}"[:2000])
    reference = save_schedule(
        project, spec, code_revision=context.code_revision, inputs=(tier.report, *ladders)
    )
    root = context.dependencies.harbor_root
    if root is None:
        root = project.paths.runtime_directory / "decision-variance"
    path = root / context.identity / "schedules" / f"{_slug(tier.tier)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.model_dump_json(indent=2) + "\n")
    return TierSchedule(tier=tier.tier, schedule=reference, path=str(path), stages=len(stages))


def _qualify(context: StageContext, task: ExecutableTask, export: HarborExport) -> HarborExport:
    """Qualify one variant's or rung's exact executable task, recording every refusal or failure.

    The task is the one its Harbor package starts from, rewarded only by its frozen judge. Its
    feasibility run executes, or replays, once; the declared controls are witnessed against the
    independent specification and the result is saved, passing or not.

    Args:
        context: The running stage.
        task: The exact executable task.
        export: The variant's or rung's export, diagnostic so far.

    Returns:
        The export, qualified when every observable-contract check passed.
    """
    project, revision = context.project, context.code_revision
    settings = context.spec.records.qualification
    assert settings is not None
    reference = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision=revision
    )
    export = export.model_copy(update={"task": reference})
    try:
        if saved_continuation_job(project, task.job, code_revision=revision) is None:
            execute_continuation_job(project, task.job, code_revision=revision)
        feasibility = continuation_result_reference(project, task.job, code_revision=revision)
        cases = witness_controls(
            project,
            ControlWitnessSpec(
                task=reference,
                feasibility=feasibility,
                specification=settings.specification,
                controls=settings.controls,
            ),
            created_at=datetime.now(UTC),
            code_revision=revision,
        )
        qualification = qualify_executable_task(
            project, cases, created_at=datetime.now(UTC), code_revision=revision
        )
        result = load_executable_qualification(project, qualification)
    except ValueError as error:
        return export.model_copy(update={"failure": f"{type(error).__name__}: {error}"})
    return export.model_copy(
        update={
            "qualification": qualification,
            "diagnostic": not result.qualified,
            "refusals": result.refusals,
        }
    )


def _crosstalk(
    context: StageContext, admitted: list[FamilyOutcome], reports: dict[str, FamilyReport]
) -> tuple[ArtifactInput | None, str | None]:
    """Probe the first admitted family's exemplars against the second's judged scores."""
    starts = []
    for row in admitted:
        report = reports[family_key(row.tier, row.point)]
        confirmed = _confirmed(report)
        if confirmed and report.judging:
            starts.append((report, report.judging[confirmed[0]]))
    if len(starts) < 2:
        return None, f"{len(starts)} admitted families with a confirmed variant; two are needed"
    records = context.spec.records
    (source, source_binding), (target, target_binding) = starts[0], starts[1]
    spec = CrosstalkProbeSpec(
        source=ProbeTarget(
            job=_frozen(context.project, source.job, source_binding, records.crosstalk_attempts),
            judge=source_binding.judge,
        ),
        targets=(
            ProbeTarget(
                job=_frozen(context.project, target.job, target_binding, records.crosstalk_repeats),
                judge=target_binding.judge,
            ),
        ),
        top_k=records.crosstalk_top_k,
    )
    reference, _ = run_crosstalk_probe(context.project, spec, code_revision=context.code_revision)
    return reference, None


def _frozen(
    project: ProjectStore, job: ContinuationJob, binding: VariantJudging, repeats: int
) -> ContinuationJob:
    """The family job started at a variant's frozen start under its frozen runtime."""
    start = frozen_start(project, binding)
    if start is None:
        raise ValueError("cross-talk needs frozen variants; this variant has no frozen start")
    checkpoint, configuration = start
    probe = GroundedRuntimeDefinition.model_validate(configuration).probe_sha256
    study = job.study.model_copy(
        update={"checkpoint": checkpoint, "probe_sha256": probe, "repeats": repeats}
    )
    return job.model_copy(update={"study": study, "configuration": configuration})


def _slug(value: str) -> str:
    """Turn a label into a lowercase directory and package name."""
    return re.sub(r"[^a-z0-9_.-]", "-", value.lower())[:120]


RECORDS_STAGE = _RecordsStage()
"""The owner-bound ``records`` executor."""
