"""One command from a trace source to environments, registered starts and the whole pipeline.

:func:`run_front_door` takes one :class:`FrontDoorSpec`, pointed at a trace file, and runs:

1. ``project-build``: the project's completed build from exactly that trace source (import,
   task mining, both RAG indexes, the grounded world model), through an injected
   :class:`ProjectBuilder` (the CLI runs ``exp build`` non-interactively). A saved
   ``front-door-source`` record binds the source digest to the build it produced, so a rerun
   over the same file skips it.
2. ``bootstrap-preparation``: :func:`~exp.optimize.evaluation.variance.bootstrap_plan.
   run_bootstrap_preparation` over that build: the variance build (decision tagging, tag
   clusters, the environment-family compatibility review and per-family scopes with probe
   routing), each selected family's rules learned once and frozen into its runtime, and one
   authored, reviewed, captured and validated start registered per selected probe task. The
   batch is created from the build; no job is registered by hand.
3. ``pipeline <task>``: one six-stage decision-variance run per registered start
   (:func:`~exp.optimize.evaluation.variance.pipeline.run_decision_variance`), its spec the
   :class:`PipelineTemplate` with the saved build pinned and ``mining.bootstrap`` set to that
   start. The bootstrapped rules stage reuses the family's frozen rules with no call.

Explicit slots (:mod:`~exp.optimize.evaluation.variance.frontdoor_slots`) sit between these for
stages other owners add: after the families review (the preparation's variance build runs first
and then replays), after the preparation and after the pipelines.

**Resume.** Every step is an existing owner that saves and replays its own work, so a rerun
replans from saved artifacts and continues where it stopped.

**Plan and consent.** :func:`plan_front_door` makes no credential or provider call. A stage is
``complete`` (saved, replays with no call), ``pending`` (priced from its owner's plan) or
``blocked`` (it reads the output of an earlier pending stage, so it can only be priced once that
exists; its notes say what bounds it). One :class:`FrontDoorApproval` of the exact plan gives one
dollar ceiling and one unpriced-attempt allowance for the whole run. Each later phase is priced
when it becomes reachable and runs under that same consent only if its bound fits what is left;
otherwise the run stops with saved work intact and a rerun shows the now-priced plan.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models.client import RequestAdmissionRefused
from exp.common.project import ProjectStore, artifact_input
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapApproval,
    BootstrapDependencies,
    BootstrapPreparationReport,
    plan_bootstrap_preparation,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.frontdoor_contracts import (
    FrontDoorApproval,
    FrontDoorPlan,
    FrontDoorReport,
    FrontDoorSpec,
    FrontDoorStage,
    PipelineOutcome,
    PipelineTemplate,
)
from exp.optimize.evaluation.variance.frontdoor_executors import (
    SlotContext,
    dense_slot_settings,
    families_review_plan,
    review_families,
    state_rag_preparation,
)
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.frontdoor_slots import (
    SLOT_ORDER,
    SLOT_POSITIONS,
    SLOT_STAGES,
    HaltingExecutor,
    SlotExecutor,
    SlotName,
    SlotPosition,
    slot_executor,
)
from exp.optimize.evaluation.variance.frontdoor_source import (
    ProjectBuilder,
    TraceSource,
    ensure_build,
    recorded_build,
)
from exp.optimize.evaluation.variance.frontdoor_states import StageState, Tracker
from exp.optimize.evaluation.variance.frontdoor_task_types import (
    saved_slot,
    saved_task_types,
    task_type_preparation,
    task_type_selection,
)
from exp.optimize.evaluation.variance.pipeline import (
    StageExecutor,
    plan_decision_variance,
    run_decision_variance,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    CostLine,
    PipelineApproval,
    PipelinePlan,
    StageDependencies,
    StageName,
)

__all__ = [
    "FrontDoorApproval",
    "FrontDoorDependencies",
    "FrontDoorPlan",
    "FrontDoorReport",
    "FrontDoorSpec",
    "FrontDoorStage",
    "PipelineOutcome",
    "PipelineTemplate",
    "TraceSource",
    "configured_slots",
    "effective_spec",
    "plan_front_door",
    "run_directory",
    "run_front_door",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrontDoorDependencies:
    """Collaborators the front door passes to its owners.

    Attributes:
        builder: Project builder, required only when the project build is pending.
        bootstrap: Preparation collaborators, or their admitted defaults.
        pipeline: Pipeline stage collaborators, or their admitted defaults.
        executors: Pipeline stage executors replacing the owner-bound defaults.
        slots: Slot executors replacing the owner-bound defaults.
        announce: Shown the dry-run view of each phase (completed work, the phase's remaining
            calls and unpriced components, the ledger left) before it is admitted.
    """

    builder: ProjectBuilder | None = None
    bootstrap: BootstrapDependencies = BootstrapDependencies()
    pipeline: StageDependencies = StageDependencies()
    executors: Mapping[StageName, StageExecutor] | None = None
    slots: dict[SlotName, SlotExecutor] = field(default_factory=dict)
    announce: Callable[[tuple[str, ...]], None] | None = None


def plan_front_door(
    project: ProjectStore,
    spec: FrontDoorSpec,
    *,
    code_revision: str,
    dependencies: FrontDoorDependencies | None = None,
) -> FrontDoorPlan:
    """Name every stage, say which are saved and price what can be priced, with no call.

    Args:
        project: Project receiving the build and every saved step.
        spec: The registration.
        code_revision: Producing source identity.
        dependencies: Pipeline and slot executors used for planning.

    Returns:
        The plan.

    Raises:
        ValueError: A configured slot has no executor, or no build exists and no trace source
            is given.
    """
    dependencies = dependencies or FrontDoorDependencies()
    built = _build_stage(project, spec)
    stages = [built]
    if any(SLOT_POSITIONS[name] == "after-families" for name in configured_slots(spec)):
        stages.append(_review_stage(project, spec, code_revision, built.status))
    if project.load_project().build is None:
        # Every later stage reads the build's traces, tasks or embeddings: name them as
        # waiting instead of asking their owners to plan over a build that does not exist.
        stages.extend(_pending_build_stages(spec, dependencies))
        return FrontDoorPlan(
            spec_sha256=sha256_json(spec), code_revision=code_revision, stages=tuple(stages)
        )
    saved = _saved_family_slots(project, spec, code_revision)
    stages.extend(
        _slot_stages(project, spec, "after-families", code_revision, dependencies, dict(saved))
    )
    report: BootstrapPreparationReport | None = None
    bound = _bound_spec(project, spec, code_revision)
    effective, selection = (None, ()) if bound is None else bound
    if effective is None:
        preparation = FrontDoorStage(
            stage="bootstrap-preparation",
            status=StageState.BLOCKED,
            lines=_PREPARATION_LINES,
            notes=(
                "Priced once the families slots saved what every start binds: each family's "
                "state-tag index and, with task types, one start per task type.",
            ),
        )
    elif built.status == StageState.DONE:
        spec = effective
        preparation, report, prepared = _preparation_stage(project, spec, code_revision)
        if prepared is not None:
            saved["preparation"] = prepared
        elif selection:
            preparation = preparation.model_copy(update={"lines": (*preparation.lines, *selection)})
    else:
        preparation = FrontDoorStage(
            stage="bootstrap-preparation",
            status=StageState.BLOCKED,
            lines=_PREPARATION_LINES,
            notes=(
                "Priced once the project build exists: tagging, compatibility review, family "
                f"rules and at most {spec.preparation.limits.maximum_tasks} starts.",
            ),
        )
    stages.append(preparation)
    stages.extend(
        _slot_stages(project, spec, "after-preparation", code_revision, dependencies, dict(saved))
    )
    if report is None:
        stages.append(
            FrontDoorStage(
                stage="pipelines",
                status=StageState.BLOCKED,
                lines=("One six-stage decision-variance run per registered start.",),
                notes=(
                    "Each start's pipeline is priced once the preparation registers it; at most "
                    f"{spec.preparation.limits.maximum_tasks} starts, "
                    f"{len(spec.pipeline.mining.actors)} actor tiers each.",
                ),
            )
        )
    else:
        for binding in report.bindings:
            pipeline = _pipeline_plan(project, spec, report, binding, code_revision, dependencies)
            stages.append(_pipeline_stage(binding, pipeline))
            if pipeline.complete:
                for row in pipeline.stages:
                    manifest = project.artifacts.read(row.identity + "-stage").manifest
                    saved[f"pipeline:{binding.task_id}:{row.stage}"] = artifact_input(manifest)
    stages.extend(_slot_stages(project, spec, "after-pipeline", code_revision, dependencies, saved))
    return FrontDoorPlan(
        spec_sha256=sha256_json(spec), code_revision=code_revision, stages=tuple(stages)
    )


def run_front_door(
    project: ProjectStore,
    spec: FrontDoorSpec,
    *,
    code_revision: str,
    approval: FrontDoorApproval | None = None,
    dependencies: FrontDoorDependencies | None = None,
) -> FrontDoorReport:
    """Run, resume or replay the whole sequence, phase by phase, under one cumulative ceiling.

    Args:
        project: Project receiving the build and every saved step.
        spec: The registration.
        code_revision: Producing source identity.
        approval: Consent to the exact current plan; required unless it is complete. Its
            ceiling and allowance become the registration's cumulative ledger ceiling.
        dependencies: Builder, preparation, pipeline and slot collaborators.

    Returns:
        The report; it stops before a phase whose bound does not fit what is left of the
        cumulative ceiling, saved work intact.

    Raises:
        ValueError: Consent is missing or names another plan, or the build is pending without a
            builder.
    """
    dependencies = dependencies or FrontDoorDependencies()
    spec = FrontDoorSpec.model_validate_json(spec.model_dump_json())
    plan = plan_front_door(project, spec, code_revision=code_revision, dependencies=dependencies)
    if not plan.complete and (approval is None or approval.plan_sha256 != plan.digest):
        raise ValueError(
            "the front door needs consent to its exact current plan; review the dry run and "
            "consent once"
        )
    directory = run_directory(project, spec)
    tracker = Tracker(TransitionJournal(directory / "transitions.jsonl"), directory.name)
    for row in plan.stages:
        tracker.stage(row.stage, row.status.value)
    ledger = PhaseLedger(
        directory / "ledger.json",
        ceiling_usd=None if approval is None else approval.ceiling_usd,
        unpriced_attempts=None if approval is None else approval.unpriced_attempts,
    )
    if plan.complete:
        tracker.run_event("replay")
    else:
        tracker.run_event("consented")
        tracker.run_event("started")
    runner = _Runner(project, spec, plan, ledger, tracker, code_revision, dependencies)
    try:
        report = runner.sequence()
    except Exception:
        tracker.run_event("raised")
        raise
    tracker.run_event("stopped" if report.stopped is not None else "finished")
    return report


def configured_slots(spec: FrontDoorSpec) -> dict[SlotName, JsonObject]:
    """Return the configured slots in their fixed order, with the dense-hybrid slot always on.

    The rubric's decision-model part is part of every exported task, so its slot is configured
    whenever a pipeline exists; the in-flow vocabulary is described in its settings. Slots at one
    position run in ``SLOT_ORDER``, so a tag monitor checkpoint follows the stage it monitors.
    """
    slots = dict(spec.slots)
    records = spec.pipeline.records
    if "dense-hybrid" not in slots and records.dense is None:
        traceml, job = spec.pipeline.traceml, spec.preparation.bootstrap.job
        slots["dense-hybrid"] = dense_slot_settings(
            records.behavior_bank,
            traceml.error_bank,
            traceml.bank_root,
            spec.pipeline.authoring.rubric_author,
            traceml.decision_alias,
            (job.maximum_model_calls or job.horizon_ceiling) * traceml.steps_per_call,
        )
    return {name: slots[name] for name in SLOT_ORDER if name in slots}


def effective_spec(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str
) -> FrontDoorSpec | None:
    """Return the registration with the families slots' results bound into its starts.

    Each family's saved state recall enters the runtime template, and with task types one start
    is registered per task type. Unchanged without those slots; None until a configured one of
    them is saved, since every start depends on it.
    """
    bound = _bound_spec(project, spec, code_revision)
    return None if bound is None else bound[0]


def _bound_spec(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str
) -> tuple[FrontDoorSpec, tuple[str, ...]] | None:
    """Return :func:`effective_spec` and the task-type start selection's plan lines."""
    preparation = spec.preparation
    recall = spec.slots.get("state-rag")
    if recall is not None:
        bound = state_rag_preparation(SlotContext(project, preparation, code_revision), recall)
        if bound is None:
            return None
        preparation = bound
    lines: tuple[str, ...] = ()
    typed = spec.slots.get("task-types")
    if typed is not None:
        context = SlotContext(project, preparation, code_revision)
        banks = saved_slot(context, "tag-banks", spec.slots.get("tag-banks", {}))
        if banks is None:
            return None
        context = SlotContext(project, preparation, code_revision, {"tag-banks": banks})
        types = saved_task_types(context, typed)
        if types is None:
            return None
        _, _, lines = task_type_selection(context, types)
        preparation = task_type_preparation(context, types)
    return spec.model_copy(update={"preparation": preparation}), lines


def run_directory(project: ProjectStore, spec: FrontDoorSpec) -> Path:
    """Return the directory holding a registration's transition journal and phase ledger."""
    identity = f"front-door-{sha256_json(spec)[:16]}"
    return project.paths.runtime_directory / "front-door" / identity


@dataclass
class _Runner:
    """One front-door run: every phase moved through its lifecycle against the ledger.

    Attributes:
        project: The project.
        spec: The registration.
        plan: The plan the run started from.
        ledger: The cumulative phase ledger.
        tracker: The checked, persisted state of the run and its phases.
        code_revision: Producing source identity.
        dependencies: Collaborators.
    """

    project: ProjectStore
    spec: FrontDoorSpec
    plan: FrontDoorPlan
    ledger: PhaseLedger
    tracker: Tracker
    code_revision: str
    dependencies: FrontDoorDependencies

    def sequence(self) -> FrontDoorReport:
        """Run every phase in order and return the report."""
        project, spec, deps = self.project, self.spec, self.dependencies
        report = FrontDoorReport(plan=self.plan)
        if spec.traces is not None:
            traces = spec.traces
            source, reason = self.phase(
                "project-build",
                (),
                lambda: ensure_build(project, traces, deps.builder, self.code_revision),
            )
            if reason is not None:
                return report.model_copy(update={"stopped": reason})
            report = report.model_copy(update={"source": source})
        if any(SLOT_POSITIONS[name] == "after-families" for name in configured_slots(spec)):
            context = SlotContext(project, spec.preparation, self.code_revision)
            _, costs, _ = families_review_plan(context)
            if self.tracker.stages.get("variance-build") == StageState.BLOCKED:
                self.tracker.stage("variance-build", "priced")
            _, reason = self.phase(
                "variance-build",
                costs,
                lambda: review_families(
                    project, spec.preparation, self.code_revision, deps.bootstrap
                ),
            )
            if reason is not None:
                return report.model_copy(update={"stopped": reason})
            report = self.slots("after-families", report)
            if report.stopped is not None:
                return report
        effective = effective_spec(project, spec, self.code_revision)
        if effective is None:
            raise ValueError("the families slots finished without their saved results")
        self.spec = spec = effective
        preparation = plan_bootstrap_preparation(
            project, spec.preparation, code_revision=self.code_revision
        )
        result_id = preparation.identity + "-result"
        stage = "bootstrap-preparation"
        if self.tracker.stages[stage] == StageState.BLOCKED:
            done = project.artifacts.exists(result_id)
            self.tracker.stage(stage, "replayed" if done else "priced")

        def prepare() -> BootstrapPreparationReport:
            """Run or replay the preparation under its own share of the ceiling."""
            return run_bootstrap_preparation(
                project,
                spec.preparation,
                code_revision=self.code_revision,
                approval=BootstrapApproval(
                    plan_sha256=preparation.digest,
                    ceiling_usd=preparation.known_usd,
                    unpriced_attempts=preparation.unpriced_attempts,
                ),
                dependencies=deps.bootstrap,
            )

        prepared, reason = self.phase(
            stage,
            preparation.costs,
            prepare,
            stopped=lambda _: None if project.artifacts.exists(result_id) else "tasks failed",
        )
        if prepared is None:
            return report.model_copy(update={"stopped": reason})
        partial = not project.artifacts.exists(result_id)
        if partial and not (spec.save_partial and prepared.bindings):
            failed = ", ".join(f"{t}" for t in (*prepared.failed_tasks, *prepared.refused_tasks))
            return report.model_copy(
                update={
                    "stopped": f"bootstrap preparation saved no result; failed: {failed}",
                    "failed_tasks": prepared.failed_tasks,
                    "refused_tasks": prepared.refused_tasks,
                }
            )
        if partial:
            # Registered starts need not wait for a failed task: save this run's partial
            # report under its own content identity and run their pipelines. No "-result" is
            # written, so the next run replays every saved outcome and retries the failed.
            saved = save_partial_preparation(
                project, preparation.identity, prepared, self.code_revision
            )
        else:
            saved = artifact_input(project.artifacts.read(result_id).manifest)
        report = report.model_copy(
            update={
                "preparation": saved,
                "bindings": prepared.bindings,
                "failed_tasks": prepared.failed_tasks,
                "refused_tasks": prepared.refused_tasks,
            }
        )
        report = self.slots("after-preparation", report)
        if report.stopped is not None:
            return report
        if self.tracker.stages.get("pipelines") == StageState.BLOCKED:
            self.tracker.stage("pipelines", "replayed")
        # Each start's pipeline is independent: one that stops, raises or is refused admission
        # (a provider or budget refusal, nothing dispatched) is recorded and the next start
        # runs, so one bad start never costs the others their phase. The run stops before the
        # after-pipeline slots when any start stopped; a rerun replays the finished starts and
        # resumes the stopped one.
        outcomes: list[PipelineOutcome] = []
        for binding in prepared.bindings:
            try:
                outcome = self.pipeline(prepared, binding)
            except (Exception, RequestAdmissionRefused) as error:  # noqa: BLE001 - per start
                logger.exception(
                    "pipeline %s raised; continuing with the next start", binding.task_id
                )
                outcome = PipelineOutcome(
                    task_id=binding.task_id,
                    binding=binding,
                    stopped=f"raised {type(error).__name__}: {str(error)[:1000]}",
                )
            outcomes.append(outcome)
        stops = [f"{row.task_id}: {row.stopped}" for row in outcomes if row.stopped is not None]
        if stops:
            return report.model_copy(
                update={"pipelines": tuple(outcomes), "stopped": "; ".join(stops)}
            )
        report = report.model_copy(update={"pipelines": tuple(outcomes)})
        if partial:
            return report.model_copy(
                update={
                    "stopped": "registered starts ran; failed tasks retry on the next run: "
                    + ", ".join(prepared.failed_tasks)
                }
            )
        return self.slots("after-pipeline", report)

    def context(self, report: FrontDoorReport) -> SlotContext:
        """Return what a slot executor reads: earlier slots, the preparation, every pipeline."""
        results: dict[str, ArtifactInput] = {str(k): v for k, v in report.slots.items()}
        if report.preparation is not None:
            results["preparation"] = report.preparation
        for outcome in report.pipelines:
            for stage_name, record in outcome.stages.items():
                results[f"pipeline:{outcome.task_id}:{stage_name}"] = record
        return SlotContext(self.project, self.spec.preparation, self.code_revision, results)

    def phase[Result](
        self,
        name: str,
        costs: tuple[CostLine, ...],
        work: Callable[[], Result],
        stopped: Callable[[Result], str | None] = lambda _: None,
    ) -> tuple[Result | None, str | None]:
        """Admit, run and settle one phase, or replay it when it is done.

        Args:
            name: The phase.
            costs: Its bound, from its owner's plan of the work it will run.
            work: Runs or replays it.
            stopped: Why a finished run of it stopped short, from its result.

        Returns:
            The result (None when refused) and why it stopped, if it did.
        """
        state = self.tracker.stages.get(name)
        if state is None:
            state = self.tracker.stage(name, StageState.PLANNED.value)
        if state == StageState.DONE:
            return work(), None
        usd = sum(row.usd for row in costs if row.usd is not None)
        attempts = sum(row.attempts for row in costs if row.usd is None)
        announce = self.dependencies.announce
        if announce is not None:
            announce(self.preview(name, costs))
        reason = self.ledger.admit(name, usd, attempts)
        if reason is not None:
            self.tracker.stage(name, "refused")
            return None, reason
        self.tracker.stage(name, "approved")
        self.tracker.stage(name, "started")
        before = measure(self.project)
        try:
            result = work()
        except BaseException:
            # A budget or admission stop is a BaseException; the phase is settled either way.
            self.ledger.settle(name, before, measure(self.project))
            self.tracker.stage(name, "raised")
            raise
        self.ledger.settle(name, before, measure(self.project))
        reason = stopped(result)
        self.tracker.stage(name, "stopped" if reason is not None else "finished")
        return result, reason

    def preview(self, name: str, costs: tuple[CostLine, ...]) -> tuple[str, ...]:
        """The dry-run view shown before one phase: completed, its remaining calls, unpriced."""
        done = [stage for stage, state in self.tracker.stages.items() if state == StageState.DONE]
        priced = sum(row.usd for row in costs if row.usd is not None)
        requests = sum(row.attempts for row in costs)
        unpriced = [f"{row.name} ({row.attempts} attempts)" for row in costs if row.usd is None]
        return (
            f"Before phase {name}:",
            f"   Completed work: {', '.join(done) if done else 'none'}.",
            f"   This phase: up to ${priced:.6f} priced, at most {requests} physical requests.",
            f"   Unpriced components: {'; '.join(unpriced) if unpriced else 'none'}.",
            f"   Cumulative ledger left: ${self.ledger.remaining_usd:.6f} and "
            f"{self.ledger.remaining_attempts} unpriced attempts.",
        )

    def slots(self, position: SlotPosition, report: FrontDoorReport) -> FrontDoorReport:
        """Run or replay the configured slots at one position.

        A slot whose executor reports an unacknowledged halt (a tag monitor checkpoint) stops the
        run after it, so no later phase is admitted until the halt is acknowledged or the
        monitored stage re-authored; a rerun replays the saved slot and stops there again.
        """
        results = dict(report.slots)
        costs = {row.stage: row.costs for row in self.plan.stages}
        for name, settings in configured_slots(self.spec).items():
            if SLOT_POSITIONS[name] != position:
                continue
            executor = slot_executor(name, self.dependencies.slots)
            stage = f"slot {name}"
            bound = costs.get(stage, ())
            if self.tracker.stages.get(stage) == StageState.BLOCKED:
                # Consented while the build was pending: price it now that its inputs exist.
                saved, bound, _ = executor.plan(
                    self.context(report.model_copy(update={"slots": results})), settings
                )
                self.tracker.stage(stage, "replayed" if saved else "priced")
            result, reason = self.phase(
                stage,
                bound,
                lambda executor=executor, settings=settings: executor.run(
                    self.context(report.model_copy(update={"slots": results})), settings
                ),
            )
            if result is None:
                return report.model_copy(update={"slots": results, "stopped": reason})
            results[name] = result
            if isinstance(executor, HaltingExecutor):
                halted = executor.halt(self.context(report), result)
                if halted is not None:
                    return report.model_copy(update={"slots": results, "stopped": halted})
        return report.model_copy(update={"slots": results})

    def pipeline(
        self, prepared: BootstrapPreparationReport, binding: BootstrapBinding
    ) -> PipelineOutcome:
        """Run, resume or replay one start's pipeline as its own phase."""
        deps = self.dependencies
        pipeline_spec = self.spec.pipeline_spec(prepared, binding)
        plan = plan_decision_variance(
            self.project, pipeline_spec, code_revision=self.code_revision, executors=deps.executors
        )
        name = f"pipeline {binding.task_id}"
        if name not in self.tracker.stages and plan.complete:
            self.tracker.stage(name, StageState.DONE.value)
        approval = None if plan.complete else PipelineApproval.for_plan(plan)
        costs = tuple(
            line.model_copy(update={"name": f"{row.stage}: {line.name}"})
            for row in plan.stages
            if not row.saved
            for line in row.costs
        )
        result, reason = self.phase(
            name,
            costs,
            lambda: run_decision_variance(
                self.project,
                pipeline_spec,
                code_revision=self.code_revision,
                approval=approval,
                executors=deps.executors,
                dependencies=replace(
                    deps.pipeline, remaining_attempts=lambda: self.ledger.allowance(name)
                ),
            ),
            stopped=lambda result: result.stopped,
        )
        if result is None:
            return PipelineOutcome(task_id=binding.task_id, binding=binding, stopped=reason)
        return PipelineOutcome(
            task_id=binding.task_id,
            binding=binding,
            run=result.run,
            complete=result.complete,
            stopped=result.stopped,
            stages={row.stage: row.record for row in result.stages},
        )


_PREPARATION_LINES = (
    "Variance build: exposure fixed before tagging, decision tags, tag clusters, environment "
    "families reviewed for compatibility, per-family scoped index, world model and probe "
    "routing.",
    "Family rules learned once per family and frozen into its runtime; one authored, reviewed, "
    "captured and validated start registered per selected probe task.",
)


def _build_stage(project: ProjectStore, spec: FrontDoorSpec) -> FrontDoorStage:
    """Plan the project build: saved for this exact source, or pending through ``exp build``."""
    if spec.traces is None:
        if project.load_project().build is None:
            raise ValueError("the project has no completed build; point traces at a trace file")
        return FrontDoorStage(
            stage="project-build",
            status=StageState.DONE,
            lines=("Uses the project's completed build as it is.",),
        )
    source = spec.traces
    if recorded_build(project, source) is not None:
        return FrontDoorStage(
            stage="project-build",
            status=StageState.DONE,
            lines=(f"Built from {source.path} ({source.sha256()[:12]}); replays with no call.",),
        )
    return FrontDoorStage(
        stage="project-build",
        status=StageState.PLANNED,
        lines=(
            f"exp build over {source.path} ({source.source}): import, task mining, both RAG "
            "indexes and the grounded world model.",
        ),
        notes=(
            "Embeddings are priced by exp build's own preflight; consent to this plan is its "
            "confirmation (--yes), including an estimate above the build budget or an "
            "undefined one from an unpriced embedder.",
        ),
    )


def _pending_build_stages(
    spec: FrontDoorSpec, dependencies: FrontDoorDependencies
) -> tuple[FrontDoorStage, ...]:
    """Name every stage after a pending project build as blocked, in run order, with no plan."""
    waiting = "Priced once the project build exists."
    slots = configured_slots(spec)

    def blocked(position: SlotPosition) -> list[FrontDoorStage]:
        rows = []
        for name in slots:
            if SLOT_POSITIONS[name] == position:
                slot_executor(name, dependencies.slots)  # still refuse a slot without executor
                rows.append(
                    FrontDoorStage(
                        stage=f"slot {name}",
                        status=StageState.BLOCKED,
                        lines=(SLOT_STAGES[name],),
                        notes=(waiting,),
                    )
                )
        return rows

    limits = spec.preparation.limits
    return (
        *blocked("after-families"),
        FrontDoorStage(
            stage="bootstrap-preparation",
            status=StageState.BLOCKED,
            lines=_PREPARATION_LINES,
            notes=(
                f"{waiting} Tagging, compatibility review, family rules and at most "
                f"{limits.maximum_tasks} starts.",
            ),
        ),
        *blocked("after-preparation"),
        FrontDoorStage(
            stage="pipelines",
            status=StageState.BLOCKED,
            lines=("One six-stage decision-variance run per registered start.",),
            notes=(
                f"{waiting} At most {limits.maximum_tasks} starts, "
                f"{len(spec.pipeline.mining.actors)} actor tiers each.",
            ),
        ),
        *blocked("after-pipeline"),
    )


def _saved_family_slots(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str
) -> dict[str, ArtifactInput]:
    """The saved results of the families slots, as later slots' plans read them."""
    saved: dict[str, ArtifactInput] = {}
    for name, depends in (("tag-banks", ()), ("task-types", ("tag-banks",)), ("state-rag", ())):
        if name in spec.slots:
            context = SlotContext(project, spec.preparation, code_revision, dict(saved))
            found = saved_slot(context, name, spec.slots[name], depends)
            if found is not None:
                saved[name] = found
    return saved


def _preparation_stage(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str
) -> tuple[FrontDoorStage, BootstrapPreparationReport | None, ArtifactInput | None]:
    """Plan the preparation and return its saved report and artifact once it completed."""
    plan = plan_bootstrap_preparation(project, spec.preparation, code_revision=code_revision)
    result_id = plan.identity + "-result"
    if project.artifacts.exists(result_id):
        report = BootstrapPreparationReport.model_validate_json(
            project.artifacts.read_bytes(result_id, "report.json")
        )
        lines = (
            f"Saved: {len(report.bindings)} registered starts, {len(report.refused_tasks)} "
            f"refused with diagnostics, {len(report.simulators)} frozen family simulators.",
        )
        saved = artifact_input(project.artifacts.read(result_id).manifest)
        return (
            FrontDoorStage(stage="bootstrap-preparation", status=StageState.DONE, lines=lines),
            report,
            saved,
        )
    progress = (
        f"Tasks completed {len(plan.completed_tasks)}, remaining {plan.task_bound}; families "
        f"learning rules {plan.family_bound}.",
    )
    return (
        FrontDoorStage(
            stage="bootstrap-preparation",
            status=StageState.PLANNED,
            costs=plan.costs,
            lines=(*_PREPARATION_LINES, *progress, *plan.lines),
        ),
        None,
        None,
    )


def _pipeline_plan(
    project: ProjectStore,
    spec: FrontDoorSpec,
    report: BootstrapPreparationReport,
    binding: BootstrapBinding,
    code_revision: str,
    dependencies: FrontDoorDependencies,
) -> PipelinePlan:
    """Plan one registered start's pipeline."""
    return plan_decision_variance(
        project,
        spec.pipeline_spec(report, binding),
        code_revision=code_revision,
        executors=dependencies.executors,
    )


def _pipeline_stage(binding: BootstrapBinding, plan: PipelinePlan) -> FrontDoorStage:
    """Describe one start's pipeline from its own plan."""
    lines = tuple(
        f"{row.stage}: "
        + ("saved, replays with no call" if row.saved else f"up to ${row.known_usd:.6f}")
        for row in plan.stages
    )
    return FrontDoorStage(
        stage=f"pipeline {binding.task_id}",
        status=StageState.DONE if plan.complete else StageState.PLANNED,
        costs=tuple(
            line.model_copy(update={"name": f"{row.stage}: {line.name}"})
            for row in plan.stages
            if not row.saved
            for line in row.costs
        ),
        lines=lines,
    )


def _review_stage(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str, built: StageState
) -> FrontDoorStage:
    """Plan the variance build the ``after-families`` slots read, as its own priced phase."""
    if built != StageState.DONE:
        return FrontDoorStage(
            stage="variance-build",
            status=StageState.BLOCKED,
            notes=("Priced once the project build exists.",),
        )
    context = SlotContext(project, spec.preparation, code_revision)
    saved, costs, lines = families_review_plan(context)
    return FrontDoorStage(
        stage="variance-build",
        status=StageState.DONE if saved else StageState.PLANNED,
        costs=() if saved else costs,
        lines=lines,
    )


def _slot_stages(
    project: ProjectStore,
    spec: FrontDoorSpec,
    position: SlotPosition,
    code_revision: str,
    dependencies: FrontDoorDependencies,
    results: dict[str, ArtifactInput] | None = None,
) -> tuple[FrontDoorStage, ...]:
    """Plan the configured slots at one position, refusing a slot without an executor."""
    rows = []
    for name, settings in configured_slots(spec).items():
        if SLOT_POSITIONS[name] != position:
            continue
        context = SlotContext(project, spec.preparation, code_revision, results or {})
        saved, costs, lines = slot_executor(name, dependencies.slots).plan(context, settings)
        rows.append(
            FrontDoorStage(
                stage=f"slot {name}",
                status=StageState.DONE if saved else StageState.PLANNED,
                costs=() if saved else costs,
                lines=(SLOT_STAGES[name], *lines),
            )
        )
    return tuple(rows)


PARTIAL_TYPE = "bootstrap-preparation-partial"
"""Artifact type of a preparation saved while some of its tasks still fail."""


def save_partial_preparation(
    project: ProjectStore, identity: str, prepared: BootstrapPreparationReport, revision: str
) -> ArtifactInput:
    """Save, or find, a partial preparation report under its content identity.

    Returns:
        The saved partial report.
    """
    artifact_id = f"{identity}-partial-{sha256_json(prepared)[:20]}"
    store = project.artifacts
    if not store.exists(artifact_id):
        refs = (prepared.build_report, *prepared.batches, *prepared.simulators)
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=PARTIAL_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision,
                inputs=tuple(sorted(set(refs), key=lambda ref: ref.artifact_id)),
            ),
            files={"report.json": prepared},
        )
    return artifact_input(store.read(artifact_id).manifest)
