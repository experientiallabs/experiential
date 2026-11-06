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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapApproval,
    BootstrapDependencies,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
    plan_bootstrap_preparation,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger, measure
from exp.optimize.evaluation.variance.frontdoor_slots import (
    SLOT_POSITIONS,
    SLOT_STAGES,
    SlotExecutor,
    SlotName,
    SlotPosition,
    slot_executor,
)
from exp.optimize.evaluation.variance.frontdoor_states import StageState, Tracker
from exp.optimize.evaluation.variance.pipeline import (
    StageExecutor,
    plan_decision_variance,
    run_decision_variance,
)
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    CostLine,
    DecisionVarianceSpec,
    MiningSettings,
    PipelineApproval,
    PipelinePlan,
    RecordsSettings,
    RulesSettings,
    StageDependencies,
    StageName,
    TraceMLSettings,
)

logger = logging.getLogger(__name__)

FRONT_DOOR_PROTOCOL = "decision-variance-front-door-v1"
"""Version of the front-door sequence and its source record."""

SOURCE_TYPE = "front-door-source"
"""Artifact type binding one trace source's digest to the project build it produced."""


class TraceSource(ContractModel):
    """The trace file the front door is pointed at.

    Attributes:
        path: Local canonical trace export, or the gateway capture file with ``identity``.
        source: Canonical trace source format, as ``exp build --source`` takes it.
        identity: Gateway identity whose captures supply the build, with ``source`` gateway.
    """

    path: Path
    source: str = Field(default="otlp", min_length=1, max_length=64)
    identity: str | None = Field(default=None, min_length=1)

    def sha256(self) -> Sha256:
        """Return the digest of the trace file's exact bytes."""
        return sha256_bytes(self.path.read_bytes())


class PipelineTemplate(ContractModel):
    """Every pipeline setting except the build and the start, which the front door supplies.

    Attributes:
        rules: Rules-stage settings, or None for the preparation's own rules registration. A
            bootstrapped pipeline reuses the family's frozen rules with no call either way.
        traceml: TraceML settings.
        mining: Error-mining settings with neither ``job`` nor ``bootstrap``: each registered
            start fills ``bootstrap``.
        authoring: Family settings.
        records: Records settings.
    """

    rules: RulesSettings | None = None
    traceml: TraceMLSettings
    mining: MiningSettings
    authoring: AuthoringSettings
    records: RecordsSettings

    @model_validator(mode="after")
    def _unbound(self) -> Self:
        """Refuse a hand-registered job or start; the preparation registers them."""
        if self.mining.job is not None or self.mining.bootstrap is not None:
            raise ValueError(
                "the front door binds every start its preparation registers; leave "
                "pipeline.mining.job and pipeline.mining.bootstrap empty"
            )
        return self


class FrontDoorSpec(ContractModel):
    """One front-door registration: a trace source and every downstream setting.

    Attributes:
        traces: The trace source, or None to use the project's completed build as it is.
        preparation: Variance build, environment families, bootstrap roles and finite limits.
        pipeline: Pipeline settings applied to every registered start.
        slots: Settings of the slot stages to run, by slot; each needs a registered executor.
        protocol: Front-door protocol.
    """

    traces: TraceSource | None = None
    preparation: BootstrapPreparationSpec
    pipeline: PipelineTemplate
    slots: dict[SlotName, JsonObject] = Field(default_factory=dict)
    protocol: Literal["decision-variance-front-door-v1"] = FRONT_DOOR_PROTOCOL

    @model_validator(mode="after")
    def _rules(self) -> Self:
        """Require a rules registration for the pipeline."""
        if self.pipeline.rules is None and self.preparation.bootstrap.rules is None:
            raise ValueError("give pipeline.rules when the preparation registers no rules")
        return self

    def pipeline_spec(
        self, report: BootstrapPreparationReport, binding: BootstrapBinding
    ) -> DecisionVarianceSpec:
        """Return one registered start's pipeline registration.

        Args:
            report: The saved preparation.
            binding: One of its registered starts.

        Returns:
            The template with the preparation's saved build pinned and the start bound.
        """
        rules = self.pipeline.rules
        if rules is None:
            assert self.preparation.bootstrap.rules is not None
            rules = RulesSettings(spec=self.preparation.bootstrap.rules)
        return DecisionVarianceSpec(
            build=self.preparation.build.model_copy(update={"saved_report": report.build_report}),
            rules=rules,
            traceml=self.pipeline.traceml,
            mining=self.pipeline.mining.model_copy(update={"bootstrap": binding}),
            authoring=self.pipeline.authoring,
            records=self.pipeline.records,
        )


class FrontDoorStage(ContractModel):
    """One front-door stage before it runs.

    Attributes:
        stage: Its name (``project-build``, ``slot <name>``, ``bootstrap-preparation`` or
            ``pipeline <task>``).
        status: Saved, priced and runnable, or waiting on an earlier stage.
        costs: Upper-bound cost inventory of a pending stage.
        lines: What it does, or did.
        notes: Work without a known bound yet, and what will bound it.
    """

    stage: str
    status: StageState
    costs: tuple[CostLine, ...] = ()
    lines: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def known_usd(self) -> float:
        """Sum of the priced components."""
        return sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Components with physical requests but no known price."""
        return tuple(row for row in self.costs if row.usd is None and row.attempts > 0)


class FrontDoorPlan(ContractModel):
    """Every front-door stage, credential-free.

    Attributes:
        spec_sha256: Digest of the registration.
        code_revision: Producing source identity.
        stages: Every stage in order.
    """

    spec_sha256: Sha256
    code_revision: str
    stages: tuple[FrontDoorStage, ...]

    @property
    def digest(self) -> Sha256:
        """Digest a consent binds."""
        return sha256_json(self)

    @property
    def known_usd(self) -> float:
        """Priced upper bound of every pending stage."""
        return sum(row.known_usd for row in self.stages if row.status == StageState.PLANNED)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Unpriced components of every pending stage, named by stage."""
        return tuple(
            line.model_copy(update={"name": f"{row.stage}: {line.name}"})
            for row in self.stages
            if row.status == StageState.PLANNED
            for line in row.unpriced
        )

    @property
    def complete(self) -> bool:
        """Whether every stage is saved, so a run replays without consent or calls."""
        return all(row.status == StageState.DONE for row in self.stages)


class FrontDoorApproval(ContractModel):
    """One consent for the whole front-door run.

    Attributes:
        plan_sha256: The consented plan.
        ceiling_usd: Dollar ceiling across every phase, including phases priced later.
        unpriced_attempts: Physical attempts allowed to unpriced components across the run.
    """

    plan_sha256: Sha256
    ceiling_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(default=0, ge=0)


class PipelineOutcome(ContractModel):
    """One registered start's pipeline.

    Attributes:
        task_id: The start's task.
        binding: Its registered start.
        run: The saved run registration, once it ran.
        complete: Whether every stage is saved.
        stopped: Why it stopped, when it did.
    """

    task_id: str
    binding: BootstrapBinding
    run: ArtifactInput | None = None
    complete: bool = False
    stopped: str | None = None


class FrontDoorReport(ContractModel):
    """What one front-door run did.

    Attributes:
        plan: The plan it ran under.
        source: The saved trace-source record, when a trace source was given.
        preparation: The saved preparation, once it completed.
        bindings: Every registered start.
        failed_tasks: Starts whose preparation failed and that a rerun retries.
        refused_tasks: Starts refused with a saved diagnostic.
        pipelines: One outcome per registered start that was reached.
        slots: Saved result of every slot stage that ran.
        stopped: Why the run stopped early, when it did.
    """

    plan: FrontDoorPlan
    source: ArtifactInput | None = None
    preparation: ArtifactInput | None = None
    bindings: tuple[BootstrapBinding, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    refused_tasks: tuple[str, ...] = ()
    pipelines: tuple[PipelineOutcome, ...] = ()
    slots: dict[SlotName, ArtifactInput] = Field(default_factory=dict)
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every start's pipeline is complete and nothing stopped or failed."""
        return (
            self.stopped is None
            and not self.failed_tasks
            and len(self.pipelines) == len(self.bindings)
            and all(row.complete for row in self.pipelines)
        )


class ProjectBuilder(Protocol):
    """Makes the project's completed build come from exactly one trace source."""

    def __call__(self, project: ProjectStore, source: TraceSource) -> None:
        """Build, or replay, the project from the source; raise when it cannot."""
        ...


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
    stages.extend(_slot_stages(project, spec, "after-families", code_revision, dependencies))
    report: BootstrapPreparationReport | None = None
    if built.status == StageState.DONE:
        preparation, report = _preparation_stage(project, spec, code_revision)
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
    stages.extend(_slot_stages(project, spec, "after-preparation", code_revision, dependencies))
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
    stages.extend(_slot_stages(project, spec, "after-pipeline", code_revision, dependencies))
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
                lambda: _ensure_build(project, traces, deps, self.code_revision),
            )
            if reason is not None:
                return report.model_copy(update={"stopped": reason})
            report = report.model_copy(update={"source": source})
        preparation = plan_bootstrap_preparation(
            project, spec.preparation, code_revision=self.code_revision
        )
        result_id = preparation.identity + "-result"
        stage = "bootstrap-preparation"
        if self.tracker.stages[stage] == StageState.BLOCKED:
            done = project.artifacts.exists(result_id)
            self.tracker.stage(stage, "replayed" if done else "priced")
        if any(SLOT_POSITIONS[name] == "after-families" for name in spec.slots):
            _review_families(project, spec, self.code_revision, deps)
            report = self.slots("after-families", report)

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
        saved = None
        if project.artifacts.exists(result_id):
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
        if self.tracker.stages.get("pipelines") == StageState.BLOCKED:
            self.tracker.stage("pipelines", "replayed")
        outcomes: list[PipelineOutcome] = []
        for binding in prepared.bindings:
            outcome = self.pipeline(prepared, binding)
            outcomes.append(outcome)
            if outcome.stopped is not None:
                return report.model_copy(
                    update={"pipelines": tuple(outcomes), "stopped": outcome.stopped}
                )
        report = report.model_copy(update={"pipelines": tuple(outcomes)})
        return self.slots("after-pipeline", report)

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
        except Exception:
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
        """Run or replay the configured slots at one position."""
        results = dict(report.slots)
        costs = {row.stage: row.costs for row in self.plan.stages}
        for name, settings in self.spec.slots.items():
            if SLOT_POSITIONS[name] != position:
                continue
            executor = slot_executor(name, self.dependencies.slots)
            stage = f"slot {name}"
            result, reason = self.phase(
                stage,
                costs.get(stage, ()),
                lambda executor=executor, settings=settings: executor.run(
                    self.project, settings, self.code_revision
                ),
            )
            if result is None:
                return report.model_copy(update={"slots": results, "stopped": reason})
            results[name] = result
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
                dependencies=deps.pipeline,
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
        )


_PREPARATION_LINES = (
    "Variance build: exposure fixed before tagging, decision tags, tag clusters, environment "
    "families reviewed for compatibility, per-family scoped index, world model and probe "
    "routing.",
    "Family rules learned once per family and frozen into its runtime; one authored, reviewed, "
    "captured and validated start registered per selected probe task.",
)


def _source_identity(source: TraceSource, build: ProjectBuildArtifacts) -> str:
    """Return the record identity binding one trace source's exact content to one build."""
    return stable_id(
        SOURCE_TYPE,
        {
            "protocol": FRONT_DOOR_PROTOCOL,
            "sha256": source.sha256(),
            "source": source.source,
            "identity": source.identity,
            "build": sha256_json(build),
        },
    )


def _recorded_build(project: ProjectStore, source: TraceSource) -> ArtifactInput | None:
    """Return the source record when the project's current build came from this source."""
    build = project.load_project().build
    if build is None:
        return None
    identity = _source_identity(source, build)
    if not project.artifacts.exists(identity):
        return None
    return artifact_input(project.artifacts.read(identity).manifest)


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
    if _recorded_build(project, source) is not None:
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
            "Embeddings are priced by exp build's own preflight; it runs non-interactively only "
            "when that estimate fits the configured build budget, otherwise it stops and asks.",
        ),
    )


def _preparation_stage(
    project: ProjectStore, spec: FrontDoorSpec, code_revision: str
) -> tuple[FrontDoorStage, BootstrapPreparationReport | None]:
    """Plan the preparation and return its saved report once it completed."""
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
        return FrontDoorStage(
            stage="bootstrap-preparation", status=StageState.DONE, lines=lines
        ), report
    progress = (
        f"Tasks completed {len(plan.completed_tasks)}, remaining {plan.task_bound}; families "
        f"learning rules {plan.family_bound}.",
    )
    return FrontDoorStage(
        stage="bootstrap-preparation",
        status=StageState.PLANNED,
        costs=plan.costs,
        lines=(*_PREPARATION_LINES, *progress, *plan.lines),
    ), None


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


def _slot_stages(
    project: ProjectStore,
    spec: FrontDoorSpec,
    position: SlotPosition,
    code_revision: str,
    dependencies: FrontDoorDependencies,
) -> tuple[FrontDoorStage, ...]:
    """Plan the configured slots at one position, refusing a slot without an executor."""
    rows = []
    for name, settings in spec.slots.items():
        if SLOT_POSITIONS[name] != position:
            continue
        saved, costs, lines = slot_executor(name, dependencies.slots).plan(
            project, settings, code_revision
        )
        rows.append(
            FrontDoorStage(
                stage=f"slot {name}",
                status=StageState.DONE if saved else StageState.PLANNED,
                costs=() if saved else costs,
                lines=(SLOT_STAGES[name], *lines),
            )
        )
    return tuple(rows)


def _review_families(
    project: ProjectStore,
    spec: FrontDoorSpec,
    code_revision: str,
    dependencies: FrontDoorDependencies,
) -> None:
    """Run the preparation's own variance build first, so its saved families exist.

    The call is the one the preparation makes, with the same collaborators, so the preparation
    replays it with no call. It runs only when an ``after-families`` slot needs the families.
    """
    collaborators = dependencies.bootstrap
    build = spec.preparation.build
    reviewer = collaborators.reviewer or family_reviewer(
        project, build, code_revision=code_revision
    )
    run_variance_build(
        project,
        build.spec,
        code_revision=code_revision,
        tagger=collaborators.tagger,
        reviewer=reviewer,
        namer=collaborators.namer,
    )


def _ensure_build(
    project: ProjectStore,
    source: TraceSource,
    dependencies: FrontDoorDependencies,
    code_revision: str,
) -> ArtifactInput:
    """Build the project from the source unless its record is current, then save the record.

    Raises:
        ValueError: The build is pending without a builder, or the builder left no build.
    """
    recorded = _recorded_build(project, source)
    if recorded is not None:
        return recorded
    if dependencies.builder is None:
        raise ValueError("the project build is pending and no project builder was given")
    logger.info("front door: building project from %s", source.path)
    dependencies.builder(project, source)
    build = project.load_project().build
    if build is None:
        raise ValueError("the project builder finished without a completed build")
    identity = _source_identity(source, build)
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=SOURCE_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(
                    sorted(
                        (build.trace_dataset, build.task_set, build.world_model),
                        key=lambda ref: ref.artifact_id,
                    )
                ),
            ),
            files={"build.json": build, "source.json": source},
        )
    return artifact_input(project.artifacts.read(identity).manifest)
