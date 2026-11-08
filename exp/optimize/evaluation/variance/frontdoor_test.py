"""The front door: one consent from a trace file to registered starts and their pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, sha256_json
from exp.common.project import ProjectStore
from exp.common.project.project import write_project_config
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapDependencies,
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.bootstrap_plan_test import (
    PreparationPredictor,
    PreparationScript,
)
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    FrontDoorDependencies,
    FrontDoorSpec,
    PipelineTemplate,
    TraceSource,
    plan_front_door,
    run_directory,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_executors import SlotContext
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    BuildOutputs,
    CostLine,
    DecisionVarianceSpec,
    MiningSettings,
    RecordsSettings,
    RulesOutputs,
    StageContext,
    StageName,
    StageRun,
    TraceMLSettings,
)
from exp.optimize.evaluation.variance.pipeline_test import write_bank


class Seen(ContractModel):
    """What a scripted downstream stage saw.

    Attributes:
        job: Digest of the mining job its spec carried.
        task: The bootstrap binding's task.
    """

    job: str
    task: str


@dataclass
class ScriptedStage:
    """A downstream stage that records the registered job it was handed.

    Attributes:
        stage: The stage it stands in for.
        usd: Its priced bound.
        runs: Specs it ran with.
    """

    stage: StageName
    usd: float = 0.01
    runs: list[DecisionVarianceSpec] = field(default_factory=list)

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Read nothing outside the spec."""
        del project, spec
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price one physical request."""
        del project, spec, code_revision
        return (CostLine(name=self.stage, attempts=1, usd=self.usd),), (f"scripted {self.stage}",)

    def run(self, context: StageContext) -> StageRun:
        """Record the spec and the upstream outputs' job."""
        self.runs.append(context.spec)
        binding = context.spec.mining.bootstrap
        assert binding is not None
        if self.stage == "error-mining":
            rules = RulesOutputs.model_validate(context.upstream["rules"].outputs)
            assert rules.job == context.spec.mining.registered
            BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        seen = Seen(job=sha256_json(context.spec.mining.registered), task=binding.task_id)
        return StageRun(seen, summary={"task": binding.task_id})


@dataclass
class Builder:
    """A project builder whose fixture project was already built from the trace file.

    Attributes:
        calls: Sources it was asked to build.
    """

    calls: list[TraceSource] = field(default_factory=list)

    def __call__(self, project: ProjectStore, source: TraceSource) -> None:
        """Record the call; the fixture build already came from this source."""
        assert project.load_project().build is not None
        self.calls.append(source)


@dataclass
class Fixture:
    """A project, its front-door spec and the scripted collaborators.

    Attributes:
        project: The project.
        spec: The front-door registration.
        builder: The scripted builder.
        stages: Scripted downstream stages.
        dependencies: Front-door collaborators.
    """

    project: ProjectStore
    spec: FrontDoorSpec
    builder: Builder
    stages: dict[StageName, ScriptedStage]
    dependencies: FrontDoorDependencies


def front_door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    """Point a front door at a trace file over the real build, family and bootstrap owners."""
    project, preparation, bound = preparation_fixture(tmp_path, monkeypatch)
    preparation = _roomy(preparation)
    script, predictor = PreparationScript(), PreparationPredictor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: script
    )
    monkeypatch.setattr(grounded, "HttpxJsonTransport", lambda: predictor)
    monkeypatch.setattr(
        "exp.runtime.models.providers.transport.HttpxJsonTransport", lambda: predictor
    )
    traces = tmp_path / "pointed-traces.jsonl"
    traces.write_text('{"fixture": "the build above was made from these traces"}\n')
    author = preparation.bootstrap.author
    spec = FrontDoorSpec(
        traces=TraceSource(path=traces),
        preparation=preparation,
        pipeline=PipelineTemplate(
            traceml=TraceMLSettings(enabled=False, bank_root=write_bank(tmp_path)),
            mining=MiningSettings(
                actors=(preparation.bootstrap.job.actor_alias,),
                judge=author,
                selector=author,
                critic=author,
            ),
            authoring=AuthoringSettings(
                decision_author=author,
                scenario_author=author,
                world_author=author,
                rubric_author=author,
            ),
            records=RecordsSettings(intent="Complete the selected family task"),
        ),
    )
    stages: dict[StageName, ScriptedStage] = {
        name: ScriptedStage(name) for name in ("traceml", "error-mining", "families", "records")
    }
    builder = Builder()
    dependencies = FrontDoorDependencies(
        builder=builder,
        bootstrap=BootstrapDependencies(reviewer=bound, tagger=_tagger(project, FakeDecisions())),
        executors=stages,
    )
    return Fixture(project, spec, builder, stages, dependencies)


def _roomy(spec: BootstrapPreparationSpec) -> BootstrapPreparationSpec:
    """Give every served fixture world room for capture and reveal checks."""
    runtime = spec.bootstrap.runtime.model_copy(update={"maximum_physical_attempts": 20})
    return spec.model_copy(
        update={"bootstrap": spec.bootstrap.model_copy(update={"runtime": runtime})}
    )


def test_a_trace_file_reaches_registered_mining_with_no_hand_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build, tagging, families, rules, starts and every start's pipeline under one consent."""
    run = front_door(tmp_path, monkeypatch)
    project, spec = run.project, run.spec
    plan = plan_front_door(project, spec, code_revision="test", dependencies=run.dependencies)
    assert [row.status for row in plan.stages] == ["planned", "blocked", "blocked", "planned"]
    assert plan.stages[-1].stage == "slot dense-hybrid"
    with pytest.raises(ValueError, match="consent"):
        run_front_door(project, spec, code_revision="test", dependencies=run.dependencies)
    approval = FrontDoorApproval(plan_sha256=plan.digest, ceiling_usd=10.0, unpriced_attempts=1_000)
    report = run_front_door(
        project, spec, code_revision="test", approval=approval, dependencies=run.dependencies
    )
    assert report.complete and report.stopped is None, report.stopped
    assert len(run.builder.calls) == 1 and report.source is not None
    assert report.preparation is not None and len(report.bindings) == 2
    mined = run.stages["error-mining"].runs
    assert [row.mining.bootstrap for row in mined] == list(report.bindings)
    for binding, registered in zip(report.bindings, mined, strict=True):
        registration = bootstrapped_registration(project, binding)
        assert registered.mining.registered == registration.job
        assert registered.build.saved_report is not None
    assert {row.task_id for row in report.pipelines} == {row.task_id for row in report.bindings}

    replay = plan_front_door(project, spec, code_revision="test", dependencies=run.dependencies)
    assert replay.complete and replay.known_usd == 0
    assert [row.stage for row in replay.stages][:2] == ["project-build", "bootstrap-preparation"]
    again = run_front_door(project, spec, code_revision="test", dependencies=run.dependencies)
    assert again.complete and len(run.builder.calls) == 1
    assert len(run.stages["error-mining"].runs) == 2


def test_a_phase_priced_on_the_way_stops_beyond_the_consented_ceiling_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved work survives a stop; the rerun's plan prices the now-reachable pipelines."""
    run = front_door(tmp_path, monkeypatch)
    project, spec, deps = run.project, run.spec, run.dependencies
    plan = plan_front_door(project, spec, code_revision="test", dependencies=deps)
    approval = FrontDoorApproval(plan_sha256=plan.digest, ceiling_usd=0.0, unpriced_attempts=1_000)
    stopped = run_front_door(
        project, spec, code_revision="test", approval=approval, dependencies=deps
    )
    assert stopped.stopped is not None and "cumulative ceiling" in stopped.stopped
    assert stopped.preparation is not None and not run.stages["error-mining"].runs

    resumed = plan_front_door(project, spec, code_revision="test", dependencies=deps)
    statuses = {row.stage: row.status for row in resumed.stages}
    assert statuses["project-build"] == statuses["bootstrap-preparation"] == "done"
    pipelines = [row for row in resumed.stages if row.stage.startswith("pipeline ")]
    assert len(pipelines) == 2 and all(row.status == "planned" for row in pipelines)
    assert resumed.known_usd == pytest.approx(2 * 4 * 0.01)
    lines = pipelines[0].lines
    assert lines[0].startswith("variance-build: up to") and lines[3].startswith("error-mining")
    finished = run_front_door(
        project,
        spec,
        code_revision="test",
        approval=FrontDoorApproval(
            plan_sha256=resumed.digest, ceiling_usd=resumed.known_usd, unpriced_attempts=1_000
        ),
        dependencies=deps,
    )
    assert finished.complete and len(run.builder.calls) == 1
    ledger = PhaseLedger(run_directory(project, spec) / "ledger.json")
    phases = [row.phase for row in ledger.state.phases]
    assert phases[:2] == ["project-build", "bootstrap-preparation"]
    pipelines = [phase for phase in phases[2:] if phase.startswith("pipeline ")]
    assert sorted(pipelines) == sorted(f"pipeline {row.task_id}" for row in finished.bindings)
    assert phases[-1] == "slot dense-hybrid"
    assert all(row.measured_attempts is not None for row in ledger.state.phases)
    journal = TransitionJournal(run_directory(project, spec) / "transitions.jsonl").rows()
    refused = [row for row in journal if row["event"] == "refused"]
    assert refused and all(row["to"] == "stopped_at_cap" for row in refused)
    assert not any(row["refused"] for row in journal)


@dataclass
class ScriptedSlot:
    """A slot owner's executor that saves nothing and counts its runs.

    Attributes:
        runs: Settings it ran with.
    """

    runs: list[JsonObject] = field(default_factory=list)

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Price nothing."""
        del context
        return bool(self.runs), (), (f"banks {settings['banks']}",)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Count the run."""
        del context
        self.runs.append(settings)
        return ArtifactInput(artifact_id="authored-banks", sha256="0" * 64)


def test_a_configured_slot_needs_its_owners_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured slot is planned at its fixed position through its executor."""
    run = front_door(tmp_path, monkeypatch)
    spec = run.spec.model_copy(update={"slots": {"tag-banks": {"banks": 2}}})
    slot = ScriptedSlot()
    deps = FrontDoorDependencies(
        builder=run.builder,
        bootstrap=run.dependencies.bootstrap,
        executors=run.dependencies.executors,
        slots={"tag-banks": slot},
    )
    plan = plan_front_door(run.project, spec, code_revision="test", dependencies=deps)
    assert [row.stage for row in plan.stages][:4] == [
        "project-build",
        "variance-build",
        "slot tag-banks",
        "bootstrap-preparation",
    ]
    assert plan.stages[1].status == "blocked"  # priced once the project build exists
    assert plan.stages[2].lines[1] == "banks 2"


def test_a_project_with_no_build_plans_its_build_and_waits_for_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new pool plans: the build is pending and every later stage waits for it.

    The families slots' saved lookups (task types among them) read the variance build, which
    needs a completed build, so they must not be asked before one exists.
    """
    run = front_door(tmp_path, monkeypatch)
    config = run.project.load_project()
    write_project_config(run.project.paths, config.model_copy(update={"build": None}))
    spec = run.spec.model_copy(
        update={"slots": {"tag-banks": {"banks": 2}, "task-types": {"namer_alias": "author"}}}
    )
    plan = plan_front_door(run.project, spec, code_revision="test", dependencies=run.dependencies)
    rows = [(row.stage, row.status) for row in plan.stages]
    assert rows == [
        ("project-build", "planned"),
        ("variance-build", "blocked"),
        ("slot tag-banks", "blocked"),
        ("slot task-types", "blocked"),
        ("bootstrap-preparation", "blocked"),
        ("pipelines", "blocked"),
        ("slot dense-hybrid", "blocked"),
    ]
    assert not plan.complete and plan.known_usd == 0


def test_one_consent_over_a_pending_build_prices_each_waiting_stage_on_the_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consented before the build existed, the run builds, then prices and runs the slots."""
    run = front_door(tmp_path, monkeypatch)
    built = run.project.load_project()
    write_project_config(run.project.paths, built.model_copy(update={"build": None}))

    def build(project: ProjectStore, source: TraceSource) -> None:
        """Make the completed build appear, as exp build would."""
        run.builder.calls.append(source)
        write_project_config(project.paths, built)

    slot = ScriptedSlot()
    deps = FrontDoorDependencies(
        builder=build,
        bootstrap=run.dependencies.bootstrap,
        executors=run.dependencies.executors,
        slots={"tag-banks": slot},
    )
    spec = run.spec.model_copy(update={"slots": {"tag-banks": {"banks": 2}}})
    plan = plan_front_door(run.project, spec, code_revision="test", dependencies=deps)
    assert plan.stages[2].stage == "slot tag-banks" and plan.stages[2].status == "blocked"
    approval = FrontDoorApproval(plan_sha256=plan.digest, ceiling_usd=10.0, unpriced_attempts=1_000)
    report = run_front_door(
        run.project, spec, code_revision="test", approval=approval, dependencies=deps
    )
    assert len(run.builder.calls) == 1 and report.source is not None
    assert slot.runs == [{"banks": 2}]
    assert report.preparation is not None, report.stopped
