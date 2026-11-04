"""The actual pipeline build executor supplies compatibility review and prices its calls."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.bootstrap_plan_test import successful_preparation
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build import plan_variance_build
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.pipeline_build import BUILD_STAGE
from exp.optimize.evaluation.variance.pipeline_inputs import world_sources
from exp.optimize.evaluation.variance.pipeline_review import review_costs
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    BuildOutputs,
    BuildSettings,
    DecisionVarianceSpec,
    MiningSettings,
    RecordsSettings,
    RulesOutputs,
    RulesSettings,
    StageContext,
    StageDependencies,
    TraceMLSettings,
)
from exp.optimize.evaluation.variance.pipeline_stages import RULES_STAGE
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired
from exp.simulation.mining.environment_families_test import answer, reviewer


def test_build_stage_delivers_the_injected_family_reviewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real scoped build produces families through BuildStage rather than a direct call."""
    run = wired(tmp_path / "pipeline", monkeypatch)
    (tmp_path / "build").mkdir()
    project = variance_project(tmp_path / "build", monkeypatch)
    spec = make_spec(run.fixture, run.rubric, run.bank)
    registered = family_spec()
    assert registered.environment_families is not None
    settings = BuildSettings(
        spec=registered,
        reviewer=run.fixture.settings.model_copy(
            update={"output_tokens": registered.environment_families.review.maximum_output_tokens}
        ),
    )
    spec = spec.model_copy(update={"build": settings})

    def compatible(members: frozenset[str], variant: str) -> JsonObject:
        """Script one compatible family for every review stance."""
        del variant
        return answer("compatible", members)

    script, bound = reviewer(compatible)
    assert settings.reviewer is not None
    settings = settings.model_copy(
        update={"reviewer": settings.reviewer.model_copy(update={"model": bound.model})}
    )
    spec = spec.model_copy(update={"build": settings})
    context = StageContext(
        project=project,
        spec=spec,
        identity="build-injected-review",
        code_revision="test",
        dependencies=StageDependencies(
            build_tagger=_tagger(project, FakeDecisions()), build_reviewer=bound
        ),
    )
    result = BUILD_STAGE.run(context)
    saved = result.outputs.model_dump(mode="json")["report"]
    assert isinstance(saved, dict) and isinstance(saved["artifact_id"], str)
    report = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(saved["artifact_id"], "report.json")
    )
    assert report.environment_families is not None and report.environment_families.scopes
    assert script.requests
    calls = len(script.requests)
    replay = BUILD_STAGE.run(context)
    assert replay.outputs == result.outputs and len(script.requests) == calls
    plan = plan_variance_build(
        project, registered, code_revision="test", reviewer_model=bound.model
    )
    assert plan.saved and review_costs(project, settings, plan) == ()


def test_prepared_start_keeps_family_inputs_and_rules_through_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real automatic registration selects scoped build inputs and reuses its frozen rules."""
    project, preparation, prepared, _ = successful_preparation(tmp_path, monkeypatch)
    assert prepared.bindings and not prepared.failed_tasks and not prepared.refused_tasks
    author = preparation.bootstrap.author
    assert preparation.bootstrap.rules is not None
    spec = DecisionVarianceSpec(
        build=preparation.build,
        rules=RulesSettings(spec=preparation.bootstrap.rules),
        traceml=TraceMLSettings(enabled=False),
        mining=MiningSettings(
            bootstrap=prepared.bindings[0],
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
    )
    assert spec.mining.job is None
    keys = BUILD_STAGE.keys(project, spec)
    other = spec.model_copy(
        update={
            "mining": spec.mining.model_copy(
                update={
                    "bootstrap": prepared.bindings[1],
                }
            )
        }
    )
    assert keys != BUILD_STAGE.keys(project, other)
    resolved = resolve_bootstrap(project, spec)

    def forbidden() -> None:
        """Saved family build and rules reuse must not construct a provider transport."""
        raise AssertionError("saved family integration attempted a provider call")

    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", forbidden)
    monkeypatch.setattr("exp.runtime.models.providers.transport.HttpxJsonTransport", forbidden)
    context = StageContext(
        project=project, spec=resolved, identity="scoped-build", code_revision="test"
    )
    result = BUILD_STAGE.run(context)
    assert isinstance(result.outputs, BuildOutputs)
    registration = bootstrapped_registration(project, prepared.bindings[0])
    report = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(prepared.build_report.artifact_id, "report.json")
    )
    assert report.environment_families is not None
    scope = report.environment_families.scope(registration.family_id)
    assert result.outputs.world_model == scope.world_model
    assert result.outputs.index == scope.index
    assert result.outputs.retrieval_scope == scope.retrieval_scope
    assert result.outputs.rules_traces == scope.rules_traces
    assert result.outputs.selected_probes == (registration.task_id,)
    assert {source.trace_id for source in world_sources(result.outputs)} == set(scope.rules_traces)
    assert RULES_STAGE.plan(project, resolved, "test")[0] == ()
    rules = RULES_STAGE.run(context)
    assert isinstance(rules.outputs, RulesOutputs)
    assert rules.outputs.job == registration.job
    assert rules.outputs.rulebook == registration.rulebook
    assert rules.outputs.rule_memory == registration.rule_memory
    assert rules.outputs.report == registration.rules_report
