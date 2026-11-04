"""The actual pipeline build executor supplies compatibility review and prices its calls."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.variance.build import plan_variance_build
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.pipeline_review import review_costs
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildSettings,
    StageContext,
    StageDependencies,
)
from exp.optimize.evaluation.variance.pipeline_stages import BUILD_STAGE
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
    plan = plan_variance_build(project, registered, code_revision="test")
    assert plan.saved and review_costs(project, settings, plan) == ()
