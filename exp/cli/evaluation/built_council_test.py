"""Public build-to-council preparation without an external council configuration."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.models import load_model_catalog, write_model_catalog
from exp.optimize.evaluation.audit.panel import source_rag_inputs
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
)
from exp.optimize.evaluation.council.built import BuiltCouncil
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


@pytest.mark.parametrize("use_defaults", [False, True])
@pytest.mark.parametrize("maximum_concurrency", [1, 2])
def test_built_project_prepares_automatic_council_without_credentials_or_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, maximum_concurrency: int, use_defaults: bool
) -> None:
    """Actual build artifacts supply public tasks and the fit-only simulator to eval."""
    provider = SimulatedProvider([episode(index) for index in range(USERS)])
    root, project, _ = build_project(tmp_path, monkeypatch, provider)
    build = project.load_project().build
    assert build is not None
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["world"]
    catalog.models["model"] = record.model_copy(update={"model": "actor-one"})
    catalog.models["other"] = record.model_copy(update={"model": "actor-two"})
    default_actors = ("actor-qwen27", "actor-glmflash", "actor-glmfull", "actor-mimo", "actor-kimi")
    for alias in default_actors:
        catalog.models[alias] = record.model_copy(update={"model": alias})
    write_model_catalog(project.model_catalog_path, catalog)
    expected, _ = source_rag_inputs(project, 2)
    before = len(provider.requests)
    monkeypatch.delenv("HELD_OUT_TEST_KEY")
    result = CliRunner().invoke(
        app,
        [
            "eval",
            PROJECT,
            "--root",
            str(root),
            "--trace-analysis",
            "auto",
            *([] if use_defaults else ["--models", "model,other"]),
            "--concurrency",
            str(maximum_concurrency),
            "--dry-run",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "without provider calls" in result.output.lower()
    assert len(provider.requests) == before
    plan_ids = project.artifacts.list_ids(artifact_type="built-trace-council-plan")
    assert len(plan_ids) == 1
    plan = BuiltCouncil.model_validate_json(project.artifacts.read_bytes(plan_ids[0], "plan.json"))
    assert plan.council.world_model.artifact_id in result.output
    assert plan.council.maximum_concurrency == maximum_concurrency
    assert plan.calibration.spec.world_model == plan.council.world_model
    actual = load_grounded_world_model_artifact(project.artifacts, plan.council.world_model)
    original = load_grounded_world_model_artifact(project.artifacts, expected)
    assert actual.serving_rag == original.serving_rag
    assert actual.model == original.model
    assert actual.knowledge == original.knowledge
    assert plan.calibration.spec.task.lineage_group_id != plan.council.tasks[0].lineage_group_id
    expected_actors = default_actors if use_defaults else ("model", "other")
    assert tuple(actor.alias for actor in plan.council.actors) == expected_actors
    assert all(alias in result.output for alias in expected_actors)
    assert plan.council.tasks[0].reference_answer is None
    assert plan.council.analyst_fidelity_plans[0].artifact_id == plan.calibration.id


def test_auto_council_rejects_unregistered_evaluation_overrides_without_writing(
    tmp_path: Path,
) -> None:
    """Worker-evaluation repeats cannot silently change an automatically derived council."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "unbuilt",
            "--root",
            str(tmp_path),
            "--trace-analysis",
            "auto",
            "--models",
            "model,other",
            "--repeats",
            "2",
            "--dry-run",
            "--non-interactive",
        ],
    )
    assert result.exit_code != 0
    assert "cannot combine" in result.output or "override" in result.output
    assert not list(tmp_path.iterdir())
