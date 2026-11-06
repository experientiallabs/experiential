"""The rubric-evolution slot prices every boundary for the plan and runs present exports."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from exp.common.models import load_model_catalog, write_model_catalog
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
from exp.optimize.evaluation.variance.frontdoor import plan_front_door
from exp.optimize.evaluation.variance.frontdoor_executors import SlotContext
from exp.optimize.evaluation.variance.frontdoor_slots import (
    DEFAULT_SLOT_EXECUTORS,
    SLOT_POSITIONS,
    SLOT_STAGES,
)
from exp.optimize.evaluation.variance.frontdoor_test import front_door
from exp.optimize.evaluation.variance.pipeline_spec import COUNCIL_ACTORS
from exp.optimize.evaluation.variance.rubric_boundary_test import EPOCH_ONE
from exp.optimize.evaluation.variance.rubric_evo_slot import (
    EvolutionProgress,
    RubricEvolutionExecutor,
)
from exp.optimize.evaluation.variance.rubric_evolution import current_version, load_state
from exp.optimize.evaluation.variance.rubric_evolution_test import (
    export,
    registration_in,
    world,
)


def test_the_slot_is_planned_priced_then_runs_the_present_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan shows the stage and its bound; the run stops at the first missing export."""
    project, registration, clef, wire = world(tmp_path, monkeypatch)
    assert SLOT_POSITIONS["rubric-evolution"] == "after-pipeline"
    assert isinstance(DEFAULT_SLOT_EXECUTORS["rubric-evolution"], RubricEvolutionExecutor)
    assert "epoch boundary" in SLOT_STAGES["rubric-evolution"]
    exports = tmp_path / "exports"
    exports.mkdir()
    settings = {
        "registration": registration.model_dump(mode="json"),
        "exports": str(exports),
        "epochs": 2,
    }
    context = SlotContext(project, cast(BootstrapPreparationSpec, None), "test")
    executor = RubricEvolutionExecutor(tagger=clef)
    saved, costs, lines = executor.plan(context, settings)
    assert not saved and costs and "2 epoch boundaries" in lines[0]
    assert {"clef", "glm-5.3-flash", "kimi-k3"} <= {row.name for row in costs}
    assert wire.requests == [] and clef.calls == 0
    executor.run(context, settings)
    state = load_state(project, registration)
    assert state is not None and state.open_epoch.epoch == 1
    v1 = current_version(project, registration)
    (exports / "epoch-1.json").write_text(
        export(1, 1, v1.identity, EPOCH_ONE).model_dump_json(), encoding="utf-8"
    )
    reference = executor.run(context, settings)
    progress = EvolutionProgress.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "slot.json")
    )
    assert progress.boundaries == {"1": "published"}
    assert progress.stopped == "epoch-2.json is not exported"
    _, remaining, _ = executor.plan(context, settings)
    assert remaining and sum(r.attempts for r in remaining) < sum(r.attempts for r in costs)


def test_the_front_door_dry_run_plan_shows_the_priced_rubric_evolution_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured after the pipelines, the stage is planned with its bound and not yet run."""
    run = front_door(tmp_path, monkeypatch)
    catalog = load_model_catalog(run.project.model_catalog_path)
    models = {**catalog.models, **{alias: catalog.models["judge"] for alias in COUNCIL_ACTORS}}
    models["clef"] = catalog.models["judge"]
    write_model_catalog(
        run.project.model_catalog_path, catalog.model_copy(update={"models": models})
    )
    registration = registration_in(run.project, tmp_path / "evolution-banks")
    settings = {
        "registration": registration.model_dump(mode="json"),
        "exports": str(tmp_path / "exports"),
        "epochs": 3,
    }
    spec = run.spec.model_copy(update={"slots": {"rubric-evolution": settings}})
    plan = plan_front_door(run.project, spec, code_revision="test", dependencies=run.dependencies)
    stages = [row.stage for row in plan.stages]
    assert sorted(stages[-2:]) == ["slot dense-hybrid", "slot rubric-evolution"]
    (row,) = [r for r in plan.stages if r.stage == "slot rubric-evolution"]
    assert row.status == "planned" and row.costs
    assert row.lines[0] == SLOT_STAGES["rubric-evolution"]
    assert "3 epoch boundaries" in row.lines[1]
    assert {"clef", "glm-5.3-flash", "gemma-4-26b-a4b-it"} <= {c.name for c in row.costs}
