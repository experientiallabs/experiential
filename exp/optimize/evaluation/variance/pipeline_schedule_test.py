"""Scheduled pipeline consent binds the exact role registration before execution."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.pipeline_schedule import (
    require_scheduled_plan,
    scheduled_line,
    scheduled_preview,
)
from exp.optimize.evaluation.variance.pipeline_spec import StageContext, StagePlan
from exp.optimize.evaluation.variance.pipeline_stages import MINING_STAGE
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec, TransitionRule
from exp.optimize.evaluation.variance.schedule_test import _schedule, _scripted_metrics, _step


def test_mining_prices_and_binds_the_scheduled_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady draws change the approved plan and a stale preview cannot authorize execution."""
    run = wired(tmp_path, monkeypatch)
    project = run.fixture.project
    _, base = _schedule(tmp_path / "schedule-fixture", monkeypatch)
    original = make_spec(run.fixture, run.rubric, run.bank, actors=("actor",))
    schedule = TrainingScheduleSpec(
        name="bound",
        base=base.base.model_copy(update={"initial": original.mining.job}),
        transition=TransitionRule(warmup_rollouts=1),
        steady_draws=15,
    )
    spec = original.model_copy(
        update={"mining": original.mining.model_copy(update={"schedule": schedule})}
    )
    _scripted_metrics(monkeypatch)
    before = scheduled_preview(project, spec)
    assert before is not None
    costs, lines = MINING_STAGE.plan(project, spec, "test")
    context = StageContext(
        project=project,
        spec=spec,
        identity="mining",
        code_revision="test",
        plan=StagePlan(
            stage="error-mining", identity="mining", saved=False, costs=costs, lines=lines
        ),
    )
    require_scheduled_plan(context, before)
    _step(project, schedule)
    after = scheduled_preview(project, spec)
    assert after is not None and "J=15" in scheduled_line(after)
    larger, updated = MINING_STAGE.plan(project, spec, "test")
    assert sum(row.attempts for row in larger) > sum(row.attempts for row in costs)
    assert scheduled_line(after) in updated
    with pytest.raises(ValueError, match="changed since spend planning"):
        require_scheduled_plan(context, after)
    _step(project, schedule)
    with pytest.raises(ValueError, match="actor-only"):
        MINING_STAGE.plan(project, spec, "test")
