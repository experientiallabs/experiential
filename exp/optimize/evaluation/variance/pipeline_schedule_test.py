"""Scheduled pipeline consent binds the exact role registration before execution."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.actor_critic_costs import point_option_costs
from exp.optimize.evaluation.variance.pipeline_schedule import (
    require_scheduled_plan,
    scheduled_line,
    scheduled_preview,
)
from exp.optimize.evaluation.variance.pipeline_spec import StageContext, StagePlan
from exp.optimize.evaluation.variance.pipeline_stages import MINING_STAGE
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired
from exp.optimize.evaluation.variance.point_options import PointOptionsSpec
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
    options = PointOptionsSpec(proposer=base.base.judges[0], classifier=base.base.critics[0])
    option_schedule = schedule.model_copy(update={"point_options": options})
    option_spec = spec.model_copy(
        update={"mining": spec.mining.model_copy(update={"schedule": option_schedule})}
    )
    option_plan = scheduled_preview(project, option_spec)
    assert option_plan is not None and option_plan.spec is not None
    option_costs, _ = MINING_STAGE.plan(project, option_spec, "test")
    expected = point_option_costs(project, option_schedule, option_plan.spec)
    assert expected and sum(row.attempts for row in option_costs) == (
        sum(row.attempts for row in costs) + sum(row.attempts for row in expected)
    )
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
