"""Cost bounds follow the actual scheduled registration and its role budgets."""

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.actor_critic_costs import actor_critic_costs
from exp.optimize.evaluation.variance.schedule import TransitionRule, preview_rollout_plan
from exp.optimize.evaluation.variance.schedule_test import _schedule, _scripted_metrics, _step


def test_steady_schedule_prices_its_actual_larger_role_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A steady registration prices its added suffixes and judges, with no provider call."""
    project, spec = _schedule(
        tmp_path, monkeypatch, transition=TransitionRule(warmup_rollouts=1), steady_draws=15
    )
    _scripted_metrics(monkeypatch)
    warmup = preview_rollout_plan(project, spec)
    assert warmup.spec is not None
    _step(project, spec)
    steady = preview_rollout_plan(project, spec)
    assert steady.spec is not None and len(steady.spec.judges) == 15
    before = actor_critic_costs(project, warmup.spec)
    after = actor_critic_costs(project, steady.spec)
    expected = steady.actor_job.study.repeats + 15 * spec.base.points * spec.base.continuations
    assert (
        after[-1].attempts == expected * spec.base.episode_judge.settings.maximum_physical_attempts
    )
    assert sum(row.attempts for row in after) > sum(row.attempts for row in before)
