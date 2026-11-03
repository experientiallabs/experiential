"""Tests for judge, critic and continuation-group rewards."""

from __future__ import annotations

import math

import pytest

from exp.optimize.evaluation.variance.rewards import (
    PointOutcome,
    critic_score,
    decision_drops,
    group_advantages,
    judge_score,
)


def _point(
    point_id: str, step: int, successes: int, failures: int, unknown: int = 0
) -> PointOutcome:
    """Build one measured point."""
    return PointOutcome(
        point_id=point_id, step=step, successes=successes, failures=failures, unknown=unknown
    )


def test_critic_reward_is_negative_brier_over_graded_points_only() -> None:
    """An ungraded point is reported, not scored as a miss."""
    outcomes = (_point("a", 1, 3, 1), _point("b", 4, 0, 4), _point("c", 6, 0, 0, unknown=4))

    score = critic_score({"a": 0.75, "b": 0.5, "c": 0.9}, outcomes)

    assert score.errors == {"a": 0.0, "b": 0.25}
    assert score.reward == pytest.approx(-0.125)
    assert score.unscored == ("c",)


def test_critic_estimates_must_match_the_measured_points() -> None:
    """A critic cannot drop a point it would rather not be scored on."""
    with pytest.raises(ValueError, match="exactly the measured points"):
        critic_score({"a": 0.5}, (_point("a", 1, 1, 1), _point("b", 2, 1, 1)))
    with pytest.raises(ValueError, match="not a probability"):
        critic_score({"a": 1.5}, (_point("a", 1, 1, 1),))


def test_judge_reward_subtracts_sampling_noise() -> None:
    """Points that truly separate earn more than points that differ only by noise."""
    separating = judge_score((_point("a", 1, 8, 0), _point("b", 5, 0, 8)))
    noisy = judge_score((_point("a", 1, 1, 0), _point("b", 5, 0, 1)))
    flat = judge_score((_point("a", 1, 4, 4), _point("b", 5, 4, 4)))

    assert separating.reward == pytest.approx(0.25)
    assert flat.reward == 0.0
    assert separating.reward >= noisy.reward
    assert judge_score((_point("a", 1, 4, 4),)).scored_points == 0


def test_group_advantages_stay_within_one_point() -> None:
    """Ungraded continuations keep no advantage, and a uniform group carries no signal."""
    group = group_advantages("a", (1.0, 0.0, None, 1.0))

    assert group.advantages[2] is None
    graded = [value for value in group.advantages if value is not None]
    assert sum(graded) == pytest.approx(0.0)
    assert math.isclose(max(graded), 1 / math.sqrt(2))
    assert group_advantages("b", (1.0, 1.0)).advantages == (0.0, 0.0)
    assert group_advantages("c", (None,)).advantages == (None,)


def test_decision_drops_bracket_the_losing_decision() -> None:
    """The largest drop sits on the point just before the decision that lost the task."""
    drops = decision_drops(
        (_point("late", 9, 0, 4), _point("early", 2, 4, 0), _point("mid", 5, 3, 1))
    )

    assert drops == {"early": pytest.approx(0.25), "mid": pytest.approx(0.75)}
    assert max(drops, key=lambda point: drops[point]) == "mid"
