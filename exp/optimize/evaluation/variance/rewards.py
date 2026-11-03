"""Rewards for the point-choosing judge, the value-estimating critic, and continuation groups.

The judge chooses estimation points in one trajectory; the critic estimates the probability of
success from each; continuations from each point measure that probability. Everything here is pure
arithmetic over those measurements, so a reward can be recomputed exactly from saved artifacts.

- **Critic** reward is the negative Brier score of its estimates against the measured rates. It is
  computed only over points that have at least one graded continuation; an unknown outcome is
  neither a success nor a failure and is left out of the rate.
- **Judge** reward is the spread of the measured rates across its points, minus the spread that
  binomial sampling alone would produce. Points whose success probability genuinely differs
  separate good decisions from bad ones; points that differ only because few continuations ran do
  not, and are not rewarded.
- **Continuation groups** are normalized within each point, so continuations from one point form
  their own group and are never compared with continuations from another starting state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel


class PointOutcome(ContractModel):
    """Graded continuations from one estimation point.

    Attributes:
        point_id: Stable identity of the estimation point.
        step: Trajectory step the continuations started before.
        successes: Continuations graded as a pass.
        failures: Continuations graded as a fail.
        unknown: Continuations that could not be graded; they are excluded from the rate.
    """

    point_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    unknown: int = Field(default=0, ge=0)

    @property
    def graded(self) -> int:
        """Continuations with a known pass or fail."""
        return self.successes + self.failures

    @property
    def rate(self) -> float | None:
        """Measured probability of success, or None when nothing was graded."""
        return None if self.graded == 0 else self.successes / self.graded


class CriticScore(ContractModel):
    """One critic attempt scored against measured outcomes.

    Attributes:
        reward: Negative mean Brier score over scored points, in [-1, 0].
        errors: Squared error per scored point identity.
        unscored: Point identities with no graded continuation.
    """

    reward: float
    errors: dict[str, float]
    unscored: tuple[str, ...]


class JudgeScore(ContractModel):
    """One judge attempt's choice of points scored by how much their outcomes separate.

    Attributes:
        reward: Rate variance across scored points minus the expected sampling variance,
            floored at zero.
        observed_variance: Variance of measured rates across scored points.
        noise_variance: Mean binomial sampling variance of those rates.
        scored_points: Points that contributed a measured rate.
    """

    reward: float
    observed_variance: float
    noise_variance: float
    scored_points: int


class GroupAdvantage(ContractModel):
    """Normalized advantages for continuations that started from one point.

    Attributes:
        point_id: Estimation point the group started from.
        advantages: One advantage per continuation in input order; None where ungraded.
    """

    point_id: str
    advantages: tuple[float | None, ...]

    @model_validator(mode="after")
    def _finite(self) -> GroupAdvantage:
        """Reject non-finite advantages before they reach a training record."""
        if any(value is not None and not math.isfinite(value) for value in self.advantages):
            raise ValueError("group advantages must be finite")
        return self


def critic_score(estimates: Mapping[str, float], outcomes: Sequence[PointOutcome]) -> CriticScore:
    """Score one critic attempt against measured continuation rates.

    Args:
        estimates: Estimated probability of success per point identity.
        outcomes: Measured outcomes for the same points.

    Returns:
        The critic's reward and per-point errors.

    Raises:
        ValueError: An estimate is outside [0, 1] or the points do not match.
    """
    by_id = {outcome.point_id: outcome for outcome in outcomes}
    if set(estimates) != set(by_id):
        raise ValueError("critic estimates must cover exactly the measured points")
    errors: dict[str, float] = {}
    unscored: list[str] = []
    for point_id in sorted(estimates):
        estimate = estimates[point_id]
        if not 0.0 <= estimate <= 1.0 or not math.isfinite(estimate):
            raise ValueError(f"critic estimate for {point_id} is not a probability")
        rate = by_id[point_id].rate
        if rate is None:
            unscored.append(point_id)
            continue
        errors[point_id] = (estimate - rate) ** 2
    reward = -sum(errors.values()) / len(errors) if errors else 0.0
    return CriticScore(reward=reward, errors=errors, unscored=tuple(unscored))


def judge_score(outcomes: Sequence[PointOutcome]) -> JudgeScore:
    """Score one judge attempt by how much its points' measured success rates separate.

    Args:
        outcomes: Measured outcomes for the points the judge chose.

    Returns:
        The judge's reward; zero when fewer than two points have a measured rate.
    """
    scored = [(o.rate, o.graded) for o in outcomes if o.rate is not None]
    if len(scored) < 2:
        return JudgeScore(reward=0.0, observed_variance=0.0, noise_variance=0.0, scored_points=0)
    rates = [rate for rate, _ in scored]
    mean = sum(rates) / len(rates)
    observed = sum((rate - mean) ** 2 for rate in rates) / len(rates)
    noise = sum(rate * (1.0 - rate) / count for rate, count in scored) / len(scored)
    return JudgeScore(
        reward=max(0.0, observed - noise),
        observed_variance=observed,
        noise_variance=noise,
        scored_points=len(scored),
    )


def group_advantages(point_id: str, rewards: Sequence[float | None]) -> GroupAdvantage:
    """Normalize one point's continuation rewards into group-relative advantages.

    Args:
        point_id: Estimation point the continuations started from.
        rewards: One reward per continuation; None marks an ungraded continuation.

    Returns:
        Mean-centred, standard-deviation-scaled advantages; all zero when every graded
        continuation earned the same reward, since such a group carries no learning signal.
    """
    graded = [reward for reward in rewards if reward is not None]
    if not graded:
        return GroupAdvantage(point_id=point_id, advantages=tuple(None for _ in rewards))
    mean = sum(graded) / len(graded)
    deviation = math.sqrt(sum((reward - mean) ** 2 for reward in graded) / len(graded))
    return GroupAdvantage(
        point_id=point_id,
        advantages=tuple(
            None if reward is None else (0.0 if deviation == 0.0 else (reward - mean) / deviation)
            for reward in rewards
        ),
    )


def decision_drops(outcomes: Sequence[PointOutcome]) -> dict[str, float]:
    """Return how much the measured success rate falls from each point to the next one.

    A large drop between consecutive points brackets the decision that lost the task, which is
    where a scenario should vary the world. The last scored point has no successor and is absent.

    Args:
        outcomes: Measured outcomes for points of one trajectory.

    Returns:
        Point identity mapped to its rate minus the next scored point's rate.
    """
    scored = sorted(
        (outcome.step, outcome.point_id, rate)
        for outcome in outcomes
        if (rate := outcome.rate) is not None
    )
    return {
        point_id: rate - following
        for (_, point_id, rate), (_, _, following) in zip(scored, scored[1:], strict=False)
    }
