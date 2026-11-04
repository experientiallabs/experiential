"""Rewards for the point-choosing judge, the value-estimating critic, and continuation groups.

The judge chooses estimation points in one trajectory; the critic estimates the probability of
success from each; continuations from each point measure that probability. Everything here is pure
arithmetic over those measurements, so a reward can be recomputed exactly from saved artifacts.

- **Critic** reward is the negative Brier score of its estimates against the measured rates. It is
  computed only over points that have at least one graded continuation; an unknown outcome is
  neither a success nor a failure and is left out of the rate. A critic with no scored point is
  ``unscored`` and has no reward, so it can never look perfect and must be left out of training.
- **Judge** screening score is the spread of the measured rates across its points, minus the
  spread binomial sampling alone would produce. It is a candidate-selection heuristic, not a
  training reward: rates also differ across points because of ordinary progress, newly revealed
  information, spent budget or an already irreversible mistake, none of which is a decision, and
  with two equal points and eight continuations each it is still positive about a fifth of the
  time. Whether a point is a decision is established only by running competing actions from the
  same checkpoint.
- **Point selector** training reward (objective :data:`SELECTOR_OBJECTIVE`) is how
  decision-relevant the draw's chosen points proved under the judged continuations: the mean, over
  its scored points, of the judged outcome variance ``4 * p * (1 - p)`` (``p`` the judged success
  rate there; a point with fewer than :attr:`SelectorObjective.minimum_scored` judged pass or fail
  continuations is unscored and left out), minus a configured penalty for each duplicate or
  unusable pick as a share of the draw's picks. The ``J`` draws over one picked trajectory form one
  group and are normalized together with :func:`group_advantages`. A point is relevant here when
  the frozen actor's continuations from it split, which the selector can be rewarded for finding;
  that is still not proof that a competing action decides the outcome. This module and the
  records built from it never run an optimizer: an external trainer consumes them.
- **Continuation groups** are normalized within each point, so continuations from one point form
  their own group and are never compared with continuations from another starting state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.evaluations.uncertainty import wilson_interval


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

    def interval(self) -> tuple[float, float] | None:
        """Two-sided 95 percent Wilson interval of the rate, or None when nothing was graded.

        Four passes in eight graded continuations gives roughly 0.22 to 0.78: screening evidence,
        not a precise value.
        """
        rate = self.rate
        return None if rate is None else wilson_interval(rate, self.graded)


class CriticScore(ContractModel):
    """One critic attempt scored against measured outcomes.

    Attributes:
        status: ``scored`` when at least one point had a graded continuation, else ``unscored``.
        reward: Negative mean Brier score over scored points, in [-1, 0]; None when unscored.
        errors: Squared error per scored point identity.
        unscored: Point identities with no graded continuation.
    """

    status: Literal["scored", "unscored"]
    reward: float | None
    errors: dict[str, float]
    unscored: tuple[str, ...]


class JudgeScore(ContractModel):
    """One judge attempt's screening score: how much its points' outcomes separate.

    Attributes:
        purpose: Always ``screening``; this score never enters a training record.
        reward: Rate variance across scored points minus the expected sampling variance,
            floored at zero.
        observed_variance: Variance of measured rates across scored points.
        noise_variance: Mean binomial sampling variance of those rates.
        scored_points: Points that contributed a measured rate.
    """

    purpose: Literal["screening"] = "screening"
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
    if not errors:
        return CriticScore(status="unscored", reward=None, errors={}, unscored=tuple(unscored))
    reward = -sum(errors.values()) / len(errors)
    return CriticScore(status="scored", reward=reward, errors=errors, unscored=tuple(unscored))


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


SELECTOR_OBJECTIVE = "selector-decision-variance-v1"
"""Version of the point-selector training objective; bump when :func:`selector_score` changes."""


class SelectorObjective(ContractModel):
    """The frozen point-selector training objective.

    Attributes:
        version: Objective version.
        minimum_scored: Fewest judged pass or fail continuations a point needs to be scored.
        duplicate_penalty: Reward removed per repeated pick, as a share of the draw's picks.
        unusable_penalty: Reward removed per pick with no continuations (not offered, or its
            continuations could not run), as a share of the draw's picks. A draw with a saved
            answer but no usable selection counts every pick as unusable.
    """

    version: Literal["selector-decision-variance-v1"] = SELECTOR_OBJECTIVE
    minimum_scored: int = Field(default=4, ge=1, le=100, strict=True)
    duplicate_penalty: float = Field(default=0.25, ge=0.0, le=1.0)
    unusable_penalty: float = Field(default=0.25, ge=0.0, le=1.0)


class SelectorScore(ContractModel):
    """One point-selector draw scored under :class:`SelectorObjective`.

    Attributes:
        objective: The objective version.
        status: ``scored`` when the reward is defined, else ``unscored``.
        reward: Mean judged outcome variance over scored points minus the penalty; None when no
            pick was scored and none was penalized.
        relevance: Mean judged outcome variance over scored points, or None when none scored.
        variances: ``4 * p * (1 - p)`` per scored point identity.
        unscored: Picks whose continuations ran but have too few judged labels.
        unusable: Picks with no continuations.
        duplicates: Repeated picks.
        penalty: Total reward removed for duplicate and unusable picks.
    """

    objective: str = SELECTOR_OBJECTIVE
    status: Literal["scored", "unscored"]
    reward: float | None
    relevance: float | None
    variances: dict[str, float]
    unscored: tuple[str, ...]
    unusable: tuple[str, ...]
    duplicates: int = Field(ge=0)
    penalty: float = Field(ge=0.0)


def decision_variance(outcome: PointOutcome, minimum_scored: int) -> float | None:
    """Return the judged outcome variance ``4 * p * (1 - p)`` of one point, in [0, 1].

    Args:
        outcome: The point's judged counts; unknown outcomes are excluded from ``p``.
        minimum_scored: Fewest judged pass or fail continuations needed.

    Returns:
        The variance, or None when fewer than ``minimum_scored`` continuations were judged.
    """
    rate = outcome.rate
    if rate is None or outcome.graded < minimum_scored:
        return None
    return 4.0 * rate * (1.0 - rate)


def selector_score(
    picks: Sequence[str],
    outcomes: Mapping[str, PointOutcome],
    objective: SelectorObjective,
) -> SelectorScore:
    """Score one point-selector draw by how decision-relevant its picks proved.

    Args:
        picks: Point identities the draw chose, in answer order, repeats kept. An empty sequence
            stands for a draw whose saved answer gave no usable selection.
        outcomes: Judged outcomes of every point whose continuations ran; a pick absent here is
            unusable.
        objective: The frozen objective.

    Returns:
        The draw's score.
    """
    seen: set[str] = set()
    duplicates = 0
    unusable: list[str] = []
    unscored: list[str] = []
    variances: dict[str, float] = {}
    for pick in picks:
        if pick in seen:
            duplicates += 1
            continue
        seen.add(pick)
        outcome = outcomes.get(pick)
        if outcome is None:
            unusable.append(pick)
            continue
        variance = decision_variance(outcome, objective.minimum_scored)
        if variance is None:
            unscored.append(pick)
        else:
            variances[pick] = variance
    if not picks:
        penalty = objective.unusable_penalty
    else:
        penalty = (
            duplicates * objective.duplicate_penalty + len(unusable) * objective.unusable_penalty
        ) / len(picks)
    relevance = sum(variances.values()) / len(variances) if variances else None
    if relevance is None and penalty == 0.0:
        reward = None
    else:
        reward = (relevance or 0.0) - penalty
    return SelectorScore(
        status="unscored" if reward is None else "scored",
        reward=reward,
        relevance=relevance,
        variances=variances,
        unscored=tuple(unscored),
        unusable=tuple(unusable),
        duplicates=duplicates,
        penalty=penalty,
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
