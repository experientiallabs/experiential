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

Outcome variance and decision variance are different quantities. ``4 * p * (1 - p)`` is largest
for a coin flip, which is also what a confusing task produces. The decision-variance quantities
below separate three things, each measured from judged evidence:

- **Choice shift** (:func:`choice_shift`): whether the frozen actor's classified first
  consequential choice changes across scenario states, as the Jensen-Shannon divergence between
  the states' choice distributions with a seeded bootstrap interval and a permutation p-value.
- **Outcome consequence** (:func:`state_consequence`): whether the choice changes eventual
  success in a state, as the largest conservative lower bound of a success difference between two
  forced options, plus whether one option is separately best.
- **Clarity**: each state has a separately best option that matches its declared correct option
  and passes an LLM clarity review; high choice entropy without a separated best option is pure
  uncertainty and is refused.

:class:`DecisionVarianceObjective` (:data:`JOINT_OBJECTIVE`) freezes their thresholds and
:func:`~exp.optimize.evaluation.variance.decision_measure.joint_verdict` combines them.
:class:`SelectorObjectiveV2` (:data:`SELECTOR_OBJECTIVE_V2`) is the point-selector objective
that rewards points where both a choice split and its consequence hold: the share of judged
outcome variance explained by the classified choice, or the joint score of a family measured from
the point. :data:`DEFAULT_SELECTOR_OBJECTIVE` is that version; the
``selector-decision-variance-v1`` objective stays available by version, so records made under it
remain referenceable.
"""

from __future__ import annotations

import math
import random
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


# Decision variance: choice shift, outcome consequence, clarity

OTHER = "other"
"""Choice label of an episode whose first consequential action is none of the declared options."""

UNCLEAR = "unclear"
"""Choice label of an episode whose first consequential action could not be classified."""

JOINT_OBJECTIVE = "decision-variance-joint-v1"
"""Version of :class:`DecisionVarianceObjective`; bump when the joint verdict changes."""


class DecisionVarianceObjective(ContractModel):
    """The frozen joint decision-variance objective: choice shift, consequence and clarity.

    Attributes:
        version: Objective version.
        minimum_choice_shift: Smallest bootstrap lower bound of the Jensen-Shannon divergence
            (base 2, in [0, 1]) between two states' choice distributions that counts as a shift.
        significance: Largest permutation p-value at which a choice shift counts.
        minimum_effect: Smallest conservative lower bound of a success difference between two
            forced options at which the choice changes eventual success in a state.
        minimum_consequential_states: Fewest states in which the choice must change success.
        entropy_ceiling: Normalized choice entropy at or above which a state without a separated
            best option is refused as pure uncertainty.
        maximum_unclear: Largest share of a state's episodes the classifier may leave unclear.
        minimum_classified: Fewest classified episodes a state needs to enter a choice shift.
        resamples: Bootstrap and permutation draws per state pair.
        seed: Seed of every bootstrap and permutation draw.
        require_flip: Whether the separated best option must differ between at least two states.
    """

    version: Literal["decision-variance-joint-v1"] = JOINT_OBJECTIVE
    minimum_choice_shift: float = Field(default=0.1, ge=0.0, le=1.0)
    significance: float = Field(default=0.05, gt=0.0, le=1.0)
    minimum_effect: float = Field(default=0.2, ge=0.0, le=1.0)
    minimum_consequential_states: int = Field(default=2, ge=1, le=16, strict=True)
    entropy_ceiling: float = Field(default=0.8, gt=0.0, le=1.0)
    maximum_unclear: float = Field(default=0.25, ge=0.0, le=1.0)
    minimum_classified: int = Field(default=4, ge=1, le=1000, strict=True)
    resamples: int = Field(default=2000, ge=100, le=100_000, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    require_flip: bool = True


class OptionCounts(ContractModel):
    """Judged episodes after forcing one option in one state.

    Attributes:
        option_id: Decision option identity.
        successes: Judged passes.
        planned: Planned episodes, the denominator.
        unknown: Planned episodes without a judged pass or fail.
    """

    option_id: str = Field(min_length=1)
    successes: int = Field(ge=0)
    planned: int = Field(ge=1)
    unknown: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _counts(self) -> OptionCounts:
        """Keep passes and unknowns within the planned episodes."""
        if self.successes + self.unknown > self.planned:
            raise ValueError("successes and unknowns cannot exceed the planned episodes")
        return self

    @property
    def bounds(self) -> tuple[float, float]:
        """Conservative interval: unknowns as failures below and as successes above."""
        return (
            wilson_interval(self.successes / self.planned, self.planned)[0],
            wilson_interval((self.successes + self.unknown) / self.planned, self.planned)[1],
        )


class StateConsequence(ContractModel):
    """Whether the choice changes eventual success in one state.

    Attributes:
        better: Option of the pair with the largest conservative difference lower bound.
        worse: The other option of that pair.
        effect: Point estimate of that pair's success difference, unknowns counted against it.
        effect_bounds: Newcombe interval of that difference over conservative rates.
        separated_best: Option whose lower bound is above every other option's upper bound, or
            None when no option is separately best.
        holds: Whether the effect's lower bound reaches the objective's minimum effect.
    """

    better: str
    worse: str
    effect: float
    effect_bounds: tuple[float, float]
    separated_best: str | None = None
    holds: bool


class ChoiceShift(ContractModel):
    """How much the classified choice distribution differs between two states.

    Attributes:
        states: The two state identities.
        divergence: Jensen-Shannon divergence (base 2) of the two choice distributions.
        interval: Seeded bootstrap 95 percent percentile interval of the divergence.
        p_value: Permutation p-value of the divergence under exchangeable labels.
        holds: Whether the p-value and the interval's lower bound meet the objective.
    """

    states: tuple[str, str]
    divergence: float = Field(ge=0.0, le=1.0)
    interval: tuple[float, float]
    p_value: float = Field(gt=0.0, le=1.0)
    holds: bool


def jensen_shannon(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the base-2 Jensen-Shannon divergence of two distributions, in [0, 1].

    Args:
        left: Probabilities over a fixed category order.
        right: Probabilities over the same order.

    Returns:
        The divergence; 0 for identical distributions, 1 for disjoint supports.
    """
    total = 0.0
    for p, q in zip(left, right, strict=True):
        m = (p + q) / 2.0
        if p > 0.0:
            total += 0.5 * p * math.log2(p / m)
        if q > 0.0:
            total += 0.5 * q * math.log2(q / m)
    return min(1.0, max(0.0, total))


def _distribution(labels: Sequence[str], categories: Sequence[str]) -> list[float]:
    """Return the empirical distribution of classified labels over the categories."""
    count = len(labels)
    return [sum(label == category for label in labels) / count for category in categories]


def normalized_entropy(labels: Sequence[str], categories: Sequence[str]) -> float | None:
    """Return the Shannon entropy of classified labels relative to an even split, in [0, 1].

    The scale is an even split over the declared options, so an even split between the two
    options of a two-option decision is 1 even though :data:`OTHER` is also a category; a spread
    beyond that is capped at 1.

    Args:
        labels: Classified choices, unclear ones already removed.
        categories: Every category a label may take: the declared options, then :data:`OTHER`.

    Returns:
        The normalized entropy, or None when there is no label.
    """
    options = len([category for category in categories if category != OTHER])
    if not labels or options < 2:
        return None
    entropy = -sum(p * math.log2(p) for p in _distribution(labels, categories) if p > 0.0)
    return min(1.0, entropy / math.log2(options))


def choice_shift(
    states: tuple[str, str],
    left: Sequence[str],
    right: Sequence[str],
    categories: Sequence[str],
    objective: DecisionVarianceObjective,
) -> ChoiceShift:
    """Measure how much classified choices differ between two states, with bounds.

    The bootstrap resamples each state's episodes with replacement; the permutation null pools
    both states' labels and redraws the split, since a plug-in divergence of two finite samples
    from one distribution is positive. Both use generators seeded from the objective seed and the
    state pair, so the result recomputes exactly.

    Args:
        states: The two state identities, in order.
        left: Classified choices of the first state, unclear ones removed.
        right: Classified choices of the second state, unclear ones removed.
        categories: Every category a label may take.
        objective: The frozen objective.

    Returns:
        The shift, holding when the p-value is at most the significance and the bootstrap lower
        bound reaches the minimum shift.

    Raises:
        ValueError: Either state has no classified choice.
    """
    if not left or not right:
        raise ValueError("a choice shift needs classified choices in both states")
    observed = jensen_shannon(_distribution(left, categories), _distribution(right, categories))
    rng = random.Random(f"{objective.seed}:{states[0]}:{states[1]}")
    boot: list[float] = []
    null_at_least = 0
    pooled = [*left, *right]
    for _ in range(objective.resamples):
        a = [left[rng.randrange(len(left))] for _ in left]
        b = [right[rng.randrange(len(right))] for _ in right]
        boot.append(jensen_shannon(_distribution(a, categories), _distribution(b, categories)))
        rng.shuffle(pooled)
        split = jensen_shannon(
            _distribution(pooled[: len(left)], categories),
            _distribution(pooled[len(left) :], categories),
        )
        null_at_least += split >= observed - 1e-12
    boot.sort()
    last = len(boot) - 1
    interval = (boot[int(0.025 * last)], boot[math.ceil(0.975 * last)])
    p_value = (1 + null_at_least) / (1 + objective.resamples)
    return ChoiceShift(
        states=states,
        divergence=observed,
        interval=interval,
        p_value=p_value,
        holds=p_value <= objective.significance and interval[0] >= objective.minimum_choice_shift,
    )


def success_difference(better: OptionCounts, worse: OptionCounts) -> tuple[float, float, float]:
    """Return a conservative success difference and its Newcombe hybrid score interval.

    Unknown episodes count as failures of ``better`` and as successes of ``worse``, so missing
    labels can only shrink the difference.

    Args:
        better: The option expected to succeed more.
        worse: The option compared against it.

    Returns:
        The point difference and its lower and upper 95 percent bounds.
    """
    p1 = better.successes / better.planned
    p2 = (worse.successes + worse.unknown) / worse.planned
    l1, u1 = wilson_interval(p1, better.planned)
    l2, u2 = wilson_interval(p2, worse.planned)
    difference = p1 - p2
    lower = difference - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = difference + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return difference, max(-1.0, lower), min(1.0, upper)


def state_consequence(
    options: Sequence[OptionCounts], objective: DecisionVarianceObjective
) -> StateConsequence:
    """Decide whether forcing different options changes judged success in one state.

    Args:
        options: Judged outcomes of every forced option in the state; at least two.
        objective: The frozen objective.

    Returns:
        The pair with the strongest conservative difference, the separately best option when
        one exists, and whether the difference's lower bound reaches the minimum effect.

    Raises:
        ValueError: Fewer than two options were contrasted.
    """
    if len(options) < 2:
        raise ValueError("an outcome consequence needs at least two forced options")
    pairs = [
        (success_difference(a, b), a.option_id, b.option_id)
        for a in options
        for b in options
        if a.option_id != b.option_id
    ]
    (effect, lower, upper), better, worse = max(pairs, key=lambda row: (row[0][1], row[1], row[2]))
    leader = max(options, key=lambda option: (option.bounds[0], option.option_id))
    others = [option.bounds[1] for option in options if option.option_id != leader.option_id]
    separated = leader.option_id if leader.bounds[0] > max(others) else None
    return StateConsequence(
        better=better,
        worse=worse,
        effect=effect,
        effect_bounds=(lower, upper),
        separated_best=separated,
        holds=lower >= objective.minimum_effect,
    )


# Point selector, version 2: reward points where a choice split and its consequence both hold

SELECTOR_OBJECTIVE_V2 = "selector-decision-variance-v2"
"""Version of :class:`SelectorObjectiveV2`; bump when :func:`selector_score_v2` changes."""


class SelectorObjectiveV2(ContractModel):
    """The point-selector objective that rewards decision variance, not outcome variance alone.

    Attributes:
        version: Objective version.
        minimum_scored: Fewest judged pass or fail continuations with a classified choice a
            point needs to be scored.
        maximum_unclear: Largest share of a point's judged continuations whose choice may be
            unclear; above it the point is unscored.
        duplicate_penalty: Reward removed per repeated pick, as a share of the draw's picks.
        unusable_penalty: Reward removed per pick with no continuations, as a share of the
            draw's picks; a draw with a saved answer but no usable selection counts every pick
            as unusable.
    """

    version: Literal["selector-decision-variance-v2"] = SELECTOR_OBJECTIVE_V2
    minimum_scored: int = Field(default=4, ge=1, le=100, strict=True)
    maximum_unclear: float = Field(default=0.25, ge=0.0, le=1.0)
    duplicate_penalty: float = Field(default=0.25, ge=0.0, le=1.0)
    unusable_penalty: float = Field(default=0.25, ge=0.0, le=1.0)


DEFAULT_SELECTOR_OBJECTIVE = SelectorObjectiveV2()
"""The current point-selector objective; version 1 stays available as :class:`SelectorObjective`."""


class PointDecision(ContractModel):
    """Classified choices of one point's judged continuations, and its family verdict.

    Attributes:
        point_id: Stable identity of the estimation point.
        choices: Classified first consequential choice per continuation, in branch order.
        rewards: Judged label per continuation in the same order: 1, 0 or None.
        joint_score: Joint decision-variance score of a family measured from this point, when
            one was measured; it then decides the point's value (0 when not admitted).
    """

    point_id: str = Field(min_length=1)
    choices: tuple[str, ...] = ()
    rewards: tuple[float | None, ...] = ()
    joint_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _aligned(self) -> PointDecision:
        """Keep rewards aligned with choices."""
        if len(self.rewards) != len(self.choices):
            raise ValueError("continuation rewards must align with their choices")
        return self


def explained_variance(point: PointDecision, objective: SelectorObjectiveV2) -> float | None:
    """Return the judged outcome variance explained by the classified choice, in [0, 1].

    With ``p`` the judged success rate and ``p_k`` the rate among continuations choosing ``k``,
    the value is ``4 * (sum_k w_k (p_k - p)^2 - (K - 1) p (1 - p) / N)`` floored at zero: the
    between-choice variance minus what sampling alone would give across ``K`` choice groups. A
    coin-flip point where every choice succeeds half the time scores 0, as does a point whose
    choices split without changing success; only a split whose choices differ in success scores.

    Args:
        point: The point's classified choices and judged labels.
        objective: The frozen objective.

    Returns:
        The explained variance, or None when too few continuations are judged and classified or
        too many choices are unclear.
    """
    judged = [(c, r) for c, r in zip(point.choices, point.rewards, strict=True) if r is not None]
    if not judged:
        return None
    unclear = sum(choice == UNCLEAR for choice, _ in judged)
    pairs = [(choice, reward) for choice, reward in judged if choice != UNCLEAR]
    if unclear / len(judged) > objective.maximum_unclear or len(pairs) < objective.minimum_scored:
        return None
    total = len(pairs)
    rate = sum(reward for _, reward in pairs) / total
    groups: dict[str, list[float]] = {}
    for choice, reward in pairs:
        groups.setdefault(choice, []).append(reward)
    between = sum(
        len(rows) / total * (sum(rows) / len(rows) - rate) ** 2 for rows in groups.values()
    )
    noise = (len(groups) - 1) * rate * (1.0 - rate) / total
    return min(1.0, max(0.0, 4.0 * (between - noise)))


def selector_score_v2(
    picks: Sequence[str],
    outcomes: Mapping[str, PointOutcome],
    decisions: Mapping[str, PointDecision],
    objective: SelectorObjectiveV2,
) -> SelectorScore:
    """Score one point-selector draw by the decision variance its picks proved to have.

    A pick's value is the joint score of a family measured from it when one exists, else the
    outcome variance its classified choice explains. A pick whose continuations ran without
    classified choices is unscored, never penalized.

    Args:
        picks: Point identities the draw chose, in answer order, repeats kept.
        outcomes: Judged outcomes of every point whose continuations ran; a pick absent here is
            unusable.
        decisions: Classified choices and family verdicts by point identity.
        objective: The frozen objective.

    Returns:
        The draw's score; ``variances`` holds each scored pick's value.
    """
    seen: set[str] = set()
    duplicates = 0
    unusable: list[str] = []
    unscored: list[str] = []
    values: dict[str, float] = {}
    for pick in picks:
        if pick in seen:
            duplicates += 1
            continue
        seen.add(pick)
        if pick not in outcomes:
            unusable.append(pick)
            continue
        decision = decisions.get(pick)
        value = (
            None
            if decision is None
            else decision.joint_score
            if decision.joint_score is not None
            else explained_variance(decision, objective)
        )
        if value is None:
            unscored.append(pick)
        else:
            values[pick] = value
    if not picks:
        penalty = objective.unusable_penalty
    else:
        penalty = (
            duplicates * objective.duplicate_penalty + len(unusable) * objective.unusable_penalty
        ) / len(picks)
    relevance = sum(values.values()) / len(values) if values else None
    reward = None if relevance is None and penalty == 0.0 else (relevance or 0.0) - penalty
    return SelectorScore(
        objective=SELECTOR_OBJECTIVE_V2,
        status="unscored" if reward is None else "scored",
        reward=reward,
        relevance=relevance,
        variances=values,
        unscored=tuple(unscored),
        unusable=tuple(unusable),
        duplicates=duplicates,
        penalty=penalty,
    )


def score_selector(
    picks: Sequence[str],
    outcomes: Mapping[str, PointOutcome],
    objective: SelectorObjective | SelectorObjectiveV2,
    decisions: Mapping[str, PointDecision] | None = None,
) -> SelectorScore:
    """Score one point-selector draw under whichever objective version it was registered with.

    Args:
        picks: Point identities the draw chose, in answer order, repeats kept.
        outcomes: Judged outcomes of every point whose continuations ran.
        objective: The frozen objective, version 1 or version 2.
        decisions: Classified choices by point identity; read by version 2 only.

    Returns:
        The draw's score under that version.
    """
    if isinstance(objective, SelectorObjective):
        return selector_score(picks, outcomes, objective)
    return selector_score_v2(picks, outcomes, decisions or {}, objective)
