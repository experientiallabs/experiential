"""Tests for judge, critic and continuation-group rewards."""

from __future__ import annotations

import math

import pytest

from exp.optimize.evaluation.variance.rewards import (
    DEFAULT_SELECTOR_OBJECTIVE,
    SELECTOR_OBJECTIVE_V2,
    DecisionVarianceObjective,
    OptionCounts,
    PointDecision,
    PointOutcome,
    SelectorObjective,
    SelectorObjectiveV2,
    choice_shift,
    critic_score,
    decision_drops,
    decision_variance,
    explained_variance,
    group_advantages,
    jensen_shannon,
    judge_score,
    normalized_entropy,
    score_selector,
    selector_score,
    selector_score_v2,
    state_consequence,
    success_difference,
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
    assert score.status == "scored"


def test_a_critic_with_nothing_graded_has_no_reward() -> None:
    """An unscored critic must not receive the best possible reward."""
    score = critic_score({"a": 0.3}, (_point("a", 1, 0, 0, unknown=8),))

    assert score.status == "unscored" and score.reward is None


def test_judge_spread_is_screening_and_noisy_under_the_null() -> None:
    """Two equal points still separate about a fifth of the time with eight continuations each."""
    from math import comb

    n = 8
    mass = [comb(n, k) / 2**n for k in range(n + 1)]
    positive = sum(
        mass[a] * mass[b]
        for a in range(n + 1)
        for b in range(n + 1)
        if judge_score((_point("a", 0, a, n - a), _point("b", 1, b, n - b))).reward > 0
    )
    assert 0.15 < positive < 0.3
    assert judge_score((_point("a", 0, 8, 0), _point("b", 1, 0, 8))).purpose == "screening"


def test_point_intervals_show_how_little_eight_continuations_say() -> None:
    """Four of eight is compatible with almost any underlying rate."""
    interval = _point("a", 1, 4, 4).interval()
    assert interval is not None
    assert interval[0] == pytest.approx(0.215, abs=0.01)
    assert interval[1] == pytest.approx(0.785, abs=0.01)
    assert _point("b", 1, 0, 0, unknown=3).interval() is None


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


def test_decision_variance_is_four_p_one_minus_p_over_judged_labels_only() -> None:
    """Unknown outcomes leave the rate, and too few judged labels leave the point unscored."""
    assert decision_variance(_point("a", 0, 4, 4, unknown=8), 4) == pytest.approx(1.0)
    assert decision_variance(_point("b", 0, 6, 2), 4) == pytest.approx(0.75)
    assert decision_variance(_point("c", 0, 8, 0), 4) == 0.0
    assert decision_variance(_point("d", 0, 2, 1, unknown=5), 4) is None
    assert decision_variance(_point("e", 0, 0, 0, unknown=8), 1) is None


def test_selector_reward_is_mean_variance_of_scored_picks_minus_penalties() -> None:
    """Scored picks average their variance; duplicate and unusable picks cost a share each."""
    outcomes = {
        "split": _point("split", 1, 4, 4),
        "lean": _point("lean", 2, 6, 2),
        "sure": _point("sure", 3, 8, 0),
        "thin": _point("thin", 4, 1, 1, unknown=6),
    }
    objective = SelectorObjective(minimum_scored=4, duplicate_penalty=0.2, unusable_penalty=0.4)

    clean = selector_score(["split", "lean", "thin"], outcomes, objective)
    assert clean.reward == pytest.approx(0.875) and clean.unscored == ("thin",)
    assert clean.variances == {"split": 1.0, "lean": 0.75} and clean.penalty == 0.0

    messy = selector_score(["split", "split", "gone", "sure"], outcomes, objective)
    assert messy.duplicates == 1 and messy.unusable == ("gone",)
    assert messy.relevance == pytest.approx(0.5)
    assert messy.penalty == pytest.approx((0.2 + 0.4) / 4)
    assert messy.reward == pytest.approx(0.5 - 0.15)

    refused = selector_score([], outcomes, objective)
    assert refused.reward == pytest.approx(-0.4) and refused.status == "scored"
    nothing = selector_score(["thin"], outcomes, objective)
    assert nothing.status == "unscored" and nothing.reward is None


def test_selector_draws_of_one_run_form_one_group() -> None:
    """Group-relative advantages rank the draws; an unrewarded draw keeps no advantage."""
    outcomes = {"a": _point("a", 1, 4, 4), "b": _point("b", 2, 8, 0)}
    objective = SelectorObjective(minimum_scored=1)
    rewards = [
        selector_score(picks, outcomes, objective).reward
        for picks in (["a"], ["b"], ["a", "b"], ["missing"])
    ]
    assert rewards == pytest.approx([1.0, 0.0, 0.5, -0.25])
    advantages = group_advantages("run", [*rewards, None]).advantages
    assert advantages[-1] is None
    graded = [value for value in advantages if value is not None]
    assert sum(graded) == pytest.approx(0.0)
    assert graded[0] > graded[2] > graded[1] > graded[3]


# Decision variance: choice shift, consequence and the version 2 selector objective


def _counts(option: str, successes: int, planned: int = 20, unknown: int = 0) -> OptionCounts:
    """Build one forced option's judged counts."""
    return OptionCounts(option_id=option, successes=successes, planned=planned, unknown=unknown)


def test_jensen_shannon_and_entropy_are_bounded_and_exact_at_the_extremes() -> None:
    """Identical distributions diverge by 0, disjoint ones by 1; entropy is normalized."""
    assert jensen_shannon([0.5, 0.5, 0.0], [0.5, 0.5, 0.0]) == 0.0
    assert jensen_shannon([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert 0.0 < jensen_shannon([0.75, 0.25], [0.25, 0.75]) < 1.0
    assert normalized_entropy(["a", "b"], ("a", "b", "other")) == pytest.approx(1.0)
    assert normalized_entropy(["a", "b", "other"], ("a", "b", "other")) == 1.0
    assert normalized_entropy(["a", "a"], ("a", "b", "other")) == 0.0
    assert normalized_entropy([], ("a", "b")) is None


def test_a_choice_shift_needs_a_significant_bounded_divergence_and_recomputes_exactly() -> None:
    """Opposite choices shift; samples from one mixture do not, despite a positive plug-in value."""
    objective = DecisionVarianceObjective(resamples=500)
    categories = ("refund", "decline", "other")
    shifted = choice_shift(
        ("s1", "s2"),
        ["refund"] * 7 + ["decline"],
        ["decline"] * 7 + ["refund"],
        categories,
        objective,
    )
    same = choice_shift(
        ("s1", "s2"),
        ["refund", "decline"] * 4,
        ["decline", "refund", "refund", "decline", "decline", "refund", "refund", "decline"],
        categories,
        objective,
    )

    assert shifted.holds and shifted.p_value <= 0.05 and shifted.interval[0] >= 0.1
    assert not same.holds and same.divergence == 0.0
    assert shifted == choice_shift(
        ("s1", "s2"),
        ["refund"] * 7 + ["decline"],
        ["decline"] * 7 + ["refund"],
        categories,
        objective,
    )
    with pytest.raises(ValueError, match="both states"):
        choice_shift(("s1", "s2"), [], ["refund"], categories, objective)


def test_consequence_uses_conservative_bounds_and_names_a_separated_best() -> None:
    """A clear gap holds with a separated best; unknowns shrink it; equal options never hold."""
    objective = DecisionVarianceObjective()
    clear = state_consequence((_counts("refund", 18), _counts("decline", 2)), objective)
    murky = state_consequence((_counts("refund", 10), _counts("decline", 10)), objective)
    unknowns = state_consequence(
        (_counts("refund", 12, unknown=6), _counts("decline", 2, unknown=6)), objective
    )

    assert clear.holds and clear.better == "refund" and clear.separated_best == "refund"
    assert clear.effect_bounds[0] > 0.4 and clear.effect == pytest.approx(0.8)
    assert not murky.holds and murky.separated_best is None
    assert unknowns.effect == pytest.approx(0.2) and not unknowns.holds
    assert success_difference(_counts("a", 20), _counts("b", 0))[1] > 0.6
    with pytest.raises(ValueError, match="two forced options"):
        state_consequence((_counts("refund", 18),), objective)
    with pytest.raises(ValueError, match="exceed"):
        _counts("refund", 18, unknown=5)


def _decision(point: str, choices: str, rewards: str) -> PointDecision:
    """Build a point's classified choices ('r', 'd', 'u') and judged labels ('1', '0', '-')."""
    names = {"r": "refund", "d": "decline", "u": "unclear"}
    labels = {"1": 1.0, "0": 0.0, "-": None}
    return PointDecision(
        point_id=point,
        choices=tuple(names[c] for c in choices),
        rewards=tuple(labels[r] for r in rewards),
    )


def test_selector_v2_rewards_choices_that_change_success_not_coin_flips() -> None:
    """A coin flip scores 1 under version 1 but 0 under version 2 unless the choice explains it."""
    objective = SelectorObjectiveV2(minimum_scored=4)
    decisive = _decision("decisive", "rrrrdddd", "11110000")
    coin_flip = _decision("coin_flip", "rrrrdddd", "10101010")
    split_no_effect = _decision("split_no_effect", "rrrrdddd", "11111111")
    agreeing = _decision("agreeing", "rrrrrrrr", "10101010")

    assert explained_variance(decisive, objective) == pytest.approx(1.0 - 4 * 0.25 / 8)
    assert explained_variance(coin_flip, objective) == 0.0
    assert explained_variance(split_no_effect, objective) == 0.0
    assert explained_variance(agreeing, objective) == 0.0
    assert explained_variance(_decision("thin", "rrdd", "1--0"), objective) is None
    assert explained_variance(_decision("unclear", "rruuuudd", "11000000"), objective) is None
    assert decision_variance(_point("coin_flip", 0, 4, 4), 4) == 1.0

    outcomes = {
        name: _point(name, i, 4, 4)
        for i, name in enumerate(("decisive", "coin_flip", "family", "silent"))
    }
    decisions = {
        "decisive": decisive,
        "coin_flip": coin_flip,
        "family": _decision("family", "", "").model_copy(update={"joint_score": 0.6}),
    }
    score = selector_score_v2(
        ["decisive", "coin_flip", "family", "silent", "gone", "decisive"],
        outcomes,
        decisions,
        objective,
    )
    assert score.objective == SELECTOR_OBJECTIVE_V2 and score.unscored == ("silent",)
    assert score.variances["coin_flip"] == 0.0 and score.variances["family"] == 0.6
    assert score.unusable == ("gone",) and score.duplicates == 1
    assert score.reward == pytest.approx((0.875 + 0.0 + 0.6) / 3 - 0.5 / 6)
    assert DEFAULT_SELECTOR_OBJECTIVE.version == SELECTOR_OBJECTIVE_V2


def test_both_objective_versions_stay_scorable_by_version() -> None:
    """Version 1 keeps scoring outcome variance; version 2 reads classified choices."""
    outcomes = {"p": _point("p", 0, 4, 4)}
    decisions = {"p": _decision("p", "rrrrdddd", "10101010")}

    old = score_selector(["p"], outcomes, SelectorObjective())
    new = score_selector(["p"], outcomes, SelectorObjectiveV2(), decisions)

    assert old.objective == "selector-decision-variance-v1" and old.reward == 1.0
    assert new.objective == SELECTOR_OBJECTIVE_V2 and new.reward == 0.0
    assert score_selector(["p"], outcomes, SelectorObjectiveV2()).status == "unscored"
    with pytest.raises(ValueError, match="align"):
        PointDecision(point_id="p", choices=("refund",), rewards=())
