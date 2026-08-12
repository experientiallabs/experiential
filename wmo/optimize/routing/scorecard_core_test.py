"""Tests for shared three-objective scorecard calculations.

Pure offline aggregation over synthetic outcome matrices: no provider, no judge, no spend.
"""

from __future__ import annotations

import pytest

from wmo.common.providers.base import ProviderKind, TokenUsage
from wmo.common.providers.pool import PoolEntry
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.scorecard_core import (
    EFFECTIVE_COST_RULE,
    Arm,
    CompletionRule,
    ConditionLabel,
    OperatingPoint,
    RowOverhead,
    build_scorecard,
    effective_cost_per_completed_task,
    rows_for_model,
)

_BASE = ConditionLabel(
    base_model="qwen3-8b",
    optimizer="none",
    dataset="tau-bench-retail",
    split="test",
    judge="tau2-verifier",
    provenance="real_episode",
)


def _row(
    sid: str,
    model: str,
    *,
    reward: float | None,
    cost: float = 0.01,
    seconds: list[float] | None = None,
    episode: int = 0,
    tokens: int = 100,
) -> ScenarioOutcome:
    return ScenarioOutcome(
        scenario_id=sid,
        task=f"task for {sid}",
        model=model,
        episode=episode,
        reward=reward,
        success=reward is not None and reward >= 0.5,
        usage=TokenUsage(input_tokens=tokens, output_tokens=tokens // 2),
        cost_usd=cost,
        call_seconds=seconds if seconds is not None else [0.5, 0.5],
        error=None if reward is not None else "sandbox timeout",
    )


def _arm(name: str, rows: list[ScenarioOutcome], **axes: str | int) -> Arm:
    return Arm(name=name, condition=_BASE.replace(optimizer=name, **axes), rows=rows)


# --- effective cost: exclusion semantics -------------------------------------------------


def test_unscored_rows_leave_both_numerator_and_denominator() -> None:
    rows = [
        _row("s1", "m", reward=1.0, cost=0.10),
        _row("s2", "m", reward=0.0, cost=0.10),
        _row("s3", "m", reward=None, cost=0.10),  # infrastructure failure, not a 0
    ]
    cost = effective_cost_per_completed_task(rows)

    assert cost.n_scored == 2
    assert cost.n_excluded == 1
    assert cost.n_completed == 1
    # The unscored row's $0.10 is out of the numerator but visible, never silently dropped.
    assert cost.provider_cost_usd == pytest.approx(0.20)
    assert cost.excluded_cost_usd == pytest.approx(0.10)
    assert cost.cost_per_completed_task_usd == pytest.approx(0.20)
    assert "unscored episode(s) were excluded" in cost.cost_assumptions


def test_no_completed_tasks_reports_undefined_not_zero_or_infinity() -> None:
    cost = effective_cost_per_completed_task([_row("s1", "m", reward=0.0, cost=0.10)])

    assert cost.n_scored == 1
    assert cost.n_completed == 0
    assert cost.cost_per_completed_task_usd is None
    assert cost.total_cost_usd == pytest.approx(0.10)
    assert "reported as undefined rather than as zero" in cost.cost_assumptions


def test_overhead_is_added_to_the_numerator_and_named_in_the_basis() -> None:
    rows = [_row("s1", "m", reward=1.0, cost=0.10), _row("s2", "m", reward=1.0, cost=0.10)]
    overheads = [
        RowOverhead(scenario_id="s1", model="m", component="compressor", cost_usd=0.01),
        RowOverhead(scenario_id="s1", model="m", component="router", cost_usd=0.002),
        RowOverhead(scenario_id="s2", model="m", component="compressor", cost_usd=0.01),
    ]
    cost = effective_cost_per_completed_task(rows, overheads=overheads)

    assert cost.provider_cost_usd == pytest.approx(0.20)
    assert cost.overhead_cost_usd == pytest.approx(0.022)
    assert cost.total_cost_usd == pytest.approx(0.222)
    assert cost.cost_per_completed_task_usd == pytest.approx(0.111)
    assert cost.overhead_components == ["compressor", "router"]
    assert "compressor, router" in cost.cost_assumptions


def test_overhead_on_an_unscored_row_is_excluded_with_it() -> None:
    rows = [_row("s1", "m", reward=1.0, cost=0.10), _row("s2", "m", reward=None, cost=0.10)]
    overheads = [
        RowOverhead(scenario_id="s1", model="m", component="compressor", cost_usd=0.01),
        RowOverhead(scenario_id="s2", model="m", component="compressor", cost_usd=0.05),
    ]
    cost = effective_cost_per_completed_task(rows, overheads=overheads)

    assert cost.overhead_cost_usd == pytest.approx(0.01)
    assert cost.excluded_cost_usd == pytest.approx(0.15)


def test_reward_threshold_completion_rule_changes_the_denominator() -> None:
    # `success` is False on both rows (reward < 0.5), so the flag rule completes nothing while a
    # partial-credit rubric threshold completes one.
    rows = [_row("s1", "m", reward=0.4, cost=0.10), _row("s2", "m", reward=0.1, cost=0.10)]

    flag = effective_cost_per_completed_task(rows)
    assert flag.n_completed == 0
    assert flag.cost_per_completed_task_usd is None

    rubric = effective_cost_per_completed_task(
        rows, completion=CompletionRule(kind="reward_at_least", threshold=0.3)
    )
    assert rubric.n_completed == 1
    assert rubric.cost_per_completed_task_usd == pytest.approx(0.20)
    assert "verified reward is at least 0.3" in rubric.cost_assumptions


def test_priced_tokens_recorded_at_zero_dollars_are_surfaced() -> None:
    rows = [_row("s1", "m", reward=1.0, cost=0.0, tokens=1000)]
    cost = effective_cost_per_completed_task(rows)

    assert cost.zero_cost_rows == 1
    assert "recorded tokens but $0" in cost.cost_assumptions


def test_overhead_naming_an_episode_the_arm_lacks_is_rejected() -> None:
    with pytest.raises(ValueError, match="overhead rows name episodes not present here"):
        Arm(
            name="a",
            condition=_BASE,
            rows=[_row("s1", "m", reward=1.0)],
            overheads=[
                RowOverhead(scenario_id="s9", model="m", component="compressor", cost_usd=1.0)
            ],
        )


# --- scorecard: same scenarios, comparability --------------------------------------------


def test_scorecard_compares_only_scenarios_scored_on_both_sides() -> None:
    # s3 is scored for the arm and unscored for the anchor, so it leaves the comparison; a
    # per-side mean would have let the arm bank an easy win the anchor never got graded on.
    arm = _arm(
        "distill",
        [
            _row("s1", "student", reward=1.0, cost=0.01),
            _row("s2", "student", reward=0.0, cost=0.01),
            _row("s3", "student", reward=1.0, cost=0.01),
        ],
    )
    anchor = _arm(
        "teacher",
        [
            _row("s1", "teacher", reward=1.0, cost=0.10),
            _row("s2", "teacher", reward=1.0, cost=0.10),
            _row("s3", "teacher", reward=None, cost=0.10),
        ],
        base_model="glm-5.2",
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.scenarios_compared == 2
    assert card.scenarios_excluded == 1
    assert card.quality.mean_reward == pytest.approx(0.5)
    assert card.anchor_quality.mean_reward == pytest.approx(1.0)
    assert card.quality_delta_points == pytest.approx(-50.0)
    # One completed of two on the arm at $0.01 each; two of two on the anchor at $0.10 each.
    assert card.cost.cost_per_completed_task_usd == pytest.approx(0.02)
    assert card.anchor_cost.cost_per_completed_task_usd == pytest.approx(0.10)
    assert card.cost_delta_percent == pytest.approx(-80.0)
    assert card.provenance == "real_episode"
    assert card.judge == "tau2-verifier"


def test_scorecard_raises_when_no_scenario_is_scored_on_both_sides() -> None:
    arm = _arm("distill", [_row("s1", "student", reward=1.0), _row("s2", "student", reward=None)])
    anchor = _arm(
        "teacher",
        [_row("s1", "teacher", reward=None), _row("s2", "teacher", reward=1.0)],
        base_model="glm-5.2",
    )
    with pytest.raises(ValueError, match="share no scenario scored on BOTH sides"):
        build_scorecard(arm=arm, anchor=anchor)


def test_simulated_arm_never_silently_scores_against_a_real_anchor() -> None:
    arm = Arm(
        name="distill",
        condition=_BASE.replace(optimizer="distill", provenance="wm_simulated"),
        rows=[_row("s1", "student", reward=1.0)],
    )
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),  # real_episode
        rows=[_row("s1", "teacher", reward=1.0)],
    )
    with pytest.raises(ValueError, match="provenance='wm_simulated'"):
        build_scorecard(arm=arm, anchor=anchor)


@pytest.mark.parametrize("axis", ["judge", "dataset", "split"])
def test_mismatched_comparability_axes_raise(axis: str) -> None:
    arm = _arm("distill", [_row("s1", "student", reward=1.0)])
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2", **{axis: "other"}),
        rows=[_row("s1", "teacher", reward=1.0)],
    )
    with pytest.raises(ValueError, match=f"{axis}='other'"):
        build_scorecard(arm=arm, anchor=anchor)


def test_arm_and_anchor_with_the_same_condition_label_raise() -> None:
    rows_a = [_row("s1", "student", reward=1.0)]
    rows_b = [_row("s1", "teacher", reward=1.0)]
    arm = Arm(name="a", condition=_BASE, rows=rows_a)
    anchor = Arm(name="b", condition=_BASE, rows=rows_b)
    with pytest.raises(ValueError, match="carry the SAME condition label"):
        build_scorecard(arm=arm, anchor=anchor)


def test_latency_is_per_task_and_includes_optimizer_overhead() -> None:
    arm = Arm(
        name="compact",
        condition=_BASE.replace(optimizer="compact"),
        rows=[
            _row("s1", "student", reward=1.0, seconds=[1.0, 1.0]),
            _row("s2", "student", reward=1.0, seconds=[2.0]),
        ],
        overheads=[
            RowOverhead(scenario_id="s1", model="student", component="compressor", latency_s=0.5),
            RowOverhead(scenario_id="s2", model="student", component="compressor", latency_s=0.5),
        ],
    )
    anchor = _arm(
        "teacher",
        [
            _row("s1", "teacher", reward=1.0, seconds=[3.0]),
            _row("s2", "teacher", reward=1.0, seconds=[3.0]),
        ],
        base_model="glm-5.2",
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    # Per task: s1 = 1.0 + 1.0 + 0.5 = 2.5, s2 = 2.0 + 0.5 = 2.5. A per-CALL p50 would have
    # read 1.0 here and understated the arm against the anchor's single 3.0s call.
    assert card.latency.p50_model_s == pytest.approx(2.5)
    assert card.latency.n_tasks == 2
    assert card.anchor_latency.p50_model_s == pytest.approx(3.0)
    assert card.latency_p50_delta_percent == pytest.approx((2.5 - 3.0) / 3.0 * 100.0)


def test_operating_point_without_a_cost_delta_refuses_to_become_an_anchor() -> None:
    point = OperatingPoint(
        name="p", quality_delta_points=1.0, cost_delta_percent=None, dial_position=0.5
    )
    with pytest.raises(ValueError, match="has no cost delta"):
        point.as_cost_quality_anchor()


# --- matrix helper ------------------------------------------------------------------------


def test_rows_for_model_selects_one_pool_model_and_names_a_bad_handle() -> None:
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="student",
                kind=ProviderKind.OPENAI,
                model="custom-student",
                tier="open",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            ),
            PoolEntry(name="teacher", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        ],
        outcomes=[
            _row("s1", "student", reward=1.0),
            _row("s1", "teacher", reward=1.0),
            _row("s2", "student", reward=0.0),
        ],
    )
    assert [o.scenario_id for o in rows_for_model(matrix, "student")] == ["s1", "s2"]
    with pytest.raises(KeyError, match="no pool model named 'ghost'"):
        rows_for_model(matrix, "ghost")


def test_multiple_episodes_of_one_scenario_are_separate_tasks() -> None:
    rows = [
        _row("s1", "m", reward=1.0, cost=0.02, episode=0),
        _row("s1", "m", reward=0.0, cost=0.02, episode=1),
    ]
    cost = effective_cost_per_completed_task(rows)

    assert cost.n_scored == 2
    assert cost.n_completed == 1
    assert cost.cost_per_completed_task_usd == pytest.approx(0.04)


def test_overhead_keys_on_episode_not_just_scenario() -> None:
    rows = [
        _row("s1", "m", reward=1.0, cost=0.02, episode=0),
        _row("s1", "m", reward=1.0, cost=0.02, episode=1),
    ]
    overheads = [
        RowOverhead(scenario_id="s1", model="m", episode=1, component="compressor", cost_usd=0.05)
    ]
    cost = effective_cost_per_completed_task(rows, overheads=overheads)

    assert cost.overhead_cost_usd == pytest.approx(0.05)
    assert cost.cost_per_completed_task_usd == pytest.approx(0.045)


# --- invariants the mutation pass found unprotected -----------------------------------------


def test_effective_cost_rule_is_shipped_verbatim_in_every_basis() -> None:
    cost = effective_cost_per_completed_task([_row("s1", "m", reward=1.0)])
    assert EFFECTIVE_COST_RULE in cost.cost_assumptions


def test_unscored_episode_inside_a_kept_scenario_is_not_averaged_in_as_zero() -> None:
    # Invariant #1 at the one place it is reachable: a scenario the anchor scored, where the arm
    # has one scored episode and one that never returned. Averaging the failure in as 0 would
    # halve the arm's reward and manufacture a regression out of an infrastructure fault.
    arm = Arm(
        name="flaky",
        condition=_BASE.replace(optimizer="flaky"),
        rows=[
            _row("s1", "flaky", reward=1.0, episode=0),
            _row("s1", "flaky", reward=None, episode=1),
        ],
    )
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),
        rows=[_row("s1", "teacher", reward=1.0)],
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.quality.mean_reward == pytest.approx(1.0)
    assert card.quality.n_scored == 1
    assert card.quality.n_excluded == 1
    assert card.quality.task_success_rate == pytest.approx(1.0)
    assert card.quality_delta_points == pytest.approx(0.0)


def test_quality_is_averaged_per_scenario_not_per_episode() -> None:
    # The arm ran three episodes on the easy scenario and one on the hard one. Episode-weighted
    # means would read 0.75 and hand the arm a win it did not earn; scenario-weighted reads 0.5.
    arm = Arm(
        name="lopsided",
        condition=_BASE.replace(optimizer="lopsided"),
        rows=[
            _row("easy", "lopsided", reward=1.0, episode=0),
            _row("easy", "lopsided", reward=1.0, episode=1),
            _row("easy", "lopsided", reward=1.0, episode=2),
            _row("hard", "lopsided", reward=0.0, episode=0),
        ],
    )
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),
        rows=[
            _row("easy", "teacher", reward=1.0),
            _row("hard", "teacher", reward=0.0),
        ],
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.quality.mean_reward == pytest.approx(0.5)
    assert card.quality.n_scenarios == 2
    assert card.quality.n_scored == 4
    assert card.quality_delta_points == pytest.approx(0.0)


# --- money is conserved ----------------------------------------------------------------------


def test_spend_on_scenarios_held_out_of_the_comparison_is_still_reported() -> None:
    # The arm scored s3 and paid $5.00 there; the anchor left s3 unscored, so the scenario drops
    # out entirely. Without the withheld bucket that $5.00 would leave no trace on the card,
    # while n_excluded stayed 0 and the assumptions string claimed a clean accounting.
    arm = Arm(
        name="expensive",
        condition=_BASE.replace(optimizer="expensive"),
        rows=[
            _row("s1", "expensive", reward=1.0, cost=0.01),
            _row("s3", "expensive", reward=1.0, cost=5.00),
        ],
    )
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),
        rows=[
            _row("s1", "teacher", reward=1.0, cost=0.10),
            _row("s3", "teacher", reward=None, cost=7.00),
        ],
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.scenarios_compared == 1
    assert card.scenarios_excluded == 1
    assert card.withheld_cost_usd == pytest.approx(5.00)
    assert card.anchor_withheld_cost_usd == pytest.approx(7.00)
    assert "held out of this comparison" in card.cost_assumptions
    # Conservation: nothing either side spent is missing from the card.
    assert card.cost.total_cost_usd + card.cost.excluded_cost_usd + card.withheld_cost_usd == (
        pytest.approx(sum(r.cost_usd for r in arm.rows))
    )
    assert (
        card.anchor_cost.total_cost_usd
        + card.anchor_cost.excluded_cost_usd
        + card.anchor_withheld_cost_usd
    ) == pytest.approx(sum(r.cost_usd for r in anchor.rows))


def test_scorecard_assumptions_quote_both_sides() -> None:
    arm = _arm("distill", [_row("s1", "student", reward=1.0)])
    anchor = _arm(
        "teacher",
        [_row("s1", "teacher", reward=1.0), _row("s2", "teacher", reward=None)],
        base_model="glm-5.2",
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert "Arm 'distill':" in card.cost_assumptions
    assert "Anchor 'teacher':" in card.cost_assumptions


def test_tiny_excluded_spend_never_renders_as_zero_dollars() -> None:
    rows = [_row("s1", "m", reward=1.0), _row("s2", "m", reward=None, cost=0.00002)]
    cost = effective_cost_per_completed_task(rows)

    assert "less than $0.0001" in cost.cost_assumptions
    assert "$0.0000" not in cost.cost_assumptions


# --- unpriced arms ----------------------------------------------------------------------------


def test_a_wholly_unpriced_arm_is_flagged_not_reported_as_free() -> None:
    # Every scored row has tokens and $0, so cost per task is $0 and the saving against a priced
    # anchor reads -100%. The number is arithmetically right and substantively meaningless, so a
    # renderer needs a field it can branch on, not a sentence it will not parse.
    arm = Arm(
        name="unpriced",
        condition=_BASE.replace(optimizer="unpriced"),
        rows=[
            _row("s1", "unpriced", reward=1.0, cost=0.0, tokens=1000),
            _row("s2", "unpriced", reward=1.0, cost=0.0, tokens=1000),
        ],
    )
    anchor = Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),
        rows=[
            _row("s1", "teacher", reward=1.0, cost=0.10),
            _row("s2", "teacher", reward=1.0, cost=0.10),
        ],
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.cost.cost_is_unpriced is True
    assert card.cost_delta_percent == pytest.approx(-100.0)
    assert "Treat the cost figures as absent, not as free" in card.cost_assumptions
    assert card.anchor_cost.cost_is_unpriced is False


def test_a_partially_priced_arm_is_not_flagged_unpriced() -> None:
    rows = [
        _row("s1", "m", reward=1.0, cost=0.0, tokens=1000),
        _row("s2", "m", reward=1.0, cost=0.01, tokens=1000),
    ]
    cost = effective_cost_per_completed_task(rows)

    assert cost.zero_cost_rows == 1
    assert cost.cost_is_unpriced is False


# --- arm hygiene --------------------------------------------------------------------------------


def test_duplicate_episode_keys_are_rejected_so_overhead_cannot_double_bill() -> None:
    with pytest.raises(ValueError, match="share the episode key"):
        Arm(
            name="dupe",
            condition=_BASE,
            rows=[_row("s1", "m", reward=1.0), _row("s1", "m", reward=1.0)],
        )


@pytest.mark.parametrize("reward", [1.5, -0.5, 8.0])
def test_rewards_outside_the_unit_interval_are_rejected(reward: float) -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        Arm(name="rubric", condition=_BASE, rows=[_row("s1", "m", reward=reward)])


def test_public_effective_cost_rejects_orphan_overhead_like_the_arm_does() -> None:
    with pytest.raises(ValueError, match="overhead rows name episodes not present here"):
        effective_cost_per_completed_task(
            [_row("s1", "m", reward=1.0)],
            overheads=[
                RowOverhead(scenario_id="TYPO", model="m", component="router", cost_usd=99.0)
            ],
        )


def test_condition_label_replace_rejects_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match="unknown condition axes"):
        _BASE.replace(optimzer="typo")


def test_condition_label_replace_revalidates_the_new_value() -> None:
    with pytest.raises(ValueError):
        _BASE.replace(base_model="")


# --- restrict_to --------------------------------------------------------------------------------


def test_restrict_to_narrows_a_standalone_scorecard() -> None:
    arm = _arm(
        "distill",
        [_row("s1", "student", reward=1.0), _row("s2", "student", reward=0.0)],
    )
    anchor = _arm(
        "teacher",
        [_row("s1", "teacher", reward=1.0), _row("s2", "teacher", reward=1.0)],
        base_model="glm-5.2",
    )
    assert build_scorecard(arm=arm, anchor=anchor).scenarios_compared == 2
    narrowed = build_scorecard(arm=arm, anchor=anchor, restrict_to=["s1"])
    assert narrowed.scenarios_compared == 1
    assert narrowed.quality.mean_reward == pytest.approx(1.0)


def test_restrict_to_that_excludes_everything_raises() -> None:
    arm = _arm("distill", [_row("s1", "student", reward=1.0)])
    anchor = _arm("teacher", [_row("s1", "teacher", reward=1.0)], base_model="glm-5.2")
    with pytest.raises(ValueError, match="share no scored scenario inside"):
        build_scorecard(arm=arm, anchor=anchor, restrict_to=["nowhere"])


def test_cost_delta_is_undefined_when_the_anchor_itself_cost_nothing() -> None:
    # A free anchor makes the ratio a division by zero. None is the only honest answer; a raw
    # divide would crash the report and a 0.0 would claim the arm matched a free baseline.
    arm = Arm(
        name="priced",
        condition=_BASE.replace(optimizer="priced"),
        rows=[_row("s1", "priced", reward=1.0, cost=0.05)],
    )
    anchor = Arm(
        name="free-teacher",
        condition=_BASE.replace(base_model="local"),
        rows=[_row("s1", "free-teacher", reward=1.0, cost=0.0, tokens=1000)],
    )
    card = build_scorecard(arm=arm, anchor=anchor)

    assert card.anchor_cost.cost_per_completed_task_usd == pytest.approx(0.0)
    assert card.cost_delta_percent is None


def test_an_episode_with_no_tokens_is_not_counted_as_unpriced() -> None:
    # $0 with zero tokens is an episode that never called the model, not a missing price.
    # Counting it would let an empty run masquerade as an unpriced pool entry.
    rows = [_row("s1", "m", reward=1.0, cost=0.0, tokens=0)]
    cost = effective_cost_per_completed_task(rows)

    assert cost.zero_cost_rows == 0
    assert cost.cost_is_unpriced is False
