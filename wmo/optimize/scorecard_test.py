"""Tests for the three-objective scorecard and the ablation ladder.

Pure offline aggregation over synthetic outcome matrices: no provider, no judge, no spend.
"""

from __future__ import annotations

import pytest

from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.scorecard import (
    Arm,
    CompletionRule,
    ConditionLabel,
    OperatingPoint,
    RowOverhead,
    build_ladder,
    build_scorecard,
    effective_cost_per_completed_task,
    rows_for_model,
)
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry

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
    with pytest.raises(ValueError, match="overhead rows name episodes this arm does not contain"):
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
    assert card.latency.p50_s == pytest.approx(2.5)
    assert card.latency.n_tasks == 2
    assert card.anchor_latency.p50_s == pytest.approx(3.0)
    assert card.latency_p50_delta_percent == pytest.approx((2.5 - 3.0) / 3.0 * 100.0)


# --- ladder: label collisions and Pareto --------------------------------------------------


def _ladder_arm(
    name: str, *, reward: float, cost: float, seconds: float, dial: float | None = None
) -> Arm:
    return Arm(
        name=name,
        condition=_BASE.replace(optimizer=name),
        rows=[
            _row(f"s{i}", name, reward=reward, cost=cost, seconds=[seconds]) for i in range(1, 3)
        ],
        dial_position=dial,
    )


def _anchor_arm(
    *, reward: float = 1.0, cost: float = 0.10, seconds: float = 3.0, scenarios: int = 2
) -> Arm:
    return Arm(
        name="teacher",
        condition=_BASE.replace(base_model="glm-5.2"),
        rows=[
            _row(f"s{i}", "teacher", reward=reward, cost=cost, seconds=[seconds])
            for i in range(1, scenarios + 1)
        ],
    )


def test_ladder_rejects_colliding_condition_labels() -> None:
    # Two rungs whose displayed names differ but whose experimental axes are identical: the
    # failure mode that cost the GEPA program runs.
    first = Arm(
        name="+routing", condition=_BASE.replace(optimizer="d"), rows=[_row("s1", "m", reward=1.0)]
    )
    second = Arm(
        name="+compaction",
        condition=_BASE.replace(optimizer="d"),
        rows=[_row("s1", "m", reward=1.0)],
    )
    with pytest.raises(ValueError, match="carry the SAME condition label"):
        build_ladder("joint-tau", anchor=_anchor_arm(), arms=[first, second])


def test_ladder_rejects_duplicate_rung_names() -> None:
    a = Arm(name="same", condition=_BASE.replace(optimizer="a"), rows=[_row("s1", "m", reward=1.0)])
    b = Arm(name="same", condition=_BASE.replace(optimizer="b"), rows=[_row("s1", "m", reward=1.0)])
    with pytest.raises(ValueError, match="two rungs are both named 'same'"):
        build_ladder("joint-tau", anchor=_anchor_arm(), arms=[a, b])


def test_ladder_rejects_a_rung_colliding_with_the_anchor() -> None:
    anchor = _anchor_arm()
    clash = Arm(name="rung", condition=anchor.condition, rows=[_row("s1", "m", reward=1.0)])
    with pytest.raises(ValueError, match="carry the SAME condition label"):
        build_ladder("joint-tau", anchor=anchor, arms=[clash])


def test_every_rung_is_measured_on_one_common_scenario_set() -> None:
    # The hole a pairwise-only intersection leaves: `patchy` goes unscored on s1 (the scenario it
    # fails) and would otherwise be graded on {s2, s3} while `full` is graded on {s1, s2, s3},
    # handing `patchy` a Pareto win it did not earn. The ladder must narrow BOTH to {s2, s3}.
    anchor = _anchor_arm(scenarios=3)
    full = Arm(
        name="full",
        condition=_BASE.replace(optimizer="full"),
        rows=[
            _row("s1", "full", reward=0.0, cost=0.01),
            _row("s2", "full", reward=1.0, cost=0.01),
            _row("s3", "full", reward=1.0, cost=0.01),
        ],
    )
    patchy = Arm(
        name="patchy",
        condition=_BASE.replace(optimizer="patchy"),
        rows=[
            _row("s1", "patchy", reward=None, cost=0.01),
            _row("s2", "patchy", reward=1.0, cost=0.01),
            _row("s3", "patchy", reward=1.0, cost=0.01),
        ],
    )
    ladder = build_ladder("joint-tau", anchor=anchor, arms=[full, patchy])

    assert ladder.scenarios_compared == 2
    assert ladder.scenarios_excluded == 1
    assert {rung.scorecard.scenarios_compared for rung in ladder.rungs} == {2}
    # Both rungs now read 1.0 on the shared pair, so neither buys a quality edge from an
    # exclusion. Without the common set `full` would have read 2/3 against `patchy`'s 2/2.
    assert [rung.scorecard.quality.mean_reward for rung in ladder.rungs] == [1.0, 1.0]
    # A standalone pairwise scorecard still reports the wider, un-narrowed comparison.
    assert build_scorecard(arm=full, anchor=anchor).scenarios_compared == 3


def test_ladder_raises_when_no_scenario_is_common_to_every_rung() -> None:
    anchor = _anchor_arm(scenarios=2)
    first = Arm(
        name="a",
        condition=_BASE.replace(optimizer="a"),
        rows=[_row("s1", "a", reward=1.0), _row("s2", "a", reward=None)],
    )
    second = Arm(
        name="b",
        condition=_BASE.replace(optimizer="b"),
        rows=[_row("s1", "b", reward=None), _row("s2", "b", reward=1.0)],
    )
    with pytest.raises(ValueError, match="no scenario was scored by anchor"):
        build_ladder("joint-tau", anchor=anchor, arms=[first, second])


def test_ladder_needs_at_least_one_rung() -> None:
    with pytest.raises(ValueError, match="needs at least one arm besides the anchor"):
        build_ladder("joint-tau", anchor=_anchor_arm(), arms=[])


def test_pareto_front_on_a_hand_built_matrix() -> None:
    # Objectives are (mean reward up, cost per completed task down, latency down).
    #   cheap-fast : 0.5 reward, $0.02/task, 1.0s  -> best on cost and latency
    #   balanced   : 0.8 reward, $0.05/task, 2.0s  -> best on nothing alone, dominated by nobody
    #   dominated  : 0.5 reward, $0.06/task, 3.0s  -> worse than cheap-fast on all three
    #   best-quality: 1.0 reward, $0.09/task, 4.0s -> best on quality
    arms = [
        _ladder_arm("cheap-fast", reward=0.5, cost=0.01, seconds=1.0),
        _ladder_arm("balanced", reward=0.8, cost=0.025, seconds=2.0),
        _ladder_arm("dominated", reward=0.5, cost=0.03, seconds=3.0),
        _ladder_arm("best-quality", reward=1.0, cost=0.045, seconds=4.0),
    ]
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=arms)

    assert [rung.index for rung in ladder.rungs] == [0, 1, 2, 3]
    assert [rung.scorecard.arm for rung in ladder.pareto()] == [
        "cheap-fast",
        "balanced",
        "best-quality",
    ]


def test_pareto_omits_rungs_whose_cost_per_completed_task_is_undefined() -> None:
    completes = _ladder_arm("completes", reward=1.0, cost=0.01, seconds=1.0)
    # Scored, cheap, and fast, but nothing completed: it cannot sit on a cost frontier at all.
    never_completes = _ladder_arm("never-completes", reward=0.0, cost=0.001, seconds=0.1)
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[completes, never_completes])

    assert len(ladder.rungs) == 2
    assert [rung.scorecard.arm for rung in ladder.pareto()] == ["completes"]


def test_pareto_can_dominate_on_p95_instead_of_p50() -> None:
    steady = Arm(
        name="steady",
        condition=_BASE.replace(optimizer="steady"),
        rows=[_row(f"s{i}", "steady", reward=1.0, cost=0.01, seconds=[1.0]) for i in range(1, 4)],
    )
    # Identical p50 and cheaper, but one very slow tail episode: only the p95 objective sees it
    # (three tasks, so the 9.0s outlier moves p95 to 15.4s and leaves the median at 1.0s).
    spiky = Arm(
        name="spiky",
        condition=_BASE.replace(optimizer="spiky"),
        rows=[
            _row("s1", "spiky", reward=1.0, cost=0.005, seconds=[1.0]),
            _row("s2", "spiky", reward=1.0, cost=0.005, seconds=[1.0]),
            _row("s3", "spiky", reward=1.0, cost=0.005, seconds=[9.0]),
        ],
    )
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(scenarios=3), arms=[steady, spiky])

    assert ladder.rungs[0].scorecard.latency.p50_s == pytest.approx(1.0)
    assert ladder.rungs[1].scorecard.latency.p50_s == pytest.approx(1.0)
    assert ladder.rungs[1].scorecard.latency.p95_s == pytest.approx(15.4)

    # On p50 the two tie on latency and spiky is strictly cheaper, so steady drops off.
    assert [r.scorecard.arm for r in ladder.pareto(latency="p50")] == ["spiky"]
    # On p95 steady's tail is better, so neither dominates.
    assert [r.scorecard.arm for r in ladder.pareto(latency="p95")] == ["steady", "spiky"]


def test_operating_points_carry_the_dial_shape_only_when_a_dial_was_measured() -> None:
    dialed = _ladder_arm("balanced", reward=0.9, cost=0.02, seconds=1.0, dial=0.25)
    undialed = _ladder_arm("+compaction", reward=1.0, cost=0.01, seconds=0.5)
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[dialed, undialed])

    points = {p.name: p for p in ladder.operating_points(pareto_only=False)}
    anchor_row = points["balanced"].as_cost_quality_anchor()
    assert anchor_row.cost_quality == pytest.approx(0.25)
    assert anchor_row.named_point == "balanced"
    assert anchor_row.quality_delta_points == pytest.approx(-10.0)
    assert anchor_row.cost_delta_percent == pytest.approx(-80.0)
    # The D-DIAL wire contract the platform reads.
    assert set(anchor_row.model_dump(by_alias=True)) == {
        "s",
        "label",
        "quality_delta_pt",
        "cost_delta_pct",
    }

    with pytest.raises(ValueError, match="has no dial position"):
        points["+compaction"].as_cost_quality_anchor()


def test_operating_point_without_a_cost_delta_refuses_to_become_an_anchor() -> None:
    point = OperatingPoint(
        name="p", quality_delta_points=1.0, cost_delta_percent=None, dial_position=0.5
    )
    with pytest.raises(ValueError, match="has no cost delta"):
        point.as_cost_quality_anchor()


def test_operating_points_default_to_the_frontier() -> None:
    good = _ladder_arm("good", reward=1.0, cost=0.01, seconds=1.0)
    worse = _ladder_arm("worse", reward=0.5, cost=0.02, seconds=2.0)
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[good, worse])

    assert [p.name for p in ladder.operating_points()] == ["good"]
    assert [p.name for p in ladder.operating_points(pareto_only=False)] == ["good", "worse"]


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
