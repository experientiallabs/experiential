"""Tests for the three-objective scorecard and the ablation ladder.

Pure offline aggregation over synthetic outcome matrices: no provider, no judge, no spend.
"""

from __future__ import annotations

import pytest

from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import ClusterRanking, EmbedderSpec, RoutingPolicy
from wmo.optimize.routing.scorecard import (
    EFFECTIVE_COST_RULE,
    Arm,
    CompletionRule,
    ConditionLabel,
    OperatingPoint,
    RowOverhead,
    build_ladder,
    build_scorecard,
    effective_cost_per_completed_task,
    rows_for_model,
    rows_for_policy,
)
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

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
    # Objectives are (mean reward up, cost per completed task down, latency down). Each arm has
    # two scenarios, both completed, so cost per task equals the per-row cost.
    #   cheap-fast  : 0.5 reward, $0.010/task, 1.0s -> best on cost and latency
    #   balanced    : 0.8 reward, $0.025/task, 2.0s -> best on nothing alone, dominated by nobody
    #   dominated   : 0.5 reward, $0.030/task, 3.0s -> worse than cheap-fast on all three
    #   best-quality: 1.0 reward, $0.045/task, 4.0s -> best on quality
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
    # (three tasks, so the 9.0s outlier moves p95 to 8.2s and leaves the median at 1.0s).
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

    assert ladder.rungs[0].scorecard.latency.p50_model_s == pytest.approx(1.0)
    assert ladder.rungs[1].scorecard.latency.p50_model_s == pytest.approx(1.0)
    assert ladder.rungs[1].scorecard.latency.p95_model_s == pytest.approx(8.2)

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


def _routing_matrix() -> OutcomeMatrix:
    return OutcomeMatrix(
        pool=[
            PoolEntry(
                name="cheap",
                kind=ProviderKind.OPENAI,
                model="custom-cheap",
                tier="open",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            ),
            PoolEntry(name="strong", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        ],
        outcomes=[
            _row("s1", "cheap", reward=0.2, cost=0.001),
            _row("s1", "strong", reward=1.0, cost=0.050),
            _row("s2", "cheap", reward=1.0, cost=0.001),
            _row("s2", "strong", reward=1.0, cost=0.050),
        ],
    )


def test_rows_for_policy_selects_the_rows_the_policy_would_have_routed_to() -> None:
    matrix = _routing_matrix()
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["task for s1", "task for s2"])
    policy = RoutingPolicy(
        kind="rank",
        default_model="cheap",
        pool=matrix.pool,
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=1,
        clusters=[
            ClusterRanking(cluster_id=0, label="hard", centroid=sql, ranking=["strong", "cheap"]),
            ClusterRanking(cluster_id=1, label="easy", centroid=prose, ranking=["cheap", "strong"]),
        ],
    )
    rows = rows_for_policy(matrix, policy, embedder=embedder)

    # s1 routes to strong, s2 to cheap: one row per scenario, from the model actually chosen.
    assert [(r.scenario_id, r.model) for r in rows] == [("s1", "strong"), ("s2", "cheap")]
    # The routed mix beats either single model on effective cost: it buys the expensive model
    # only where the cheap one fails.
    routed = effective_cost_per_completed_task(rows)
    assert routed.n_completed == 2
    assert routed.cost_per_completed_task_usd == pytest.approx(0.0255)
    assert effective_cost_per_completed_task(
        rows_for_model(matrix, "strong")
    ).cost_per_completed_task_usd == pytest.approx(0.050)


def test_rows_for_policy_honors_a_static_policy_and_an_id_subset() -> None:
    matrix = _routing_matrix()
    policy = RoutingPolicy(kind="static", default_model="cheap", pool=matrix.pool)

    assert [(r.scenario_id, r.model) for r in rows_for_policy(matrix, policy)] == [
        ("s1", "cheap"),
        ("s2", "cheap"),
    ]
    assert [r.scenario_id for r in rows_for_policy(matrix, policy, ids=["s2"])] == ["s2"]


def test_a_routed_rung_composes_into_a_ladder() -> None:
    matrix = _routing_matrix()
    policy = RoutingPolicy(kind="static", default_model="cheap", pool=matrix.pool)
    routed = Arm(
        name="+routing",
        condition=_BASE.replace(optimizer="distill+routing"),
        rows=rows_for_policy(matrix, policy),
    )
    anchor = Arm(
        name="strong-only",
        condition=_BASE.replace(base_model="claude-fable-5"),
        rows=rows_for_model(matrix, "strong"),
    )
    ladder = build_ladder("joint-tau", anchor=anchor, arms=[routed])

    assert ladder.scenarios_compared == 2
    card = ladder.rungs[0].scorecard
    # Static-to-cheap completes only s2, so its effective cost per completed task is $0.002
    # against the anchor's $0.050 while quality drops 40 points. Both objectives visible.
    assert card.cost.n_completed == 1
    assert card.cost.cost_per_completed_task_usd == pytest.approx(0.002)
    assert card.quality_delta_points == pytest.approx(-40.0)


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


def test_tied_rungs_both_stay_on_the_frontier() -> None:
    # Strict improvement is what keeps mutual domination from emptying the front. Two rungs
    # equal on all three objectives dominate nobody, so both belong on it.
    first = _ladder_arm("tie-a", reward=1.0, cost=0.01, seconds=1.0)
    second = _ladder_arm("tie-b", reward=1.0, cost=0.01, seconds=1.0)
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[first, second])

    assert [r.scorecard.arm for r in ladder.pareto()] == ["tie-a", "tie-b"]


def test_one_strict_improvement_is_enough_to_dominate() -> None:
    # Equal on quality and latency, strictly cheaper: that alone removes the costlier rung.
    cheap = _ladder_arm("cheap", reward=1.0, cost=0.01, seconds=1.0)
    dear = _ladder_arm("dear", reward=1.0, cost=0.02, seconds=1.0)
    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[cheap, dear])

    assert [r.scorecard.arm for r in ladder.pareto()] == ["cheap"]


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


# --- the anchor is an operating point too ------------------------------------------------------


def test_a_rung_the_anchor_dominates_is_kept_off_the_frontier() -> None:
    # Worse than the teacher on quality, cost, and latency. Reporting it as a frontier point
    # would advertise an operating point the untouched baseline beats on every axis.
    anchor = _anchor_arm(reward=1.0, cost=0.001, seconds=0.5)
    loser = _ladder_arm("worse-everywhere", reward=0.5, cost=1.0, seconds=9.0)
    ladder = build_ladder("joint-tau", anchor=anchor, arms=[loser])

    assert ladder.rungs[0].dominated_by_anchor is True
    assert ladder.pareto() == []
    assert ladder.operating_points() == []
    # It is still on the ladder: suppressed from the frontier, not hidden from the report.
    assert [r.scorecard.arm for r in ladder.rungs] == ["worse-everywhere"]


def test_a_rung_that_beats_the_anchor_on_one_axis_stays_on_the_frontier() -> None:
    anchor = _anchor_arm(reward=1.0, cost=0.10, seconds=3.0)
    cheaper = _ladder_arm("cheaper", reward=0.9, cost=0.01, seconds=1.0)
    ladder = build_ladder("joint-tau", anchor=anchor, arms=[cheaper])

    assert ladder.rungs[0].dominated_by_anchor is False
    assert [r.scorecard.arm for r in ladder.pareto()] == ["cheaper"]


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


# --- routed rungs: unmeasured choices ------------------------------------------------------------


def test_a_routed_choice_the_matrix_never_measured_becomes_an_unscored_row() -> None:
    # The policy sends everything to "strong", but the matrix only measured "strong" on s1.
    # Emitting nothing for s2 would shrink the routed arm's scenario set invisibly.
    matrix = OutcomeMatrix(
        pool=[
            PoolEntry(
                name="cheap",
                kind=ProviderKind.OPENAI,
                model="custom-cheap",
                tier="open",
                input_per_mtok=0.1,
                output_per_mtok=0.2,
            ),
            PoolEntry(name="strong", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        ],
        outcomes=[
            _row("s1", "cheap", reward=0.2),
            _row("s1", "strong", reward=1.0),
            _row("s2", "cheap", reward=1.0),
        ],
    )
    policy = RoutingPolicy(kind="static", default_model="strong", pool=matrix.pool)
    rows = rows_for_policy(matrix, policy)

    assert [(r.scenario_id, r.model, r.reward) for r in rows] == [
        ("s1", "strong", 1.0),
        ("s2", "strong", None),
    ]
    assert "never measured on this scenario" in (rows[1].error or "")
    cost = effective_cost_per_completed_task(rows)
    assert cost.n_excluded == 1


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
