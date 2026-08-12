"""Tests for scorecard policy replay and ablation-ladder assembly."""

from __future__ import annotations

import pytest

from wmo.common.providers.base import ProviderKind, TokenUsage
from wmo.common.providers.pool import PoolEntry
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import ClusterRanking, EmbedderSpec, RoutingPolicy
from wmo.optimize.routing.scorecard_core import (
    Arm,
    ConditionLabel,
    build_scorecard,
    effective_cost_per_completed_task,
    rows_for_model,
)
from wmo.optimize.routing.scorecard_ladder import build_ladder, rows_for_policy
from wmo.simulation.retrieval.embedders import HashingEmbedder

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


def test_ladder_module_exports_the_replay_and_assembly_entrypoints() -> None:
    """The independently importable ladder module owns both ladder operations."""
    assert callable(build_ladder)
    assert callable(rows_for_policy)


def test_ladder_rejects_colliding_condition_labels() -> None:
    """Ladder assembly rejects two names for the same experiment."""
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
    """Ladder assembly requires distinct display names."""
    first = Arm(
        name="same", condition=_BASE.replace(optimizer="a"), rows=[_row("s1", "m", reward=1.0)]
    )
    second = Arm(
        name="same", condition=_BASE.replace(optimizer="b"), rows=[_row("s1", "m", reward=1.0)]
    )

    with pytest.raises(ValueError, match="two rungs are both named 'same'"):
        build_ladder("joint-tau", anchor=_anchor_arm(), arms=[first, second])


def test_ladder_rejects_a_rung_colliding_with_the_anchor() -> None:
    """Ladder assembly distinguishes the anchor's experimental condition."""
    anchor = _anchor_arm()
    clash = Arm(name="rung", condition=anchor.condition, rows=[_row("s1", "m", reward=1.0)])

    with pytest.raises(ValueError, match="carry the SAME condition label"):
        build_ladder("joint-tau", anchor=anchor, arms=[clash])


def test_every_rung_is_measured_on_one_common_scenario_set() -> None:
    """Ladder assembly narrows every rung to one shared scored scenario set."""
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
    assert [rung.scorecard.quality.mean_reward for rung in ladder.rungs] == [1.0, 1.0]
    assert build_scorecard(arm=full, anchor=anchor).scenarios_compared == 3


def test_ladder_raises_when_no_scenario_is_common_to_every_rung() -> None:
    """Ladder assembly refuses incomparable rungs."""
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
    """Ladder assembly requires an arm in addition to its anchor."""
    with pytest.raises(ValueError, match="needs at least one arm besides the anchor"):
        build_ladder("joint-tau", anchor=_anchor_arm(), arms=[])


def test_pareto_front_on_a_hand_built_matrix() -> None:
    """A constructed ladder retains exactly the non-dominated rungs."""
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


def test_pareto_omits_rungs_with_undefined_cost_per_completed_task() -> None:
    """A rung without a completed task cannot occupy a cost frontier."""
    completes = _ladder_arm("completes", reward=1.0, cost=0.01, seconds=1.0)
    never_completes = _ladder_arm("never-completes", reward=0.0, cost=0.001, seconds=0.1)

    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[completes, never_completes])

    assert len(ladder.rungs) == 2
    assert [rung.scorecard.arm for rung in ladder.pareto()] == ["completes"]


def test_pareto_can_dominate_on_p95_instead_of_p50() -> None:
    """Ladder Pareto selection changes when callers choose the tail percentile."""
    steady = Arm(
        name="steady",
        condition=_BASE.replace(optimizer="steady"),
        rows=[_row(f"s{i}", "steady", reward=1.0, cost=0.01, seconds=[1.0]) for i in range(1, 4)],
    )
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
    assert [rung.scorecard.arm for rung in ladder.pareto(latency="p50")] == ["spiky"]
    assert [rung.scorecard.arm for rung in ladder.pareto(latency="p95")] == ["steady", "spiky"]


def test_operating_points_carry_the_dial_shape_only_when_a_dial_was_measured() -> None:
    """Ladder operating points retain measured dial metadata only."""
    dialed = _ladder_arm("balanced", reward=0.9, cost=0.02, seconds=1.0, dial=0.25)
    undialed = _ladder_arm("+compaction", reward=1.0, cost=0.01, seconds=0.5)

    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[dialed, undialed])

    points = {point.name: point for point in ladder.operating_points(pareto_only=False)}
    anchor_row = points["balanced"].as_cost_quality_anchor()
    assert anchor_row.cost_quality == pytest.approx(0.25)
    assert anchor_row.named_point == "balanced"
    assert anchor_row.quality_delta_points == pytest.approx(-10.0)
    assert anchor_row.cost_delta_percent == pytest.approx(-80.0)
    assert set(anchor_row.model_dump(by_alias=True)) == {
        "s",
        "label",
        "quality_delta_pt",
        "cost_delta_pct",
    }
    with pytest.raises(ValueError, match="has no dial position"):
        points["+compaction"].as_cost_quality_anchor()


def test_operating_points_default_to_the_frontier() -> None:
    """Ladder operating points omit dominated rungs by default."""
    good = _ladder_arm("good", reward=1.0, cost=0.01, seconds=1.0)
    worse = _ladder_arm("worse", reward=0.5, cost=0.02, seconds=2.0)

    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[good, worse])

    assert [point.name for point in ladder.operating_points()] == ["good"]
    assert [point.name for point in ladder.operating_points(pareto_only=False)] == ["good", "worse"]


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
    """Policy replay returns the measured rows for each selected model."""
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

    assert [(row.scenario_id, row.model) for row in rows] == [("s1", "strong"), ("s2", "cheap")]
    routed = effective_cost_per_completed_task(rows)
    assert routed.n_completed == 2
    assert routed.cost_per_completed_task_usd == pytest.approx(0.0255)
    assert effective_cost_per_completed_task(
        rows_for_model(matrix, "strong")
    ).cost_per_completed_task_usd == pytest.approx(0.050)


def test_rows_for_policy_honors_a_static_policy_and_an_id_subset() -> None:
    """Policy replay accepts static policies and explicit scenario subsets."""
    matrix = _routing_matrix()
    policy = RoutingPolicy(kind="static", default_model="cheap", pool=matrix.pool)

    assert [(row.scenario_id, row.model) for row in rows_for_policy(matrix, policy)] == [
        ("s1", "cheap"),
        ("s2", "cheap"),
    ]
    assert [row.scenario_id for row in rows_for_policy(matrix, policy, ids=["s2"])] == ["s2"]


def test_a_routed_rung_composes_into_a_ladder() -> None:
    """Policy replay feeds the ladder's normal scored-arm contract."""
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
    assert card.cost.n_completed == 1
    assert card.cost.cost_per_completed_task_usd == pytest.approx(0.002)
    assert card.quality_delta_points == pytest.approx(-40.0)


def test_tied_rungs_both_stay_on_the_frontier() -> None:
    """Mutually equal ladder rungs do not dominate each other."""
    first = _ladder_arm("tie-a", reward=1.0, cost=0.01, seconds=1.0)
    second = _ladder_arm("tie-b", reward=1.0, cost=0.01, seconds=1.0)

    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[first, second])

    assert [rung.scorecard.arm for rung in ladder.pareto()] == ["tie-a", "tie-b"]


def test_one_strict_improvement_is_enough_to_dominate() -> None:
    """A strictly cheaper otherwise equal rung dominates its peer."""
    cheap = _ladder_arm("cheap", reward=1.0, cost=0.01, seconds=1.0)
    dear = _ladder_arm("dear", reward=1.0, cost=0.02, seconds=1.0)

    ladder = build_ladder("joint-tau", anchor=_anchor_arm(), arms=[cheap, dear])

    assert [rung.scorecard.arm for rung in ladder.pareto()] == ["cheap"]


def test_a_rung_the_anchor_dominates_is_kept_off_the_frontier() -> None:
    """An anchor-dominated rung remains reported but leaves the frontier."""
    anchor = _anchor_arm(reward=1.0, cost=0.001, seconds=0.5)
    loser = _ladder_arm("worse-everywhere", reward=0.5, cost=1.0, seconds=9.0)

    ladder = build_ladder("joint-tau", anchor=anchor, arms=[loser])

    assert ladder.rungs[0].dominated_by_anchor is True
    assert ladder.pareto() == []
    assert ladder.operating_points() == []
    assert [rung.scorecard.arm for rung in ladder.rungs] == ["worse-everywhere"]


def test_a_rung_that_beats_the_anchor_on_one_axis_stays_on_the_frontier() -> None:
    """A rung that improves any objective remains an operating point."""
    anchor = _anchor_arm(reward=1.0, cost=0.10, seconds=3.0)
    cheaper = _ladder_arm("cheaper", reward=0.9, cost=0.01, seconds=1.0)

    ladder = build_ladder("joint-tau", anchor=anchor, arms=[cheaper])

    assert ladder.rungs[0].dominated_by_anchor is False
    assert [rung.scorecard.arm for rung in ladder.pareto()] == ["cheaper"]


def test_an_unmeasured_routed_choice_becomes_an_unscored_row() -> None:
    """Policy replay preserves an unmeasured decision as visible missing evidence."""
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

    assert [(row.scenario_id, row.model, row.reward) for row in rows] == [
        ("s1", "strong", 1.0),
        ("s2", "strong", None),
    ]
    assert "never measured on this scenario" in (rows[1].error or "")
    assert effective_cost_per_completed_task(rows).n_excluded == 1
