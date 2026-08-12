"""Tests for the kNN routing fitter: bank contents, baseline choice, and the sidecar contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmo.common.providers.base import ProviderKind
from wmo.common.providers.pool import PoolEntry
from wmo.optimize.routing import evaluate_policy
from wmo.optimize.routing.knn import (
    COST_QUALITY_ANCHORS,
    COST_QUALITY_BALANCED,
    apply_cost_quality,
    bank_floor_sim,
    best_single_on_fit,
    build_knn_bank,
    cost_quality_knobs,
    cost_quality_named_point,
    fit_knn_policy,
)
from wmo.optimize.routing.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.routing.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
    knn_decision,
)
from wmo.simulation.retrieval.embedders import HashingEmbedder

_SQL_TASKS = [
    "SELECT count(*) FROM superheroes WHERE height > 190",
    "SELECT name FROM users ORDER BY created_at DESC LIMIT 10",
    "SELECT avg(price) FROM orders GROUP BY customer_id",
    "SELECT id FROM events WHERE ts > '2026-01-01' AND kind = 'click'",
    "SELECT t.name, count(*) FROM teams t JOIN players p ON p.team_id = t.id GROUP BY 1",
    "SELECT max(score) FROM matches WHERE season = 2025",
    "SELECT sum(total) FROM invoices WHERE paid_at IS NULL",
    "SELECT city, count(*) FROM stores GROUP BY city HAVING count(*) > 2",
    "SELECT p.title FROM posts p WHERE p.author_id = 42 ORDER BY p.views DESC",
    "SELECT distinct category FROM products WHERE stock > 0",
]
_PROSE_TASKS = [
    "write a friendly email to the team about the offsite",
    "draft a short thank-you note for the conference organizers",
    "compose a birthday message for a colleague",
    "write a warm welcome paragraph for new employees",
    "draft an apology note for the delayed shipment",
    "write a cheerful newsletter intro about spring",
    "write a gentle reminder about the expense deadline",
    "draft a congratulations message for the launch",
    "compose a polite decline to a speaking invitation",
    "write a short farewell note for a departing teammate",
]


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="cheap",
            kind=ProviderKind.OPENAI,
            model="c",
            input_per_mtok=0.1,
            output_per_mtok=0.4,
        ),
        PoolEntry(
            name="pricey",
            kind=ProviderKind.OPENAI,
            model="p",
            input_per_mtok=10.0,
            output_per_mtok=40.0,
        ),
    ]


def _matrix(*, pricey_cost: float = 0.01) -> OutcomeMatrix:
    """cheap aces SQL and flunks prose; pricey is the mirror image at 10x the cost."""
    outcomes: list[ScenarioOutcome] = []
    for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]:
        for index, task in enumerate(tasks):
            for model in ("cheap", "pricey"):
                wins = (model == "cheap") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=f"{group}:{index}",
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001 if model == "cheap" else pricey_cost,
                    )
                )
    return OutcomeMatrix(pool=_pool(), outcomes=outcomes)


def _close_matrix() -> OutcomeMatrix:
    """Both models scored on every scenario, pricey better by 0.02: a near-tie everywhere.

    The regime a cost knob is for. The 0.02 gap is inside the guard's economic bar, and small
    enough that a 10x price difference outweighs it once the knob is priced in.
    """
    return OutcomeMatrix(
        pool=_pool(),
        outcomes=[
            ScenarioOutcome(
                scenario_id=f"{group}:{index}",
                task=task,
                model=model,
                reward=0.52 if model == "pricey" else 0.50,
                success=True,
                cost_usd=0.001 if model == "cheap" else 0.01,
            )
            for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]
            for index, task in enumerate(tasks)
            for model in ("cheap", "pricey")
        ],
    )


def _fit(
    tmp_path: Path,
    *,
    fit_ids: list[str] | None = None,
    guard_model: str | None = "cheap",
    rag_num: int = 5,
    rag_thres: float = 0.95,
    z: float = 0.5,
    min_pairs: int = 3,
    se_floor: bool = True,
) -> RoutingPolicy:
    """Fit on the toy matrix, with a neighbor budget scaled to a 20-row bank.

    The budget has to be a small fraction of the bank, and min_pairs below it: at the production
    default (50) every row is a neighbor of every query (see
    test_the_adaptive_cap_prevents_the_whole_bank_neighborhood_collapse).
    """
    return fit_knn_policy(
        _matrix(),
        bank_path=tmp_path / KNN_BANK_FILENAME,
        fit_ids=fit_ids,
        embedder=EmbedderSpec(dim=256),
        guard_model=guard_model,
        rag_num=rag_num,
        rag_thres=rag_thres,
        z=z,
        min_pairs=min_pairs,
        se_floor=se_floor,
    )


def test_fit_writes_a_sidecar_and_pins_the_requested_fallback(tmp_path: Path) -> None:
    policy = _fit(tmp_path, guard_model="pricey")
    assert policy.kind == "knn"
    assert policy.default_model == "pricey" == policy.guard_model
    assert policy.knn_bank_path == KNN_BANK_FILENAME  # a filename, so the artifact dir is portable
    assert (tmp_path / KNN_BANK_FILENAME).is_file()
    bank = policy.knn_bank()
    assert bank.models == ["cheap", "pricey"]
    assert len(bank.scenario_ids) == 20
    assert bank.dim == 256


def test_bank_cells_hold_scored_means_and_nan_elsewhere(tmp_path: Path) -> None:
    matrix = _matrix()
    # Two episodes of one cell (mean 0.5) and one cell never scored at all.
    matrix.outcomes.append(
        ScenarioOutcome(
            scenario_id="sql:0",
            task=_SQL_TASKS[0],
            model="cheap",
            reward=0.0,
            success=False,
            cost_usd=0.003,
        )
    )
    matrix.outcomes = [
        outcome
        for outcome in matrix.outcomes
        if not (outcome.scenario_id == "prose:0" and outcome.model == "pricey")
    ]
    bank = build_knn_bank(matrix, matrix.scenario_ids(), embedder=HashingEmbedder(dim=64))
    sql0 = bank.scenario_ids.index("sql:0")
    prose0 = bank.scenario_ids.index("prose:0")
    cheap, pricey = bank.models.index("cheap"), bank.models.index("pricey")
    assert bank.rewards[sql0, cheap] == pytest.approx(0.5)  # mean of 1.0 and 0.0
    assert bank.costs[sql0, cheap] == pytest.approx(0.002)
    assert np.isnan(bank.rewards[prose0, pricey])
    assert np.isnan(bank.costs[prose0, pricey])
    # Rows are L2-normalized at fit time so serve-time retrieval is one matrix product.
    np.testing.assert_allclose(np.linalg.norm(bank.embeddings, axis=1), 1.0, atol=1e-6)


def test_bank_mean_costs_ignore_unscored_cells(tmp_path: Path) -> None:
    bank = _fit(tmp_path).knn_bank()
    costs = bank.mean_costs()
    assert costs[bank.models.index("cheap")] == pytest.approx(0.001)
    assert costs[bank.models.index("pricey")] == pytest.approx(0.01)


def test_unpinned_fallback_is_the_best_single_model_on_the_fit_split(tmp_path: Path) -> None:
    # Both models score 0.5 overall, so the tie goes to the cheaper one: routing has to beat the
    # model a cost-conscious user would have picked, not the one that sorts first.
    assert best_single_on_fit(_matrix(), _matrix().scenario_ids()) == "cheap"
    assert _fit(tmp_path, guard_model=None).default_model == "cheap"
    # Give pricey the better fit-set reward and it becomes the baseline on merit.
    strong = _matrix()
    for outcome in strong.outcomes:
        if outcome.model == "pricey":
            outcome.reward = 1.0
    assert best_single_on_fit(strong, strong.scenario_ids()) == "pricey"


def test_fit_restricted_to_fit_ids_banks_only_those_scenarios(tmp_path: Path) -> None:
    fit_ids = [sid for sid in _matrix().scenario_ids() if sid.startswith("sql:")]
    policy = _fit(tmp_path, fit_ids=fit_ids)
    assert policy.knn_bank().scenario_ids == fit_ids


def test_fit_rejects_a_fallback_outside_the_pool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the matrix pool"):
        _fit(tmp_path, guard_model="not-a-model")


def test_fit_rejects_a_fallback_the_matrix_never_scored(tmp_path: Path) -> None:
    matrix = _matrix()
    matrix.outcomes = [outcome for outcome in matrix.outcomes if outcome.model != "pricey"]
    with pytest.raises(ValueError, match="no scored reward for baseline"):
        fit_knn_policy(
            matrix,
            bank_path=tmp_path / KNN_BANK_FILENAME,
            embedder=EmbedderSpec(dim=64),
            guard_model="pricey",
        )


def test_fit_rejects_unknown_fit_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the matrix"):
        _fit(tmp_path, fit_ids=["sql:0", "nope:1"])


def test_fitted_policy_recovers_the_specialists_through_the_serving_path(tmp_path: Path) -> None:
    # In-sample replay: each request retrieves its own row, so a working fit must route SQL to
    # cheap and prose to pricey. The point is that the whole chain (bank -> knn_decision ->
    # evaluate_policy) agrees, not that 100% is a held-out claim.
    policy = _fit(tmp_path, guard_model="cheap")
    matrix = _matrix()
    result = evaluate_policy(policy, matrix, matrix.scenario_ids())
    assert result.accuracy == pytest.approx(1.0)
    assert result.model_mix["pricey"] == pytest.approx(0.5)


def test_a_saved_knn_policy_reloads_and_routes_from_disk(tmp_path: Path) -> None:
    # The artifact contract serving depends on: policy.json plus its sidecar, nothing else.
    policy = _fit(tmp_path, guard_model="cheap")
    policy.save(tmp_path / POLICY_FILENAME)
    reloaded = RoutingPolicy.load(tmp_path / POLICY_FILENAME)
    assert reloaded.knn_bank().scenario_ids == policy.knn_bank().scenario_ids
    matrix = _matrix()
    assert evaluate_policy(reloaded, matrix, matrix.scenario_ids()).accuracy == pytest.approx(1.0)


def test_the_adaptive_cap_prevents_the_whole_bank_neighborhood_collapse(tmp_path: Path) -> None:
    # Pre-hardening behavior: a budget covering the bank made every profile the global
    # average and routing inert. The adaptive rule (R1 promotion hardening) caps the budget
    # at half the bank, so local evidence survives and the router still routes.
    policy = _fit(tmp_path, rag_num=20)
    assert policy.rag_num == 10  # capped at ceil(20-row bank / 2)
    matrix = _matrix()
    result = evaluate_policy(policy, matrix, matrix.scenario_ids())
    assert set(result.model_mix) == {"cheap", "pricey"}  # routing is alive, not collapsed
    assert result.accuracy > 0.8  # far above the fallback's 0.5; one boundary miss is fine


def test_fit_carries_the_guard_knobs_onto_the_policy(tmp_path: Path) -> None:
    policy = _fit(tmp_path, rag_num=17, rag_thres=0.9, z=1.25, min_pairs=3, se_floor=False)
    # rag_num 17 exceeds the adaptive cap (ceil(20 / 2) = 10) and is capped there.
    assert (policy.rag_num, policy.rag_thres) == (10, 0.9)
    assert (policy.knn_z, policy.knn_min_pairs, policy.se_floor) == (1.25, 3, False)


def test_adaptive_rule_scales_neighborhood_to_small_banks(tmp_path: Path) -> None:
    # A 50-neighbor budget on a tiny bank makes every profile the global mean and routing
    # inert; the fitted policy must scale the budget and the evidence bar to the bank.
    policy = fit_knn_policy(
        _matrix(),
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=256),
        guard_model="cheap",
    )
    assert policy.rag_num == 10  # ceil(20-row bank / 2), not the 50 default
    assert policy.knn_min_pairs == 5  # max(3, 10 // 2)


def test_adaptive_rule_keeps_caller_values_below_the_cap(tmp_path: Path) -> None:
    # Explicit smaller settings pass through untouched: min() semantics, never a raise.
    policy = _fit(tmp_path, rag_num=5, min_pairs=2)
    assert policy.rag_num == 5
    assert policy.knn_min_pairs == 2


def test_novelty_floor_abstains_on_far_queries(tmp_path: Path) -> None:
    policy = fit_knn_policy(
        _matrix(),
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=256),
        guard_model="cheap",
        rag_num=5,
        min_pairs=3,
        floor_q=0.99,  # floor at the 99th pct of self-NN sims: nearly everything abstains
    )
    assert policy.floor_sim is not None
    embedder = policy.embedder.build() if policy.embedder is not None else None
    assert embedder is not None
    query = embedder.embed(["zzz qqq xww utterly unrelated novel gibberish"])[0]
    decision = knn_decision(policy, np.asarray(query))
    assert decision.model == "cheap"
    assert "novelty abstain" in decision.reason


def test_floor_off_by_default(tmp_path: Path) -> None:
    policy = _fit(tmp_path)
    assert policy.floor_sim is None


def test_operator_min_pairs_passes_through_when_the_bank_does_not_cap(tmp_path: Path) -> None:
    # An explicit tightening on a bank the budget does not swallow must arrive verbatim:
    # min_pairs > rag_num is legitimate because the relative rule can keep more neighbors
    # than the budget.
    policy = _fit(tmp_path, rag_num=5, min_pairs=8)
    assert policy.rag_num == 5
    assert policy.knn_min_pairs == 8


def test_floor_q_on_a_single_row_bank_stays_off(tmp_path: Path) -> None:
    matrix = _matrix()
    only = matrix.scenario_ids()[:1]
    policy = fit_knn_policy(
        matrix,
        bank_path=tmp_path / KNN_BANK_FILENAME,
        fit_ids=only,
        embedder=EmbedderSpec(dim=256),
        guard_model="cheap",
        floor_q=0.5,
    )
    assert policy.floor_sim is None  # never NaN: a 1-row bank has no self-NN distribution


# --- the operator dial ---------------------------------------------------------------------


def test_fit_records_the_cost_unit_the_knob_divides_by(tmp_path: Path) -> None:
    # cheap costs $0.001 a call and pricey $0.01, so one "average call" is $0.0055: the mean of
    # the per-model means, not of the cells (which would tilt toward whichever model ran more).
    assert _fit(tmp_path).cost_scale == pytest.approx(0.0055)


def test_a_costless_matrix_leaves_the_cost_unit_at_zero(tmp_path: Path) -> None:
    matrix = _matrix()
    for outcome in matrix.outcomes:
        outcome.cost_usd = 0.0
    policy = fit_knn_policy(
        matrix,
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=64),
        guard_model="cheap",
    )
    assert policy.cost_scale == 0.0
    with pytest.raises(ValueError, match="no cost_scale"):
        apply_cost_quality(policy, 1.0)
    # The coverage leg needs no prices, so it still works on a costless fit.
    assert apply_cost_quality(policy, 0.1).pick_lam == 0.0


def test_the_dial_endpoints_are_the_measured_knobs() -> None:
    quality_max = cost_quality_knobs(0.0)
    assert (quality_max.knn_z, quality_max.floor_q, quality_max.pick_lam) == (0.5, 0.5, 0.0)
    assert quality_max.guard_mode == "symmetric"
    balanced = cost_quality_knobs(COST_QUALITY_BALANCED)
    assert (balanced.knn_z, balanced.floor_q, balanced.pick_lam) == (0.5, 0.05, 0.0)
    assert balanced.guard_mode == "symmetric"  # the shipped default keeps the strict bar
    savings_max = cost_quality_knobs(1.0)
    assert (savings_max.knn_z, savings_max.floor_q, savings_max.pick_lam) == (0.5, 0.05, 0.03)
    assert savings_max.guard_mode == "asymmetric"


def test_the_dial_is_monotone_in_both_knobs() -> None:
    dials = [index / 20 for index in range(21)]
    knobs = [cost_quality_knobs(dial) for dial in dials]
    floors = [knob.floor_q for knob in knobs]
    lams = [knob.pick_lam for knob in knobs]
    assert floors == sorted(floors, reverse=True)  # coverage opens up
    assert lams == sorted(lams)  # price pressure only ever rises
    assert cost_quality_named_point(0.0) == "Quality max"
    assert cost_quality_named_point(COST_QUALITY_BALANCED) == "Balanced (default)"
    assert cost_quality_named_point(0.5) == "Cost saver"
    assert cost_quality_named_point(0.75) == "Deep saver"
    assert cost_quality_named_point(1.0) == "Max savings"
    # A position next to an anchor is "Custom", never the anchor's name: the label travels with
    # that anchor's measured quality and cost, so borrowing it borrows the numbers.
    assert cost_quality_named_point(0.4) == "Custom"
    assert cost_quality_named_point(COST_QUALITY_BALANCED + 1e-6) == "Custom"


def test_the_anchor_table_covers_the_dial_and_is_sorted() -> None:
    # The table is what the endpoint hands the platform UI and what the docstring promises: the
    # measured positions span the dial, in order, and cost falls across them as advertised.
    dials = [anchor.cost_quality for anchor in COST_QUALITY_ANCHORS]
    assert dials == sorted(dials)
    assert (dials[0], dials[-1]) == (0.0, 1.0)
    assert COST_QUALITY_BALANCED in dials
    # Every anchor is a measured point, so every anchor row carries a real name: a client renders
    # one detent per row, and "Custom" is reserved for the positions in between.
    for anchor in COST_QUALITY_ANCHORS:
        assert anchor.named_point == cost_quality_named_point(anchor.cost_quality)
        assert anchor.named_point != "Custom"
    assert [anchor.named_point for anchor in COST_QUALITY_ANCHORS] == [
        "Quality max",
        "Balanced (default)",
        "Cost saver",
        "Deep saver",
        "Max savings",
    ]
    costs = [anchor.cost_delta_percent for anchor in COST_QUALITY_ANCHORS]
    assert costs == sorted(costs, reverse=True)  # every step up the dial measured cheaper
    assert all(cost < 0.0 for cost in costs)  # and every anchor beats the best single model


def test_the_anchor_table_serializes_under_the_platform_field_names() -> None:
    # The wire contract the platform card is built against: four fields per anchor, no knobs and
    # no provenance, so nothing on it can be mistaken for a measurement of another position.
    row = COST_QUALITY_ANCHORS[1].model_dump(by_alias=True)
    assert row == {
        "s": 0.25,
        "label": "Balanced (default)",
        "quality_delta_pt": 0.99,
        "cost_delta_pct": -24.7,
    }


def test_the_dial_rejects_settings_outside_its_range(tmp_path: Path) -> None:
    policy = _fit(tmp_path)
    for outside in (-0.01, 1.5):
        with pytest.raises(ValueError, match="between 0.0"):
            apply_cost_quality(policy, outside)


def test_the_dial_only_applies_to_knn_policies() -> None:
    static = RoutingPolicy(kind="static", default_model="cheap", pool=_pool())
    with pytest.raises(ValueError, match="kind='static'"):
        apply_cost_quality(static, 0.5)


def test_the_dial_returns_a_copy_and_leaves_the_original_alone(tmp_path: Path) -> None:
    fitted = _fit(tmp_path)
    slid = apply_cost_quality(fitted, 1.0)
    assert (fitted.pick_lam, fitted.guard_mode, fitted.cost_quality) == (0.0, "symmetric", None)
    assert (slid.pick_lam, slid.guard_mode, slid.cost_quality) == (0.03, "asymmetric", 1.0)
    assert slid.floor_sim is not None  # dial 1.0 keeps the shipped novelty floor
    assert fitted.floor_sim is None
    # Everything the fit measured travels unchanged: no refit, just a re-priced pick.
    assert slid.pool == fitted.pool
    assert slid.default_model == fitted.default_model == slid.guard_model
    assert (slid.rag_num, slid.rag_thres, slid.knn_min_pairs) == (
        fitted.rag_num,
        fitted.rag_thres,
        fitted.knn_min_pairs,
    )
    assert slid.knn_bank() is fitted.knn_bank()  # the bank is shared, never re-read
    assert "cost_quality=1" in (slid.fitted_from or "")


def test_the_dial_is_absolute_so_re_applying_never_compounds(tmp_path: Path) -> None:
    fitted = _fit(tmp_path)
    once = apply_cost_quality(fitted, 0.4)
    twice = apply_cost_quality(apply_cost_quality(fitted, 1.0), 0.4)
    assert once.model_dump() == twice.model_dump()


def test_the_dial_buys_cheap_calls_where_the_models_are_close(tmp_path: Path) -> None:
    # End to end: on a matrix where pricey is a hair better everywhere (0.52 vs 0.50) the
    # balanced dial pays for that hair on every call, and the savings end spends the 0.02 to
    # take the 10x cheaper model instead.
    matrix = _close_matrix()
    fitted = fit_knn_policy(
        matrix,
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=256),
        guard_model="pricey",
        rag_num=5,
        min_pairs=3,
    )
    ids = matrix.scenario_ids()
    balanced = evaluate_policy(apply_cost_quality(fitted, COST_QUALITY_BALANCED), matrix, ids)
    savings = evaluate_policy(apply_cost_quality(fitted, 1.0), matrix, ids)
    assert balanced.model_mix == {"pricey": 1.0}
    assert savings.model_mix == {"cheap": 1.0}
    assert savings.cost_per_scenario < balanced.cost_per_scenario / 5


def test_the_dial_cannot_flip_decisive_evidence(tmp_path: Path) -> None:
    # The specialist matrix: cheap aces SQL 1.0 to 0.0 and pricey aces prose. No dial position
    # may sell those calls to the wrong model, because the guard reads the untilted evidence.
    fitted = _fit(tmp_path, guard_model="pricey")
    matrix = _matrix()
    ids = matrix.scenario_ids()
    for dial in (0.0, COST_QUALITY_BALANCED, 0.5, 1.0):
        result = evaluate_policy(apply_cost_quality(fitted, dial), matrix, ids)
        assert result.accuracy == pytest.approx(1.0), dial


def test_the_novelty_floor_helper_reads_the_bank_not_the_policy(tmp_path: Path) -> None:
    # The dial recomputes the floor from the bank at every position (floor_q is a quantile of
    # the bank's own similarities, so it cannot be interpolated from the stored threshold).
    bank = _fit(tmp_path).knn_bank()
    assert bank_floor_sim(bank, 0.0) is None
    tight, loose = bank_floor_sim(bank, 0.5), bank_floor_sim(bank, 0.05)
    assert tight is not None and loose is not None
    assert tight > loose  # a higher quantile abstains more often


def test_fit_records_the_coverage_quantile_beside_its_threshold(tmp_path: Path) -> None:
    # The threshold alone cannot be read back (0.4 similarity is strict on one bank and loose on
    # another), so the quantile that produced it travels with it.
    wide = fit_knn_policy(
        _matrix(),
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=256),
        guard_model="cheap",
        rag_num=5,
        min_pairs=3,
        floor_q=0.5,
    )
    assert wide.floor_q == 0.5
    assert wide.floor_sim is not None
    off = _fit(tmp_path)  # the function default leaves the floor off
    assert (off.floor_q, off.floor_sim) == (0.0, None)


def test_the_dial_records_the_coverage_quantile_it_applied(tmp_path: Path) -> None:
    fitted = _fit(tmp_path)
    for dial in (0.0, COST_QUALITY_BALANCED, 1.0):
        slid = apply_cost_quality(fitted, dial)
        assert slid.floor_q == pytest.approx(cost_quality_knobs(dial).floor_q)
        assert (slid.floor_sim is None) == (slid.floor_q == 0.0)


# --- zero-cost (local) arm ---------------------------------------------------------------
# A locally hosted candidate legitimately prices at $0 per Mtok. These pin that a free arm
# does not degenerate the cost machinery: the tilt stays finite, the guard reads "free" as
# "cheaper", and an ALL-free pool refuses the cost knob loudly instead of dividing by zero.


def _pool_with_free() -> list[PoolEntry]:
    return [
        *_pool(),
        PoolEntry(
            name="free-local",
            kind=ProviderKind.OPENAI,
            model="qwen3:4b",
            endpoint="http://localhost:11434/v1",
            input_per_mtok=0.0,
            output_per_mtok=0.0,
        ),
    ]


def _free_arm_matrix() -> OutcomeMatrix:
    """A near-tie everywhere, with the free arm scored beside the paid two.

    pricey leads by 0.02 (inside the guard's economic bar), so at the savings end of the dial
    the argmax must land on the free arm on price alone, exactly the situation a local model
    creates.
    """
    rewards = {"cheap": 0.50, "pricey": 0.52, "free-local": 0.50}
    costs = {"cheap": 0.001, "pricey": 0.01, "free-local": 0.0}
    return OutcomeMatrix(
        pool=_pool_with_free(),
        outcomes=[
            ScenarioOutcome(
                scenario_id=f"{group}:{index}",
                task=task,
                model=model,
                reward=rewards[model],
                success=True,
                cost_usd=costs[model],
            )
            for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]
            for index, task in enumerate(tasks)
            for model in rewards
        ],
    )


def _fit_free(tmp_path: Path, matrix: OutcomeMatrix) -> RoutingPolicy:
    return fit_knn_policy(
        matrix,
        bank_path=tmp_path / KNN_BANK_FILENAME,
        embedder=EmbedderSpec(dim=256),
        guard_model="pricey",
        rag_num=5,
        rag_thres=0.95,
        min_pairs=3,
    )


def test_a_zero_cost_arm_keeps_the_cost_unit_positive(tmp_path: Path) -> None:
    # cost_scale is the mean of the per-model mean costs; a free arm pulls it down but the
    # priced arms keep it a usable unit, so the tilt never divides by zero.
    policy = _fit_free(tmp_path, _free_arm_matrix())
    assert policy.cost_scale == pytest.approx((0.001 + 0.01 + 0.0) / 3)


def test_the_savings_dial_routes_to_the_free_arm_without_degenerating(tmp_path: Path) -> None:
    # At the savings end the free arm's tilt is exactly 0 (it costs nothing), so it outbids the
    # near-tied paid arms on price; the guard's economic bar accepts it because the evidence
    # says "not significantly worse", and the whole decision stays finite.
    policy = apply_cost_quality(_fit_free(tmp_path, _free_arm_matrix()), 1.0)
    embedder = HashingEmbedder(dim=256)
    decision = knn_decision(
        policy, np.asarray(embedder.embed([_SQL_TASKS[0]])[0], dtype=np.float32)
    )
    assert decision.model == "free-local"
    assert decision.evidence is not None and decision.evidence.gate == "passed"


def test_a_free_pick_faces_the_cheaper_bar_not_the_pricier_one(tmp_path: Path) -> None:
    # Under the as-fitted symmetric guard a PRICIER pick pays a doubled z; a free pick is by
    # definition cheaper than any paid baseline, so its bar is the single z and the reason
    # never carries the pricier annotation.
    rewards = {"cheap": 0.2, "pricey": 0.2, "free-local": 0.9}
    costs = {"cheap": 0.001, "pricey": 0.01, "free-local": 0.0}
    matrix = OutcomeMatrix(
        pool=_pool_with_free(),
        outcomes=[
            ScenarioOutcome(
                scenario_id=f"sql:{index}",
                task=task,
                model=model,
                reward=rewards[model],
                success=rewards[model] > 0.5,
                cost_usd=costs[model],
            )
            for index, task in enumerate(_SQL_TASKS)
            for model in rewards
        ],
    )
    policy = _fit_free(tmp_path, matrix)
    embedder = HashingEmbedder(dim=256)
    decision = knn_decision(
        policy, np.asarray(embedder.embed([_SQL_TASKS[0]])[0], dtype=np.float32)
    )
    assert decision.model == "free-local"
    assert "pricier" not in decision.reason
    assert "0.5xSE" in decision.reason


def test_an_all_free_pool_refuses_the_cost_knob_loudly(tmp_path: Path) -> None:
    # Every arm free (an all-local pool): there is no price to trade against, so the savings
    # half of the dial must refuse with the refit instruction instead of dividing by zero.
    free_everything = OutcomeMatrix(
        pool=[
            entry.model_copy(update={"input_per_mtok": 0.0, "output_per_mtok": 0.0})
            for entry in _pool_with_free()
        ],
        outcomes=[
            ScenarioOutcome(
                scenario_id=f"sql:{index}",
                task=task,
                model=model,
                reward=0.5,
                success=True,
                cost_usd=0.0,
            )
            for index, task in enumerate(_SQL_TASKS)
            for model in ("cheap", "pricey", "free-local")
        ],
    )
    policy = _fit_free(tmp_path, free_everything)
    assert policy.cost_scale == 0.0
    # The coverage leg (dial at or below the balanced point) never prices anything, so it works.
    assert apply_cost_quality(policy, COST_QUALITY_BALANCED).pick_lam == 0.0
    with pytest.raises(ValueError, match="cost_scale"):
        apply_cost_quality(policy, 1.0)


def test_w05_guarded_router_decision_fixture_preserves_evidence_and_fallback(
    tmp_path: Path,
) -> None:
    """Map current `knn_decision` and `RoutingDecision` to the approved guarded router decision.

    The fixture exercises one paired candidate and baseline row, freezes the evidence gate, and
    leaves private cached vectors and the future policy/request identity envelope out of scope.
    """
    task = "Refund order A-42"
    matrix = OutcomeMatrix(
        pool=_pool(),
        outcomes=[
            ScenarioOutcome(
                scenario_id="scenario-w05-refund",
                task=task,
                model="cheap",
                reward=1.0,
                success=True,
                steps=1,
                stop_reason="agent_done",
                cost_usd=0.001,
            ),
            ScenarioOutcome(
                scenario_id="scenario-w05-refund",
                task=task,
                model="pricey",
                reward=0.0,
                success=False,
                steps=1,
                stop_reason="agent_done",
                cost_usd=0.01,
            ),
        ],
    )
    policy = fit_knn_policy(
        matrix,
        bank_path=tmp_path / KNN_BANK_FILENAME,
        fit_ids=["scenario-w05-refund"],
        embedder=EmbedderSpec(dim=64),
        guard_model="pricey",
        rag_num=1,
        min_pairs=1,
        z=0.5,
    )
    query = np.asarray(HashingEmbedder(dim=64).embed([task])[0], dtype=np.float32)

    decision = knn_decision(policy, query)

    assert decision.model == "cheap"
    assert decision.cluster_id is None
    assert decision.cluster_label == ""
    assert decision.reason == "knn: 1 neighbors, delta=+1.000 > 0.5xSE=0.250"
    assert decision.evidence is not None
    assert decision.evidence.mean_diff == pytest.approx(1.0)
    assert decision.evidence.se == pytest.approx(0.5)
    assert decision.evidence.n_pairs == 1
    assert decision.evidence.gate == "passed"
    assert decision.evidence.propensity == "greedy"
