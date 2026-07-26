"""Tests for the kNN routing fitter: bank contents, baseline choice, and the sidecar contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmh.optimize.knn import best_single_on_fit, build_knn_bank, fit_knn_policy
from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
    knn_decision,
)
from wmh.optimize.routing import evaluate_policy
from wmh.providers.base import ProviderKind
from wmh.providers.pool import PoolEntry
from wmh.retrieval.embedders import HashingEmbedder

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
    test_a_budget_as_large_as_the_bank_collapses_to_the_fallback).
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
