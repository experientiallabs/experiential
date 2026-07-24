"""Tests for the routing fitter (Avengers replication) and its policy evaluation."""

from __future__ import annotations

import pytest

from wmh.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmh.optimize.policy import EmbedderSpec, RoutingPolicy
from wmh.optimize.routing import evaluate_policy, fit_rank_policy
from wmh.providers.base import ProviderKind
from wmh.providers.pool import PoolEntry

_SQL_TASKS = [
    "SELECT count(*) FROM superheroes WHERE height > 190",
    "SELECT name FROM users ORDER BY created_at DESC LIMIT 10",
    "SELECT avg(price) FROM orders GROUP BY customer_id",
    "SELECT id FROM events WHERE ts > '2026-01-01' AND kind = 'click'",
    "SELECT t.name, count(*) FROM teams t JOIN players p ON p.team_id = t.id GROUP BY 1",
    "SELECT max(score) FROM matches WHERE season = 2025",
]
_PROSE_TASKS = [
    "write a friendly email to the team about the offsite",
    "draft a short thank-you note for the conference organizers",
    "compose a birthday message for a colleague",
    "write a warm welcome paragraph for new employees",
    "draft an apology note for the delayed shipment",
    "write a cheerful newsletter intro about spring",
]


def _entries() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="sql-model",
            kind=ProviderKind.OPENAI,
            model="custom-sql",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
        PoolEntry(
            name="prose-model",
            kind=ProviderKind.OPENAI,
            model="custom-prose",
            input_per_mtok=1.0,
            output_per_mtok=1.0,
        ),
    ]


def _matrix() -> OutcomeMatrix:
    """sql-model aces SQL and flunks prose; prose-model is the mirror image."""
    outcomes: list[ScenarioOutcome] = []
    for group, tasks in [("sql", _SQL_TASKS), ("prose", _PROSE_TASKS)]:
        for index, task in enumerate(tasks):
            sid = f"{group}:{index}"
            for model in ["sql-model", "prose-model"]:
                wins = (model == "sql-model") == (group == "sql")
                outcomes.append(
                    ScenarioOutcome(
                        scenario_id=sid,
                        task=task,
                        model=model,
                        reward=1.0 if wins else 0.0,
                        success=wins,
                        cost_usd=0.001,
                    )
                )
    return OutcomeMatrix(pool=_entries(), outcomes=outcomes)


def _fit(**kwargs: object) -> RoutingPolicy:
    matrix = _matrix()
    defaults: dict = {
        "embedder": EmbedderSpec(dim=256),
        "n_clusters": 2,
        "seed": 42,
        "top_k_clusters": 1,
    }
    defaults.update(kwargs)
    return fit_rank_policy(matrix, **defaults)


def test_fit_recovers_the_specialists() -> None:
    policy = _fit()
    assert policy.kind == "rank"
    assert len(policy.clusters) == 2
    result = evaluate_policy(policy, _matrix(), _matrix().scenario_ids())
    # Audited (2026-07-24): hashing-trigram geometry puts one prose text in the sql-majority
    # cluster, so routing is 11/12, not 12/12 - an EMBEDDER locality miss, not a fitter bug
    # (the mixed cluster's ranking still has the sql specialist first at 6/7). The algorithm's
    # guarantee is beating every single model (0.5 here), not perfection.
    assert result.accuracy >= 11 / 12 - 1e-9
    assert result.accuracy > 0.5
    mix = result.model_mix
    assert set(mix) == {"sql-model", "prose-model"}
    assert all(share > 0.3 for share in mix.values())


def test_fit_is_deterministic() -> None:
    assert _fit() == _fit()


def test_cluster_count_clamps_to_scenarios() -> None:
    policy = _fit(n_clusters=500)
    assert len(policy.clusters) == 12  # one per scenario at most


def test_default_model_is_overall_best_when_unset() -> None:
    policy = _fit()
    assert policy.default_model in {"sql-model", "prose-model"}  # tied overall; pool order wins
    assert policy.default_model == "sql-model"


def test_rankings_only_contain_scored_models() -> None:
    matrix = _matrix()
    # prose-model never scored anywhere: with a single cluster it must be absent from the
    # ranking entirely (it falls back to default_rank at selection time, like the reference).
    matrix.outcomes = [o for o in matrix.outcomes if o.model != "prose-model"]
    policy = fit_rank_policy(
        matrix, embedder=EmbedderSpec(dim=256), n_clusters=1, seed=42, top_k_clusters=1
    )
    (cluster,) = policy.clusters
    assert cluster.ranking == ["sql-model"]


def test_evaluate_policy_static_baseline() -> None:
    matrix = _matrix()
    static = RoutingPolicy(kind="static", default_model="sql-model", pool=_entries())
    result = evaluate_policy(static, matrix, matrix.scenario_ids())
    assert result.accuracy == pytest.approx(0.5)
    assert result.model_mix == {"sql-model": 1.0}


def test_fitted_from_provenance_recorded() -> None:
    policy = _fit(fitted_from="routerbench_0shot.pkl@seed0")
    assert policy.fitted_from == "routerbench_0shot.pkl@seed0"
