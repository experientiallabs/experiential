"""Tests for the routing policy artifact and serve-time model selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.optimize.policy import (
    ClusterAssignment,
    EmbedderSpec,
    RoutingPolicy,
    select_model,
)
from wmh.providers.base import ProviderKind
from wmh.providers.pool import PoolEntry
from wmh.retrieval.embedders import HashingEmbedder


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(name="fable-5", kind=ProviderKind.ANTHROPIC, model="claude-fable-5"),
        PoolEntry(name="haiku-4-5", kind=ProviderKind.ANTHROPIC, model="claude-haiku-4-5"),
    ]


def _static() -> RoutingPolicy:
    return RoutingPolicy(kind="static", default_model="haiku-4-5", pool=_pool())


def _cluster_policy() -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["SELECT count(*) FROM superheroes", "write a friendly email"])
    return RoutingPolicy(
        kind="cluster",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        clusters=[
            ClusterAssignment(cluster_id=0, label="sql", centroid=sql, model="fable-5"),
            ClusterAssignment(cluster_id=1, label="prose", centroid=prose, model="haiku-4-5"),
        ],
    )


def test_static_policy_routes_to_default() -> None:
    decision = select_model(_static(), "anything at all")
    assert decision.model == "haiku-4-5"
    assert decision.cluster_id is None
    assert "static" in decision.reason


def test_cluster_policy_routes_to_nearest_centroid() -> None:
    policy = _cluster_policy()
    sql_decision = select_model(policy, "SELECT name FROM superheroes WHERE power = 'flight'")
    assert sql_decision.model == "fable-5"
    assert sql_decision.cluster_label == "sql"
    prose_decision = select_model(policy, "write a friendly email to the team")
    assert prose_decision.model == "haiku-4-5"
    assert prose_decision.cluster_label == "prose"


def test_incumbent_sticks_by_default() -> None:
    policy = _cluster_policy()
    decision = select_model(policy, "SELECT 1", incumbent="haiku-4-5")
    assert decision.model == "haiku-4-5"  # affinity wins over the cluster preference
    assert "sticky" in decision.reason


def test_retired_incumbent_reroutes() -> None:
    decision = select_model(_cluster_policy(), "SELECT 1", incumbent="gone-model")
    assert decision.model == "fable-5"


def test_default_model_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="default_model"):
        RoutingPolicy(kind="static", default_model="nope", pool=_pool())


def test_cluster_model_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="cluster"):
        RoutingPolicy(
            kind="cluster",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=4),
            clusters=[ClusterAssignment(cluster_id=0, centroid=[1, 0, 0, 0], model="missing")],
        )


def test_cluster_kind_requires_clusters_and_matching_dims() -> None:
    with pytest.raises(ValueError, match="cluster"):
        RoutingPolicy(kind="cluster", default_model="haiku-4-5", pool=_pool())
    with pytest.raises(ValueError, match="dim"):
        RoutingPolicy(
            kind="cluster",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=8),
            clusters=[ClusterAssignment(cluster_id=0, centroid=[1.0, 0.0], model="fable-5")],
        )


def test_static_kind_rejects_clusters() -> None:
    with pytest.raises(ValueError, match="static"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=2),
            clusters=[ClusterAssignment(cluster_id=0, centroid=[1.0, 0.0], model="fable-5")],
        )


def test_policy_round_trips_through_json(tmp_path: Path) -> None:
    policy = _cluster_policy()
    path = tmp_path / "policy.json"
    policy.save(path)
    assert RoutingPolicy.load(path) == policy
