"""Tests for the routing policy artifact and the Avengers-style rank selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wmh.optimize.policy import (
    ClusterRanking,
    EmbedderSpec,
    RoutingPolicy,
    rank_decision,
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


def _rank_policy(top_k_clusters: int = 1) -> RoutingPolicy:
    embedder = HashingEmbedder(dim=64)
    sql, prose = embedder.embed(["SELECT count(*) FROM superheroes", "write a friendly email"])
    return RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=top_k_clusters,
        clusters=[
            ClusterRanking(
                cluster_id=0,
                label="sql",
                centroid=sql,
                ranking=["fable-5", "haiku-4-5"],
                scores={"fable-5": 0.9, "haiku-4-5": 0.5},
                total=10,
            ),
            ClusterRanking(
                cluster_id=1,
                label="prose",
                centroid=prose,
                ranking=["haiku-4-5", "fable-5"],
                scores={"haiku-4-5": 0.8, "fable-5": 0.7},
                total=10,
            ),
        ],
    )


def test_static_policy_routes_to_default() -> None:
    decision = select_model(_static(), "anything at all")
    assert decision.model == "haiku-4-5"
    assert decision.cluster_id is None
    assert "static" in decision.reason


def test_rank_policy_routes_by_nearest_cluster_ranking() -> None:
    policy = _rank_policy()
    sql_decision = select_model(policy, "SELECT name FROM superheroes WHERE power = 'flight'")
    assert sql_decision.model == "fable-5"
    assert sql_decision.cluster_label == "sql"
    assert "rank router" in sql_decision.reason
    prose_decision = select_model(policy, "write a friendly email to the team")
    assert prose_decision.model == "haiku-4-5"
    assert prose_decision.cluster_label == "prose"


def test_rank_policy_soft_mixing_follows_the_closer_cluster() -> None:
    # With both clusters mixed in (top_k=2) and symmetric opposite rankings, the winner is
    # decided by which cluster is closer: score(m) = sum_c p_c / (rank_c(m) + 0.1).
    policy = _rank_policy(top_k_clusters=2)
    assert select_model(policy, "SELECT count(*) FROM superheroes").model == "fable-5"
    assert select_model(policy, "write a friendly email").model == "haiku-4-5"


def test_models_missing_from_rankings_score_default_rank() -> None:
    # Ranking mentions only fable-5; haiku scores 1/default_rank and cannot win.
    embedder = HashingEmbedder(dim=32)
    (centroid,) = embedder.embed(["anything"])
    policy = RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=32),
        top_k_clusters=1,
        clusters=[ClusterRanking(cluster_id=0, centroid=centroid, ranking=["fable-5"])],
    )
    assert select_model(policy, "anything").model == "fable-5"


def test_incumbent_sticks_by_default() -> None:
    decision = select_model(_rank_policy(), "SELECT 1", incumbent="haiku-4-5")
    assert decision.model == "haiku-4-5"  # affinity wins over the cluster preference
    assert "sticky" in decision.reason


def test_retired_incumbent_reroutes() -> None:
    decision = select_model(_rank_policy(), "SELECT 1", incumbent="gone-model")
    assert decision.model == "fable-5"


def test_default_model_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="default_model"):
        RoutingPolicy(kind="static", default_model="nope", pool=_pool())


def test_ranking_models_must_be_in_pool() -> None:
    with pytest.raises(ValueError, match="missing"):
        RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=4),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1, 0, 0, 0], ranking=["missing"])],
        )


def test_rank_kind_requires_clusters_and_matching_dims() -> None:
    with pytest.raises(ValueError, match="cluster"):
        RoutingPolicy(kind="rank", default_model="haiku-4-5", pool=_pool())
    with pytest.raises(ValueError, match="dim"):
        RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=8),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1.0, 0.0], ranking=["fable-5"])],
        )


def test_static_kind_rejects_clusters() -> None:
    with pytest.raises(ValueError, match="static"):
        RoutingPolicy(
            kind="static",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=2),
            clusters=[ClusterRanking(cluster_id=0, centroid=[1.0, 0.0], ranking=["fable-5"])],
        )


def test_policy_round_trips_through_json(tmp_path: Path) -> None:
    policy = _rank_policy()
    path = tmp_path / "policy.json"
    policy.save(path)
    assert RoutingPolicy.load(path) == policy


def test_azure_embedder_spec_requires_backend_fields() -> None:
    with pytest.raises(ValueError, match="deployment"):
        EmbedderSpec(kind="azure", dim=3072)


def test_azure_embedder_spec_builds_a_batched_provider_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZ_EMBED_KEY", "sk-test")
    spec = EmbedderSpec(
        kind="azure",
        dim=3072,
        deployment="text-embedding-3-large",
        endpoint="https://example.openai.azure.com",
        api_key_env="AZ_EMBED_KEY",
        batch=128,
    )
    embedder = spec.build()  # constructs lazily; no network until embed()
    from wmh.retrieval.embedders import BatchedEmbedder

    assert isinstance(embedder, BatchedEmbedder)


def test_azure_embedder_spec_missing_key_env_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZ_EMBED_KEY", raising=False)
    spec = EmbedderSpec(
        kind="azure",
        dim=8,
        deployment="d",
        endpoint="https://example.openai.azure.com",
        api_key_env="AZ_EMBED_KEY",
    )
    with pytest.raises(ValueError, match="AZ_EMBED_KEY"):
        spec.build()


def test_support_tilt_shifts_weight_to_supported_clusters() -> None:
    # Two near-equidistant clusters with opposite rankings; the tiny cluster (total=1) wins
    # untilted (slightly closer), the big one (total=400) wins under tilt.
    embedder = HashingEmbedder(dim=64)
    near, far = embedder.embed(["SELECT count(*) FROM t", "SELECT sum(x) FROM t"])

    def build(gamma: float) -> RoutingPolicy:
        return RoutingPolicy(
            kind="rank",
            default_model="haiku-4-5",
            pool=_pool(),
            embedder=EmbedderSpec(dim=64),
            top_k_clusters=2,
            beta=1.0,
            support_tilt_gamma=gamma,
            clusters=[
                ClusterRanking(
                    cluster_id=0, centroid=near, ranking=["fable-5", "haiku-4-5"], total=1
                ),
                ClusterRanking(
                    cluster_id=1, centroid=far, ranking=["haiku-4-5", "fable-5"], total=400
                ),
            ],
        )

    query = "SELECT count(*) FROM t"
    assert select_model(build(0.0), query).model == "fable-5"
    assert select_model(build(1.0), query).model == "haiku-4-5"


def test_select_model_uses_a_caller_supplied_embedder() -> None:
    # The seam a many-request caller needs: build the policy's embedder once and hand it in.
    # Proven by handing in one that maps SQL text onto the prose centroid.
    class _ProseEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            (prose,) = HashingEmbedder(dim=64).embed(["write a friendly email"])
            return [prose for _ in texts]

    policy = _rank_policy()
    assert select_model(policy, "SELECT 1").model == "fable-5"
    assert select_model(policy, "SELECT 1", embedder=_ProseEmbedder()).model == "haiku-4-5"


def test_support_tilt_gamma_rejects_negative_values() -> None:
    # A negative tilt would reweight AWAY from supported clusters, silently inverting the lever.
    with pytest.raises(ValidationError):
        RoutingPolicy(
            kind="static", default_model="haiku-4-5", pool=_pool(), support_tilt_gamma=-1.0
        )


def test_rank_decision_normalizes_the_query_itself() -> None:
    # `beta * distance` is a softmax temperature, so an unnormalized query silently changes how
    # sharply the top clusters mix (with the support tilt on, that flips the winner). The
    # normalization is enforced inside rank_decision, so no caller can skip it.
    embedder = HashingEmbedder(dim=64)
    near, far = embedder.embed(["SELECT count(*) FROM t", "SELECT sum(x) FROM t"])
    policy = RoutingPolicy(
        kind="rank",
        default_model="haiku-4-5",
        pool=_pool(),
        embedder=EmbedderSpec(dim=64),
        top_k_clusters=2,
        beta=1.0,
        support_tilt_gamma=1.0,
        clusters=[
            ClusterRanking(cluster_id=0, centroid=near, ranking=["fable-5", "haiku-4-5"], total=1),
            ClusterRanking(cluster_id=1, centroid=far, ranking=["haiku-4-5", "fable-5"], total=400),
        ],
    )
    unit = np.asarray(embedder.embed(["SELECT count(*) FROM t"])[0])
    expected = rank_decision(policy, unit)
    assert expected.model == "haiku-4-5"
    for scale in (0.02, 50.0):  # 50.0 selected fable-5 before the fix
        assert rank_decision(policy, unit * scale) == expected
