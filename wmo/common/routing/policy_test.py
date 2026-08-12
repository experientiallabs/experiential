"""Tests for frozen guarded-kNN router policy and decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.routing import KnnGuard, KnnRouterPolicy, RouterPolicy, RoutingDecision

_DIGEST = "a" * 64


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    return RoutedCandidateSnapshot(
        alias=alias,
        model=ModelSnapshot(
            provider="openai",
            model_id=f"{alias}-model",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
    )


def _policy() -> KnnRouterPolicy:
    return KnnRouterPolicy(
        schema_version=1,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="e7aad17",
        policy_id="router-policy-v1",
        baseline_alias="candidate-incumbent",
        candidates=(_candidate("candidate-economy"), _candidate("candidate-incumbent")),
        embedder=ModelSnapshot(
            provider="openai",
            model_id="text-embedding-3-small",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        feature_extractor_id="request-visible-v1",
        feature_schema_sha256=_DIGEST,
        pricing_snapshot_id="pricing-v1",
        bank_artifact_id="bank-v1",
        bank_sha256=_DIGEST,
        guard=KnnGuard(
            maximum_neighbors=50,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0,
        ),
        fit_evaluation_id="evaluation-v1",
        judgment_status="provisional",
    )


def test_policy_and_decision_round_trip() -> None:
    """A policy pins all fit-time identities and a decision names its policy digest."""
    policy = _policy()
    decision = RoutingDecision(
        decision_id="decision-1",
        policy_id=policy.policy_id,
        policy_sha256=_DIGEST,
        request_sha256=_DIGEST,
        episode_id="episode-1",
        selected_alias="candidate-economy",
        baseline_alias="candidate-incumbent",
        neighbor_count=12,
        paired_count=10,
        best_similarity=0.98,
        estimated_quality_difference=0.01,
        uncertainty=0.02,
    )

    assert KnnRouterPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert RoutingDecision.model_validate_json(decision.model_dump_json()) == decision
    assert TypeAdapter(RouterPolicy).validate_json(policy.model_dump_json()) == policy


def test_policy_rejects_unpinned_baseline_and_nonfinite_guard_values() -> None:
    """The online runtime can rely on a baseline candidate and finite guard thresholds."""
    with pytest.raises(ValidationError, match="baseline_alias"):
        KnnRouterPolicy.model_validate(
            {**_policy().model_dump(), "baseline_alias": "candidate-missing"}
        )
    with pytest.raises(ValidationError, match="finite"):
        KnnGuard(
            maximum_neighbors=50,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=float("inf"),
            quality_tolerance=0,
        )
    with pytest.raises(ValidationError, match="cannot exceed maximum"):
        KnnGuard(
            maximum_neighbors=7,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0,
        )
    with pytest.raises(ValidationError, match="more paired observations"):
        RoutingDecision(
            decision_id="decision-1",
            policy_id="router-policy-v1",
            policy_sha256=_DIGEST,
            request_sha256=_DIGEST,
            episode_id="episode-1",
            selected_alias="candidate-economy",
            baseline_alias="candidate-incumbent",
            neighbor_count=2,
            paired_count=3,
        )
