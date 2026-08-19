"""Tests for frozen guarded-kNN router policy and decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from wmo.common.models import BillingSource, ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.routing import KnnGuard, KnnRouterPolicy, RouterPolicy, RoutingDecision

_DIGEST = "a" * 64


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    return RoutedCandidateSnapshot(
        alias=alias,
        model=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
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
        embedder_alias="embedder",
        embedder=ModelSnapshot(
            billing_source=BillingSource.CUSTOMER_MANAGED,
            provider="openai",
            model_id="text-embedding-3-small",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        feature_extractor_id="request-visible-v1",
        feature_schema_sha256=_DIGEST,
        pricing_snapshot_id="pricing-v1",
        pricing_snapshot_sha256=_DIGEST,
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
        evaluation_plan_id="plan-v1",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="task-set-v1",
        task_set_sha256=_DIGEST,
        evaluation_protocols_sha256=_DIGEST,
        judgment_status="provisional",
    )


def test_persisted_policy_resolves_through_the_router_policy_union() -> None:
    """A stored policy document loads as its concrete kind without the caller naming the class."""
    policy = _policy()

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
            episode_id_sha256=_DIGEST,
            selected_alias="candidate-economy",
            baseline_alias="candidate-incumbent",
            neighbor_count=2,
            paired_count=3,
        )


def test_stored_policy_rejects_a_pre_design_threshold_guard() -> None:
    """Loading an older policy cannot bypass the enforced eight-pair design threshold."""
    payload = _policy().model_dump(mode="json")
    payload["guard"]["minimum_paired_observations"] = 2

    with pytest.raises(ValidationError, match="minimum_paired_observations"):
        KnnRouterPolicy.model_validate(payload)


@pytest.mark.parametrize("multiplier", [0, -0.5, float("inf"), float("nan")])
def test_guard_and_stored_policy_reject_nonpositive_or_nonfinite_uncertainty(
    multiplier: float,
) -> None:
    """New guards and persisted policy payloads cannot erase the uncertainty floor."""
    guard_payload = _policy().guard.model_dump(mode="json")
    guard_payload["uncertainty_multiplier"] = multiplier
    with pytest.raises(ValidationError, match="uncertainty_multiplier|finite"):
        KnnGuard.model_validate(guard_payload)

    policy_payload = _policy().model_dump(mode="json")
    policy_payload["guard"]["uncertainty_multiplier"] = multiplier
    with pytest.raises(ValidationError, match="uncertainty_multiplier|finite"):
        KnnRouterPolicy.model_validate(policy_payload)
