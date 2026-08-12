"""Deterministic adversarial tests for conservative guarded kNN decisions."""

import hashlib
from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from wmo.common.models import ModelSnapshot, RoutedCandidateSnapshot
from wmo.common.routing import KnnGuard, KnnRouterPolicy, RoutingDecision
from wmo.common.routing.bank import (
    CandidateEvidenceCount,
    KnnBankManifest,
    KnnEvidenceBank,
    bank_bytes,
)
from wmo.common.routing.decision import RouterDecisionError, select_from_bank

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64
_REQUEST_DIGEST = "d" * 64


def test_cheapest_safe_candidate_wins_deterministic_quality_and_alias_ties() -> None:
    """Equal-cost, equal-quality candidates break their tie by stable alias."""
    scores = np.asarray(
        (
            (0.8, 0.9, 0.9),
            (0.8, 0.9, 0.9),
            (0.8, 0.9, 0.9),
        ),
        dtype=np.float32,
    )
    costs = np.asarray(((0.5, 0.1, 0.1),) * 3, dtype=np.float64)
    policy, manifest, bank = _fixture(scores, costs)

    decision = _select(policy, manifest, bank, np.asarray((1.0, 0.0)))

    assert decision.selected_alias == "candidate-a"
    assert decision.baseline_alias == "candidate-baseline"
    assert decision.neighbor_count == 3
    assert decision.paired_count == 3
    assert decision.estimated_quality_difference == pytest.approx(0.1)
    assert decision.fallback_reason is None


def test_novelty_and_bank_alias_drift_fail_closed() -> None:
    """Out-of-distribution vectors fall back and mutated bank axes are rejected."""
    scores = np.asarray(((0.8, 0.9, 0.9),) * 3, dtype=np.float32)
    costs = np.asarray(((0.5, 0.1, 0.2),) * 3, dtype=np.float64)
    policy, manifest, bank = _fixture(scores, costs)

    decision = _select(policy, manifest, bank, np.asarray((-1.0, 0.0)))

    assert decision.selected_alias == policy.baseline_alias
    assert decision.fallback_reason == "novelty"
    assert decision.best_similarity == pytest.approx(-1.0)
    drifted = manifest.model_copy(
        update={"candidate_aliases": ("candidate-baseline", "candidate-x", "candidate-a")}
    )
    with pytest.raises(RouterDecisionError, match="candidate aliases has drifted"):
        _select(policy, drifted, bank, np.asarray((1.0, 0.0)))


def test_bank_owns_read_only_copies_and_rejects_forced_mutation() -> None:
    """Caller arrays cannot alter a bank and forced in-memory mutation fails before selection."""
    scores = np.asarray(((0.8, 0.9, 0.9),) * 3, dtype=np.float32)
    costs = np.asarray(((0.5, 0.1, 0.2),) * 3, dtype=np.float64)
    policy, manifest, bank = _fixture(scores, costs)
    scores[0, 0] = 0.0
    assert bank.scores[0, 0] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        bank.scores[0, 0] = 0.0
    bank.scores.setflags(write=True)
    bank.scores[0, 0] = 0.0
    with pytest.raises(RouterDecisionError, match="content has mutated"):
        _select(policy, manifest, bank, np.asarray((1.0, 0.0)))


def test_guard_rejects_a_single_paired_observation() -> None:
    """A one-sample estimate cannot represent measurable routing uncertainty."""
    with pytest.raises(ValidationError, match="minimum_paired_observations"):
        KnnGuard(
            maximum_neighbors=3,
            minimum_paired_observations=1,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0.0,
        )


def test_one_pair_falls_back_but_two_consistent_pairs_can_route() -> None:
    """The minimum valid guard measures variance and never routes from one comparison."""
    scores = np.asarray(
        (
            (0.5, 0.9, 0.0),
            (0.5, float("nan"), 0.0),
            (0.5, float("nan"), 0.0),
        ),
        dtype=np.float32,
    )
    costs = np.asarray(((0.5, 0.1, 0.6),) * 3, dtype=np.float64)
    policy, manifest, bank = _fixture(scores, costs)
    guard = policy.guard.model_copy(update={"minimum_paired_observations": 2})
    policy = policy.model_copy(update={"guard": guard})

    one_pair = _select(policy, manifest, bank, np.asarray((1.0, 0.0)))

    assert one_pair.selected_alias == policy.baseline_alias
    assert one_pair.paired_count == 1
    assert one_pair.fallback_reason == "insufficient_pairs"

    two_scores = scores.copy()
    two_scores[1, 1] = 0.9
    two_bank = _fixture(two_scores, costs)[2]
    two_manifest = manifest.model_copy(
        update={"bank_sha256": hashlib.sha256(bank_bytes(two_bank)).hexdigest()}
    )
    two_policy = policy.model_copy(update={"bank_sha256": two_manifest.bank_sha256})

    two_pairs = _select(two_policy, two_manifest, two_bank, np.asarray((1.0, 0.0)))

    assert two_pairs.selected_alias == "candidate-b"
    assert two_pairs.paired_count == 2
    assert two_pairs.uncertainty == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("candidate_scores", "candidate_costs", "expected_reason"),
    [
        ((0.7, 0.3, 0.7), (0.1, 0.1, 0.1), "neighbor_disagreement"),
        ((0.7, float("nan"), float("nan")), (0.1, 0.1, 0.1), "insufficient_pairs"),
        ((0.4, 0.4, 0.4), (0.1, 0.1, 0.1), "uncertainty"),
        ((0.7, 0.7, 0.7), (0.5, 0.5, 0.5), "no_cheaper_candidate"),
    ],
)
def test_each_guard_reverts_to_full_quality_baseline(
    candidate_scores: tuple[float, float, float],
    candidate_costs: tuple[float, float, float],
    expected_reason: str,
) -> None:
    """Disagreement, sparse pairs, uncertainty, and no savings all retain baseline quality."""
    scores = np.asarray(tuple((0.5, score, 0.0) for score in candidate_scores), dtype=np.float32)
    costs = np.asarray(tuple((0.5, cost, 0.6) for cost in candidate_costs), dtype=np.float64)
    policy, manifest, bank = _fixture(scores, costs)

    decision = _select(policy, manifest, bank, np.asarray((1.0, 0.0)))

    assert decision.selected_alias == "candidate-baseline"
    assert decision.fallback_reason == expected_reason


def _fixture(
    scores: np.ndarray,
    costs: np.ndarray,
) -> tuple[KnnRouterPolicy, KnnBankManifest, KnnEvidenceBank]:
    """Create one aligned three-task, three-candidate policy and bank."""
    aliases = ("candidate-baseline", "candidate-b", "candidate-a")
    bank = KnnEvidenceBank(
        task_ids=("task-a", "task-b", "task-c"),
        candidate_aliases=aliases,
        embeddings=np.asarray(((1.0, 0.0),) * 3, dtype=np.float32),
        scores=scores,
        candidate_costs=costs,
        score_counts=(~np.isnan(scores)).astype(np.int32),
        cost_counts=(~np.isnan(costs)).astype(np.int32),
        workload_weights=np.ones(3, dtype=np.float64),
        novelty_floor=0.5,
    )
    bank.validate()
    counts = tuple(
        CandidateEvidenceCount(
            candidate_alias=alias,
            scored_task_count=int(np.count_nonzero(~np.isnan(scores[:, column]))),
            costed_task_count=int(np.count_nonzero(~np.isnan(costs[:, column]))),
        )
        for column, alias in enumerate(aliases)
    )
    manifest = KnnBankManifest(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        bank_artifact_id="knn-bank-a",
        fit_evaluation_id="evaluation-a",
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        task_set_id="task-set-a",
        task_set_sha256=_DIGEST,
        task_ids=bank.task_ids,
        candidate_aliases=aliases,
        evaluation_protocols_sha256=_DIGEST,
        embedder_alias="embedder",
        embedder=_snapshot("embedder"),
        feature_extractor_id="request-visible-v2",
        feature_schema_sha256=_DIGEST,
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        bank_sha256=hashlib.sha256(bank_bytes(bank)).hexdigest(),
        embedding_dimension=2,
        novelty_floor=bank.novelty_floor,
        evidence_counts=counts,
    )
    policy = KnnRouterPolicy(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        policy_id="router-policy-a",
        baseline_alias="candidate-baseline",
        candidates=tuple(_candidate(alias) for alias in aliases),
        embedder_alias=manifest.embedder_alias,
        embedder=manifest.embedder,
        feature_extractor_id=manifest.feature_extractor_id,
        feature_schema_sha256=manifest.feature_schema_sha256,
        pricing_snapshot_id=manifest.pricing_snapshot_id,
        pricing_snapshot_sha256=manifest.pricing_snapshot_sha256,
        bank_artifact_id=manifest.bank_artifact_id,
        bank_sha256=manifest.bank_sha256,
        guard=KnnGuard(
            maximum_neighbors=3,
            minimum_paired_observations=3,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0.0,
        ),
        fit_evaluation_id=manifest.fit_evaluation_id,
        evaluation_plan_id=manifest.evaluation_plan_id,
        evaluation_plan_sha256=manifest.evaluation_plan_sha256,
        task_set_id=manifest.task_set_id,
        task_set_sha256=manifest.task_set_sha256,
        evaluation_protocols_sha256=manifest.evaluation_protocols_sha256,
        judgment_status="provisional",
    )
    return policy, manifest, bank


def _select(
    policy: KnnRouterPolicy,
    manifest: KnnBankManifest,
    bank: KnnEvidenceBank,
    query: np.ndarray,
) -> RoutingDecision:
    """Select one deterministic test request."""
    return select_from_bank(
        policy,
        manifest,
        bank,
        query,
        request_sha256=_REQUEST_DIGEST,
        episode_id="episode-a",
    )


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    """Create a candidate with its exact model and connection snapshot."""
    return RoutedCandidateSnapshot(alias=alias, model=_snapshot(alias))


def _snapshot(alias: str) -> ModelSnapshot:
    """Create a secret-free model snapshot."""
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_DIGEST,
        connection_sha256="c" * 64,
    )
