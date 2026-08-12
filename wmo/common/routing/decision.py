"""Pure guarded kNN selection against one immutable offline evidence bank."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from wmo.common.core.artifacts import Sha256, sha256_json, stable_id
from wmo.common.routing import KnnRouterPolicy, RoutingDecision
from wmo.common.routing.bank import KnnBankManifest, KnnEvidenceBank, bank_bytes


class RouterDecisionError(ValueError):
    """A policy, bank, or request vector cannot support a trustworthy decision."""


@dataclass(frozen=True)
class _CandidateEstimate:
    """One cheaper candidate's paired-neighbor guard result."""

    alias: str
    cost: float
    paired_count: int
    mean_difference: float
    uncertainty: float
    conservative_difference: float
    rejection: str | None


def select_from_bank(
    policy: KnnRouterPolicy,
    bank_manifest: KnnBankManifest,
    bank: KnnEvidenceBank,
    query: np.ndarray,
    *,
    request_sha256: Sha256,
    episode_id: str,
) -> RoutingDecision:
    """Choose a candidate with the frozen conservative policy and no online learning.

    Args:
        policy: Locked offline policy containing every identity and guard parameter.
        bank_manifest: Verified manifest for the policy's numeric evidence sidecar.
        bank: Verified fit-only task, score, and candidate-cost arrays.
        query: One vector from the policy's exact request-visible embedder path.
        request_sha256: Digest of the canonical request-visible feature record.
        episode_id: Stable request or conversation identity for the decision log.

    Returns:
        A deterministic selection or an explained fallback to the quality baseline.

    Raises:
        RouterDecisionError: Policy pins drifted or the request vector is unusable.
    """
    _require_policy_bank_match(policy, bank_manifest, bank)
    vector = _normalized_query(query, bank.embeddings.shape[1])
    similarities = bank.embeddings.astype(np.float64) @ vector
    order = tuple(
        sorted(
            range(len(bank.task_ids)),
            key=lambda index: (-float(similarities[index]), bank.task_ids[index]),
        )
    )
    best_similarity = float(similarities[order[0]])
    if best_similarity < bank.novelty_floor:
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=policy.baseline_alias,
            neighbor_count=0,
            paired_count=0,
            best_similarity=best_similarity,
            fallback_reason="novelty",
        )
    if best_similarity <= 0:
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=policy.baseline_alias,
            neighbor_count=0,
            paired_count=0,
            best_similarity=best_similarity,
            fallback_reason="distance",
        )
    neighbor_rows = _neighbor_rows(policy, similarities, order)
    if not neighbor_rows:
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=policy.baseline_alias,
            neighbor_count=0,
            paired_count=0,
            best_similarity=best_similarity,
            fallback_reason="distance",
        )
    baseline_cost = bank.complete_weighted_cost(policy.baseline_alias)
    if baseline_cost is None:
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=policy.baseline_alias,
            neighbor_count=len(neighbor_rows),
            paired_count=0,
            best_similarity=best_similarity,
            fallback_reason="missing_cost",
        )
    cheaper, missing_cost = _cheaper_candidates(bank, policy.baseline_alias, baseline_cost)
    if not cheaper:
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=policy.baseline_alias,
            neighbor_count=len(neighbor_rows),
            paired_count=0,
            best_similarity=best_similarity,
            fallback_reason="missing_cost" if missing_cost else "no_cheaper_candidate",
        )
    estimates = tuple(
        _estimate_candidate(policy, bank, neighbor_rows, alias, cost) for alias, cost in cheaper
    )
    eligible = tuple(item for item in estimates if item.rejection is None)
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (item.cost, -item.conservative_difference, item.alias),
        )
        return _decision(
            policy,
            request_sha256=request_sha256,
            episode_id=episode_id,
            selected_alias=selected.alias,
            neighbor_count=len(neighbor_rows),
            paired_count=selected.paired_count,
            best_similarity=best_similarity,
            estimated_quality_difference=selected.mean_difference,
            uncertainty=selected.uncertainty,
        )
    explained = min(estimates, key=lambda item: (item.cost, item.alias))
    fallback_reason = _fallback_reason(estimates)
    return _decision(
        policy,
        request_sha256=request_sha256,
        episode_id=episode_id,
        selected_alias=policy.baseline_alias,
        neighbor_count=len(neighbor_rows),
        paired_count=explained.paired_count,
        best_similarity=best_similarity,
        estimated_quality_difference=(
            explained.mean_difference if explained.paired_count else None
        ),
        uncertainty=explained.uncertainty if explained.paired_count else None,
        fallback_reason=fallback_reason,
    )


def policy_content_sha256(policy: KnnRouterPolicy) -> Sha256:
    """Return the canonical content digest recorded in decisions and reports."""
    return sha256_json(policy)


def _require_policy_bank_match(
    policy: KnnRouterPolicy,
    manifest: KnnBankManifest,
    bank: KnnEvidenceBank,
) -> None:
    """Fail before selection when any fit-time identity or sidecar axis drifted."""
    if (
        not math.isfinite(policy.guard.uncertainty_multiplier)
        or policy.guard.uncertainty_multiplier <= 0
    ):
        raise RouterDecisionError("router uncertainty multiplier must be finite and positive")
    content_sha256 = hashlib.sha256(bank_bytes(bank)).hexdigest()
    if content_sha256 != policy.bank_sha256 or content_sha256 != manifest.bank_sha256:
        raise RouterDecisionError("router bank content has mutated from the frozen policy")
    policy_aliases = tuple(candidate.alias for candidate in policy.candidates)
    checks = (
        (policy.bank_artifact_id, manifest.bank_artifact_id, "bank artifact"),
        (policy.bank_sha256, manifest.bank_sha256, "bank digest"),
        (policy.fit_evaluation_id, manifest.fit_evaluation_id, "fit evaluation"),
        (policy.evaluation_plan_id, manifest.evaluation_plan_id, "evaluation plan"),
        (
            policy.evaluation_plan_sha256,
            manifest.evaluation_plan_sha256,
            "evaluation plan digest",
        ),
        (policy.task_set_id, manifest.task_set_id, "task set"),
        (policy.task_set_sha256, manifest.task_set_sha256, "task-set digest"),
        (
            policy.evaluation_protocols_sha256,
            manifest.evaluation_protocols_sha256,
            "evaluation protocol scope",
        ),
        (policy.fidelity_report_ids, manifest.fidelity_report_ids, "fidelity scope"),
        (policy.embedder_alias, manifest.embedder_alias, "embedder alias"),
        (policy.embedder, manifest.embedder, "embedder snapshot"),
        (policy.feature_extractor_id, manifest.feature_extractor_id, "feature extractor"),
        (policy.feature_schema_sha256, manifest.feature_schema_sha256, "feature schema"),
        (policy.pricing_snapshot_id, manifest.pricing_snapshot_id, "pricing snapshot"),
        (
            policy.pricing_snapshot_sha256,
            manifest.pricing_snapshot_sha256,
            "pricing snapshot digest",
        ),
        (policy_aliases, manifest.candidate_aliases, "candidate aliases"),
        (manifest.candidate_aliases, bank.candidate_aliases, "bank candidate columns"),
        (manifest.task_ids, bank.task_ids, "bank task rows"),
    )
    for expected, actual, label in checks:
        if expected != actual:
            raise RouterDecisionError(f"router {label} has drifted from the frozen policy")
    if policy.baseline_alias not in bank.candidate_aliases:
        raise RouterDecisionError("router bank does not contain the policy baseline")


def _normalized_query(query: np.ndarray, dimension: int) -> np.ndarray:
    """Return one finite unit vector with the bank's exact embedding dimension."""
    vector = np.asarray(query, dtype=np.float64)
    if vector.shape != (dimension,) or not np.all(np.isfinite(vector)):
        raise RouterDecisionError(
            f"request embedding must be one finite {dimension}-dimensional vector"
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0 or not math.isfinite(norm):
        raise RouterDecisionError("request embedding must have nonzero finite norm")
    return vector / norm


def _neighbor_rows(
    policy: KnnRouterPolicy,
    similarities: np.ndarray,
    order: tuple[int, ...],
) -> tuple[int, ...]:
    """Apply the native bounded relative-distance neighborhood deterministically."""
    budget = min(policy.guard.maximum_neighbors, len(order))
    nearest = order[:budget]
    best_similarity = float(similarities[nearest[0]])
    cutoff = policy.guard.relative_similarity_threshold * best_similarity
    return tuple(index for index in nearest if float(similarities[index]) >= cutoff)


def _cheaper_candidates(
    bank: KnnEvidenceBank,
    baseline_alias: str,
    baseline_cost: float,
) -> tuple[tuple[tuple[str, float], ...], bool]:
    """Return fully costed, strictly cheaper candidates and whether cost was missing."""
    cheaper = []
    missing_cost = False
    for alias in bank.candidate_aliases:
        if alias == baseline_alias:
            continue
        cost = bank.complete_weighted_cost(alias)
        if cost is None:
            missing_cost = True
        elif cost < baseline_cost:
            cheaper.append((alias, cost))
    return tuple(sorted(cheaper)), missing_cost


def _estimate_candidate(
    policy: KnnRouterPolicy,
    bank: KnnEvidenceBank,
    rows: tuple[int, ...],
    alias: str,
    cost: float,
) -> _CandidateEstimate:
    """Estimate one cheaper candidate only on paired scored neighbors."""
    candidate_column = bank.candidate_aliases.index(alias)
    baseline_column = bank.candidate_aliases.index(policy.baseline_alias)
    candidate_scores = bank.scores[list(rows), candidate_column].astype(np.float64)
    baseline_scores = bank.scores[list(rows), baseline_column].astype(np.float64)
    paired = ~np.isnan(candidate_scores) & ~np.isnan(baseline_scores)
    differences = candidate_scores[paired] - baseline_scores[paired]
    paired_count = int(differences.size)
    if paired_count:
        mean_difference = float(np.mean(differences))
        empirical_standard_error = (
            float(np.std(differences, ddof=1)) / math.sqrt(paired_count)
            if paired_count > 1
            else 0.0
        )
        # Score differences lie in [-1, 1]. With no variance prior, Popoviciu's
        # bound gives a maximum population standard deviation of range / 2 = 1.
        # Use its finite-sample width, 1 / sqrt(n), as a conservative floor on
        # empirical SE so constant samples never manufacture zero uncertainty.
        standard_error = max(empirical_standard_error, 1.0 / math.sqrt(paired_count))
    else:
        mean_difference = 0.0
        standard_error = 0.0
    uncertainty = policy.guard.uncertainty_multiplier * standard_error
    conservative = mean_difference - uncertainty
    threshold = -policy.guard.quality_tolerance
    rejection = None
    if paired_count < max(8, policy.guard.minimum_paired_observations):
        rejection = "insufficient_pairs"
    elif bool(np.any(differences < threshold) and np.any(differences > threshold)):
        rejection = "neighbor_disagreement"
    elif conservative < threshold:
        rejection = "uncertainty"
    return _CandidateEstimate(
        alias=alias,
        cost=cost,
        paired_count=paired_count,
        mean_difference=mean_difference,
        uncertainty=uncertainty,
        conservative_difference=conservative,
        rejection=rejection,
    )


def _fallback_reason(estimates: tuple[_CandidateEstimate, ...]) -> str:
    """Choose one stable summary when every cheaper candidate was vetoed."""
    reasons = {item.rejection for item in estimates}
    for reason in ("neighbor_disagreement", "insufficient_pairs", "uncertainty"):
        if reason in reasons:
            return reason
    return "uncertainty"


def _decision(
    policy: KnnRouterPolicy,
    *,
    request_sha256: Sha256,
    episode_id: str,
    selected_alias: str,
    neighbor_count: int,
    paired_count: int,
    best_similarity: float | None,
    estimated_quality_difference: float | None = None,
    uncertainty: float | None = None,
    fallback_reason: str | None = None,
) -> RoutingDecision:
    """Create one content-addressed decision with its exact policy digest."""
    policy_sha256 = policy_content_sha256(policy)
    material = {
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256,
        "request_sha256": request_sha256,
        "episode_id": episode_id,
        "selected_alias": selected_alias,
        "baseline_alias": policy.baseline_alias,
        "neighbor_count": neighbor_count,
        "paired_count": paired_count,
        "best_similarity": best_similarity,
        "estimated_quality_difference": estimated_quality_difference,
        "uncertainty": uncertainty,
        "fallback_reason": fallback_reason,
    }
    return RoutingDecision(
        decision_id=stable_id("routing-decision", material),
        policy_id=policy.policy_id,
        policy_sha256=policy_sha256,
        request_sha256=request_sha256,
        episode_id=episode_id,
        selected_alias=selected_alias,
        baseline_alias=policy.baseline_alias,
        neighbor_count=neighbor_count,
        paired_count=paired_count,
        best_similarity=best_similarity,
        estimated_quality_difference=estimated_quality_difference,
        uncertainty=uncertainty,
        fallback_reason=fallback_reason,
    )
