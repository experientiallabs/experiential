"""Build and latency-audit a fit-selected BigCodeBench WMO kNN winner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
from coding_model_router_bigcodebench_fit import (
    FitData,
    LatencyMetric,
    artifact_size,
    measure_route_latency,
    outcome_matrix,
)
from coding_model_router_bigcodebench_select_run import CandidateRecord
from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy, knn_decision


class KnnArtifactAudit(BaseModel):
    """Content and one-core latency evidence for one selected WMO kNN artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: str = "bigcodebench-knn-artifact-audit-v1"
    candidate_name: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bank_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_bytes: int = Field(gt=0)
    decisions: int = Field(gt=0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_passed: bool
    network_calls_per_route: int = Field(default=0, ge=0, le=0)
    foundation_model_weights_persisted: bool = False
    target_outcomes_used: bool = False
    outer_heldout_evaluated: bool = False


def _sha256(path: Path) -> str:
    """Return one artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(record: CandidateRecord) -> dict[str, str | int | float | bool | None]:
    """Load and type-check one canonical candidate configuration."""
    value = json.loads(record.config_json)
    if not isinstance(value, dict):
        raise ValueError("candidate config must be one JSON object")
    return {str(key): cast(str | int | float | bool | None, item) for key, item in value.items()}


def _number(config: dict[str, str | int | float | bool | None], key: str) -> float:
    """Read one numeric config field without accepting booleans."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"kNN config field {key} is not numeric")
    return float(value)


def fit_knn_winner(
    data: FitData,
    fit_indices: np.ndarray,
    record: CandidateRecord,
    *,
    baseline_arm: str,
    artifact_dir: Path,
) -> tuple[RoutingPolicy, Path, Path]:
    """Refit one selected kNN config on the complete outer-fit partition."""
    if record.family != "knn":
        raise ValueError("native kNN audit received a non-kNN winner")
    config = _config(record)
    guard = config.get("guard_model")
    guard_model = baseline_arm if guard == "fit-best" else guard
    if not isinstance(guard_model, str):
        raise ValueError("kNN config has no valid guard model")
    guard_mode = config.get("guard_mode")
    if guard_mode not in {"symmetric", "asymmetric"}:
        raise ValueError("kNN config has no valid guard mode")
    dim = int(_number(config, "dim"))
    artifact_dir.mkdir(parents=True, exist_ok=False)
    bank_path = artifact_dir / "knn-bank.npz"
    policy_path = artifact_dir / "policy.json"
    policy = fit_knn_policy(
        outcome_matrix(data),
        bank_path=bank_path,
        fit_ids=[data.task_ids[int(index)] for index in fit_indices],
        embedder=EmbedderSpec(kind="hashing", dim=dim),
        guard_model=guard_model,
        rag_num=int(_number(config, "rag_num")),
        rag_thres=_number(config, "rag_thres"),
        z=_number(config, "z"),
        min_pairs=int(_number(config, "min_pairs")),
        se_floor=True,
        floor_q=0.0,
        pick_lam=_number(config, "pick_lam"),
        fitted_from="bigcodebench-v0.2.4 selected outer fit only",
    ).model_copy(update={"guard_mode": guard_mode})
    policy.save(policy_path)
    return policy, policy_path, bank_path


def latency_audit(
    policy: RoutingPolicy,
    texts: list[str],
    *,
    decisions: int = 10_000,
) -> LatencyMetric:
    """Measure the actual single-request WMO kNN route path without network calls."""
    embedder = policy.embedder.build()

    def route_one(text: str) -> int:
        vector = np.asarray(embedder.embed([text])[0], dtype=np.float64)
        model = knn_decision(policy, vector).model
        return next(index for index, entry in enumerate(policy.pool) if entry.name == model)

    return measure_route_latency(route_one, texts, decisions=decisions)


def audit_knn_winner(
    data: FitData,
    fit_indices: np.ndarray,
    record: CandidateRecord,
    *,
    baseline_arm: str,
    artifact_dir: Path,
    decisions: int = 10_000,
) -> KnnArtifactAudit:
    """Build, measure, and fingerprint one selected outer-fit kNN artifact."""
    policy, policy_path, bank_path = fit_knn_winner(
        data,
        fit_indices,
        record,
        baseline_arm=baseline_arm,
        artifact_dir=artifact_dir,
    )
    latency = latency_audit(
        policy,
        [data.texts[int(index)] for index in fit_indices],
        decisions=decisions,
    )
    return KnnArtifactAudit(
        candidate_name=record.name,
        config_sha256=record.config_sha256,
        policy_sha256=_sha256(policy_path),
        bank_sha256=_sha256(bank_path),
        artifact_bytes=artifact_size([policy_path, bank_path]),
        decisions=latency.decisions,
        latency_p50_ms=latency.p50_ms,
        latency_p95_ms=latency.p95_ms,
        latency_passed=latency.passed,
    )
