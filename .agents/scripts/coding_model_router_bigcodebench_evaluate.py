"""Replay locked BigCodeBench effort routers on one outer-heldout partition.

The fitting data and evaluation outcomes are explicit inputs so shuffled-label
controls can train on destroyed source labels while every route is valued on
the original heldout outcomes. DeepSWE artifacts are outside this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from coding_model_router_bigcodebench_fit import (
    ARMS,
    CandidateMetric,
    FitData,
    PolicyValue,
    evaluate_choices,
    feature_matrix,
    fit_native_knn_replay,
    fit_selected_static,
)
from coding_model_router_bigcodebench_select import (
    CandidateSpec,
    Estimator,
    Family,
    KnnCandidateSpec,
    _candidate_choices,
)


@dataclass(frozen=True)
class HeldoutReplay:
    """One locked candidate's routes and observed outer-heldout value."""

    spec: CandidateSpec | KnnCandidateSpec
    choices: np.ndarray
    value: PolicyValue
    baseline: PolicyValue
    metric: CandidateMetric


def _number(config: dict[str, object], key: str) -> float:
    """Read one finite numeric candidate field without accepting booleans."""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate config field {key} is not numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"candidate config field {key} is not finite")
    return result


def _integer(config: dict[str, object], key: str) -> int:
    """Read one exact integer candidate field."""
    value = _number(config, key)
    if not value.is_integer():
        raise ValueError(f"candidate config field {key} is not an integer")
    return int(value)


def candidate_spec_from_lock(
    family: str,
    config_json: str,
    *,
    name: str,
    order: int,
) -> CandidateSpec | KnnCandidateSpec:
    """Rebuild one exact frozen candidate from canonical lock fields."""
    raw = json.loads(config_json)
    if not isinstance(raw, dict):
        raise ValueError("locked candidate config must be one JSON object")
    config = {str(key): value for key, value in raw.items()}
    if family == "knn":
        guard_value = config.get("guard_model")
        if guard_value == "fit-best":
            guard_model = None
        elif isinstance(guard_value, str) and guard_value in ARMS:
            guard_model = guard_value
        else:
            raise ValueError("locked kNN candidate has an invalid guard model")
        guard_mode = config.get("guard_mode")
        if guard_mode not in {"symmetric", "asymmetric"}:
            raise ValueError("locked kNN candidate has an invalid guard mode")
        spec: CandidateSpec | KnnCandidateSpec = KnnCandidateSpec(
            dim=_integer(config, "dim"),
            rag_num=_integer(config, "rag_num"),
            rag_thres=_number(config, "rag_thres"),
            z=_number(config, "z"),
            min_pairs=_integer(config, "min_pairs"),
            order=order,
            guard_model=guard_model,
            guard_mode=cast(Literal["symmetric", "asymmetric"], guard_mode),
            pick_lam=_number(config, "pick_lam"),
        )
    else:
        candidate_family = config.get("family")
        estimator = config.get("estimator")
        max_features = config.get("max_features")
        if candidate_family not in {"ordinal", "doubly-robust", "empirical-bayes"}:
            raise ValueError("locked candidate has an invalid non-kNN family")
        if candidate_family != family:
            raise ValueError("locked candidate family differs from its config")
        if estimator not in {"ridge", "extra-trees", "histogram"}:
            raise ValueError("locked candidate has an invalid estimator")
        if max_features not in {"", "sqrt", "third"}:
            raise ValueError("locked candidate has invalid max_features")
        spec = CandidateSpec(
            family=cast(Family, candidate_family),
            estimator=cast(Estimator, estimator),
            dim=_integer(config, "dim"),
            order=order,
            alpha=_number(config, "alpha"),
            n_estimators=_integer(config, "n_estimators"),
            min_samples_leaf=_integer(config, "min_samples_leaf"),
            max_features=cast(Literal["", "sqrt", "third"], max_features),
            max_leaf_nodes=_integer(config, "max_leaf_nodes"),
            learning_rate=_number(config, "learning_rate"),
            lam=_number(config, "lam"),
            prior_strength=_number(config, "prior_strength"),
            z=_number(config, "z"),
        )
    canonical = json.dumps(spec.config(), sort_keys=True, separators=(",", ":"))
    if canonical != config_json or spec.name != name:
        raise ValueError("locked candidate config does not reproduce its identity")
    return spec


def _partitions(
    data: FitData,
    fit_indices: np.ndarray,
    heldout_indices: np.ndarray,
    evaluation_data: FitData,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate one grouped train and heldout boundary."""
    fit = np.asarray(fit_indices, dtype=np.int64)
    heldout = np.asarray(heldout_indices, dtype=np.int64)
    if fit.size == 0 or heldout.size == 0:
        raise ValueError("outer replay needs nonempty fit and heldout partitions")
    if len(set(fit.tolist())) != len(fit) or len(set(heldout.tolist())) != len(heldout):
        raise ValueError("outer replay partitions contain duplicate indices")
    if set(fit.tolist()) & set(heldout.tolist()):
        raise ValueError("outer replay fit and heldout partitions overlap")
    if np.any(fit < 0) or np.any(heldout < 0):
        raise ValueError("outer replay partition contains a negative index")
    if np.any(fit >= len(data.task_ids)) or np.any(heldout >= len(data.task_ids)):
        raise ValueError("outer replay partition contains an unknown task index")
    if evaluation_data.task_ids != data.task_ids:
        raise ValueError("outer replay evaluation data has different task identities")
    fit_groups = {data.groups[int(index)] for index in fit}
    heldout_groups = {data.groups[int(index)] for index in heldout}
    if fit_groups & heldout_groups:
        raise ValueError("task-family group crossed the outer replay boundary")
    return fit, heldout


def replay_outer_heldout(
    data: FitData,
    fit_indices: np.ndarray,
    heldout_indices: np.ndarray,
    spec: CandidateSpec | KnnCandidateSpec,
    *,
    seed: int,
    work_dir: Path,
    evaluation_data: FitData | None = None,
) -> HeldoutReplay:
    """Fit one frozen candidate on source-fit rows and replay heldout once."""
    observed = evaluation_data or data
    fit, heldout = _partitions(data, fit_indices, heldout_indices, observed)
    baseline = fit_selected_static(data, fit)
    if isinstance(spec, KnnCandidateSpec):
        guard_arm = spec.guard_model or baseline.name
        native = fit_native_knn_replay(
            data,
            fit,
            heldout,
            bank_path=work_dir / "outer-knn.bank.npz",
            dim=spec.dim,
            guard_arm=guard_arm,
            rag_num=spec.rag_num,
            rag_thres=spec.rag_thres,
            z=spec.z,
            min_pairs=spec.min_pairs,
            se_floor=True,
            floor_q=0.0,
            pick_lam=spec.pick_lam,
            guard_mode=spec.guard_mode,
        )
        choices = native.choices
    else:
        features = feature_matrix(data, dim=spec.dim, scale_indices=fit)
        choices = _candidate_choices(
            spec,
            data,
            fit,
            heldout,
            features[fit],
            features[heldout],
            seed=seed,
        )
    rewards = observed.rewards[heldout].mean(axis=2)
    costs = observed.costs[heldout].mean(axis=2)
    value = evaluate_choices(rewards, costs, choices)
    baseline_choices = np.full(
        len(heldout),
        ARMS.index(baseline.name),
        dtype=np.int64,
    )
    baseline_value = evaluate_choices(rewards, costs, baseline_choices)
    metric = CandidateMetric(
        name=spec.name,
        reward=value.reward,
        cost_usd=value.cost_usd,
        latency_p95_ms=0.0,
        artifact_bytes=0,
        order=spec.order,
    )
    return HeldoutReplay(
        spec=spec,
        choices=choices,
        value=value,
        baseline=baseline_value,
        metric=metric,
    )
