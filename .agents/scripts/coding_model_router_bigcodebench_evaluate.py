"""Replay locked BigCodeBench effort routers on one outer-heldout partition.

The fitting data and evaluation outcomes are explicit inputs so shuffled-label
controls can train on destroyed source labels while every route is valued on
the original heldout outcomes. DeepSWE artifacts are outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
