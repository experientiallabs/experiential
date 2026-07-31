"""Replay locked BigCodeBench effort routers on one outer-heldout partition.

The fitting data and evaluation outcomes are explicit inputs so shuffled-label
controls can train on destroyed source labels while every route is valued on
the original heldout outcomes. DeepSWE artifacts are outside this module.
"""

from __future__ import annotations

import hashlib
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
    TaskSplit,
    cost_only_choices,
    evaluate_choices,
    feature_matrix,
    fit_native_knn_replay,
    fit_selected_static,
    random_choices,
    seed_split_provenance,
    shuffled_task_rewards,
)
from coding_model_router_bigcodebench_select import (
    CandidateSpec,
    Estimator,
    Family,
    KnnCandidateSpec,
    _candidate_choices,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

ControlKind = Literal["static", "matched-task-blind", "random", "cost-only", "shuffled-label"]


@dataclass(frozen=True)
class HeldoutReplay:
    """One locked candidate's routes and observed outer-heldout value."""

    spec: CandidateSpec | KnnCandidateSpec
    choices: np.ndarray
    value: PolicyValue
    baseline: PolicyValue
    metric: CandidateMetric


class ValueRecord(BaseModel):
    """Observed reward, cost, and effort traffic for one heldout policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reward: float = Field(ge=0.0, le=1.0)
    cost_usd: float = Field(ge=0.0)
    arm_counts: dict[str, int]

    @model_validator(mode="after")
    def _complete_nonnegative_traffic(self) -> ValueRecord:
        if set(self.arm_counts) != set(ARMS):
            raise ValueError("heldout traffic must contain every frozen effort arm")
        if any(count < 0 for count in self.arm_counts.values()):
            raise ValueError("heldout traffic counts must be nonnegative")
        return self


class HeldoutControlRecord(ValueRecord):
    """One preregistered outer-heldout negative or static control."""

    kind: ControlKind
    name: str = Field(min_length=1)


class HeldoutTaskRecord(BaseModel):
    """Paired task-level evidence retained for grouped promotion statistics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    group: str = Field(min_length=1)
    arm_rewards: dict[str, float]
    arm_costs_usd: dict[str, float]
    router_arm: str
    baseline_arm: str
    random_arm: str
    cost_only_arm: str
    shuffled_label_arm: str

    @model_validator(mode="after")
    def _complete_valid_effort_rows(self) -> HeldoutTaskRecord:
        if set(self.arm_rewards) != set(ARMS) or set(self.arm_costs_usd) != set(ARMS):
            raise ValueError("heldout task row must contain every frozen effort arm")
        if any(not 0.0 <= reward <= 1.0 for reward in self.arm_rewards.values()):
            raise ValueError("heldout task row contains an invalid reward")
        if any(cost < 0.0 or not np.isfinite(cost) for cost in self.arm_costs_usd.values()):
            raise ValueError("heldout task row contains an invalid cost")
        choices = (
            self.router_arm,
            self.baseline_arm,
            self.random_arm,
            self.cost_only_arm,
            self.shuffled_label_arm,
        )
        if any(choice not in ARMS for choice in choices):
            raise ValueError("heldout task row selected an unknown effort arm")
        return self


class SeedHeldoutReport(BaseModel):
    """Immutable result of one locked candidate's single outer replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["bigcodebench-outer-heldout-seed-v1"] = "bigcodebench-outer-heldout-seed-v1"
    seed: int = Field(ge=0, le=4)
    code_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    selection_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_fit_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    winner_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    heldout_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_tasks: int = Field(gt=0)
    heldout_tasks: int = Field(gt=0)
    candidate_family: str = Field(min_length=1)
    candidate_name: str = Field(min_length=1)
    candidate_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    router: ValueRecord
    baseline: ValueRecord
    controls: list[HeldoutControlRecord] = Field(min_length=9, max_length=9)
    tasks: list[HeldoutTaskRecord] = Field(min_length=1)
    target_outcomes_used: Literal[False] = False
    outer_heldout_evaluated: Literal[True] = True

    @model_validator(mode="after")
    def _exact_controls_and_traffic(self) -> SeedHeldoutReport:
        expected = {f"static-{arm}" for arm in ARMS} | {
            "selected-matched-task-blind",
            "seeded-uniform-random",
            "fit-cost-only",
            "selected-shuffled-labels",
        }
        names = [control.name for control in self.controls]
        if set(names) != expected or len(names) != len(set(names)):
            raise ValueError("heldout report does not contain the exact frozen controls")
        values: list[ValueRecord] = [self.router, self.baseline, *self.controls]
        if any(sum(value.arm_counts.values()) != self.heldout_tasks for value in values):
            raise ValueError("heldout policy traffic does not cover every heldout task")
        task_ids = [task.task_id for task in self.tasks]
        if len(self.tasks) != self.heldout_tasks or len(task_ids) != len(set(task_ids)):
            raise ValueError("heldout task evidence does not match the heldout task count")
        return self


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


def _value_record(value: PolicyValue) -> ValueRecord:
    """Convert one in-memory policy value to a durable report row."""
    return ValueRecord(
        reward=value.reward,
        cost_usd=value.cost_usd,
        arm_counts=value.arm_counts,
    )


def _control_record(
    kind: ControlKind,
    name: str,
    value: PolicyValue,
) -> HeldoutControlRecord:
    """Convert one in-memory control value to a durable report row."""
    return HeldoutControlRecord(
        kind=kind,
        name=name,
        reward=value.reward,
        cost_usd=value.cost_usd,
        arm_counts=value.arm_counts,
    )


def seed_heldout_report(
    data: FitData,
    split: TaskSplit,
    spec: CandidateSpec | KnnCandidateSpec,
    *,
    code_commit: str,
    selection_lock_sha256: str,
    seed_fit_report_sha256: str,
    winner_audit_sha256: str,
    candidate_config_sha256: str,
    work_dir: Path,
) -> SeedHeldoutReport:
    """Run one outer replay and all nine preregistered heldout controls."""
    canonical_config = json.dumps(spec.config(), sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical_config.encode()).hexdigest() != candidate_config_sha256:
        raise ValueError("heldout candidate config differs from the locked digest")
    fit = split.train_indices
    heldout = split.test_indices
    replay = replay_outer_heldout(
        data,
        fit,
        heldout,
        spec,
        seed=split.seed,
        work_dir=work_dir / "router",
    )
    rewards = data.rewards[heldout].mean(axis=2)
    costs = data.costs[heldout].mean(axis=2)
    controls = [
        _control_record(
            "static",
            f"static-{arm}",
            evaluate_choices(
                rewards,
                costs,
                np.full(len(heldout), arm_index, dtype=np.int64),
            ),
        )
        for arm_index, arm in enumerate(ARMS)
    ]
    controls.append(
        HeldoutControlRecord(
            kind="matched-task-blind",
            name="selected-matched-task-blind",
            reward=replay.value.matched_blind_reward,
            cost_usd=replay.value.matched_blind_cost_usd,
            arm_counts=replay.value.arm_counts,
        )
    )
    random_assignment = random_choices(len(heldout), seed=40_000 + split.seed)
    controls.append(
        _control_record(
            "random",
            "seeded-uniform-random",
            evaluate_choices(rewards, costs, random_assignment),
        )
    )
    fit_costs = data.costs[fit].mean(axis=2)
    cost_only_arm = int(cost_only_choices(fit_costs)[0])
    controls.append(
        _control_record(
            "cost-only",
            "fit-cost-only",
            evaluate_choices(
                rewards,
                costs,
                np.full(len(heldout), cost_only_arm, dtype=np.int64),
            ),
        )
    )
    shuffled_rewards = data.rewards.copy()
    shuffled_rewards[fit] = shuffled_task_rewards(
        data.rewards[fit],
        seed=50_000 + split.seed,
    )
    shuffled = FitData(
        task_ids=data.task_ids,
        groups=data.groups,
        texts=data.texts,
        is_hard=data.is_hard,
        rewards=shuffled_rewards,
        costs=data.costs,
    )
    shuffled_replay = replay_outer_heldout(
        shuffled,
        fit,
        heldout,
        spec,
        seed=60_000 + split.seed,
        work_dir=work_dir / "shuffled-label",
        evaluation_data=data,
    )
    controls.append(
        _control_record(
            "shuffled-label",
            "selected-shuffled-labels",
            shuffled_replay.value,
        )
    )
    fit_ids_sha256, heldout_ids_sha256 = seed_split_provenance(data, split)
    family = "knn" if isinstance(spec, KnnCandidateSpec) else spec.family
    baseline_arm = next(
        arm for arm, count in replay.baseline.arm_counts.items() if count == len(heldout)
    )
    task_rows = [
        HeldoutTaskRecord(
            task_id=data.task_ids[int(task_index)],
            group=data.groups[int(task_index)],
            arm_rewards={arm: float(rewards[row, index]) for index, arm in enumerate(ARMS)},
            arm_costs_usd={arm: float(costs[row, index]) for index, arm in enumerate(ARMS)},
            router_arm=ARMS[int(replay.choices[row])],
            baseline_arm=baseline_arm,
            random_arm=ARMS[int(random_assignment[row])],
            cost_only_arm=ARMS[cost_only_arm],
            shuffled_label_arm=ARMS[int(shuffled_replay.choices[row])],
        )
        for row, task_index in enumerate(heldout)
    ]
    return SeedHeldoutReport(
        seed=split.seed,
        code_commit=code_commit,
        selection_lock_sha256=selection_lock_sha256,
        seed_fit_report_sha256=seed_fit_report_sha256,
        winner_audit_sha256=winner_audit_sha256,
        fit_ids_sha256=fit_ids_sha256,
        heldout_ids_sha256=heldout_ids_sha256,
        fit_tasks=len(fit),
        heldout_tasks=len(heldout),
        candidate_family=family,
        candidate_name=spec.name,
        candidate_config_sha256=candidate_config_sha256,
        router=_value_record(replay.value),
        baseline=_value_record(replay.baseline),
        controls=controls,
        tasks=task_rows,
    )


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
