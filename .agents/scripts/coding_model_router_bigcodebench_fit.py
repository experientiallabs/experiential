"""Fit latency-neutral routers on a promoted external BigCodeBench matrix.

The module enforces the held-out-oracle promotion boundary before it reads any
score row. It contains the shared data and evaluation primitives for the frozen
router families. DeepSWE artifacts are outside this script's contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ARMS = ("luna-low", "luna-medium", "luna-high", "luna-xhigh", "luna-max")
ATTEMPTS = 5
EXPECTED_TASKS = 300
EXPECTED_CELLS = EXPECTED_TASKS * len(ARMS) * ATTEMPTS


@dataclass(frozen=True)
class FitData:
    """Dense external task, reward, and cost tensors for router fitting."""

    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    is_hard: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class PolicyValue:
    """Observed policy value and its matched task-blind control."""

    reward: float
    cost_usd: float
    matched_blind_reward: float
    matched_blind_cost_usd: float
    arm_counts: dict[str, int]


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_id(row: dict[str, object]) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row has no task_id")
    return value


def _require_target_safe(value: dict[str, object], *, label: str) -> None:
    if value.get("target_outcomes_used") is not False:
        raise ValueError(f"{label} crossed the target outcome boundary")


def _expected_cell_ids(task_ids: list[str]) -> set[str]:
    return {
        f"{task_id}:{arm}:attempt-{attempt}"
        for task_id in task_ids
        for arm in ARMS
        for attempt in range(ATTEMPTS)
    }


def load_fit_data(root: Path) -> FitData:
    """Load a promoted external matrix after validating every frozen boundary."""
    oracle = _read_object(root / "oracle-report.json")
    protocol = oracle.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("oracle report has no protocol")
    _require_target_safe(
        {str(key): item for key, item in protocol.items()},
        label="oracle report",
    )
    if oracle.get("passed") is not True:
        raise ValueError("external oracle did not pass; router fitting is forbidden")

    matrix = _read_object(root / "matrix-manifest.json")
    score_manifest = _read_object(root / "score-manifest.json")
    _require_target_safe(matrix, label="matrix manifest")
    _require_target_safe(score_manifest, label="score manifest")
    tasks = _read_rows(root / "tasks.jsonl")
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} tasks, found {len(tasks)}")
    if (
        matrix.get("cells") != EXPECTED_CELLS
        or score_manifest.get("cells") != EXPECTED_CELLS
        or score_manifest.get("scores_sha256") != _sha256(root / "scores.jsonl")
    ):
        raise ValueError("matrix or score manifest is incomplete or changed")

    task_ids = [_task_id(task) for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task manifest contains duplicate ids")
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    expected = _expected_cell_ids(task_ids)

    score_rows = _read_rows(root / "scores.jsonl")
    outcome_rows = _read_rows(root / "outcomes.jsonl")
    score_by_cell: dict[str, dict[str, object]] = {}
    outcome_by_cell: dict[str, dict[str, object]] = {}
    for label, rows, destination in (
        ("score", score_rows, score_by_cell),
        ("outcome", outcome_rows, outcome_by_cell),
    ):
        for row in rows:
            _require_target_safe(row, label=f"{label} row")
            cell_id = row.get("cell_id")
            if not isinstance(cell_id, str) or cell_id in destination:
                raise ValueError(f"{label} rows contain a missing or duplicate cell id")
            destination[cell_id] = row
        if set(destination) != expected:
            raise ValueError(f"{label} rows do not match the frozen dense matrix")

    rewards = np.full((len(tasks), len(ARMS), ATTEMPTS), np.nan)
    costs = np.full_like(rewards, np.nan)
    for cell_id in sorted(expected):
        score = score_by_cell[cell_id]
        outcome = outcome_by_cell[cell_id]
        task_id = _task_id(score)
        arm = score.get("arm")
        attempt = score.get("attempt")
        if (
            task_id != _task_id(outcome)
            or arm != outcome.get("arm")
            or attempt != outcome.get("attempt")
            or not isinstance(arm, str)
            or arm not in arm_index
            or not isinstance(attempt, int)
            or not 0 <= attempt < ATTEMPTS
        ):
            raise ValueError(f"score and outcome identity differ for {cell_id}")
        index = (task_index[task_id], arm_index[arm], attempt)
        rewards[index] = float(cast(float, score["reward"]))
        costs[index] = float(cast(float, outcome["cost_usd"]))
    if not np.isfinite(rewards).all() or not np.isfinite(costs).all():
        raise ValueError("reward or cost tensor is not finite and dense")

    return FitData(
        task_ids=task_ids,
        groups=[str(task["library_group"]) for task in tasks],
        texts=[str(task["instruct_prompt"]) for task in tasks],
        is_hard=np.asarray([bool(task.get("is_hard")) for task in tasks]),
        rewards=rewards,
        costs=costs,
    )


def _structural_row(text: str, *, is_hard: bool) -> list[float]:
    lower = text.casefold()
    lines = text.splitlines()
    words = text.split()
    return [
        math.log1p(len(text)),
        math.log1p(len(words)),
        math.log1p(len(lines)),
        float(text.count("`")),
        float(text.count("\n")),
        float(lower.count("import ")),
        float(lower.count("raise ")),
        float(lower.count("assert ")),
        float(lower.count("example")),
        float(lower.count("test")),
        float(lower.count("exception")),
        float(lower.count("recursive")),
        float("async" in lower),
        float("class " in lower),
        float(is_hard),
    ]


def feature_matrix(data: FitData, *, dim: int) -> sparse.csr_matrix:
    """Build deterministic local prompt and structural features."""
    if dim not in {512, 2_048, 8_192}:
        raise ValueError("hash dimension is outside the frozen search space")
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=dim,
        alternate_sign=True,
        norm="l2",
    )
    text = cast(sparse.csr_matrix, vectorizer.transform(data.texts))
    structural = np.asarray(
        [
            _structural_row(prompt, is_hard=bool(data.is_hard[index]))
            for index, prompt in enumerate(data.texts)
        ],
        dtype=np.float64,
    )
    scale = np.maximum(np.std(structural, axis=0), 1.0)
    structural /= scale
    return sparse.hstack([text, sparse.csr_matrix(structural)], format="csr")


def grouped_folds(groups: list[str], *, splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic family-grouped folds and prove zero overlap."""
    if len(set(groups)) < splits:
        raise ValueError("too few task-family groups for the frozen fold count")
    indices = np.arange(len(groups))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for train, test in GroupKFold(n_splits=splits).split(indices, groups=groups):
        train_groups = {groups[index] for index in train}
        test_groups = {groups[index] for index in test}
        if train_groups & test_groups:
            raise AssertionError("task-family group crossed a fit boundary")
        result.append((train, test))
    return result


def ordinal_ridge_predictions(
    train_features: sparse.csr_matrix,
    test_features: sparse.csr_matrix,
    train_rewards: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Predict monotone rewards from one low-effort and four adjacent-uplift heads."""
    if train_rewards.ndim != 2 or train_rewards.shape[1] != len(ARMS):
        raise ValueError("ordinal training rewards have the wrong shape")
    targets = np.column_stack([train_rewards[:, 0], np.diff(train_rewards, axis=1)])
    predicted_parts: list[np.ndarray] = []
    for column in range(targets.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(train_features, targets[:, column])
        predicted_parts.append(np.asarray(model.predict(test_features), dtype=np.float64))
    parts = np.column_stack(predicted_parts)
    absolute = np.column_stack([parts[:, 0], parts[:, 0, None] + np.cumsum(parts[:, 1:], axis=1)])
    return np.maximum.accumulate(np.clip(absolute, 0.0, 1.0), axis=1)


def evaluate_choices(
    rewards: np.ndarray,
    costs: np.ndarray,
    choices: np.ndarray,
) -> PolicyValue:
    """Evaluate routes and the task-blind mixture with identical arm traffic."""
    if rewards.shape != costs.shape or rewards.ndim != 2:
        raise ValueError("policy evaluation matrices differ or are not two-dimensional")
    if rewards.shape[1] != len(ARMS) or choices.shape != (rewards.shape[0],):
        raise ValueError("policy choices do not match the effort matrix")
    if np.any(choices < 0) or np.any(choices >= len(ARMS)):
        raise ValueError("policy selected an unknown effort")
    rows = np.arange(len(choices))
    counts = np.bincount(choices, minlength=len(ARMS))
    traffic = counts / len(choices)
    return PolicyValue(
        reward=float(np.mean(rewards[rows, choices])),
        cost_usd=float(np.mean(costs[rows, choices])),
        matched_blind_reward=float(np.sum(traffic * rewards.mean(axis=0))),
        matched_blind_cost_usd=float(np.sum(traffic * costs.mean(axis=0))),
        arm_counts={arm: int(counts[index]) for index, arm in enumerate(ARMS)},
    )
