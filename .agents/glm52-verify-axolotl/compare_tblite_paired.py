#!/usr/bin/env python3
"""Compare matched Harbor TBLite arms without diluting subset scores."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from compare_tmax_paired_exclusions import exact_sign_test, quantile


def load_arm(root: Path) -> dict[str, dict[str, Any]]:
    """Load one Harbor trial result per unique task name."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task_name = row.get("task_name")
        if not isinstance(task_name, str) or not task_name:
            raise ValueError(f"missing task_name in {path}")
        if task_name in rows:
            raise ValueError(f"duplicate task {task_name!r} under {root}")
        rows[task_name] = row
    if not rows:
        raise ValueError(f"no Harbor trial results under {root}")
    return rows


def reward(row: dict[str, Any]) -> float:
    """Return verifier reward, counting a failed trial without one as zero."""
    verifier_result = row.get("verifier_result")
    rewards = (
        verifier_result.get("rewards", {})
        if isinstance(verifier_result, dict)
        else {}
    )
    value = rewards.get("reward")
    if not isinstance(value, (int, float)):
        failure = exception_type(row)
        if failure is not None:
            return 0.0
        raise ValueError(
            f"task {row.get('task_name')!r} has neither a numeric verifier "
            "reward nor a recorded trial exception"
        )
    return float(value)


def exception_type(row: dict[str, Any]) -> str | None:
    """Return Harbor's exception class without copying exception payloads."""
    value = row.get("exception_info")
    if not isinstance(value, dict):
        return None
    result = value.get("exception_type")
    return str(result) if result else None


def validate_provenance(
    task_name: str, base: dict[str, Any], adapter: dict[str, Any]
) -> None:
    """Require the paired trials to refer to the exact same task artifact."""
    keys = ("task_checksum", "source")
    for key in keys:
        if base.get(key) != adapter.get(key):
            raise ValueError(
                f"task {task_name!r} provenance mismatch for {key}: "
                f"{base.get(key)!r} != {adapter.get(key)!r}"
            )
    if base.get("task_id") != adapter.get("task_id"):
        raise ValueError(f"task {task_name!r} task_id provenance mismatch")


def arm_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize every completed trial, including failed trials as zero."""
    rewards = [reward(row) for row in rows.values()]
    exceptions = Counter(
        value for row in rows.values() if (value := exception_type(row)) is not None
    )
    return {
        "observed_task_count": len(rows),
        "strict_count": sum(value == 1.0 for value in rewards),
        "strict_rate": sum(value == 1.0 for value in rewards) / len(rewards),
        "graded_mean": sum(rewards) / len(rewards),
        "reward_sum": sum(rewards),
        "partial_credit_count": sum(0.0 < value < 1.0 for value in rewards),
        "zero_reward_count": sum(value == 0.0 for value in rewards),
        "exception_counts": dict(sorted(exceptions.items())),
    }


def compare(
    *,
    base_root: Path,
    adapter_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Return a strict task-paired TBLite comparison."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    base = load_arm(base_root)
    adapter = load_arm(adapter_root)
    if set(base) != set(adapter):
        raise ValueError(
            "task sets differ: "
            f"missing_adapter={sorted(set(base) - set(adapter))}, "
            f"missing_base={sorted(set(adapter) - set(base))}"
        )

    task_names = sorted(base)
    per_task: list[dict[str, Any]] = []
    graded_deltas: list[float] = []
    strict_deltas: list[float] = []
    for task_name in task_names:
        validate_provenance(task_name, base[task_name], adapter[task_name])
        base_reward = reward(base[task_name])
        adapter_reward = reward(adapter[task_name])
        delta = adapter_reward - base_reward
        graded_deltas.append(delta)
        strict_deltas.append(
            float(adapter_reward == 1.0) - float(base_reward == 1.0)
        )
        per_task.append(
            {
                "task_name": task_name,
                "task_checksum": base[task_name].get("task_checksum"),
                "base_reward": base_reward,
                "adapter_reward": adapter_reward,
                "delta": delta,
                "base_exception_type": exception_type(base[task_name]),
                "adapter_exception_type": exception_type(adapter[task_name]),
            }
        )

    rng = random.Random(bootstrap_seed)
    graded_bootstraps: list[float] = []
    strict_bootstraps: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(task_names)) for _ in task_names]
        graded_bootstraps.append(
            sum(graded_deltas[index] for index in indices) / len(indices)
        )
        strict_bootstraps.append(
            sum(strict_deltas[index] for index in indices) / len(indices)
        )

    wins = sum(value > 0 for value in graded_deltas)
    ties = sum(value == 0 for value in graded_deltas)
    losses = sum(value < 0 for value in graded_deltas)
    base_summary = arm_summary(base)
    adapter_summary = arm_summary(adapter)
    return {
        "schema": "xtoken-tblite-task-paired-v1",
        "evidence_scope": "held_out_tblite_subset_screen_not_official_tb2",
        "base_root": str(base_root),
        "adapter_root": str(adapter_root),
        "task_count": len(task_names),
        "base": base_summary,
        "adapter": adapter_summary,
        "paired": {
            "graded_mean_delta": adapter_summary["graded_mean"]
            - base_summary["graded_mean"],
            "strict_rate_delta": adapter_summary["strict_rate"]
            - base_summary["strict_rate"],
            "adapter_better_tasks": wins,
            "tied_tasks": ties,
            "base_better_tasks": losses,
            "exact_sign_test_p": exact_sign_test(wins, losses),
            "graded_task_bootstrap_95ci": [
                quantile(graded_bootstraps, 0.025),
                quantile(graded_bootstraps, 0.975),
            ],
            "strict_task_bootstrap_95ci": [
                quantile(strict_bootstraps, 0.025),
                quantile(strict_bootstraps, 0.975),
            ],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "per_task": per_task,
    }


def main() -> int:
    """Run the paired TBLite comparison CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        base_root=args.base,
        adapter_root=args.adapter,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
