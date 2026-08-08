#!/usr/bin/env python3
"""Aggregate repeated matched TBLite reports with task-clustered uncertainty."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from aggregate_tblite_paired import load_report, summarize_arm
from compare_tmax_paired_exclusions import exact_sign_test, quantile


def _task_map(path: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return one validated row per unique task name in a paired report."""
    rows: dict[str, dict[str, Any]] = {}
    checksums: set[str] = set()
    for source_row in report["per_task"]:
        row = dict(source_row)
        name = row.get("task_name")
        checksum = row.get("task_checksum")
        if not isinstance(name, str) or not name:
            raise ValueError(f"missing task_name in {path}")
        if not isinstance(checksum, str) or not checksum:
            raise ValueError(f"missing task_checksum for {name!r} in {path}")
        if name in rows:
            raise ValueError(f"duplicate task_name {name!r} in {path}")
        if checksum in checksums:
            raise ValueError(f"duplicate task_checksum {checksum!r} in {path}")
        for key in ("base_reward", "adapter_reward"):
            if not isinstance(row.get(key), (int, float)):
                raise ValueError(f"non-numeric {key} for {name!r} in {path}")
        rows[name] = row
        checksums.add(checksum)
    return rows


def aggregate(
    *,
    input_paths: list[Path],
    expected_task_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Combine repeated runs while treating task, not run-task, as independent."""
    if len(input_paths) < 3:
        raise ValueError("at least three paired reports are required")
    if expected_task_count <= 0:
        raise ValueError("expected_task_count must be positive")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    reports = [load_report(path) for path in input_paths]
    run_maps = [
        _task_map(path, report)
        for path, report in zip(input_paths, reports, strict=True)
    ]
    reference_names = set(run_maps[0])
    if len(reference_names) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} tasks, got {len(reference_names)} "
            f"in {input_paths[0]}"
        )
    reference_checksums = {
        name: run_maps[0][name]["task_checksum"] for name in reference_names
    }
    for path, rows in zip(input_paths[1:], run_maps[1:], strict=True):
        if set(rows) != reference_names:
            raise ValueError(
                f"task set mismatch in {path}: "
                f"missing={sorted(reference_names - set(rows))}, "
                f"extra={sorted(set(rows) - reference_names)}"
            )
        for name in reference_names:
            if rows[name]["task_checksum"] != reference_checksums[name]:
                raise ValueError(
                    f"task checksum mismatch for {name!r} in {path}: "
                    f"{rows[name]['task_checksum']!r} != "
                    f"{reference_checksums[name]!r}"
                )

    run_summaries: list[dict[str, Any]] = []
    pooled_base_rewards: list[float] = []
    pooled_adapter_rewards: list[float] = []
    for path, rows in zip(input_paths, run_maps, strict=True):
        base_rewards = [float(rows[name]["base_reward"]) for name in sorted(rows)]
        adapter_rewards = [
            float(rows[name]["adapter_reward"]) for name in sorted(rows)
        ]
        pooled_base_rewards.extend(base_rewards)
        pooled_adapter_rewards.extend(adapter_rewards)
        base = summarize_arm(base_rewards)
        adapter = summarize_arm(adapter_rewards)
        run_summaries.append(
            {
                "report": str(path),
                "task_count": len(rows),
                "base": base,
                "adapter": adapter,
                "graded_mean_delta": adapter["graded_mean"]
                - base["graded_mean"],
                "strict_rate_delta": adapter["strict_rate"]
                - base["strict_rate"],
            }
        )

    per_task: list[dict[str, Any]] = []
    task_graded_deltas: list[float] = []
    task_strict_deltas: list[float] = []
    for name in sorted(reference_names):
        base_rewards = [float(rows[name]["base_reward"]) for rows in run_maps]
        adapter_rewards = [float(rows[name]["adapter_reward"]) for rows in run_maps]
        base_mean = sum(base_rewards) / len(base_rewards)
        adapter_mean = sum(adapter_rewards) / len(adapter_rewards)
        base_strict_rate = sum(value == 1.0 for value in base_rewards) / len(
            base_rewards
        )
        adapter_strict_rate = sum(value == 1.0 for value in adapter_rewards) / len(
            adapter_rewards
        )
        graded_delta = adapter_mean - base_mean
        strict_delta = adapter_strict_rate - base_strict_rate
        task_graded_deltas.append(graded_delta)
        task_strict_deltas.append(strict_delta)
        per_task.append(
            {
                "task_name": name,
                "task_checksum": reference_checksums[name],
                "base_reward_mean": base_mean,
                "adapter_reward_mean": adapter_mean,
                "graded_mean_delta": graded_delta,
                "base_strict_rate": base_strict_rate,
                "adapter_strict_rate": adapter_strict_rate,
                "strict_rate_delta": strict_delta,
                "base_rewards_by_run": base_rewards,
                "adapter_rewards_by_run": adapter_rewards,
            }
        )

    rng = random.Random(bootstrap_seed)
    graded_bootstraps: list[float] = []
    strict_bootstraps: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(per_task)) for _ in per_task]
        graded_bootstraps.append(
            sum(task_graded_deltas[index] for index in indices) / len(indices)
        )
        strict_bootstraps.append(
            sum(task_strict_deltas[index] for index in indices) / len(indices)
        )

    wins = sum(delta > 0 for delta in task_graded_deltas)
    ties = sum(delta == 0 for delta in task_graded_deltas)
    losses = sum(delta < 0 for delta in task_graded_deltas)
    base = summarize_arm(pooled_base_rewards)
    adapter = summarize_arm(pooled_adapter_rewards)
    graded_ci = [
        quantile(graded_bootstraps, 0.025),
        quantile(graded_bootstraps, 0.975),
    ]
    strict_ci = [
        quantile(strict_bootstraps, 0.025),
        quantile(strict_bootstraps, 0.975),
    ]
    sign_p = exact_sign_test(wins, losses)
    graded_delta = adapter["graded_mean"] - base["graded_mean"]
    strict_delta = adapter["strict_rate"] - base["strict_rate"]
    gate_checks = {
        "three_complete_reports": len(reports) >= 3,
        "expected_task_count_each": all(
            len(rows) == expected_task_count for rows in run_maps
        ),
        "positive_pooled_strict_delta": strict_delta > 0,
        "positive_pooled_graded_delta": graded_delta > 0,
        "task_mean_sign_test_p_below_0_05": sign_p < 0.05,
        "graded_task_cluster_bootstrap_lower_above_zero": graded_ci[0] > 0,
    }
    return {
        "schema": "xtoken-tblite-repeated-task-clustered-v1",
        "evidence_scope": "repeated_held_out_tblite_screen_not_official_tb2",
        "input_reports": [str(path) for path in input_paths],
        "run_count": len(reports),
        "task_count_per_run": expected_task_count,
        "pooled_observation_count_per_arm": len(pooled_base_rewards),
        "runs": run_summaries,
        "base": base,
        "adapter": adapter,
        "paired": {
            "graded_mean_delta": graded_delta,
            "strict_rate_delta": strict_delta,
            "adapter_better_task_means": wins,
            "tied_task_means": ties,
            "base_better_task_means": losses,
            "task_mean_exact_sign_test_p": sign_p,
            "graded_task_cluster_bootstrap_95ci": graded_ci,
            "strict_task_cluster_bootstrap_95ci": strict_ci,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "promotion_gate": {
            "checks": gate_checks,
            "credible_tbilite_win": all(gate_checks.values()),
            "official_tb2_authorized": False,
        },
        "per_task": per_task,
    }


def main() -> int:
    """Run repeated matched TBLite aggregation from paired report files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--expected-task-count", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        input_paths=args.input,
        expected_task_count=args.expected_task_count,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
