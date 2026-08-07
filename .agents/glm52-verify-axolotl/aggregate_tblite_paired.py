#!/usr/bin/env python3
"""Pool disjoint task-paired TBLite screens with duplicate guards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from compare_tmax_paired_exclusions import exact_sign_test, quantile


def load_report(path: Path) -> dict[str, Any]:
    """Load and minimally validate one task-paired report."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "xtoken-tblite-task-paired-v1":
        raise ValueError(f"unsupported report schema in {path}")
    rows = report.get("per_task")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"report has no per_task rows: {path}")
    if report.get("task_count") != len(rows):
        raise ValueError(f"task_count does not match per_task rows: {path}")
    return report


def summarize_arm(rewards: list[float]) -> dict[str, Any]:
    """Summarize an arm over the exact observed task denominator."""
    return {
        "observed_task_count": len(rewards),
        "strict_count": sum(value == 1.0 for value in rewards),
        "strict_rate": sum(value == 1.0 for value in rewards) / len(rewards),
        "graded_mean": sum(rewards) / len(rewards),
        "reward_sum": sum(rewards),
        "partial_credit_count": sum(0.0 < value < 1.0 for value in rewards),
        "zero_reward_count": sum(value == 0.0 for value in rewards),
    }


def aggregate(
    *,
    input_paths: list[Path],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Pool exact disjoint TBLite tasks and recompute paired uncertainty."""
    if len(input_paths) < 2:
        raise ValueError("at least two paired reports are required")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    reports = [load_report(path) for path in input_paths]
    seen_names: dict[str, Path] = {}
    seen_checksums: dict[str, Path] = {}
    per_task: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    for path, report in zip(input_paths, reports, strict=True):
        run_summaries.append(
            {
                "report": str(path),
                "task_count": report["task_count"],
                "base_strict_rate": report["base"]["strict_rate"],
                "adapter_strict_rate": report["adapter"]["strict_rate"],
                "base_graded_mean": report["base"]["graded_mean"],
                "adapter_graded_mean": report["adapter"]["graded_mean"],
                "graded_mean_delta": report["paired"]["graded_mean_delta"],
            }
        )
        for source_row in report["per_task"]:
            row = dict(source_row)
            name = row.get("task_name")
            checksum = row.get("task_checksum")
            if not isinstance(name, str) or not name:
                raise ValueError(f"missing task_name in {path}")
            if not isinstance(checksum, str) or not checksum:
                raise ValueError(f"missing task_checksum for {name!r} in {path}")
            if name in seen_names:
                raise ValueError(
                    f"duplicate task_name {name!r} in {seen_names[name]} and {path}"
                )
            if checksum in seen_checksums:
                raise ValueError(
                    f"duplicate task_checksum {checksum!r} in "
                    f"{seen_checksums[checksum]} and {path}"
                )
            seen_names[name] = path
            seen_checksums[checksum] = path
            row["source_report"] = str(path)
            per_task.append(row)

    per_task.sort(key=lambda row: row["task_name"])
    base_rewards = [float(row["base_reward"]) for row in per_task]
    adapter_rewards = [float(row["adapter_reward"]) for row in per_task]
    graded_deltas = [
        adapter_reward - base_reward
        for base_reward, adapter_reward in zip(
            base_rewards, adapter_rewards, strict=True
        )
    ]
    strict_deltas = [
        float(adapter_reward == 1.0) - float(base_reward == 1.0)
        for base_reward, adapter_reward in zip(
            base_rewards, adapter_rewards, strict=True
        )
    ]

    rng = random.Random(bootstrap_seed)
    graded_bootstraps: list[float] = []
    strict_bootstraps: list[float] = []
    for _ in range(bootstrap_samples):
        indices = [rng.randrange(len(per_task)) for _ in per_task]
        graded_bootstraps.append(
            sum(graded_deltas[index] for index in indices) / len(indices)
        )
        strict_bootstraps.append(
            sum(strict_deltas[index] for index in indices) / len(indices)
        )

    wins = sum(value > 0 for value in graded_deltas)
    ties = sum(value == 0 for value in graded_deltas)
    losses = sum(value < 0 for value in graded_deltas)
    base = summarize_arm(base_rewards)
    adapter = summarize_arm(adapter_rewards)
    return {
        "schema": "xtoken-tblite-disjoint-paired-aggregate-v1",
        "evidence_scope": "held_out_tblite_disjoint_subset_screen_not_official_tb2",
        "input_reports": [str(path) for path in input_paths],
        "run_count": len(reports),
        "task_count": len(per_task),
        "runs": run_summaries,
        "base": base,
        "adapter": adapter,
        "paired": {
            "graded_mean_delta": adapter["graded_mean"] - base["graded_mean"],
            "strict_rate_delta": adapter["strict_rate"] - base["strict_rate"],
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
    """Run the disjoint paired aggregation CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(
        input_paths=args.input,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
