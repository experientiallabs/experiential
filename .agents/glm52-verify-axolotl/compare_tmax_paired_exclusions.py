#!/usr/bin/env python3
"""Compare verifier-scored task intersections with explicit exclusions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


def exact_sign_test(wins: int, losses: int) -> float:
    """Return the exact two-sided binomial sign-test p-value."""
    n = wins + losses
    if n == 0:
        return 1.0
    observed = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(observed + 1)) / (2**n)
    return min(1.0, 2 * tail)


def quantile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated quantile."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_arm(root: Path) -> dict[str, dict[str, Any]]:
    """Load every attempted episode result, keyed by unique task ID."""
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("episode_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(row.get("task_id") or "")
        if not task_id:
            raise ValueError(f"missing task_id in {path}")
        if task_id in rows:
            raise ValueError(f"duplicate task {task_id!r} in {root}")
        rows[task_id] = row
    if not rows:
        raise ValueError(f"no episode results under {root}")
    return rows


def is_scored(row: dict[str, Any]) -> bool:
    """Return whether a row has an official environment verifier score."""
    return row.get("status") == "scored" and isinstance(
        row.get("reward"), (int, float)
    )


def error_type(row: dict[str, Any]) -> str | None:
    """Return a non-sensitive error class, if present."""
    value = row.get("agent_error") or row.get("error")
    if not isinstance(value, str) or not value:
        return None
    return value.split(":", 1)[0]


def attempt_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize attempts without treating exclusions as zero reward."""
    scored = [row for row in rows.values() if is_scored(row)]
    rewards = [float(row["reward"]) for row in scored]
    return {
        "attempted_count": len(rows),
        "scored_count": len(scored),
        "excluded_count": len(rows) - len(scored),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows.values())),
        "scored_mean_reward": sum(rewards) / len(rewards) if rewards else None,
    }


def matched_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize the common verifier-scored subset."""
    rewards = [float(row["reward"]) for row in rows.values()]
    output_tokens = [
        int(row.get("context", {}).get("n_output_tokens") or 0)
        for row in rows.values()
    ]
    tool_calls = [
        int(row.get("trajectory", {}).get("assistant_tool_calls") or 0)
        for row in rows.values()
    ]
    return {
        "task_count": len(rows),
        "mean_reward": sum(rewards) / len(rewards),
        "reward_sum": sum(rewards),
        "zero_reward_tasks": sum(value == 0 for value in rewards),
        "perfect_reward_tasks": sum(value == 1 for value in rewards),
        "output_tokens_total": sum(output_tokens),
        "assistant_tool_calls_total": sum(tool_calls),
        "unparsed_tool_call_tasks": sum(
            bool(row.get("trajectory", {}).get("has_unparsed_tool_call"))
            for row in rows.values()
        ),
    }


def compare(
    *,
    base_root: Path,
    adapter_root: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
    task_ids_path: Path | None = None,
) -> dict[str, Any]:
    """Compare the exact intersection of verifier-scored tasks."""
    base_all = load_arm(base_root)
    adapter_all = load_arm(adapter_root)
    if task_ids_path is not None:
        requested = [
            line.strip()
            for line in task_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not requested:
            raise ValueError("task manifest is empty")
        if len(requested) != len(set(requested)):
            raise ValueError("task manifest contains duplicate task IDs")
        requested_set = set(requested)
        missing_base = requested_set - set(base_all)
        missing_adapter = requested_set - set(adapter_all)
        extra_adapter = set(adapter_all) - requested_set
        if missing_base or missing_adapter or extra_adapter:
            raise ValueError(
                "task manifest does not match available attempts: "
                f"missing_base={sorted(missing_base)}, "
                f"missing_adapter={sorted(missing_adapter)}, "
                f"extra_adapter={sorted(extra_adapter)}"
            )
        base_all = {task_id: base_all[task_id] for task_id in requested}
        adapter_all = {task_id: adapter_all[task_id] for task_id in requested}
    elif set(base_all) != set(adapter_all):
        raise ValueError(
            "attempted task sets differ: "
            f"missing_adapter={sorted(set(base_all) - set(adapter_all))}, "
            f"missing_base={sorted(set(adapter_all) - set(base_all))}"
        )
    task_ids = sorted(
        task_id
        for task_id in base_all
        if is_scored(base_all[task_id]) and is_scored(adapter_all[task_id])
    )
    if not task_ids:
        raise ValueError("no common verifier-scored tasks")
    base = {task_id: base_all[task_id] for task_id in task_ids}
    adapter = {task_id: adapter_all[task_id] for task_id in task_ids}
    exclusions = []
    for task_id in sorted(set(base_all) - set(task_ids)):
        base_row = base_all[task_id]
        adapter_row = adapter_all[task_id]
        exclusions.append(
            {
                "task_id": task_id,
                "base_status": base_row.get("status"),
                "base_error_type": error_type(base_row),
                "adapter_status": adapter_row.get("status"),
                "adapter_error_type": error_type(adapter_row),
                "reason": "excluded_unless_both_arms_have_environment_verifier_scores",
            }
        )

    per_task = []
    deltas = []
    for task_id in task_ids:
        base_reward = float(base[task_id]["reward"])
        adapter_reward = float(adapter[task_id]["reward"])
        delta = adapter_reward - base_reward
        deltas.append(delta)
        per_task.append(
            {
                "task_id": task_id,
                "base_reward": base_reward,
                "adapter_reward": adapter_reward,
                "delta": delta,
            }
        )
    rng = random.Random(bootstrap_seed)
    bootstraps = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    ]
    wins = sum(delta > 0 for delta in deltas)
    ties = sum(delta == 0 for delta in deltas)
    losses = sum(delta < 0 for delta in deltas)
    return {
        "schema": "xtoken-tmax-paired-with-explicit-exclusions-v1",
        "evidence_scope": "training_overlap_diagnostic_not_held_out",
        "attempted_task_count": len(base_all),
        "matched_scored_task_count": len(task_ids),
        "base_attempts": attempt_summary(base_all),
        "adapter_attempts": attempt_summary(adapter_all),
        "explicit_pair_exclusions": exclusions,
        "base_matched": matched_summary(base),
        "adapter_matched": matched_summary(adapter),
        "paired": {
            "mean_reward_delta": sum(deltas) / len(deltas),
            "adapter_better_tasks": wins,
            "tied_tasks": ties,
            "base_better_tasks": losses,
            "exact_sign_test_p": exact_sign_test(wins, losses),
            "task_bootstrap_95ci": [
                quantile(bootstraps, 0.025),
                quantile(bootstraps, 0.975),
            ],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "per_task": per_task,
    }


def main() -> int:
    """Run the paired comparison CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    parser.add_argument("--task-ids", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        base_root=args.base,
        adapter_root=args.adapter,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        task_ids_path=args.task_ids,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
