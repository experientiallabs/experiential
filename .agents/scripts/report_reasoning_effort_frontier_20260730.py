"""Report a held-out reasoning-effort frontier from the available DeepSWE ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

SEEDS = (11, 23, 37, 41, 59)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--candidate-model", default="gpt-5-6-sol")
    parser.add_argument("--candidate-effort", default="high")
    parser.add_argument("--baseline-model", default="claude-opus-5")
    parser.add_argument("--baseline-effort", default="high")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _arm_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["task_name"]), str(row["model"]), str(row["reasoning_effort"]))


def _load(path: Path) -> tuple[dict[tuple[str, str, str], tuple[float, float]], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    cells: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("source") != "deep-swe":
            continue
        if not row.get("included_in_score") or row.get("errored"):
            continue
        cells.setdefault(_arm_key(row), []).append(
            (float(row.get("reward") or 0.0), float(row.get("cost_usd") or 0.0))
        )
    means = {
        key: (
            statistics.mean(value[0] for value in observations),
            statistics.mean(value[1] for value in observations),
        )
        for key, observations in cells.items()
    }
    tasks = sorted({key[0] for key in means})
    return means, tasks


def _split(tasks: list[str], seed: int) -> list[str]:
    ordered = sorted(
        tasks,
        key=lambda task: hashlib.sha256(f"{seed}|deepswe|{task}".encode()).hexdigest(),
    )
    return ordered[math.ceil(len(ordered) * 0.7) :]


def _arm(
    means: dict[tuple[str, str, str], tuple[float, float]],
    tasks: list[str],
    model: str,
    effort: str,
) -> tuple[float, float]:
    values = [means[(task, model, effort)] for task in tasks]
    return (
        statistics.mean(value[0] for value in values),
        statistics.mean(value[1] for value in values),
    )


def _bootstrap(values: list[float], *, seed: int, repeats: int = 10_000) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        samples.append(statistics.mean(rng.choice(values) for _ in values))
    samples.sort()
    return samples[int(repeats * 0.025)], samples[int(repeats * 0.975)]


def _main() -> None:
    args = _parser().parse_args()
    means, tasks = _load(args.trials)
    candidate_key = (args.candidate_model, args.candidate_effort)
    baseline_key = (args.baseline_model, args.baseline_effort)
    missing = [
        (task, arm)
        for task in tasks
        for arm in (candidate_key, baseline_key)
        if (task, *arm) not in means
    ]
    if missing:
        raise ValueError(f"missing matched cells, first entries: {missing[:5]}")

    all_candidate = _arm(means, tasks, *candidate_key)
    all_baseline = _arm(means, tasks, *baseline_key)
    quality_deltas = [
        means[(task, *candidate_key)][0] - means[(task, *baseline_key)][0]
        for task in tasks
    ]
    split_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        heldout = _split(tasks, seed)
        candidate_q, candidate_cost = _arm(means, heldout, *candidate_key)
        baseline_q, baseline_cost = _arm(means, heldout, *baseline_key)
        split_rows.append(
            {
                "seed": seed,
                "heldout_task_count": len(heldout),
                "candidate_quality": candidate_q,
                "baseline_quality": baseline_q,
                "quality_ratio": candidate_q / baseline_q if baseline_q else None,
                "candidate_cost_usd_per_task": candidate_cost,
                "baseline_cost_usd_per_task": baseline_cost,
                "cost_savings": 1.0 - candidate_cost / baseline_cost
                if baseline_cost
                else None,
            }
        )

    report = {
        "provenance": {
            "ledger": str(args.trials),
            "ledger_scope": "available historical/shared DeepSWE ledger; not a fresh run",
            "task_count": len(tasks),
            "split_rule": "sha256(seed|deepswe|task), 70 percent fit and 30 percent held out",
            "seeds": list(SEEDS),
        },
        "candidate": {
            "model": args.candidate_model,
            "reasoning_effort": args.candidate_effort,
            "quality": all_candidate[0],
            "cost_usd_per_task": all_candidate[1],
        },
        "baseline": {
            "model": args.baseline_model,
            "reasoning_effort": args.baseline_effort,
            "quality": all_baseline[0],
            "cost_usd_per_task": all_baseline[1],
        },
        "overall": {
            "quality_ratio": all_candidate[0] / all_baseline[0],
            "cost_savings": 1.0 - all_candidate[1] / all_baseline[1],
            "quality_delta_bootstrap_95ci": list(
                _bootstrap(quality_deltas, seed=20260730)
            ),
        },
        "heldout_splits": split_rows,
        "summary": {
            "mean_split_quality_ratio": statistics.mean(
                row["quality_ratio"] for row in split_rows
            ),
            "mean_split_cost_savings": statistics.mean(
                row["cost_savings"] for row in split_rows
            ),
            "worst_split_quality_ratio": min(row["quality_ratio"] for row in split_rows),
            "worst_split_cost_savings": min(row["cost_savings"] for row in split_rows),
        },
        "promotion_gate": {
            "quality_retention_minimum": 0.95,
            "cost_savings_minimum": 0.30,
            "cost_savings_target_range": [0.30, 0.40],
            "overall_quality_pass": all_candidate[0] / all_baseline[0] >= 0.95,
            "overall_cost_pass": 1.0 - all_candidate[1] / all_baseline[1] >= 0.30,
            "pass_on_available_ledger": (
                all_candidate[0] / all_baseline[0] >= 0.95
                and 1.0 - all_candidate[1] / all_baseline[1] >= 0.30
            ),
            "fresh_execution_required": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    _main()
