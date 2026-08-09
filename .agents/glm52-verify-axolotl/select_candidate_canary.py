#!/usr/bin/env python3
"""Apply the predeclared two-training-seed TBLite canary gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(inputs: dict[str, Path]) -> dict[str, object]:
    reports = {seed: json.loads(path.read_text()) for seed, path in inputs.items()}
    if set(reports) != {"20260809", "20260810"}:
        raise ValueError("exactly training seeds 20260809 and 20260810 are required")
    for seed, report in reports.items():
        if report.get("schema") != "xtoken-tblite-task-paired-v1":
            raise ValueError(f"seed {seed}: wrong paired-report schema")
        if int(report.get("task_count", 0)) != 10:
            raise ValueError(f"seed {seed}: expected exactly 10 matched tasks")

    checks = {
        seed: {
            "nonnegative_strict_delta": report["paired"]["strict_rate_delta"] >= 0,
            "positive_graded_delta": report["paired"]["graded_mean_delta"] > 0,
            "wins_not_below_losses": report["paired"]["adapter_better_tasks"]
            >= report["paired"]["base_better_tasks"],
        }
        for seed, report in reports.items()
    }
    return {
        "schema": "candidate-step100-two-training-seed-canary-gate-v1",
        "evidence_scope": "held_out_tblite_canary_only_not_official_tb2",
        "selection_rule": (
            "both training seeds must pass every directional check; primary is "
            "predeclared seed 20260809, never best-of-two"
        ),
        "reports": reports,
        "checks": checks,
        "credible_direction": all(all(values.values()) for values in checks.values()),
        "predeclared_primary_if_credible": "seed20260809-step100",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    inputs = {}
    for raw in args.input:
        seed, separator, path = raw.partition("=")
        if not separator or seed in inputs:
            raise ValueError(f"invalid or duplicate input: {raw}")
        inputs[seed] = Path(path)
    payload = evaluate(inputs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
