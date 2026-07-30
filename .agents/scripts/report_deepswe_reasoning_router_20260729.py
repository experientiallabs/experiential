"""Evaluate trace-guided reasoning-effort routing on DeepSWE v1.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

SEEDS = (11, 23, 37, 41, 59)
MODEL_PRICES = {"gpt-5.5": (5.0, 30.0), "gpt-5.4-mini": (0.75, 3.0)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", action="append", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trace-model", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def _metadata(root: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
            continue
        config = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        result[task_dir.name] = {
            "language": str(config.get("metadata", {}).get("language") or "unknown"),
            "instruction": instruction,
        }
    return result


def _load(paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["logical_model"])
        if "@" not in model:
            continue
        for record in payload.get("trial_records", []):
            task = str(record["task_id"]).rsplit("/", 1)[-1]
            if task in matrix and model in matrix[task]:
                raise ValueError(f"duplicate outcome for {task}/{model}")
            token_cost = record.get("token_cost") or {}
            reported = token_cost.get("cost_usd")
            base_model = model.split("@", 1)[0]
            input_price, output_price = MODEL_PRICES.get(base_model, (0.0, 0.0))
            cost = float(reported or 0.0) or (
                int(token_cost.get("input_tokens") or 0) * input_price
                + int(token_cost.get("output_tokens") or 0) * output_price
            ) / 1_000_000
            matrix.setdefault(task, {})[model] = {
                "reward": float(record.get("reward") or 0.0),
                "cost_usd": cost,
                "infrastructure_failure": float(record.get("reward") is None),
            }
    variants = sorted({variant for rows in matrix.values() for variant in rows})
    if len(variants) < 2:
        raise ValueError("need at least two reasoning-effort variants")
    complete_tasks = sorted(
        task for task, rows in matrix.items() if all(variant in rows for variant in variants)
    )
    if len(complete_tasks) < 2:
        raise ValueError("reasoning matrix has fewer than two common complete tasks")
    return {task: matrix[task] for task in complete_tasks}


def _probabilities(
    matrix: dict[str, dict[str, dict[str, float]]],
    metadata: dict[str, dict[str, str]],
    model_path: Path,
) -> dict[str, float]:
    model = joblib.load(model_path)
    task_ids = sorted(matrix)
    rows = []
    for task in task_ids:
        instruction = metadata[task]["instruction"]
        rows.append(
            {
                "prompt": instruction,
                "language": metadata[task]["language"],
                "trajectory_steps": 0,
                "tool_calls": 0,
                "prompt_chars": len(instruction),
            }
        )
    return dict(zip(task_ids, model.predict_proba(pd.DataFrame(rows))[:, 1], strict=True))


def _split(task_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    ranked = sorted(
        task_ids,
        key=lambda task: hashlib.sha256(f"{seed}|{task}".encode()).digest(),
    )
    cut = min(len(ranked) - 1, max(1, round(len(ranked) * 0.7)))
    return sorted(ranked[:cut]), sorted(ranked[cut:])


def _evaluate(
    matrix: dict[str, dict[str, dict[str, float]]],
    tasks: list[str],
    choices: dict[str, str],
) -> dict[str, Any]:
    rewards = [matrix[task][choices[task]]["reward"] for task in tasks]
    costs = [matrix[task][choices[task]]["cost_usd"] for task in tasks]
    mix: dict[str, int] = {}
    for variant in choices.values():
        mix[variant] = mix.get(variant, 0) + 1
    return {
        "accuracy": statistics.mean(rewards),
        "cost_usd": statistics.mean(costs),
        "model_mix": mix,
    }


def _best_single(
    matrix: dict[str, dict[str, dict[str, float]]],
    tasks: list[str],
    variants: list[str],
) -> str:
    return max(
        variants,
        key=lambda variant: (
            statistics.mean(matrix[task][variant]["reward"] for task in tasks),
            -statistics.mean(matrix[task][variant]["cost_usd"] for task in tasks),
        ),
    )


def _choose(
    probability: float,
    low: str,
    medium: str,
    high: str,
    easy_threshold: float,
    hard_threshold: float,
) -> str:
    if probability >= easy_threshold:
        return low
    if probability >= hard_threshold:
        return medium
    return high


def main() -> None:
    args = _parser().parse_args()
    matrix = _load(args.outcome)
    metadata = _metadata(args.dataset_root.resolve())
    probabilities = _probabilities(matrix, metadata, args.trace_model.resolve())
    variants = sorted(matrix[next(iter(matrix))])
    by_effort = {variant.rsplit("@", 1)[1]: variant for variant in variants}
    required = {"low", "medium", "high"}
    if not required.issubset(by_effort):
        raise ValueError(f"need low, medium, and high variants; got {sorted(by_effort)}")
    task_ids = sorted(matrix)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        fit, report = _split(task_ids, seed)
        baseline = _best_single(matrix, fit, variants)
        baseline_fit = _evaluate(matrix, fit, {task: baseline for task in fit})
        candidates: list[dict[str, Any]] = []
        for easy_index in range(21):
            easy_threshold = easy_index / 20
            for hard_index in range(easy_index + 1):
                hard_threshold = hard_index / 20
                choices = {
                    task: _choose(
                        probabilities[task],
                        by_effort["low"],
                        by_effort["medium"],
                        by_effort["high"],
                        easy_threshold,
                        hard_threshold,
                    )
                    for task in fit
                }
                candidate_fit = _evaluate(matrix, fit, choices)
                if candidate_fit["accuracy"] < baseline_fit["accuracy"] * 0.95:
                    continue
                savings = 1.0 - candidate_fit["cost_usd"] / baseline_fit["cost_usd"]
                candidates.append(
                    {
                        "easy_threshold": easy_threshold,
                        "hard_threshold": hard_threshold,
                        "fit": candidate_fit,
                        "fit_savings": savings,
                    }
                )
        candidate = max(candidates, key=lambda item: (item["fit_savings"], item["fit"]["accuracy"]))
        baseline_report = _evaluate(matrix, report, {task: baseline for task in report})
        routed_choices = {
            task: _choose(
                probabilities[task],
                by_effort["low"],
                by_effort["medium"],
                by_effort["high"],
                candidate["easy_threshold"],
                candidate["hard_threshold"],
            )
            for task in report
        }
        routed = _evaluate(matrix, report, routed_choices)
        rows.append(
            {
                "seed": seed,
                "fit_tasks": fit,
                "report_tasks": report,
                "baseline_variant": baseline,
                "baseline": baseline_report,
                "router": candidate,
                "routed": routed,
                "savings": 1.0 - routed["cost_usd"] / baseline_report["cost_usd"],
                "quality_ratio": routed["accuracy"] / baseline_report["accuracy"]
                if baseline_report["accuracy"]
                else 1.0,
            }
        )
    mean_quality = statistics.mean(row["quality_ratio"] for row in rows)
    mean_savings = statistics.mean(row["savings"] for row in rows)
    report = {
        "benchmark": "deepswe-1.1",
        "task_count": len(task_ids),
        "variants": variants,
        "infrastructure_failures": {
            variant: sum(
                int(matrix[task][variant]["infrastructure_failure"]) for task in task_ids
            )
            for variant in variants
        },
        "trace_model": str(args.trace_model.resolve()),
        "trace_probability": probabilities,
        "splits": rows,
        "mean_baseline_accuracy": statistics.mean(row["baseline"]["accuracy"] for row in rows),
        "mean_router_accuracy": statistics.mean(row["routed"]["accuracy"] for row in rows),
        "mean_router_savings": mean_savings,
        "mean_quality_ratio": mean_quality,
        "gate": {
            "quality_floor": 0.95,
            "savings_target": 0.40,
            "quality_pass": mean_quality >= 0.95,
            "savings_pass": mean_savings >= 0.40,
            "pass": mean_quality >= 0.95 and mean_savings >= 0.40,
        },
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    output = args.artifact_root / "coding-router-deepswe-20260729-reasoning-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
