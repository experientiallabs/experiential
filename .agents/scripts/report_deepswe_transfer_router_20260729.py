"""Evaluate a trace-trained difficulty router on DeepSWE v1.1 outcomes."""

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

MODELS = ("gpt-5.5", "gpt-5.4-mini", "claude-opus-4-8", "claude-sonnet-4-6")
PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 3.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
SEEDS = (11, 23, 37, 41, 59)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", action="append", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--trace-model", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def _task_metadata(dataset_root: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for task_dir in sorted(dataset_root.iterdir()):
        if not task_dir.is_dir() or not (task_dir / "task.toml").is_file():
            continue
        task_id = task_dir.name
        task = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        language = str(task.get("metadata", {}).get("language") or "unknown")
        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text(encoding="utf-8").strip()
        metadata[task_id] = {"language": language, "instruction": instruction}
    return metadata


def _cost(model: str, record: dict[str, Any]) -> float:
    token_cost = record.get("token_cost") or {}
    reported = token_cost.get("cost_usd")
    if reported is not None and float(reported) > 0:
        return float(reported)
    input_price, output_price = PRICES[model]
    return (
        int(token_cost.get("input_tokens") or 0) * input_price
        + int(token_cost.get("output_tokens") or 0) * output_price
    ) / 1_000_000


def _load_outcomes(paths: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["logical_model"])
        if model not in MODELS:
            raise ValueError(f"unsupported model {model!r} in {path}")
        for record in payload.get("trial_records", []):
            task_id = str(record["task_id"]).rsplit("/", 1)[-1]
            if task_id in matrix and model in matrix[task_id]:
                raise ValueError(f"duplicate outcome for {task_id}/{model}")
            reward = float(record.get("reward") or 0.0)
            matrix.setdefault(task_id, {})[model] = {
                "reward": reward,
                "cost_usd": _cost(model, record),
            }
    missing = {
        f"{task}/{model}"
        for task, rows in matrix.items()
        for model in MODELS
        if model not in rows
    }
    if missing:
        raise ValueError(f"outcome matrix is incomplete: {sorted(missing)}")
    return matrix


def _probabilities(
    matrix: dict[str, dict[str, dict[str, float]]],
    metadata: dict[str, dict[str, str]],
    model_path: Path,
) -> dict[str, float]:
    model = joblib.load(model_path)
    rows = []
    task_ids = sorted(matrix)
    for task_id in task_ids:
        task = metadata[task_id]
        prompt = task["instruction"]
        rows.append(
            {
                "prompt": prompt,
                "language": task["language"],
                "trajectory_steps": 0,
                "tool_calls": 0,
                "prompt_chars": len(prompt),
            }
        )
    probabilities = model.predict_proba(pd.DataFrame(rows))[:, 1]
    return dict(zip(task_ids, (float(value) for value in probabilities), strict=True))


def _split(task_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    ranked = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(f"{seed}|{task_id}".encode()).digest(),
    )
    cut = min(len(ranked) - 1, max(1, round(len(ranked) * 0.7)))
    return sorted(ranked[:cut]), sorted(ranked[cut:])


def _evaluate(
    matrix: dict[str, dict[str, dict[str, float]]],
    task_ids: list[str],
    choices: dict[str, str],
) -> dict[str, Any]:
    if not task_ids:
        return {"accuracy": 0.0, "cost_usd": 0.0, "model_mix": {}}
    rewards = [matrix[task][choices[task]]["reward"] for task in task_ids]
    costs = [matrix[task][choices[task]]["cost_usd"] for task in task_ids]
    mix: dict[str, int] = {}
    for model in choices.values():
        mix[model] = mix.get(model, 0) + 1
    return {
        "accuracy": statistics.mean(rewards),
        "cost_usd": statistics.mean(costs),
        "model_mix": mix,
    }


def _best_single(matrix: dict[str, dict[str, dict[str, float]]], task_ids: list[str]) -> str:
    return max(
        MODELS,
        key=lambda model: (
            statistics.mean(matrix[task][model]["reward"] for task in task_ids),
            -statistics.mean(matrix[task][model]["cost_usd"] for task in task_ids),
        ),
    )


def _fit_router(
    matrix: dict[str, dict[str, dict[str, float]]],
    probabilities: dict[str, float],
    fit_ids: list[str],
) -> tuple[dict[str, Any], str]:
    baseline = _best_single(matrix, fit_ids)
    baseline_choices = {task: baseline for task in fit_ids}
    baseline_eval = _evaluate(matrix, fit_ids, baseline_choices)
    cheap = "gpt-5.4-mini"
    candidates: list[dict[str, Any]] = []
    for fallback in MODELS:
        for threshold in [index / 20 for index in range(21)]:
            choices = {
                task: (cheap if probabilities[task] >= threshold else fallback) for task in fit_ids
            }
            report = _evaluate(matrix, fit_ids, choices)
            quality_ok = report["accuracy"] >= baseline_eval["accuracy"] * 0.95
            savings = (
                1.0 - report["cost_usd"] / baseline_eval["cost_usd"]
                if baseline_eval["cost_usd"]
                else 0.0
            )
            if quality_ok:
                candidates.append(
                    {
                        "fallback_model": fallback,
                        "cheap_model": cheap,
                        "threshold": threshold,
                        "fit_accuracy": report["accuracy"],
                        "fit_cost_usd": report["cost_usd"],
                        "fit_savings": savings,
                    }
                )
    if not candidates:
        return {"kind": "static", "model": baseline}, baseline
    return max(candidates, key=lambda row: (row["fit_savings"], row["fit_accuracy"])), baseline


def main() -> None:
    args = _parser().parse_args()
    matrix = _load_outcomes(args.outcome)
    metadata = _task_metadata(args.dataset_root.resolve())
    missing_metadata = sorted(set(matrix) - set(metadata))
    if missing_metadata:
        raise ValueError(f"DeepSWE metadata missing for tasks: {missing_metadata}")
    probabilities = _probabilities(matrix, metadata, args.trace_model.resolve())
    task_ids = sorted(matrix)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        fit_ids, report_ids = _split(task_ids, seed)
        router, baseline = _fit_router(matrix, probabilities, fit_ids)
        baseline_choices = {task: baseline for task in report_ids}
        if router.get("kind") == "static":
            routed_choices = dict(baseline_choices)
        else:
            routed_choices = {
                task: (
                    router["cheap_model"]
                    if probabilities[task] >= router["threshold"]
                    else router["fallback_model"]
                )
                for task in report_ids
            }
        baseline_report = _evaluate(matrix, report_ids, baseline_choices)
        routed_report = _evaluate(matrix, report_ids, routed_choices)
        rows.append(
            {
                "seed": seed,
                "fit_tasks": fit_ids,
                "report_tasks": report_ids,
                "baseline_model": baseline,
                "baseline": baseline_report,
                "router": router,
                "routed": routed_report,
                "savings": (
                    1.0 - routed_report["cost_usd"] / baseline_report["cost_usd"]
                    if baseline_report["cost_usd"]
                    else 0.0
                ),
                "quality_ratio": (
                    routed_report["accuracy"] / baseline_report["accuracy"]
                    if baseline_report["accuracy"]
                    else 1.0
                ),
            }
        )
    report = {
        "benchmark": "deepswe-1.1",
        "task_count": len(task_ids),
        "models": list(MODELS),
        "trace_model": str(args.trace_model.resolve()),
        "trace_probability": probabilities,
        "tasks": {
            task: {
                "language": metadata[task]["language"],
                "instruction_chars": len(metadata[task]["instruction"]),
            }
            for task in task_ids
        },
        "splits": rows,
        "mean_baseline_accuracy": statistics.mean(row["baseline"]["accuracy"] for row in rows),
        "mean_router_accuracy": statistics.mean(row["routed"]["accuracy"] for row in rows),
        "mean_router_savings": statistics.mean(row["savings"] for row in rows),
        "mean_quality_ratio": statistics.mean(row["quality_ratio"] for row in rows),
        "gate": {
            "quality_floor": 0.95,
            "savings_target": 0.40,
            "quality_pass": statistics.mean(row["quality_ratio"] for row in rows) >= 0.95,
            "savings_pass": statistics.mean(row["savings"] for row in rows) >= 0.40,
        },
    }
    report["gate"]["pass"] = report["gate"]["quality_pass"] and report["gate"]["savings_pass"]
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    output = args.artifact_root / "coding-router-deepswe-20260729-transfer-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
