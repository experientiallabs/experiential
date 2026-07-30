"""Select a fast DeepSWE proxy and evaluate reasoning-effort routing."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import TypedDict

from scipy.stats import pearsonr, spearmanr

EFFORTS = ("low", "medium", "high", "xhigh")
MAPPED_ARMS = {
    "gpt-5.5-2026-04-23-medium": ("gpt-5-5", "medium"),
    "gpt-5.5-2026-04-23-xhigh": ("gpt-5-5", "xhigh"),
    "Claude Sonnet 4.6": ("claude-sonnet-4-6", "high"),
    "Claude Opus 4.8-xhigh": ("claude-opus-4-8", "xhigh"),
    "gpt-5.4-2026-03-05-medium": ("gpt-5-4", "xhigh"),
    "GLM-5.2 [high]": ("glm-5-2", "high"),
    "GPT-5.6 Luna [medium]": ("gpt-5-6-luna", "medium"),
    "GPT-5.6 Sol [medium]": ("gpt-5-6-sol", "medium"),
    "Fable 5 [high]": ("claude-fable-5", "high"),
    "Opus 5 [high]": ("claude-opus-5", "high"),
    "Sonnet 5 [high]": ("claude-sonnet-5", "high"),
    "Grok 4.5 [high]": ("grok-4-5", "high"),
}


class Trial(TypedDict):
    task_name: str
    model: str
    reasoning_effort: str
    passed: bool
    cost_usd: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--swe-leaderboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proxy-sizes", default="8,12,20")
    parser.add_argument("--candidates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def _load_trials(path: Path) -> dict[str, dict[str, list[Trial]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, list[Trial]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for raw in payload["rows"]:
        if raw.get("source") != "deep-swe" or not raw.get("included_in_score"):
            continue
        if raw.get("model") == "gpt-5-5" and raw.get("reasoning_effort") in EFFORTS:
            effort = str(raw["reasoning_effort"])
            grouped[raw["task_name"]][effort].append(
                {
                    "task_name": raw["task_name"],
                    "model": "gpt-5-5",
                    "reasoning_effort": effort,
                    "passed": bool(raw["passed"]),
                    "cost_usd": float(raw["cost_usd"] or 0.0),
                }
            )
        for arm, (model, effort) in MAPPED_ARMS.items():
            if raw.get("model") == model and raw.get("reasoning_effort") == effort:
                grouped[raw["task_name"]][arm].append(
                    {
                        "task_name": raw["task_name"],
                        "model": model,
                        "reasoning_effort": effort,
                        "passed": bool(raw["passed"]),
                        "cost_usd": float(raw["cost_usd"] or 0.0),
                    }
                )
                break
    arms = tuple(MAPPED_ARMS)
    tasks = sorted(task for task, rows in grouped.items() if all(arm in rows for arm in arms))
    if len(tasks) < 20:
        raise ValueError(f"need at least 20 complete mapped tasks, found {len(tasks)}")
    return {task: dict(grouped[task]) for task in tasks}


def _task_scores(
    grouped: dict[str, dict[str, list[Trial]]],
) -> tuple[list[str], list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    tasks = sorted(grouped)
    arms = list(MAPPED_ARMS)
    quality = {
        task: {
            arm: statistics.mean(row["passed"] for row in grouped[task][arm])
            for arm in arms
        }
        for task in tasks
    }
    cost = {
        task: {
            arm: statistics.mean(row["cost_usd"] for row in grouped[task][arm])
            for arm in arms
        }
        for task in tasks
    }
    return tasks, arms, quality, cost


def _latest_swe_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, float] = {}
    for row in payload:
        name = row.get("modelName")
        ranges = row.get("rangeStats", {}).get("all", {})
        if not name or not ranges:
            continue
        latest = max(ranges, key=lambda key: int(key.split(":", 1)[1]))
        score = ranges[latest].get("resolvedRate")
        if isinstance(score, (float, int)) and score > 0:
            result[str(name)] = float(score)
    return result


def _correlation(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) < 3:
        raise ValueError("correlation needs at least three arms")
    spearman = spearmanr(left, right)
    pearson = pearsonr(left, right)
    return {
        "n": len(left),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
    }


def _arm_scores(
    tasks: list[str],
    arms: list[str],
    quality: dict[str, dict[str, float]],
) -> dict[str, float]:
    return {arm: statistics.mean(quality[task][arm] for task in tasks) for arm in arms}


def _proxy_score(
    proxy: list[str],
    tasks: list[str],
    arms: list[str],
    quality: dict[str, dict[str, float]],
    train_arms: list[str],
) -> float:
    full = _arm_scores(tasks, arms, quality)
    small = _arm_scores(proxy, arms, quality)
    return statistics.mean(
        _correlation(
            [small[arm] for arm in train_arms if arm in small],
            [full[arm] for arm in train_arms if arm in full],
        )["spearman_rho"]
        for _ in (0,)
    )


def _choose_proxy(
    tasks: list[str],
    arms: list[str],
    quality: dict[str, dict[str, float]],
    size: int,
    candidates: int,
    seed: int,
) -> list[str]:
    arm_order = sorted(arms, key=lambda arm: hashlib.sha256(arm.encode()).digest())
    train_arms = arm_order[: max(6, len(arm_order) * 2 // 3)]
    rng = random.Random(seed + size)
    best: tuple[float, tuple[str, ...]] | None = None
    for _ in range(candidates):
        candidate = tuple(sorted(rng.sample(tasks, size)))
        score = _proxy_score(list(candidate), tasks, arms, quality, train_arms)
        key = (score, tuple(candidate))
        if best is None or key > best:
            best = key
    if best is None:
        raise ValueError("proxy search produced no candidate")
    return list(best[1])


def _router_report(
    grouped: dict[str, dict[str, list[Trial]]],
) -> dict[str, object]:
    tasks = sorted(grouped)
    effort = {
        task: {
            variant: {
                "quality": statistics.mean(row["passed"] for row in grouped[task][variant]),
                "cost_usd": statistics.mean(row["cost_usd"] for row in grouped[task][variant]),
            }
            for variant in EFFORTS
        }
        for task in tasks
        if all(variant in grouped[task] for variant in EFFORTS)
    }
    tasks = sorted(effort)

    def split(seed: int) -> tuple[list[str], list[str]]:
        ranked = sorted(tasks, key=lambda task: hashlib.sha256(f"{seed}|{task}".encode()).digest())
        cut = min(len(ranked) - 1, max(1, round(len(ranked) * 0.7)))
        return ranked[:cut], ranked[cut:]

    def evaluate(task_ids: list[str], choices: dict[str, str], baseline: str) -> dict[str, object]:
        quality = statistics.mean(
            effort[task][choices[task]]["quality"] for task in task_ids
        )
        cost = statistics.mean(effort[task][choices[task]]["cost_usd"] for task in task_ids)
        baseline_quality = statistics.mean(effort[task][baseline]["quality"] for task in task_ids)
        baseline_cost = statistics.mean(effort[task][baseline]["cost_usd"] for task in task_ids)
        return {
            "quality": quality,
            "quality_retained": quality / baseline_quality if baseline_quality else 1.0,
            "quality_delta": quality - baseline_quality,
            "cost_usd_per_task": cost,
            "cost_savings": 1.0 - cost / baseline_cost if baseline_cost else 0.0,
            "model_mix": dict(collections.Counter(choices[task] for task in task_ids)),
        }

    rows: list[dict[str, object]] = []
    for seed in (11, 23, 37, 41, 59):
        fit, heldout = split(seed)
        baseline = max(
            EFFORTS,
            key=lambda variant: statistics.mean(effort[task][variant]["quality"] for task in fit),
        )
        low_to_high = {task: "high" for task in heldout}
        low_to_high_result = evaluate(heldout, low_to_high, baseline)
        low_to_high_result["cascade_semantics"] = "low first, high only after low failure"
        low_to_high_result["quality"] = statistics.mean(
            effort[task]["low"]["quality"]
            + (1.0 - effort[task]["low"]["quality"])
            * effort[task][low_to_high[task]]["quality"]
            for task in heldout
        )
        low_to_high_result["cost_usd_per_task"] = statistics.mean(
            effort[task]["low"]["cost_usd"]
            + (1.0 - effort[task]["low"]["quality"])
            * effort[task][low_to_high[task]]["cost_usd"]
            for task in heldout
        )
        baseline_quality = statistics.mean(effort[task][baseline]["quality"] for task in heldout)
        baseline_cost = statistics.mean(effort[task][baseline]["cost_usd"] for task in heldout)
        low_to_high_result["quality_retained"] = low_to_high_result["quality"] / baseline_quality
        low_to_high_result["quality_delta"] = low_to_high_result["quality"] - baseline_quality
        low_to_high_result["cost_savings"] = (
            1.0 - low_to_high_result["cost_usd_per_task"] / baseline_cost
        )
        rows.append(
            {
                "seed": seed,
                "fit_tasks": len(fit),
                "heldout_tasks": len(heldout),
                "fit_selected_baseline": baseline,
                "heldout_baseline": evaluate(
                    heldout, {task: baseline for task in heldout}, baseline
                ),
                "low_to_high_cascade": low_to_high_result,
            }
        )
    return {
        "benchmark": "deepswe-1.1",
        "task_count": len(tasks),
        "splits": rows,
        "mean_cascade_quality_retained": statistics.mean(
            row["low_to_high_cascade"]["quality_retained"] for row in rows
        ),
        "mean_cascade_cost_savings": statistics.mean(
            row["low_to_high_cascade"]["cost_savings"] for row in rows
        ),
        "promotion_gate": {
            "quality_retention_floor": 0.95,
            "cost_savings_floor": 0.40,
            "pass": all(
                row["low_to_high_cascade"]["quality_retained"] >= 0.95
                and row["low_to_high_cascade"]["cost_savings"] >= 0.40
                for row in rows
            ),
        },
    }


def main() -> None:
    args = _parser().parse_args()
    grouped = _load_trials(args.trials.resolve())
    tasks, arms, quality, _ = _task_scores(grouped)
    swe_scores = _latest_swe_scores(args.swe_leaderboard.resolve())
    deep_scores = _arm_scores(tasks, arms, quality)
    external_arms = [arm for arm in arms if arm in swe_scores]
    external = _correlation(
        [deep_scores[arm] for arm in external_arms],
        [swe_scores[arm] for arm in external_arms],
    ) if len(external_arms) >= 3 else {"n": len(external_arms)}
    proxy_results: dict[str, object] = {}
    for size_text in args.proxy_sizes.split(","):
        size = int(size_text)
        proxy = _choose_proxy(tasks, arms, quality, size, args.candidates, args.seed)
        proxy_scores = _arm_scores(proxy, arms, quality)
        proxy_results[str(size)] = {
            "task_ids": proxy,
            "deep_swe_correlation": _correlation(
                [proxy_scores[arm] for arm in arms], [deep_scores[arm] for arm in arms]
            ),
            "swe_bench_correlation": (
                _correlation(
                    [proxy_scores[arm] for arm in external_arms],
                    [swe_scores[arm] for arm in external_arms],
                )
                if len(external_arms) >= 3
                else {"n": len(external_arms)}
            ),
        }
    report = {
        "protocol": {
            "selection_uses_swe_labels": False,
            "selection_seed": args.seed,
            "candidate_count_per_size": args.candidates,
            "deep_swe_trials": str(args.trials.resolve()),
            "swe_bench_snapshot": str(args.swe_leaderboard.resolve()),
        },
        "deep_swe": {
            "task_count": len(tasks),
            "mapped_arm_count": len(arms),
            "mapped_arms": arms,
            "full_correlation_to_swe_bench": external,
            "deep_swe_mean_scores": deep_scores,
            "swe_bench_latest_resolved_rates": {
                arm: swe_scores[arm] for arm in external_arms
            },
        },
        "fast_proxy": proxy_results,
        "reasoning_effort_router": _router_report(grouped),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
