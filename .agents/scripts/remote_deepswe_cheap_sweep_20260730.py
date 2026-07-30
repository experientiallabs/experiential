"""Run the guarded cheap-arm DeepSWE sweep on remote compute.

This file deliberately has no WMO import. The remote job only needs numpy and the frozen
ledger, which keeps the fitting job independent from the workstation checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np


GUARD = "gpt-5.6-luna__max"
SEEDS = (11, 23, 37, 41, 59)
DIM = 512
NGRAM = 3
ARMS = (
    ("gpt-5.6-luna__low", "gpt-5.6-luna", "low"),
    ("gpt-5.6-luna__medium", "gpt-5.6-luna", "medium"),
    ("gpt-5.6-luna__high", "gpt-5.6-luna", "high"),
    ("gpt-5.6-luna__xhigh", "gpt-5.6-luna", "xhigh"),
    ("gpt-5.6-luna__max", "gpt-5.6-luna", "max"),
    ("gpt-5.6-terra__high", "gpt-5.6-terra", "high"),
    ("gpt-5.6-terra__xhigh", "gpt-5.6-terra", "xhigh"),
    ("gpt-5.6-terra__max", "gpt-5.6-terra", "max"),
    ("gpt-5.6-sol__high", "gpt-5.6-sol", "high"),
    ("gpt-5.6-sol__xhigh", "gpt-5.6-sol", "xhigh"),
    ("claude-opus-5__low", "claude-opus-5", "low"),
    ("claude-opus-5__medium", "claude-opus-5", "medium"),
    ("claude-opus-5__high", "claude-opus-5", "high"),
)
ARM_INDEX = {name: index for index, (name, _model, _effort) in enumerate(ARMS)}
ARM_LOOKUP = {(model, effort): name for name, model, effort in ARMS}
CHEAP_POOLS = (
    ("xhigh_only", ("gpt-5.6-luna__xhigh",)),
    ("opus_low_only", ("claude-opus-5__low",)),
    ("terra_xhigh_only", ("gpt-5.6-terra__xhigh",)),
    (
        "cheap_reasoning",
        (
            "gpt-5.6-luna__xhigh",
            "claude-opus-5__low",
            "gpt-5.6-terra__xhigh",
            "gpt-5.6-terra__high",
        ),
    ),
    (
        "under_luna_cost",
        (
            "gpt-5.6-luna__low",
            "gpt-5.6-luna__medium",
            "gpt-5.6-luna__high",
            "gpt-5.6-luna__xhigh",
            "gpt-5.6-terra__high",
            "gpt-5.6-terra__xhigh",
            "claude-opus-5__low",
        ),
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--task-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _embed(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float64)
    normalized = text.lower()
    if len(normalized) < NGRAM:
        normalized = normalized.ljust(NGRAM)
    for start in range(len(normalized) - NGRAM + 1):
        digest = hashlib.blake2b(normalized[start : start + NGRAM].encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % DIM
        vector[bucket] += 1.0 if digest[0] & 1 else -1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _load(
    trials_path: Path, meta_path: Path
) -> tuple[list[str], dict[str, str], dict[str, str], dict[str, str], np.ndarray, np.ndarray]:
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = json.loads(trials_path.read_text(encoding="utf-8"))["rows"]
    cells: dict[tuple[str, str], list[tuple[float, float]]] = {}
    tasks = set()
    for row in rows:
        if row.get("source") != "deep-swe" or row.get("eval_scope") != "full" or not row.get("included_in_score"):
            continue
        model = str(row["model"]).replace("gpt-5-6-", "gpt-5.6-")
        handle = ARM_LOOKUP.get((model, str(row.get("reasoning_effort"))))
        if handle is None or not row.get("f2p_total") or row.get("cost_usd") is None:
            continue
        task = str(row["task_name"])
        tasks.add(task)
        cells.setdefault((task, handle), []).append(
            (float(row["f2p_passed"]) / float(row["f2p_total"]), float(row["cost_usd"]))
        )
    task_ids = sorted(tasks)
    missing = [
        (task, arm)
        for task in task_ids
        for arm, _model, _effort in ARMS
        if (task, arm) not in cells
    ]
    if missing:
        raise ValueError(f"incomplete matrix: {len(missing)} missing cells, first={missing[:3]}")
    texts = {task: str(metadata[task]["instruction"]) for task in task_ids}
    repos = {task: str(metadata[task]["repo"]) for task in task_ids}
    languages = {task: str(metadata[task].get("language", "unknown")) for task in task_ids}
    rewards = np.asarray(
        [[statistics.mean(v[0] for v in cells[(task, arm)]) for arm, _m, _e in ARMS] for task in task_ids],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[statistics.mean(v[1] for v in cells[(task, arm)]) for arm, _m, _e in ARMS] for task in task_ids],
        dtype=np.float64,
    )
    return task_ids, texts, repos, languages, rewards, costs


def _split(tasks: list[str], repos: dict[str, str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    repo_names = sorted(
        set(repos.values()),
        key=lambda repo: (hashlib.sha256(f"{seed}|{repo}".encode()).digest(), repo),
    )
    cut = min(len(repo_names) - 1, max(1, round(len(repo_names) * 0.7)))
    fit_repos = set(repo_names[:cut])
    fit = np.asarray([i for i, task in enumerate(tasks) if repos[task] in fit_repos], dtype=np.int64)
    report = np.asarray([i for i, task in enumerate(tasks) if repos[task] not in fit_repos], dtype=np.int64)
    if set(repos[tasks[i]] for i in fit) & set(repos[tasks[i]] for i in report):
        raise AssertionError("repository leakage in grouped split")
    return fit, report


def _choose(
    query: np.ndarray,
    fit_embeddings: np.ndarray,
    fit_rewards: np.ndarray,
    fit_costs: np.ndarray,
    arms: tuple[str, ...],
    *,
    z: float,
    pick_lam: float,
    min_pairs: int,
    se_floor: bool,
) -> str:
    names = (GUARD, *arms)
    indices = np.asarray([ARM_INDEX[name] for name in names], dtype=np.int64)
    similarities = fit_embeddings @ query
    budget = min(50, len(similarities))
    kth = float(np.sort(similarities)[-budget])
    neighbor_rows = np.flatnonzero(similarities > 0.95 * kth)
    if neighbor_rows.size == 0:
        neighbor_rows = np.asarray([int(np.argmax(similarities))])
    weights = np.clip(similarities[neighbor_rows], 0.0, None)
    rewards = fit_rewards[neighbor_rows][:, indices]
    costs = fit_costs[neighbor_rows][:, indices]
    denominator = weights.sum()
    profile = (rewards * weights[:, None]).sum(axis=0) / denominator if denominator else rewards.mean(axis=0)
    mean_cost = costs.mean(axis=0)
    cost_scale = float(mean_cost.mean())
    tilt = pick_lam * mean_cost / cost_scale if pick_lam else np.zeros_like(mean_cost)
    pick_index = int(np.argmax(profile - tilt))
    if pick_index == 0:
        return GUARD
    paired = np.isfinite(rewards[:, pick_index]) & np.isfinite(rewards[:, 0])
    diffs = rewards[paired, pick_index] - rewards[paired, 0]
    pairs = int(diffs.size)
    if pairs < min_pairs:
        return GUARD
    error = float(diffs.std(ddof=1)) / pairs**0.5 if pairs > 1 else 0.0
    if se_floor and pairs < 30:
        error = max(error, (0.25 / pairs) ** 0.5)
    candidate_cost = float(mean_cost[pick_index])
    guard_cost = float(mean_cost[0])
    z_effective = z if candidate_cost > guard_cost or not pick_lam else -z
    return names[pick_index] if float(diffs.mean()) > z_effective * error else GUARD


def _evaluate(
    choices: list[str], report: np.ndarray, rewards: np.ndarray, costs: np.ndarray
) -> tuple[float, float]:
    indices = np.asarray([ARM_INDEX[name] for name in choices], dtype=np.int64)
    return float(rewards[report, indices].mean()), float(costs[report, indices].mean())


def _profile_choices(
    fit: np.ndarray,
    report: np.ndarray,
    groups: list[str],
    rewards: np.ndarray,
    costs: np.ndarray,
    tolerance: float,
) -> list[str]:
    """Choose the cheapest arm within a fit-set quality band for each task profile."""
    choices: dict[str, str] = {}
    guard_index = ARM_INDEX[GUARD]
    for group in sorted(set(groups[index] for index in fit)):
        members = np.asarray([index for index in fit if groups[index] == group], dtype=np.int64)
        guard_quality = float(rewards[members, guard_index].mean())
        guard_cost = float(costs[members, guard_index].mean())
        eligible = []
        for arm, _model, _effort in ARMS:
            arm_index = ARM_INDEX[arm]
            quality = float(rewards[members, arm_index].mean())
            cost = float(costs[members, arm_index].mean())
            if cost < guard_cost and quality >= guard_quality - tolerance:
                eligible.append((cost, -quality, arm))
        choices[group] = min(eligible)[2] if eligible else GUARD
    return [choices.get(groups[index], GUARD) for index in report]


def main() -> None:
    args = _parser().parse_args()
    tasks, texts, repos, languages, rewards, costs = _load(args.trials, args.task_meta)
    embeddings = np.asarray([_embed(texts[task]) for task in tasks])
    results: list[dict[str, object]] = []
    grid = tuple(
        (z, lam, min_pairs, se_floor)
        for z in (0.0, 0.25, 0.5, 1.0, 1.5)
        for lam in (0.0, 0.0005, 0.001, 0.002, 0.005)
        for min_pairs in (4, 8, 12)
        for se_floor in (False, True)
    )
    for pool_name, cheap_arms in CHEAP_POOLS:
        for z, pick_lam, min_pairs, se_floor in grid:
            split_rows = []
            for seed in SEEDS:
                fit, report = _split(tasks, repos, seed)
                luna_q = float(rewards[report, ARM_INDEX[GUARD]].mean())
                luna_cost = float(costs[report, ARM_INDEX[GUARD]].mean())
                choices = [
                    _choose(
                        embeddings[index],
                        embeddings[fit],
                        rewards[fit],
                        costs[fit],
                        cheap_arms,
                        z=z,
                        pick_lam=pick_lam,
                        min_pairs=min_pairs,
                        se_floor=se_floor,
                    )
                    for index in report
                ]
                router_q, router_cost = _evaluate(choices, report, rewards, costs)
                split_rows.append(
                    {
                        "seed": seed,
                        "router_quality": router_q,
                        "router_cost": router_cost,
                        "luna_quality": luna_q,
                        "luna_cost": luna_cost,
                        "quality_diff": router_q - luna_q,
                        "cost_savings": 1.0 - router_cost / luna_cost,
                        "model_mix": {
                            name: choices.count(name) / len(choices) for name in sorted(set(choices))
                        },
                    }
                )
            results.append(
                {
                    "pool": pool_name,
                    "arms": list(cheap_arms),
                    "z": z,
                    "pick_lam": pick_lam,
                    "min_pairs": min_pairs,
                    "se_floor": se_floor,
                    "mean_quality_diff": statistics.mean(row["quality_diff"] for row in split_rows),
                    "min_quality_diff": min(row["quality_diff"] for row in split_rows),
                    "mean_cost_savings": statistics.mean(row["cost_savings"] for row in split_rows),
                    "min_cost_savings": min(row["cost_savings"] for row in split_rows),
                    "splits": split_rows,
                }
            )
    profile_results: list[dict[str, object]] = []
    lengths = np.asarray([len(texts[task]) for task in tasks], dtype=np.float64)
    for profile_name in ("language", "length", "language_length"):
        for tolerance in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
            split_rows = []
            for seed in SEEDS:
                fit, report = _split(tasks, repos, seed)
                if profile_name == "language":
                    groups = [languages[task] for task in tasks]
                else:
                    thresholds = np.quantile(lengths[fit], (1 / 3, 2 / 3))
                    length_groups = np.digitize(lengths, thresholds).astype(str).tolist()
                    groups = length_groups
                    if profile_name == "language_length":
                        groups = [f"{languages[task]}|{length_groups[index]}" for index, task in enumerate(tasks)]
                choices = _profile_choices(fit, report, groups, rewards, costs, tolerance)
                router_q, router_cost = _evaluate(choices, report, rewards, costs)
                luna_q = float(rewards[report, ARM_INDEX[GUARD]].mean())
                luna_cost = float(costs[report, ARM_INDEX[GUARD]].mean())
                split_rows.append(
                    {
                        "seed": seed,
                        "router_quality": router_q,
                        "router_cost": router_cost,
                        "luna_quality": luna_q,
                        "luna_cost": luna_cost,
                        "quality_diff": router_q - luna_q,
                        "cost_savings": 1.0 - router_cost / luna_cost,
                        "model_mix": {
                            name: choices.count(name) / len(choices) for name in sorted(set(choices))
                        },
                    }
                )
            profile_results.append(
                {
                    "profile": profile_name,
                    "tolerance": tolerance,
                    "mean_quality_diff": statistics.mean(row["quality_diff"] for row in split_rows),
                    "min_quality_diff": min(row["quality_diff"] for row in split_rows),
                    "mean_cost_savings": statistics.mean(row["cost_savings"] for row in split_rows),
                    "min_cost_savings": min(row["cost_savings"] for row in split_rows),
                    "splits": split_rows,
                }
            )
    results.sort(
        key=lambda row: (
            float(row["mean_quality_diff"]),
            float(row["mean_cost_savings"]),
        ),
        reverse=True,
    )
    eligible = [
        row
        for row in results
        if float(row["mean_quality_diff"]) >= -0.005
        and float(row["min_quality_diff"]) >= -0.015
        and float(row["mean_cost_savings"]) > 0.0
    ]
    eligible.sort(key=lambda row: float(row["mean_cost_savings"]), reverse=True)
    profile_eligible = [
        row
        for row in profile_results
        if float(row["mean_quality_diff"]) >= -0.005
        and float(row["min_quality_diff"]) >= -0.015
        and float(row["mean_cost_savings"]) > 0.0
    ]
    profile_eligible.sort(key=lambda row: float(row["mean_cost_savings"]), reverse=True)
    report = {
        "benchmark": "DeepSWE 1.1",
        "tasks": len(tasks),
        "repos": len(set(repos.values())),
        "baseline": GUARD,
        "criterion": {
            "mean_quality_diff_at_least": -0.005,
            "min_quality_diff_at_least": -0.015,
            "mean_cost_savings_positive": True,
        },
        "candidate_count": len(results),
        "eligible_count": len(eligible),
        "best_eligible": eligible[:20],
        "best_quality": results[:20],
        "profile_candidates": sorted(
            profile_results,
            key=lambda row: (float(row["mean_quality_diff"]), float(row["mean_cost_savings"])),
            reverse=True,
        ),
        "profile_eligible": profile_eligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
