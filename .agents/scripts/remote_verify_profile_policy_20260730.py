"""Verify the deployable profile policy on DeepSWE in remote compute."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import report_deepswe_knn_router_20260730 as base
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import RoutingPolicy
from wmo.optimize.profile import fit_profile_policy
from wmo.optimize.routing import evaluate_policy
from wmo.providers.base import TokenUsage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--task-meta", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def _matrix(trials_path: Path, meta_path: Path) -> tuple[OutcomeMatrix, list[str], dict[str, str]]:
    raw_rows = json.loads(trials_path.read_text(encoding="utf-8"))["rows"]
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    wanted_models = {(model, effort): name for name, model, effort, *_ in base.ARMS}
    by_cell: dict[tuple[str, str], list[dict[str, object]]] = collections.defaultdict(list)
    for row in raw_rows:
        if (
            row.get("source") != "deep-swe"
            or row.get("eval_scope") != "full"
            or not row.get("included_in_score")
        ):
            continue
        model = str(row["model"]).replace("gpt-5-6-", "gpt-5.6-")
        key = (model, str(row.get("reasoning_effort")))
        if key in wanted_models:
            by_cell[(str(row["task_name"]), wanted_models[key])].append(row)
    tasks = sorted(metadata)
    outcomes: list[ScenarioOutcome] = []
    for task in tasks:
        for arm, _model, _effort, _kind, _input, _output in base.ARMS:
            rows = by_cell[(task, arm)]
            rewards = [
                float(row["f2p_passed"]) / float(row["f2p_total"])
                for row in rows
                if row.get("f2p_total")
            ]
            costs = [float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=f"deepswe:{task}",
                    task=str(metadata[task]["instruction"]),
                    model=arm,
                    reward=statistics.mean(rewards),
                    success=statistics.mean(rewards) >= 1.0,
                    usage=TokenUsage(),
                    cost_usd=statistics.mean(costs),
                )
            )
    matrix = OutcomeMatrix(pool=base._pool(), outcomes=outcomes)
    repos = {task: str(metadata[task]["repo"]) for task in tasks}
    return matrix, tasks, repos


def main() -> None:
    args = _parser().parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    matrix, tasks, repos = _matrix(args.trials, args.task_meta)
    baseline = RoutingPolicy(kind="static", default_model=base.CHEAP_GUARD, pool=matrix.pool)
    rows: list[dict[str, object]] = []
    for seed in base.SEEDS:
        fit_ids, report_ids = base._group_split(tasks, repos, seed)
        policy = fit_profile_policy(
            matrix,
            fit_ids=fit_ids,
            guard_model=base.CHEAP_GUARD,
            quality_tolerance=0.02,
            bins=3,
            fitted_from="DeepSWE 1.1 grouped-repo profile length router",
        )
        routed = evaluate_policy(policy, matrix, report_ids)
        luna = evaluate_policy(baseline, matrix, report_ids)
        rows.append(
            {
                "seed": seed,
                "fit_tasks": len(fit_ids),
                "report_tasks": len(report_ids),
                "router_quality": routed.accuracy,
                "router_cost": routed.cost_per_scenario,
                "luna_quality": luna.accuracy,
                "luna_cost": luna.cost_per_scenario,
                "quality_ratio": routed.accuracy / luna.accuracy,
                "cost_savings": 1.0 - routed.cost_per_scenario / luna.cost_per_scenario,
                "model_mix": routed.model_mix,
                "profile_bins": policy.profile_bins,
                "profile_models": policy.profile_models,
            }
        )

    final_policy = fit_profile_policy(
        matrix,
        fit_ids=[f"deepswe:{task}" for task in tasks],
        guard_model=base.CHEAP_GUARD,
        quality_tolerance=0.02,
        bins=3,
        fitted_from="DeepSWE 1.1 full-fit profile length router",
    )
    final_policy.save(args.artifact_root / "deepswe-profile-router-policy.json")
    final_eval = evaluate_policy(final_policy, matrix, [f"deepswe:{task}" for task in tasks])
    report = {
        "benchmark": "DeepSWE 1.1",
        "tasks": len(tasks),
        "repos": len(set(repos.values())),
        "baseline": base.CHEAP_GUARD,
        "router": "profile text length, 3 fit-derived bins, quality tolerance 0.02",
        "splits": rows,
        "mean_quality_ratio": statistics.mean(row["quality_ratio"] for row in rows),
        "min_quality_ratio": min(row["quality_ratio"] for row in rows),
        "mean_cost_savings": statistics.mean(row["cost_savings"] for row in rows),
        "min_cost_savings": min(row["cost_savings"] for row in rows),
        "final_fit": {
            "quality": final_eval.accuracy,
            "cost_per_scenario": final_eval.cost_per_scenario,
            "model_mix": final_eval.model_mix,
            "unscored_scenarios": final_eval.unscored_scenarios,
            "profile_bins": final_policy.profile_bins,
            "profile_models": final_policy.profile_models,
        },
    }
    (args.artifact_root / "deepswe-profile-router-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
