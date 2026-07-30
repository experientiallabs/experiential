"""Fit a grouped-CV DeepSWE model-effort kNN router with graded f2p reward."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import evaluate_policy
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

# ruff: noqa: E501, ANN401, B905

SEEDS = (11, 23, 37, 41, 59)
BASELINE = "claude-opus-5__high"
CHEAP_GUARD = "gpt-5.6-luna__max"

# These arms are model x reasoning effort handles. The underlying runtime id remains in
# PoolEntry.model, so the fitted artifact can later be wired to the actual provider runtime.
ARMS: tuple[tuple[str, str, str, ProviderKind, float, float], ...] = (
    ("gpt-5.6-luna__low", "gpt-5.6-luna", "low", ProviderKind.OPENAI_RESPONSES, 1.0, 6.0),
    ("gpt-5.6-luna__medium", "gpt-5.6-luna", "medium", ProviderKind.OPENAI_RESPONSES, 1.0, 6.0),
    ("gpt-5.6-luna__high", "gpt-5.6-luna", "high", ProviderKind.OPENAI_RESPONSES, 1.0, 6.0),
    ("gpt-5.6-luna__xhigh", "gpt-5.6-luna", "xhigh", ProviderKind.OPENAI_RESPONSES, 1.0, 6.0),
    ("gpt-5.6-luna__max", "gpt-5.6-luna", "max", ProviderKind.OPENAI_RESPONSES, 1.0, 6.0),
    ("gpt-5.6-terra__high", "gpt-5.6-terra", "high", ProviderKind.OPENAI_RESPONSES, 2.5, 15.0),
    ("gpt-5.6-terra__xhigh", "gpt-5.6-terra", "xhigh", ProviderKind.OPENAI_RESPONSES, 2.5, 15.0),
    ("gpt-5.6-terra__max", "gpt-5.6-terra", "max", ProviderKind.OPENAI_RESPONSES, 2.5, 15.0),
    ("gpt-5.6-sol__high", "gpt-5.6-sol", "high", ProviderKind.OPENAI_RESPONSES, 5.0, 30.0),
    ("gpt-5.6-sol__xhigh", "gpt-5.6-sol", "xhigh", ProviderKind.OPENAI_RESPONSES, 5.0, 30.0),
    ("claude-opus-5__low", "claude-opus-5", "low", ProviderKind.ANTHROPIC, 5.0, 25.0),
    ("claude-opus-5__medium", "claude-opus-5", "medium", ProviderKind.ANTHROPIC, 5.0, 25.0),
    ("claude-opus-5__high", "claude-opus-5", "high", ProviderKind.ANTHROPIC, 5.0, 25.0),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--pick-lam", type=float, default=0.002)
    return parser


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name=name,
            kind=kind,
            model=model,
            reasoning_effort=effort if kind is ProviderKind.OPENAI_RESPONSES else None,
            input_per_mtok=input_price,
            output_per_mtok=output_price,
        )
        for name, model, effort, kind, input_price, output_price in ARMS
    ]


def _task_metadata(root: Path, task: str) -> tuple[str, str]:
    task_dir = root / task
    instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
    metadata = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    repo = str(metadata["metadata"]["repository_url"])
    return instruction, repo


def _load_matrix(trials_path: Path, task_root: Path, artifact_root: Path) -> tuple[OutcomeMatrix, list[str], dict[str, str]]:
    raw_rows = json.loads(trials_path.read_text(encoding="utf-8"))["rows"]
    wanted_models = {(model, effort): name for name, model, effort, *_ in ARMS}
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in raw_rows:
        if row.get("source") != "deep-swe" or row.get("eval_scope") != "full" or not row.get("included_in_score"):
            continue
        ledger_model = str(row["model"]).replace("gpt-5-6-", "gpt-5.6-")
        key = (ledger_model, str(row.get("reasoning_effort")))
        if key in wanted_models:
            by_cell[(str(row["task_name"]), wanted_models[key])].append(row)

    all_tasks = sorted({task for task, _ in by_cell})
    task_texts: dict[str, str] = {}
    task_repos: dict[str, str] = {}
    for task in all_tasks:
        task_texts[task], task_repos[task] = _task_metadata(task_root, task)

    complete_tasks: list[str] = []
    dropped: dict[str, list[str]] = {}
    outcomes: list[ScenarioOutcome] = []
    for task in all_tasks:
        missing: list[str] = []
        for arm, *_ in ARMS:
            rows = by_cell.get((task, arm), [])
            rewards = [float(row["f2p_passed"]) / float(row["f2p_total"]) for row in rows if row.get("f2p_total")]
            costs = [float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
            if not rewards or not costs:
                missing.append(arm)
        if missing:
            dropped[task] = missing
            continue
        complete_tasks.append(task)
        for arm, *_ in ARMS:
            rows = by_cell[(task, arm)]
            rewards = [float(row["f2p_passed"]) / float(row["f2p_total"]) for row in rows if row.get("f2p_total")]
            costs = [float(row["cost_usd"]) for row in rows if row.get("cost_usd") is not None]
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=f"deepswe:{task}",
                    task=task_texts[task],
                    model=arm,
                    reward=statistics.mean(rewards),
                    success=statistics.mean(rewards) >= 1.0,
                    usage=TokenUsage(),
                    cost_usd=statistics.mean(costs),
                )
            )
    matrix = OutcomeMatrix(pool=_pool(), outcomes=outcomes)
    matrix_path = artifact_root / "deepswe-graded-model-effort-matrix.json"
    matrix.save(matrix_path)
    (artifact_root / "matrix-filter.json").write_text(
        json.dumps(
            {
                "source": str(trials_path.resolve()),
                "arms": len(ARMS),
                "input_tasks": len(all_tasks),
                "kept_tasks": len(complete_tasks),
                "dropped_tasks": dropped,
                "missing_cell_count": sum(len(v) for v in dropped.values()),
                "reward": "graded f2p_passed/f2p_total",
                "cost": "measured cost_usd mean per task-arm cell",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix, complete_tasks, task_repos


def _group_split(tasks: list[str], repos: dict[str, str], seed: int) -> tuple[list[str], list[str]]:
    repo_names = sorted(set(repos[task] for task in tasks), key=lambda repo: (hashlib.sha256(f"{seed}|{repo}".encode()).digest(), repo))
    cut = min(len(repo_names) - 1, max(1, round(len(repo_names) * 0.7)))
    fit_repos = set(repo_names[:cut])
    fit = sorted(task for task in tasks if repos[task] in fit_repos)
    report = sorted(task for task in tasks if repos[task] not in fit_repos)
    if set(repos[t] for t in fit) & set(repos[t] for t in report):
        raise AssertionError("grouped split leaked a repository")
    return [f"deepswe:{task}" for task in fit], [f"deepswe:{task}" for task in report]


def _fit(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    artifact_root: Path,
    embedder: HashingEmbedder,
    *,
    z: float,
    pick_lam: float,
) -> RoutingPolicy:
    policy = fit_knn_policy(
        matrix,
        bank_path=artifact_root / f"bank-{z:g}-{pick_lam:g}.npz",
        fit_ids=fit_ids,
        embedder=EmbedderSpec(kind="hashing", dim=512),
        embed_with=embedder,
        guard_model=CHEAP_GUARD,
        z=z,
        pick_lam=pick_lam,
        min_pairs=8,
        se_floor=True,
        floor_q=0.0,
        fitted_from="DeepSWE 1.1 graded f2p grouped-repo CV",
    )
    if pick_lam > 0.0:
        policy = policy.model_copy(update={"guard_mode": "asymmetric"})
    return policy


def _static(matrix: OutcomeMatrix, model: str) -> RoutingPolicy:
    return RoutingPolicy(kind="static", default_model=model, pool=matrix.pool)


def main() -> None:
    args = _parser().parse_args()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    matrix, tasks, repos = _load_matrix(args.trials.resolve(), args.task_root.resolve(), artifact_root)
    embedder = HashingEmbedder(dim=512)
    split_rows: list[dict[str, Any]] = []
    traffic: collections.Counter[str] = collections.Counter()
    for seed in SEEDS:
        fit_ids, report_ids = _group_split(tasks, repos, seed)
        baseline_report = evaluate_policy(_static(matrix, BASELINE), matrix, report_ids)
        policy = _fit(matrix, fit_ids, artifact_root, embedder, z=args.z, pick_lam=args.pick_lam)
        fit_eval = evaluate_policy(policy, matrix, fit_ids, embedder=embedder)
        if fit_eval.unscored_scenarios != 0:
            raise AssertionError("fit evaluation has unscored routed scenarios")
        routed = evaluate_policy(policy, matrix, report_ids, embedder=embedder)
        if routed.unscored_scenarios != 0:
            raise AssertionError("report evaluation has unscored routed scenarios")
        traffic.update({model: round(share * routed.scenarios) for model, share in routed.model_mix.items()})
        split_rows.append(
            {
                "seed": seed,
                "fit_tasks": len(fit_ids),
                "report_tasks": len(report_ids),
                "fit_repos": len(set(repos[sid.split(":", 1)[1]] for sid in fit_ids)),
                "report_repos": len(set(repos[sid.split(":", 1)[1]] for sid in report_ids)),
                "baseline_quality": baseline_report.accuracy,
                "baseline_cost": baseline_report.cost_per_scenario,
                "router_quality": routed.accuracy,
                "router_cost": routed.cost_per_scenario,
                "quality_ratio": routed.accuracy / baseline_report.accuracy,
                "cost_savings": 1.0 - routed.cost_per_scenario / baseline_report.cost_per_scenario,
                "model_mix": routed.model_mix,
                "params": {"z": args.z, "pick_lam": args.pick_lam},
                "fit_quality": fit_eval.accuracy,
                "fit_cost": fit_eval.cost_per_scenario,
            }
        )
    final_policy = _fit(matrix, [f"deepswe:{task}" for task in tasks], artifact_root, embedder, z=args.z, pick_lam=args.pick_lam)
    final_policy.save(artifact_root / "deepswe-knn-router-policy.json")
    final_eval = evaluate_policy(final_policy, matrix, [f"deepswe:{task}" for task in tasks], embedder=embedder)
    result = {
        "benchmark": "DeepSWE 1.1",
        "reward": "graded f2p_passed/f2p_total",
        "arms": [name for name, *_ in ARMS],
        "baseline": BASELINE,
        "cheap_guard": CHEAP_GUARD,
        "knn_z": args.z,
        "pick_lam": args.pick_lam,
        "seeds": list(SEEDS),
        "splits": split_rows,
        "mean_quality": statistics.mean(row["router_quality"] for row in split_rows),
        "mean_cost": statistics.mean(row["router_cost"] for row in split_rows),
        "mean_quality_ratio": statistics.mean(row["quality_ratio"] for row in split_rows),
        "mean_cost_savings": statistics.mean(row["cost_savings"] for row in split_rows),
        "min_quality_ratio": min(row["quality_ratio"] for row in split_rows),
        "min_cost_savings": min(row["cost_savings"] for row in split_rows),
        "traffic_counts_across_splits": dict(sorted(traffic.items())),
        "traffic_share_across_splits": {
            model: count / sum(traffic.values()) for model, count in sorted(traffic.items())
        },
        "final_fit": {
            "scenarios": final_eval.scenarios,
            "quality": final_eval.accuracy,
            "cost_per_scenario": final_eval.cost_per_scenario,
            "model_mix": final_eval.model_mix,
            "unscored_scenarios": final_eval.unscored_scenarios,
        },
        "promotion_gate": {
            "quality_floor": 0.95,
            "savings_floor": 0.30,
            "quality_pass": statistics.mean(row["quality_ratio"] for row in split_rows) >= 0.95,
            "savings_pass": statistics.mean(row["cost_savings"] for row in split_rows) >= 0.30,
        },
    }
    (artifact_root / "deepswe-knn-router-report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
