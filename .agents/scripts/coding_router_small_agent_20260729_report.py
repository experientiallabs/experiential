"""Build and evaluate the isolated coding-router outcome matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from wmo.optimize.knn import best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.routing import evaluate_policy
from wmo.providers.base import ProviderKind, TokenUsage
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

SEEDS = (11, 23, 37, 41, 59)
PRICES = {
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 3.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcome", action="append", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--task-cache-root",
        type=Path,
        help="Directory containing Harbor task-cache subdirectories with instruction.md files.",
    )
    parser.add_argument("--matrix-name", default="coding-router-small-agent-20260729")
    return parser


def _pool() -> list[PoolEntry]:
    return [
        PoolEntry(
            name="gpt-5.5",
            kind=ProviderKind.OPENAI,
            model="gpt-5.5",
            input_per_mtok=5.0,
            output_per_mtok=30.0,
        ),
        PoolEntry(
            name="gpt-5.4-mini",
            kind=ProviderKind.OPENAI,
            model="gpt-5.4-mini",
            input_per_mtok=0.75,
            output_per_mtok=3.0,
        ),
        PoolEntry(
            name="claude-opus-4-8",
            kind=ProviderKind.ANTHROPIC,
            model="claude-opus-4-8",
            input_per_mtok=5.0,
            output_per_mtok=25.0,
        ),
        PoolEntry(
            name="claude-sonnet-4-6",
            kind=ProviderKind.ANTHROPIC,
            model="claude-sonnet-4-6",
            input_per_mtok=3.0,
            output_per_mtok=15.0,
        ),
    ]


def _cost(model: str, token_cost: dict[str, Any]) -> float:
    input_price, output_price = PRICES[model]
    return int(token_cost.get("input_tokens") or 0) * input_price / 1_000_000 + int(
        token_cost.get("output_tokens") or 0
    ) * output_price / 1_000_000


def _load_task_texts(root: Path | None) -> dict[str, str]:
    """Recover stable semantic task text from the run-owned Harbor cache."""
    if root is None or not root.is_dir():
        return {}
    texts: dict[str, str] = {}
    for path in sorted(root.rglob("instruction.md")):
        task_id = path.parent.name
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        previous = texts.get(task_id)
        if previous is not None and previous != text:
            raise ValueError(f"task cache has conflicting instruction text for {task_id!r}")
        texts[task_id] = text
    return texts


def _load_matrix(paths: list[Path], task_texts: dict[str, str]) -> OutcomeMatrix:
    pool = _pool()
    known_models = {entry.name for entry in pool}
    outcomes: list[ScenarioOutcome] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = str(payload["logical_model"])
        if model not in known_models:
            raise ValueError(f"{path} names unsupported matrix model {model!r}")
        for record in payload["trial_records"]:
            task = str(record["task_id"])
            key = (task, model)
            if key in seen:
                raise ValueError(f"duplicate matrix cell {key} from {path}")
            seen.add(key)
            token_cost = record.get("token_cost") or {}
            input_tokens = int(token_cost.get("input_tokens") or 0)
            output_tokens = int(token_cost.get("output_tokens") or 0)
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=f"terminal-bench-2:{task}",
                    task=task_texts.get(task, task),
                    model=model,
                    reward=float(record["reward"]) if record.get("reward") is not None else None,
                    success=float(record["reward"] or 0.0) >= 1.0,
                    stop_reason=str(record.get("exception_type") or ""),
                    usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    cost_usd=_cost(model, token_cost),
                    error=record.get("exception_type"),
                )
            )
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def _split(ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    ranked = sorted(
        ids,
        key=lambda item: (hashlib.sha256(f"{seed}|{item}".encode()).digest(), item),
    )
    cut = min(len(ranked) - 1, max(1, round(len(ranked) * 0.7)))
    return sorted(ranked[:cut]), sorted(ranked[cut:])


def _fit_candidate(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    embedder: HashingEmbedder,
    root: Path,
    *,
    z: float,
    floor_q: float,
    pick_lam: float,
) -> tuple[Any, str]:
    baseline = best_single_on_fit(matrix, fit_ids)
    policy = fit_knn_policy(
        matrix,
        bank_path=root / f"knn-bank-{z:g}-{floor_q:g}-{pick_lam:g}.npz",
        fit_ids=fit_ids,
        embedder=EmbedderSpec(kind="hashing", dim=512),
        embed_with=embedder,
        guard_model=baseline,
        z=z,
        floor_q=floor_q,
        pick_lam=pick_lam,
        min_pairs=1,
        se_floor=False,
        fitted_from=f"{matrix.__class__.__name__} fit seed",
    )
    if pick_lam > 0.0:
        # The measured cost-quality frontier uses the asymmetric guard: a cheaper challenger
        # may trade a small amount of quality for a material price reduction, while a pricier
        # challenger still needs positive evidence over the fallback.
        policy = policy.model_copy(update={"guard_mode": "asymmetric"})
    return policy, baseline


def main() -> None:
    args = _parser().parse_args()
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    task_texts = _load_task_texts(args.task_cache_root.resolve() if args.task_cache_root else None)
    matrix = _load_matrix(args.outcome, task_texts)
    matrix_path = root / f"{args.matrix_name}-matrix.json"
    matrix.save(matrix_path)
    embedder = HashingEmbedder(dim=512)
    rows: list[dict[str, Any]] = []
    grid = (
        (0.25, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.75, 0.0, 0.0),
        (0.5, 0.05, 0.0),
        (0.5, 0.0, 0.02),
        (0.5, 0.05, 0.02),
        (0.5, 0.0, 0.05),
        (0.5, 0.05, 0.05),
        (0.5, 0.0, 0.1),
        (0.5, 0.05, 0.1),
        (0.5, 0.0, 0.2),
        (0.5, 0.05, 0.2),
    )
    for seed in SEEDS:
        fit_ids, report_ids = _split(matrix.scenario_ids(), seed)
        baseline = best_single_on_fit(matrix, fit_ids)
        base_policy = RoutingPolicy(kind="static", default_model=baseline, pool=matrix.pool)
        fit_base = evaluate_policy(base_policy, matrix, fit_ids)
        base = evaluate_policy(base_policy, matrix, report_ids)
        best: dict[str, Any] | None = None
        for z, floor_q, pick_lam in grid:
            policy, _ = _fit_candidate(
                matrix,
                fit_ids,
                embedder,
                root,
                z=z,
                floor_q=floor_q,
                pick_lam=pick_lam,
            )
            result = evaluate_policy(policy, matrix, fit_ids, embedder=embedder)
            fit_quality_ok = result.accuracy >= fit_base.accuracy * 0.95
            fit_savings = 1.0 - result.cost_per_scenario / fit_base.cost_per_scenario
            candidate = {
                "policy": policy,
                "fit_accuracy": result.accuracy,
                "fit_cost_usd": result.cost_per_scenario,
                "fit_savings": fit_savings,
                "fit_quality_ok": fit_quality_ok,
                "z": z,
                "floor_q": floor_q,
                "pick_lam": pick_lam,
            }
            if fit_quality_ok and (best is None or candidate["fit_savings"] > best["fit_savings"]):
                best = candidate
        if best is None:
            raise RuntimeError(f"no quality-feasible router candidate on split seed={seed}")
        routed = evaluate_policy(best["policy"], matrix, report_ids, embedder=embedder)
        baseline_report = base
        rows.append(
            {
                "seed": seed,
                "fit_scenarios": len(fit_ids),
                "report_scenarios": len(report_ids),
                "baseline_model": baseline,
                "baseline_accuracy": baseline_report.accuracy,
                "baseline_cost_usd": baseline_report.cost_per_scenario,
                "router_accuracy": routed.accuracy,
                "router_cost_usd": routed.cost_per_scenario,
                "router_savings": (
                    1.0 - routed.cost_per_scenario / baseline_report.cost_per_scenario
                ),
                "router_model_mix": routed.model_mix,
                "router_params": {
                    key: best[key]
                    for key in ("z", "floor_q", "pick_lam", "fit_accuracy", "fit_savings")
                },
            }
        )
    report = {
        "matrix": str(matrix_path),
        "models": matrix.model_names(),
        "scenarios": matrix.scenario_ids(),
        "splits": rows,
        "mean_baseline_accuracy": statistics.mean(row["baseline_accuracy"] for row in rows),
        "mean_router_accuracy": statistics.mean(row["router_accuracy"] for row in rows),
        "mean_router_savings": statistics.mean(row["router_savings"] for row in rows),
        "quality_ratio": statistics.mean(row["router_accuracy"] for row in rows)
        / statistics.mean(row["baseline_accuracy"] for row in rows),
        "task_texts_loaded": len(task_texts),
    }
    report["promotion_gate"] = {
        "minimum_quality_ratio": 0.95,
        "minimum_savings": 0.40,
        "target_savings": [0.30, 0.40],
        "quality_pass": report["quality_ratio"] >= 0.95,
        "savings_pass": report["mean_router_savings"] >= 0.40,
        "pass": report["quality_ratio"] >= 0.95 and report["mean_router_savings"] >= 0.40,
    }
    (root / f"{args.matrix_name}-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
