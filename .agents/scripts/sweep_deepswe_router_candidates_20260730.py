"""Sweep guarded cheap-arm routers against Luna @ max on grouped DeepSWE splits."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import tempfile
from pathlib import Path

from wmo.optimize.knn import fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec
from wmo.optimize.routing import evaluate_policy
from wmo.retrieval.embedders import HashingEmbedder

import report_deepswe_knn_router_20260730 as base


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
    parser.add_argument("--trials", type=Path, default=Path("/private/tmp/deepswe_trials.json"))
    parser.add_argument(
        "--task-root", type=Path, default=Path("/private/tmp/deep-swe-reference-20260729/tasks")
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def _submatrix(matrix: OutcomeMatrix, names: tuple[str, ...]) -> OutcomeMatrix:
    wanted = set(names) | {base.CHEAP_GUARD}
    return matrix.model_copy(
        update={
            "pool": [entry for entry in matrix.pool if entry.name in wanted],
            "outcomes": [outcome for outcome in matrix.outcomes if outcome.model in wanted],
        }
    )


def _fit(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    artifact_root: Path,
    embedder: HashingEmbedder,
    *,
    z: float,
    pick_lam: float,
    min_pairs: int,
    se_floor: bool,
) -> object:
    policy = fit_knn_policy(
        matrix,
        bank_path=artifact_root / f"bank-{z:g}-{pick_lam:g}-{min_pairs}-{int(se_floor)}.npz",
        fit_ids=fit_ids,
        embedder=EmbedderSpec(kind="hashing", dim=512),
        embed_with=embedder,
        guard_model=base.CHEAP_GUARD,
        z=z,
        pick_lam=pick_lam,
        min_pairs=min_pairs,
        se_floor=se_floor,
        floor_q=0.0,
        fitted_from="DeepSWE guarded cheap-arm sweep",
    )
    if pick_lam > 0.0:
        policy = policy.model_copy(update={"guard_mode": "asymmetric"})
    return policy


def main() -> None:
    args = _parser().parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        matrix, tasks, repos = base._load_matrix(
            args.trials.resolve(), args.task_root.resolve(), Path(scratch)
        )
        embedder = HashingEmbedder(dim=512)
        split_data = []
        for seed in base.SEEDS:
            fit_ids, report_ids = base._group_split(tasks, repos, seed)
            luna = evaluate_policy(base._static(matrix, base.CHEAP_GUARD), matrix, report_ids)
            split_data.append((seed, fit_ids, report_ids, luna))

        results: list[dict[str, object]] = []
        for pool_name, cheap_names in CHEAP_POOLS:
            candidate_matrix = _submatrix(matrix, cheap_names)
            for z, pick_lam, min_pairs, se_floor in itertools.product(
                (0.0, 0.25, 0.5, 1.0, 1.5),
                (0.0, 0.0005, 0.001, 0.002, 0.005),
                (4, 8, 12),
                (False, True),
            ):
                rows: list[dict[str, float | int]] = []
                for seed, fit_ids, report_ids, luna in split_data:
                    policy = _fit(
                        candidate_matrix,
                        fit_ids,
                        args.artifact_root,
                        embedder,
                        z=z,
                        pick_lam=pick_lam,
                        min_pairs=min_pairs,
                        se_floor=se_floor,
                    )
                    routed = evaluate_policy(
                        policy, candidate_matrix, report_ids, embedder=embedder
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "router_quality": routed.accuracy,
                            "router_cost": routed.cost_per_scenario,
                            "luna_quality": luna.accuracy,
                            "luna_cost": luna.cost_per_scenario,
                            "quality_diff": routed.accuracy - luna.accuracy,
                            "cost_savings": 1.0 - routed.cost_per_scenario / luna.cost_per_scenario,
                        }
                    )
                results.append(
                    {
                        "pool": pool_name,
                        "arms": list(cheap_names),
                        "z": z,
                        "pick_lam": pick_lam,
                        "min_pairs": min_pairs,
                        "se_floor": se_floor,
                        "mean_quality_diff": statistics.mean(row["quality_diff"] for row in rows),
                        "min_quality_diff": min(row["quality_diff"] for row in rows),
                        "mean_cost_savings": statistics.mean(row["cost_savings"] for row in rows),
                        "min_cost_savings": min(row["cost_savings"] for row in rows),
                        "rows": rows,
                    }
                )

    results.sort(
        key=lambda row: (
            -float(row["mean_cost_savings"])
            if float(row["mean_quality_diff"]) >= -0.005
            and float(row["min_quality_diff"]) >= -0.015
            else 1.0,
            -float(row["mean_quality_diff"]),
        )
    )
    report = {
        "benchmark": "DeepSWE 1.1",
        "baseline": base.CHEAP_GUARD,
        "criterion": {
            "mean_quality_diff_floor": -0.005,
            "min_quality_diff_floor": -0.015,
            "positive_mean_savings": True,
        },
        "candidate_count": len(results),
        "top_candidates": results[:50],
    }
    output = args.artifact_root / "deepswe-cheap-arm-sweep.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
