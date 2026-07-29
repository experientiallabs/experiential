"""Compare real and WMO-simulated matrices at cell, model, route, and decision levels."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from wmo.core.files import write_text_atomic
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import RoutingDecision, RoutingPolicy
from wmo.optimize.routing import evaluate_policy, route_scenarios

DIALS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = range(5)
BOOTSTRAP_DRAWS = 10_000


class CachedEmbedder:
    def __init__(self, path: Path) -> None:
        meta = _dict(_json(path / "meta.json"))
        texts = [str(value) for value in _list(meta["tasks"])]
        with np.load(path / "vectors.npz") as payload:
            vectors = np.asarray(payload["vectors"], dtype=np.float64)
        self._vectors = {text: vectors[index] for index, text in enumerate(texts)}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text].tolist() for text in texts]


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected object, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value).__name__}")
    return list(value)


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected number, got {type(value).__name__}")
    return float(value)


def _canonical(benchmark: str, matrix: OutcomeMatrix) -> OutcomeMatrix:
    if benchmark != "tau2":
        return matrix
    rows = []
    for row in matrix.outcomes:
        domain, separator, task_id = row.scenario_id.partition(":")
        scenario_id = f"{domain}/{task_id}" if separator else row.scenario_id
        rows.append(row.model_copy(update={"scenario_id": scenario_id}))
    return matrix.model_copy(update={"outcomes": rows})


def _cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], tuple[float, float]]:
    grouped: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in matrix.outcomes:
        if row.reward is not None:
            grouped[(row.scenario_id, row.model)].append((row.reward, row.cost_usd))
    return {
        key: (
            statistics.mean(reward for reward, _cost in values),
            statistics.mean(cost for _reward, cost in values),
        )
        for key, values in grouped.items()
    }


def _groups(path: Path) -> dict[str, str]:
    root = _dict(_json(path))
    groups = {}
    for value in _list(root["tasks"]):
        row = _dict(value)
        groups[str(row["task_id"])] = str(row["group"])
    return groups


def _split(path: Path, benchmark: str) -> list[str]:
    root = _dict(_json(path))
    split = _dict(root[benchmark])
    return [str(value) for value in _list(split["heldout"])]


def _rank_corr(left: dict[str, float], right: dict[str, float], method: str) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    frame = pd.DataFrame(
        {"left": [left[key] for key in keys], "right": [right[key] for key in keys]}
    )
    value = frame["left"].corr(frame["right"], method=method)
    return None if pd.isna(value) else float(value)


def _cell_and_model(
    benchmark: str,
    real: OutcomeMatrix,
    simulated: dict[int, OutcomeMatrix],
) -> dict[str, object]:
    real_cells = _cells(real)
    seed_rows = []
    model_rows = []
    for seed in SEEDS:
        sim_cells = _cells(simulated[seed])
        keys = sorted(set(real_cells) & set(sim_cells))
        actual = [real_cells[key][0] for key in keys]
        predicted = [sim_cells[key][0] for key in keys]
        fp = sum(p >= 0.5 and a < 0.5 for a, p in zip(actual, predicted, strict=True))
        fn = sum(p < 0.5 and a >= 0.5 for a, p in zip(actual, predicted, strict=True))
        calibration = []
        for low, high in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
            band = [
                (a, p)
                for a, p in zip(actual, predicted, strict=True)
                if low <= p < high
            ]
            calibration.append(
                {
                    "low": low,
                    "high": min(high, 1.0),
                    "cells": len(band),
                    "predicted_mean": (
                        statistics.mean(p for _a, p in band) if band else None
                    ),
                    "actual_mean": statistics.mean(a for a, _p in band) if band else None,
                }
            )
        seed_rows.append(
            {
                "seed": seed,
                "paired_cells": len(keys),
                "binary_agreement": statistics.mean(
                    float((a >= 0.5) == (p >= 0.5))
                    for a, p in zip(actual, predicted, strict=True)
                ),
                "mae": statistics.mean(
                    abs(a - p) for a, p in zip(actual, predicted, strict=True)
                ),
                "false_positive_rate": fp / len(keys),
                "false_negative_rate": fn / len(keys),
                "spearman": _rank_corr(
                    {str(index): value for index, value in enumerate(actual)},
                    {str(index): value for index, value in enumerate(predicted)},
                    "spearman",
                ),
                "calibration": calibration,
            }
        )
        real_quality = {
            model: statistics.mean(
                value[0] for key, value in real_cells.items() if key[1] == model
            )
            for model in real.model_names()
        }
        sim_quality = {
            model: statistics.mean(
                value[0] for key, value in sim_cells.items() if key[1] == model
            )
            for model in simulated[seed].model_names()
        }
        real_cost = {
            model: statistics.mean(
                value[1] for key, value in real_cells.items() if key[1] == model
            )
            for model in real.model_names()
        }
        sim_cost = {
            model: statistics.mean(
                value[1] for key, value in sim_cells.items() if key[1] == model
            )
            for model in simulated[seed].model_names()
        }
        model_rows.append(
            {
                "seed": seed,
                "quality_spearman": _rank_corr(real_quality, sim_quality, "spearman"),
                "quality_kendall": _rank_corr(real_quality, sim_quality, "kendall"),
                "cost_spearman": _rank_corr(real_cost, sim_cost, "spearman"),
                "best_single_agreement": max(
                    real_quality, key=lambda model: real_quality[model]
                )
                == max(sim_quality, key=lambda model: sim_quality[model]),
                "real_quality": real_quality,
                "simulated_quality": sim_quality,
            }
        )
    return {
        "benchmark": benchmark,
        "cell": {
            "by_seed": seed_rows,
            "binary_agreement_mean": statistics.mean(
                _number(row["binary_agreement"]) for row in seed_rows
            ),
            "mae_mean": statistics.mean(_number(row["mae"]) for row in seed_rows),
            "false_positive_rate_mean": statistics.mean(
                _number(row["false_positive_rate"]) for row in seed_rows
            ),
            "false_negative_rate_mean": statistics.mean(
                _number(row["false_negative_rate"]) for row in seed_rows
            ),
        },
        "model": {
            "by_seed": model_rows,
            "best_single_agreement_share": statistics.mean(
                float(bool(row["best_single_agreement"])) for row in model_rows
            ),
        },
    }


def _gate(decision: RoutingDecision) -> str | None:
    return decision.evidence.gate if decision.evidence else None


def _route_rows(
    benchmark: str,
    real: OutcomeMatrix,
    simulated: dict[int, OutcomeMatrix],
    freeze_dir: Path,
    ground_analysis: Path,
    sim_analysis: Path,
    embedder: CachedEmbedder,
    groups: dict[str, str],
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    primary_rows = []
    decision_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    real_cells = _cells(real)
    for seed in SEEDS:
        heldout = _split(freeze_dir / "splits" / f"seed-{seed}.json", benchmark)
        for dial in DIALS:
            arm = f"dial-{dial:g}"
            ground_policy = RoutingPolicy.load(
                ground_analysis
                / "policies"
                / benchmark
                / f"seed-{seed}"
                / "semantic-3072"
                / f"{arm}.json"
            )
            sim_policy = RoutingPolicy.load(
                sim_analysis
                / "policies"
                / benchmark
                / f"seed-{seed}"
                / "semantic-3072"
                / f"{arm}.json"
            )
            ground_decisions = route_scenarios(
                ground_policy, real, heldout, embedder=embedder
            )
            sim_decisions = route_scenarios(
                sim_policy, simulated[seed], heldout, embedder=embedder
            )
            ground_eval = evaluate_policy(
                ground_policy, real, heldout, embedder=embedder
            )
            sim_on_real = evaluate_policy(sim_policy, real, heldout, embedder=embedder)
            baseline = ground_policy.guard_model
            if baseline is None:
                raise ValueError(f"ground policy for {benchmark} seed {seed} has no guard model")
            baseline_rewards = [
                real_cells[(scenario_id, baseline)][0] for scenario_id in heldout
            ]
            baseline_costs = [
                real_cells[(scenario_id, baseline)][1] for scenario_id in heldout
            ]
            decision_cells = _cells(simulated[seed])
            decision_baseline = sim_policy.guard_model
            if decision_baseline is None:
                raise ValueError(
                    f"simulated policy for {benchmark} seed {seed} has no guard model"
                )
            per_scenario: list[dict[str, object]] = []
            for scenario_id in heldout:
                ground = ground_decisions[scenario_id]
                predicted = sim_decisions[scenario_id]
                predicted_key = (scenario_id, predicted.model)
                baseline_key = (scenario_id, decision_baseline)
                if predicted_key not in decision_cells or baseline_key not in decision_cells:
                    continue
                predicted_reward, predicted_cost = decision_cells[predicted_key]
                base_reward, base_cost = decision_cells[baseline_key]
                per_scenario.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "dial": dial,
                        "scenario_id": scenario_id,
                        "group": groups[scenario_id],
                        "real_route_model": ground.model,
                        "sim_route_model": predicted.model,
                        "routed_reward": predicted_reward,
                        "baseline_reward": base_reward,
                        "routed_cost": predicted_cost,
                        "baseline_cost": base_cost,
                    }
                )
            decision_rows[arm].extend(per_scenario)
            if dial == 0.25:
                primary_rows.append(
                    {
                        "seed": seed,
                        "simulated_decision_gradeable": len(per_scenario),
                        "simulated_decision_coverage": len(per_scenario) / len(heldout),
                        "selected_model_agreement": statistics.mean(
                            float(ground_decisions[sid].model == sim_decisions[sid].model)
                            for sid in heldout
                        ),
                        "guard_gate_agreement": statistics.mean(
                            float(_gate(ground_decisions[sid]) == _gate(sim_decisions[sid]))
                            for sid in heldout
                        ),
                        "route_away_agreement": statistics.mean(
                            float(
                                (ground_decisions[sid].model != ground_policy.guard_model)
                                == (sim_decisions[sid].model != sim_policy.guard_model)
                            )
                            for sid in heldout
                        ),
                        "real_router_reward": ground_eval.accuracy,
                        "sim_selected_realized_reward": sim_on_real.accuracy,
                        "quality_consequence_points": 100
                        * (sim_on_real.accuracy - ground_eval.accuracy),
                        "real_router_cost": ground_eval.cost_per_scenario,
                        "sim_selected_realized_cost": sim_on_real.cost_per_scenario,
                        "cost_consequence_percent": 100
                        * (
                            sim_on_real.cost_per_scenario / ground_eval.cost_per_scenario - 1
                        ),
                        "baseline_reward": statistics.mean(baseline_rewards),
                        "baseline_cost": statistics.mean(baseline_costs),
                    }
                )
    return (
        {
            "by_seed": primary_rows,
            "selected_model_agreement_mean": statistics.mean(
                _number(row["selected_model_agreement"]) for row in primary_rows
            ),
            "guard_gate_agreement_mean": statistics.mean(
                _number(row["guard_gate_agreement"]) for row in primary_rows
            ),
            "route_away_agreement_mean": statistics.mean(
                _number(row["route_away_agreement"]) for row in primary_rows
            ),
            "simulated_decision_coverage_mean": statistics.mean(
                _number(row["simulated_decision_coverage"]) for row in primary_rows
            ),
            "quality_consequence_points_mean": statistics.mean(
                _number(row["quality_consequence_points"]) for row in primary_rows
            ),
            "cost_consequence_percent_mean": statistics.mean(
                _number(row["cost_consequence_percent"]) for row in primary_rows
            ),
        },
        decision_rows,
    )


def _bootstrap(rows: list[dict[str, object]], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = []
    by_benchmark_seed: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        by_benchmark_seed[(str(row["benchmark"]), int(_number(row["seed"])))][
            str(row["group"])
        ].append(row)
    benchmarks = sorted({benchmark for benchmark, _seed in by_benchmark_seed})
    for _ in range(BOOTSTRAP_DRAWS):
        benchmark_values = []
        for benchmark in benchmarks:
            seed_values = []
            for seed_index in SEEDS:
                clusters = by_benchmark_seed[(benchmark, seed_index)]
                names = sorted(clusters)
                sampled_names = [rng.choice(names) for _name in names]
                sampled = [
                    row
                    for name in sampled_names
                    for row in clusters[name]
                ]
                seed_values.append(
                    100
                    * statistics.mean(
                        _number(row["routed_reward"]) - _number(row["baseline_reward"])
                        for row in sampled
                    )
                )
            benchmark_values.append(statistics.mean(seed_values))
        draws.append(statistics.mean(benchmark_values))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _decision(
    rows_by_benchmark: dict[str, dict[str, list[dict[str, object]]]],
) -> dict[str, object]:
    dial_rows = {
        arm: [
            row
            for benchmark_rows in rows_by_benchmark.values()
            for row in benchmark_rows[arm]
        ]
        for arm in (f"dial-{dial:g}" for dial in DIALS)
    }
    points = []
    for dial in DIALS:
        arm = f"dial-{dial:g}"
        rows = dial_rows[arm]
        by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
        by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_benchmark[str(row["benchmark"])].append(row)
            by_seed[int(_number(row["seed"]))].append(row)
        benchmark_quality = {
            benchmark: 100
            * statistics.mean(
                _number(row["routed_reward"]) - _number(row["baseline_reward"])
                for row in selected
            )
            for benchmark, selected in by_benchmark.items()
        }
        benchmark_relative = {
            benchmark: 100
            * (
                statistics.mean(_number(row["routed_reward"]) for row in selected)
                / statistics.mean(_number(row["baseline_reward"]) for row in selected)
                - 1
            )
            if statistics.mean(_number(row["baseline_reward"]) for row in selected) > 0
            else 0.0
            for benchmark, selected in by_benchmark.items()
        }
        quality = statistics.mean(benchmark_quality.values())
        cost = statistics.mean(
            100
            * (
                statistics.mean(_number(row["routed_cost"]) for row in selected)
                / statistics.mean(_number(row["baseline_cost"]) for row in selected)
                - 1
            )
            for selected in by_benchmark.values()
        )
        ci = _bootstrap(rows, seed=91_000 + round(dial * 100))
        seed_gates = []
        for selected in by_seed.values():
            selected_by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
            for row in selected:
                selected_by_benchmark[str(row["benchmark"])].append(row)
            seed_quality = statistics.mean(
                statistics.mean(
                    _number(row["routed_reward"]) - _number(row["baseline_reward"])
                    for row in benchmark_rows
                )
                for benchmark_rows in selected_by_benchmark.values()
            )
            seed_cost = statistics.mean(
                statistics.mean(_number(row["routed_cost"]) for row in benchmark_rows)
                / statistics.mean(_number(row["baseline_cost"]) for row in benchmark_rows)
                - 1
                for benchmark_rows in selected_by_benchmark.values()
            )
            seed_gates.append(seed_quality >= 0 and seed_cost < 0)
        gates = {
            "pooled_quality_nonnegative": quality >= 0,
            "ci_lower_noninferior": ci[0] >= -0.5,
            "pooled_cost_lower": cost < 0,
            "benchmark_loss_limits": all(
                benchmark_quality[name] > -5 and benchmark_relative[name] > -10
                for name in benchmark_quality
            ),
            "all_seed_point_gates": all(seed_gates),
        }
        points.append(
            {
                "dial": dial,
                "quality_points": quality,
                "cost_percent": cost,
                "quality_ci95": list(ci),
                "benchmark_quality_points": benchmark_quality,
                "benchmark_relative_quality_percent": benchmark_relative,
                "gates": gates,
                "passes": all(gates.values()),
            }
        )
    passing = [point for point in points if point["passes"]]
    selected = min(passing, key=lambda point: _number(point["cost_percent"])) if passing else None
    return {
        "points": points,
        "promotes": selected is not None,
        "selected_dial": selected["dial"] if selected else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--real", action="append", nargs=2, metavar=("BENCHMARK", "MATRIX"))
    parser.add_argument(
        "--sim",
        action="append",
        nargs=3,
        metavar=("BENCHMARK", "SEED", "MATRIX"),
    )
    parser.add_argument("--ground-analysis", type=Path, required=True)
    parser.add_argument("--sim-analysis", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cell-only",
        action="store_true",
        help=(
            "write cell and model agreement without fitting or replaying simulated policies; "
            "use this fail-closed path when simulated coverage cannot support a routing decision"
        ),
    )
    args = parser.parse_args()

    real = {
        benchmark: _canonical(benchmark, OutcomeMatrix.load(Path(path)))
        for benchmark, path in args.real
    }
    simulated: dict[str, dict[int, OutcomeMatrix]] = defaultdict(dict)
    for benchmark, raw_seed, path in args.sim:
        simulated[benchmark][int(raw_seed)] = _canonical(
            benchmark, OutcomeMatrix.load(Path(path))
        )
    manifest_names = {
        "routerbench": "routerbench.json",
        "tau2": "tau2.json",
        "terminal_bench_2": "terminal_bench_2.json",
    }
    benchmarks = {}
    decision_ground_rows: dict[str, dict[str, list[dict[str, object]]]] = {}
    decision_sim_rows: dict[str, dict[str, list[dict[str, object]]]] = {}
    for benchmark, matrix in real.items():
        if set(simulated[benchmark]) != set(SEEDS):
            raise ValueError(f"{benchmark} simulated matrices do not cover seeds 0..4")
        groups = _groups(
            args.freeze_dir / "tasks" / manifest_names[benchmark]
        )
        result = _cell_and_model(benchmark, matrix, simulated[benchmark])
        if args.cell_only:
            benchmarks[benchmark] = result
            continue
        cache_path = args.embedding_cache / benchmark
        if not (cache_path / "meta.json").is_file():
            ground_report = _dict(
                _json(args.ground_analysis / f"{benchmark}.json")
            )
            cache_path = Path(
                str(_dict(ground_report["embedding_cache"])["path"])
            )
        embedder = CachedEmbedder(cache_path)
        route, ground_rows = _route_rows(
            benchmark,
            matrix,
            {seed: matrix for seed in SEEDS},
            args.freeze_dir,
            args.ground_analysis,
            args.ground_analysis,
            embedder,
            groups,
        )
        sim_route, sim_rows = _route_rows(
            benchmark,
            matrix,
            simulated[benchmark],
            args.freeze_dir,
            args.ground_analysis,
            args.sim_analysis,
            embedder,
            groups,
        )
        result["routing_ground_self_check"] = route
        result["routing_world_model"] = sim_route
        benchmarks[benchmark] = result
        decision_ground_rows[benchmark] = ground_rows
        decision_sim_rows[benchmark] = sim_rows
    if args.cell_only:
        result = {
            "mode": "cell-only",
            "benchmarks": benchmarks,
            "ground_truth_decision": None,
            "world_model_decision": None,
            "promotion_decision_agreement": None,
            "selected_operating_point_agreement": None,
        }
        write = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        write_text_atomic(args.out, write)
        return 0
    ground_decision = _decision(decision_ground_rows)
    simulated_decision = _decision(decision_sim_rows)
    result = {
        "benchmarks": benchmarks,
        "ground_truth_decision": ground_decision,
        "world_model_decision": simulated_decision,
        "promotion_decision_agreement": ground_decision["promotes"]
        == simulated_decision["promotes"],
        "selected_operating_point_agreement": ground_decision["selected_dial"]
        == simulated_decision["selected_dial"],
    }
    write = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_text_atomic(args.out, write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
