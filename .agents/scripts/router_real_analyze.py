"""Paired five-seed analysis for the real router reproduction matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from wmo.optimize.knn import apply_cost_quality, best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec, RoutingDecision
from wmo.optimize.routing import route_scenarios
from wmo.retrieval.embedders import HashingEmbedder

SEEDS = range(5)
DIALS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_DRAWS = 10_000


class CachedEmbedder:
    """Serve a fixed task corpus from a paid embedding cache."""

    def __init__(self, texts: list[str], vectors: np.ndarray) -> None:
        self._vectors = {
            text: vectors[index].astype(np.float64) for index, text in enumerate(texts)
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text].tolist() for text in texts]


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(key): item for key, item in value.items()}


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


def _task_map(matrix: OutcomeMatrix) -> dict[str, str]:
    tasks: dict[str, str] = {}
    for row in matrix.outcomes:
        tasks.setdefault(row.scenario_id, row.task)
    return tasks


def _cached_semantic_embedder(
    matrix: OutcomeMatrix,
    cache_dir: Path,
) -> tuple[EmbedderSpec, CachedEmbedder, dict[str, object]]:
    tasks = sorted(set(_task_map(matrix).values()))
    digest = hashlib.sha256("\0".join(tasks).encode()).hexdigest()
    meta_path = cache_dir / "meta.json"
    vectors_path = cache_dir / "vectors.npz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = EmbedderSpec(
        kind="openai",
        dim=3072,
        deployment="text-embedding-3-large",
        api_key_env="OPENAI_API_KEY",
    )
    cache_hit = False
    if meta_path.is_file() and vectors_path.is_file():
        meta = _json(meta_path)
        cache_hit = meta.get("task_digest") == digest and meta.get("tasks") == tasks
    if cache_hit:
        with np.load(vectors_path) as data:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
    else:
        vectors = np.asarray(spec.build().embed(tasks), dtype=np.float32)
        np.savez_compressed(vectors_path, vectors=vectors)
        meta_path.write_text(
            json.dumps(
                {
                    "backend": "openai",
                    "model": spec.deployment,
                    "dimensions": spec.dim,
                    "task_digest": digest,
                    "tasks": tasks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if vectors.shape != (len(tasks), spec.dim):
        raise ValueError(
            f"embedding cache shape {vectors.shape} != expected {(len(tasks), spec.dim)}"
        )
    return spec, CachedEmbedder(tasks, vectors), {
        "cache_hit": cache_hit,
        "tasks": len(tasks),
        "dimensions": spec.dim,
        "task_digest": digest,
        "path": str(cache_dir),
    }


def _hashing_embedder() -> tuple[EmbedderSpec, HashingEmbedder]:
    return EmbedderSpec(kind="hashing", dim=1024), HashingEmbedder(dim=1024)


def _cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], dict[str, object]]:
    grouped: dict[tuple[str, str], list[ScenarioOutcome]] = defaultdict(list)
    for row in matrix.outcomes:
        if row.reward is not None:
            grouped[(row.scenario_id, row.model)].append(row)
    cells: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows in grouped.items():
        call_seconds = [value for row in rows for value in row.call_seconds]
        cells[key] = {
            "reward": statistics.mean(float(row.reward) for row in rows if row.reward is not None),
            "cost": statistics.mean(row.cost_usd for row in rows),
            "latency": statistics.mean(
                sum(row.call_seconds) if row.call_seconds else row.wall_seconds for row in rows
            ),
            "call_p95": (
                float(np.percentile(call_seconds, 95)) if call_seconds else None
            ),
            "steps": statistics.mean(row.steps for row in rows),
            "tool_calls": statistics.mean(row.tool_calls for row in rows),
            "completion": statistics.mean(
                1.0
                if row.completion_status.lower() in {"completed", "success", "submitted"}
                or row.success
                else 0.0
                for row in rows
            ),
        }
    return cells


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _evaluate_choices(
    ids: list[str],
    choices: dict[str, str],
    cells: dict[tuple[str, str], dict[str, object]],
    groups: dict[str, str],
    *,
    baseline: str,
    decisions: dict[str, RoutingDecision] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for scenario_id in ids:
        model = choices[scenario_id]
        cell = cells.get((scenario_id, model))
        if cell is None:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "group": groups[scenario_id],
                    "model": model,
                    "reward": None,
                    "cost": None,
                }
            )
            continue
        decision = decisions.get(scenario_id) if decisions else None
        evidence = decision.evidence if decision else None
        rows.append(
            {
                "scenario_id": scenario_id,
                "group": groups[scenario_id],
                "model": model,
                **cell,
                "route_away": model != baseline,
                "guard_reversion": bool(evidence and evidence.gate == "reverted"),
                "novelty_abstain": bool(evidence and evidence.gate == "novelty-abstain"),
                "fallback_forced": bool(
                    evidence and evidence.propensity == "fallback-forced"
                ),
            }
        )
    scored = [row for row in rows if row["reward"] is not None]
    rewards = [_number(row["reward"]) for row in scored]
    costs = [_number(row["cost"]) for row in scored]
    latencies = [_number(row["latency"]) for row in scored]
    successes = sum(value >= 0.5 for value in rewards)
    mix = {
        model: sum(row["model"] == model for row in rows) / len(rows)
        for model in sorted(set(str(row["model"]) for row in rows))
    }
    metrics: dict[str, object] = {
        "scenarios": len(rows),
        "gradeable": len(scored),
        "gradeability": len(scored) / len(rows),
        "reward": statistics.mean(rewards),
        "success_rate": successes / len(scored),
        "cost_per_task": statistics.mean(costs),
        "total_cost": sum(costs),
        "effective_cost_per_success": sum(costs) / successes if successes else None,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "steps_mean": statistics.mean(_number(row["steps"]) for row in scored),
        "tool_calls_mean": statistics.mean(_number(row["tool_calls"]) for row in scored),
        "completion_rate": statistics.mean(_number(row["completion"]) for row in scored),
        "model_mix": mix,
        "route_away_share": statistics.mean(
            float(bool(row.get("route_away"))) for row in scored
        ),
        "guard_reversion_share": statistics.mean(
            float(bool(row.get("guard_reversion"))) for row in scored
        ),
        "novelty_abstain_share": statistics.mean(
            float(bool(row.get("novelty_abstain"))) for row in scored
        ),
        "fallback_forced_share": statistics.mean(
            float(bool(row.get("fallback_forced"))) for row in scored
        ),
    }
    return metrics, rows


def _single_choice(ids: list[str], model: str) -> dict[str, str]:
    return dict.fromkeys(ids, model)


def _mean_fit_cost(
    matrix: OutcomeMatrix,
    ids: list[str],
    model: str,
) -> float:
    wanted = set(ids)
    costs = [
        row.cost_usd
        for row in matrix.outcomes
        if row.scenario_id in wanted and row.model == model and row.reward is not None
    ]
    return statistics.mean(costs)


def _cheapest_on_fit(matrix: OutcomeMatrix, ids: list[str]) -> str:
    return min(matrix.model_names(), key=lambda model: (_mean_fit_cost(matrix, ids, model), model))


def _oracle_choices(
    ids: list[str],
    models: list[str],
    cells: dict[tuple[str, str], dict[str, object]],
) -> dict[str, str]:
    return {
        scenario_id: min(
            (model for model in models if (scenario_id, model) in cells),
            key=lambda model: (
                -_number(cells[(scenario_id, model)]["reward"]),
                _number(cells[(scenario_id, model)]["cost"]),
                model,
            ),
        )
        for scenario_id in ids
    }


def _paired_delta(
    routed: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, float]:
    return {
        "quality_points": 100
        * (_number(routed["reward"]) - _number(baseline["reward"])),
        "cost_percent": 100
        * (_number(routed["cost_per_task"]) / _number(baseline["cost_per_task"]) - 1),
        "effective_cost_per_success_percent": (
            100
            * (
                _number(routed["effective_cost_per_success"])
                / _number(baseline["effective_cost_per_success"])
                - 1
            )
            if routed["effective_cost_per_success"] is not None
            and baseline["effective_cost_per_success"] is not None
            else math.nan
        ),
    }


def _bootstrap(
    paired: list[tuple[list[dict[str, object]], list[dict[str, object]]]],
    *,
    seed: int,
) -> dict[str, object]:
    """Hierarchical paired cluster bootstrap over split seeds and benchmark groups."""
    rng = random.Random(seed)
    draws_q: list[float] = []
    draws_c: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled_seed_indexes = [rng.randrange(len(paired)) for _ in paired]
        seed_q: list[float] = []
        seed_c: list[float] = []
        for seed_index in sampled_seed_indexes:
            routed, baseline = paired[seed_index]
            route_by_id = {str(row["scenario_id"]): row for row in routed}
            base_by_id = {str(row["scenario_id"]): row for row in baseline}
            by_group: dict[str, list[str]] = defaultdict(list)
            for row in routed:
                if (
                    row["reward"] is not None
                    and base_by_id[str(row["scenario_id"])]["reward"] is not None
                ):
                    by_group[str(row["group"])].append(str(row["scenario_id"]))
            group_names = sorted(by_group)
            sampled_groups = [rng.choice(group_names) for _ in group_names]
            sampled_ids = [
                scenario_id
                for group in sampled_groups
                for scenario_id in (
                    rng.choice(by_group[group]) for _ in by_group[group]
                )
            ]
            route_rewards = [_number(route_by_id[sid]["reward"]) for sid in sampled_ids]
            base_rewards = [_number(base_by_id[sid]["reward"]) for sid in sampled_ids]
            route_costs = [_number(route_by_id[sid]["cost"]) for sid in sampled_ids]
            base_costs = [_number(base_by_id[sid]["cost"]) for sid in sampled_ids]
            seed_q.append(100 * (statistics.mean(route_rewards) - statistics.mean(base_rewards)))
            seed_c.append(100 * (statistics.mean(route_costs) / statistics.mean(base_costs) - 1))
        draws_q.append(statistics.mean(seed_q))
        draws_c.append(statistics.mean(seed_c))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "method": "hierarchical paired cluster bootstrap: seeds then benchmark task groups",
        "quality_points_ci95": [_percentile(draws_q, 2.5), _percentile(draws_q, 97.5)],
        "cost_percent_ci95": [_percentile(draws_c, 2.5), _percentile(draws_c, 97.5)],
        "quality_probability_gt_zero": sum(value > 0 for value in draws_q) / len(draws_q),
        "cost_probability_lt_zero": sum(value < 0 for value in draws_c) / len(draws_c),
    }


def _arm_summary(
    seed_rows: list[dict[str, object]],
    paired_rows: list[tuple[list[dict[str, object]], list[dict[str, object]]]],
    *,
    bootstrap_seed: int,
) -> dict[str, object]:
    deltas = [_dict(row["delta"]) for row in seed_rows]
    quality = [_number(delta["quality_points"]) for delta in deltas]
    cost = [_number(delta["cost_percent"]) for delta in deltas]
    return {
        "quality_points_mean": statistics.mean(quality),
        "quality_points_sem": statistics.stdev(quality) / len(quality) ** 0.5,
        "cost_percent_mean": statistics.mean(cost),
        "cost_percent_sem": statistics.stdev(cost) / len(cost) ** 0.5,
        "quality_wins": sum(value > 0 for value in quality),
        "cost_wins": sum(value < 0 for value in cost),
        "joint_wins": sum(q >= 0 and c < 0 for q, c in zip(quality, cost, strict=True)),
        "bootstrap": _bootstrap(paired_rows, seed=bootstrap_seed),
    }


def analyze(
    benchmark: str,
    matrix: OutcomeMatrix,
    split_dir: Path,
    task_manifest: Path,
    out_dir: Path,
    cache_root: Path,
) -> dict[str, object]:
    task_manifest_data = _json(task_manifest)
    groups = {}
    for raw_row in _list(task_manifest_data["tasks"]):
        row = _dict(raw_row)
        groups[str(row["task_id"])] = str(row["group"])
    missing_groups = sorted(set(matrix.scenario_ids()) - set(groups))
    if missing_groups:
        raise ValueError(f"task manifest lacks groups for {missing_groups[:5]}")
    cells = _cells(matrix)
    semantic_spec, semantic_embedder, cache_info = _cached_semantic_embedder(
        matrix, cache_root / benchmark
    )
    hashing_spec, hashing_embedder = _hashing_embedder()
    representations = {
        "semantic-3072": (semantic_spec, semantic_embedder),
        "hashing-1024": (hashing_spec, hashing_embedder),
    }
    seed_results: list[dict[str, object]] = []
    paired_by_arm: dict[str, list[tuple[list[dict[str, object]], list[dict[str, object]]]]] = (
        defaultdict(list)
    )
    for seed in SEEDS:
        split = _dict(_json(split_dir / f"seed-{seed}.json")[benchmark])
        fit_ids = [str(value) for value in _list(split["fit"])]
        heldout_ids = [str(value) for value in _list(split["heldout"])]
        baseline = best_single_on_fit(matrix, fit_ids)
        baseline_metrics, baseline_rows = _evaluate_choices(
            heldout_ids,
            _single_choice(heldout_ids, baseline),
            cells,
            groups,
            baseline=baseline,
        )
        arms: dict[str, dict[str, object]] = {}
        static_models = {}
        for model in matrix.model_names():
            metrics, _ = _evaluate_choices(
                heldout_ids,
                _single_choice(heldout_ids, model),
                cells,
                groups,
                baseline=baseline,
            )
            static_models[model] = metrics
        auxiliary = {
            "cheapest-single": _single_choice(
                heldout_ids, _cheapest_on_fit(matrix, fit_ids)
            ),
            "seeded-random": {
                scenario_id: random.Random(seed * 1_000_003 + index).choice(matrix.model_names())
                for index, scenario_id in enumerate(heldout_ids)
            },
            "oracle-upper-bound": _oracle_choices(
                heldout_ids, matrix.model_names(), cells
            ),
        }
        for name, choices in auxiliary.items():
            metrics, rows = _evaluate_choices(
                heldout_ids, choices, cells, groups, baseline=baseline
            )
            arms[name] = {
                "metrics": metrics,
                "delta": _paired_delta(metrics, baseline_metrics),
            }
            paired_by_arm[name].append((rows, baseline_rows))
        for representation, (spec, embedder) in representations.items():
            with tempfile.TemporaryDirectory() as temp:
                strict = fit_knn_policy(
                    matrix,
                    bank_path=Path(temp) / KNN_BANK_FILENAME,
                    fit_ids=fit_ids,
                    embedder=spec,
                    embed_with=embedder,
                    guard_model=baseline,
                    rag_num=50,
                    rag_thres=0.95,
                    z=0.5,
                    min_pairs=8,
                    se_floor=True,
                    floor_q=0.05,
                    fitted_from=f"{benchmark} seed={seed} representation={representation}",
                )
                policies = {
                    "unguarded": strict.model_copy(
                        update={
                            "knn_z": 0.0,
                            "knn_min_pairs": 0,
                            "se_floor": False,
                            "floor_sim": None,
                            "floor_q": 0.0,
                        }
                    ),
                    **{
                        f"dial-{dial:g}": apply_cost_quality(strict, dial)
                        for dial in DIALS
                    },
                }
                for arm, policy in policies.items():
                    policy.attach_bank(strict.knn_bank())
                    decisions = route_scenarios(
                        policy, matrix, heldout_ids, embedder=embedder
                    )
                    metrics, rows = _evaluate_choices(
                        heldout_ids,
                        {
                            scenario_id: decision.model
                            for scenario_id, decision in decisions.items()
                        },
                        cells,
                        groups,
                        baseline=baseline,
                        decisions=decisions,
                    )
                    arm_name = f"{representation}/{arm}"
                    arms[arm_name] = {
                        "metrics": metrics,
                        "delta": _paired_delta(metrics, baseline_metrics),
                    }
                    paired_by_arm[arm_name].append((rows, baseline_rows))
        seed_results.append(
            {
                "seed": seed,
                "fit_scenarios": len(fit_ids),
                "heldout_scenarios": len(heldout_ids),
                "baseline": baseline,
                "baseline_metrics": baseline_metrics,
                "static_models": static_models,
                "arms": arms,
            }
        )
    summaries = {}
    for arm, paired in sorted(paired_by_arm.items()):
        arm_seed_rows = []
        for seed_result in seed_results:
            seed_arms_obj = _dict(seed_result["arms"])
            arm_result = _dict(seed_arms_obj[arm])
            arm_seed_rows.append(
                {"seed": seed_result["seed"], "delta": arm_result["delta"]}
            )
        summaries[arm] = _arm_summary(
            arm_seed_rows,
            paired,
            bootstrap_seed=int(hashlib.sha256(f"{benchmark}:{arm}".encode()).hexdigest()[:8], 16),
        )
    primary = summaries["semantic-3072/dial-0.25"]
    primary_bootstrap = _dict(primary["bootstrap"])
    quality_ci = _list(primary_bootstrap["quality_points_ci95"])
    cost_ci = _list(primary_bootstrap["cost_percent_ci95"])
    promotion = {
        "nonnegative_quality": _number(primary["quality_points_mean"]) >= 0,
        "lower_cost": _number(primary["cost_percent_mean"]) < 0,
        "quality_wins_5_of_5": _number(primary["quality_wins"]) == 5,
        "joint_wins_5_of_5": _number(primary["joint_wins"]) == 5,
        "bootstrap_quality_nonnegative": _number(quality_ci[0]) >= 0,
        "bootstrap_cost_negative": _number(cost_ci[1]) < 0,
    }
    result = {
        "benchmark": benchmark,
        "matrix_scenarios": len(matrix.scenario_ids()),
        "models": matrix.model_names(),
        "split_seeds": list(SEEDS),
        "baseline_definition": "best mean-reward single model on fit only, ties to lower cost",
        "primary_arm": "semantic-3072/dial-0.25",
        "embedding_cache": cache_info,
        "seed_results": seed_results,
        "summaries": summaries,
        "promotion": promotion,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{benchmark}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--matrix", action="append", nargs=2, metavar=("BENCHMARK", "PATH"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    args = parser.parse_args()
    manifest_names = {
        "routerbench": "routerbench.json",
        "tau2": "tau2.json",
        "terminal_bench_2": "terminal_bench_2.json",
    }
    results: dict[str, dict[str, object]] = {}
    for benchmark, value in args.matrix:
        matrix = OutcomeMatrix.load(Path(value))
        results[benchmark] = analyze(
            benchmark,
            matrix,
            args.freeze_dir / "splits",
            args.freeze_dir / "tasks" / manifest_names[benchmark],
            args.out_dir,
            args.embedding_cache,
        )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                name: {
                    "primary": _dict(result["summaries"])[str(result["primary_arm"])],
                    "promotion": result["promotion"],
                }
                for name, result in results.items()
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
