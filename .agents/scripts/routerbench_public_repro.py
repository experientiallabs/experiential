"""Reproduce guarded kNN routing on a public RouterBench slice.

This is an artifact-recovery fallback, not the promoted ours9 gate. It loads the public
withmartian/routerbench pickle, samples a fixed 1,199-scenario cohort, and compares the current
production kNN path with the fit-selected best single model over five paired heldout splits.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import tempfile
from pathlib import Path
from typing import TypedDict, cast

import pandas as pd

from wmo.optimize.knn import apply_cost_quality, best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec
from wmo.optimize.routing import evaluate_policy
from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry
from wmo.retrieval.embedders import HashingEmbedder

logger = logging.getLogger("routerbench-public-repro")

MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]
SEEDS = [0, 1, 2, 3, 4]
DIALS = [0.0, 0.25, 0.5, 0.75, 1.0]


class RunRow(TypedDict):
    """One arm on one paired split seed."""

    seed: int
    arm: str
    fit_scenarios: int
    heldout_scenarios: int
    baseline: str
    baseline_reward: float
    baseline_cost: float
    router_reward: float
    router_cost: float
    quality_delta: float
    cost_delta: float
    routed_share: float
    model_mix: dict[str, float]


class ArmSummary(TypedDict):
    """Five-seed aggregate for one arm."""

    quality_delta: dict[str, float]
    cost_delta: dict[str, float]
    seed_wins: int
    routed_share: dict[str, float]


def load_matrix(path: Path, sample: int, sample_seed: int) -> OutcomeMatrix:
    """Load the public pickle into the current OutcomeMatrix contract."""
    frame = cast("pd.DataFrame", pd.read_pickle(path))
    if sample < len(frame):
        frame = frame.sample(n=sample, random_state=sample_seed).sort_index()
    pool = [
        PoolEntry(
            name=name,
            kind=ProviderKind.OPENAI,
            model=name,
            tier="frontier" if name == "gpt-4-1106-preview" else "open",
            input_per_mtok=0.0,
            output_per_mtok=0.0,
        )
        for name in MODELS
    ]
    outcomes: list[ScenarioOutcome] = []
    for row in frame.to_dict("records"):
        scenario_id = f"{row['eval_name']}:{row['sample_id']}"
        for model in MODELS:
            reward = float(row[model])
            outcomes.append(
                ScenarioOutcome(
                    scenario_id=scenario_id,
                    task=str(row["prompt"]),
                    model=model,
                    reward=reward,
                    success=reward >= 0.5,
                    steps=1,
                    stop_reason="routerbench-public",
                    cost_usd=float(row[f"{model}|total_cost"]),
                )
            )
    return OutcomeMatrix(pool=pool, outcomes=outcomes)


def stratified_split(matrix: OutcomeMatrix, seed: int) -> tuple[list[str], list[str]]:
    """Return the fixed 70/30 per-benchmark split used by the promotion research."""
    by_eval: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        by_eval.setdefault(scenario_id.split(":", 1)[0], []).append(scenario_id)
    rng = random.Random(seed)
    fit: list[str] = []
    heldout: list[str] = []
    for _, ids in sorted(by_eval.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = min(max(1, round(len(shuffled) * 0.7)), len(shuffled) - 1)
        fit.extend(shuffled[:cut])
        heldout.extend(shuffled[cut:])
    return sorted(fit), sorted(heldout)


def single_stats(matrix: OutcomeMatrix, model: str, scenario_ids: list[str]) -> tuple[float, float]:
    """Return scenario-mean reward and cost for one model."""
    wanted = set(scenario_ids)
    rewards = [
        float(row.reward)
        for row in matrix.outcomes
        if row.scenario_id in wanted and row.model == model and row.reward is not None
    ]
    costs = [
        row.cost_usd
        for row in matrix.outcomes
        if row.scenario_id in wanted and row.model == model and row.reward is not None
    ]
    return statistics.mean(rewards), statistics.mean(costs)


def summarize(values: list[float]) -> dict[str, float]:
    """Summarize a five-seed paired measurement."""
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values),
        "sem": statistics.stdev(values) / len(values) ** 0.5,
    }


def run(matrix: OutcomeMatrix) -> dict[str, object]:
    """Run guarded, unguarded, and dial arms over five paired splits."""
    embedder = HashingEmbedder(dim=1024)
    spec = EmbedderSpec(kind="hashing", dim=1024)
    rows: list[RunRow] = []
    for seed in SEEDS:
        fit_ids, heldout_ids = stratified_split(matrix, seed)
        baseline = best_single_on_fit(matrix, fit_ids)
        baseline_reward, baseline_cost = single_stats(matrix, baseline, heldout_ids)
        with tempfile.TemporaryDirectory() as temp:
            strict = fit_knn_policy(
                matrix,
                bank_path=Path(temp) / KNN_BANK_FILENAME,
                fit_ids=fit_ids,
                embedder=spec,
                embed_with=embedder,
                guard_model=baseline,
                z=0.5,
                min_pairs=8,
                se_floor=True,
                floor_q=0.05,
                fitted_from=f"public-routerbench sample1199 seed={seed}",
            )
            strict.knn_bank()
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
                **{f"dial-{dial:g}": apply_cost_quality(strict, dial) for dial in DIALS},
            }
            for name, policy in policies.items():
                policy.attach_bank(strict.knn_bank())
                result = evaluate_policy(policy, matrix, heldout_ids, embedder=embedder)
                rows.append(
                    {
                        "seed": seed,
                        "arm": name,
                        "fit_scenarios": len(fit_ids),
                        "heldout_scenarios": len(heldout_ids),
                        "baseline": baseline,
                        "baseline_reward": baseline_reward,
                        "baseline_cost": baseline_cost,
                        "router_reward": result.accuracy,
                        "router_cost": result.cost_per_scenario,
                        "quality_delta": result.accuracy - baseline_reward,
                        "cost_delta": result.cost_per_scenario / baseline_cost - 1.0,
                        "routed_share": 1.0 - result.model_mix.get(baseline, 0.0),
                        "model_mix": result.model_mix,
                    }
                )
    summaries: dict[str, ArmSummary] = {}
    for arm in ["unguarded", *[f"dial-{dial:g}" for dial in DIALS]]:
        selected = [row for row in rows if row["arm"] == arm]
        quality = [row["quality_delta"] for row in selected]
        cost = [row["cost_delta"] for row in selected]
        summaries[arm] = {
            "quality_delta": summarize(quality),
            "cost_delta": summarize(cost),
            "seed_wins": sum(value > 0.0 for value in quality),
            "routed_share": summarize([row["routed_share"] for row in selected]),
        }
    return {
        "cohort": {
            "source": "withmartian/routerbench routerbench_0shot.pkl",
            "sample": len(matrix.scenario_ids()),
            "models": len(MODELS),
            "sample_seed": 7,
            "split_seeds": SEEDS,
            "embedder": "hashing-1024",
            "note": "Public fallback cohort, not routerbench-ours9 and not its semantic embedder.",
        },
        "rows": rows,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=1199)
    parser.add_argument("--sample-seed", type=int, default=7)
    args = parser.parse_args()
    matrix = load_matrix(args.pickle, args.sample, args.sample_seed)
    result = run(matrix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summaries = cast("dict[str, ArmSummary]", result["summaries"])
    for arm, summary in summaries.items():
        quality = summary["quality_delta"]["mean"] * 100
        cost = summary["cost_delta"]["mean"] * 100
        routed = summary["routed_share"]["mean"] * 100
        logger.info(
            "%-12s quality %+.2f pt, cost %+.1f%%, routed %.1f%%, wins %d/5",
            arm,
            quality,
            cost,
            routed,
            summary["seed_wins"],
        )
    logger.info("result -> %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
