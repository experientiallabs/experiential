"""Sim2real gap benchmark: (model x domain) disagreement cells, dev/holdout split, metrics.

r3 mandate (2026-07-26): close the useful part of the sim-to-real gap. This module owns
step 1 (the benchmark) and the data plumbing for steps 2-4:

- Cells are (model, domain, cohort): real tasks and wm scenarios are DIFFERENT task sets
  (master's caveat), so the honest unit is the domain aggregate, not per-scenario pairs.
- Dev/holdout split: scenarios split ~50/50 within each (cohort, domain) stratum, seeded;
  every (model, domain) cell then has a dev half and an untouched holdout half. The split
  is registered in findings/r3.md (seed + id-list SHA256) BEFORE any diagnosis.
- Gap metrics (computed on dev and holdout separately, per cohort):
    per-model MAE   = mean over domains of |sim_mean(m,d) - real_mean(m,d)|
    rank agreement  = Spearman + Kendall between sim and real model means (9 models)
    spread ratio    = std(sim model means) / std(real model means)  (1.0 = faithful)

Subcommands: benchmark (build + register), cells (dump top-disagreement dev cells with
episodes for transcript reading), metrics (recompute on either split). All offline, $0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from wmh.optimize.outcomes import OutcomeMatrix

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r3.gap")

DATA = Path("~/Desktop/Projects/wmh-routing-data").expanduser()
OUT = DATA / "findings" / "r3_sim2real"
REAL = "tau-bench-real"
SIM_COHORTS = ["tau-bench", "tau-bench-s80"]
SPLIT_SEED = 20260726


def load(name: str) -> OutcomeMatrix:
    return OutcomeMatrix.load(DATA / "matrices" / f"{name}_matrix.json")


def domain_of(matrix_name: str, sid: str, task: str) -> str:
    if matrix_name == REAL:
        return sid.split(":", 1)[0]
    try:
        return json.loads(task)["domain"]
    except (json.JSONDecodeError, KeyError):
        return "unknown"


def scenario_domains(matrix: OutcomeMatrix, name: str) -> dict[str, str]:
    tasks: dict[str, str] = {}
    for o in matrix.outcomes:
        tasks.setdefault(o.scenario_id, o.task)
    return {sid: domain_of(name, sid, task) for sid, task in tasks.items()}


def split_scenarios(
    matrix: OutcomeMatrix, name: str, seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """50/50 dev/holdout, stratified by domain, deterministic in seed."""
    domains = scenario_domains(matrix, name)
    by_dom: dict[str, list[str]] = {}
    for sid, dom in sorted(domains.items()):
        by_dom.setdefault(dom, []).append(sid)
    rng = np.random.default_rng(seed)
    dev, hold = [], []
    for _dom, sids in sorted(by_dom.items()):
        perm = rng.permutation(len(sids))
        cut = len(sids) // 2
        dev.extend(sids[i] for i in perm[:cut])
        hold.extend(sids[i] for i in perm[cut:])
    return sorted(dev), sorted(hold)


def cell_means(
    matrix: OutcomeMatrix, name: str, ids: set[str]
) -> dict[tuple[str, str], tuple[float, int]]:
    """(model, domain) -> (mean reward, n episodes) over the given scenario ids."""
    domains = scenario_domains(matrix, name)
    acc: dict[tuple[str, str], list[float]] = {}
    for o in matrix.outcomes:
        if o.reward is None or o.scenario_id not in ids:
            continue
        acc.setdefault((o.model, domains[o.scenario_id]), []).append(o.reward)
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def gap_metrics(
    real_cells: dict, sim_cells: dict, models: list[str], domains: list[str]
) -> dict:
    mae = {}
    for m in models:
        errs = [
            abs(sim_cells[(m, d)][0] - real_cells[(m, d)][0])
            for d in domains
            if (m, d) in sim_cells and (m, d) in real_cells
        ]
        if errs:
            mae[m] = round(float(np.mean(errs)), 4)
    real_means = [
        np.mean([real_cells[(m, d)][0] for d in domains if (m, d) in real_cells])
        for m in models
    ]
    sim_means = [
        np.mean([sim_cells[(m, d)][0] for d in domains if (m, d) in sim_cells])
        for m in models
    ]
    rho, _ = spearmanr(sim_means, real_means)
    tau, _ = kendalltau(sim_means, real_means)
    spread = float(np.std(sim_means) / np.std(real_means))
    return {
        "mae_per_model": mae,
        "mae_overall": round(float(np.mean(list(mae.values()))), 4),
        "spearman": round(float(rho), 3),
        "kendall": round(float(tau), 3),
        "spread_ratio": round(spread, 3),
        "sim_means": {m: round(float(v), 3) for m, v in zip(models, sim_means, strict=True)},
        "real_means": {m: round(float(v), 3) for m, v in zip(models, real_means, strict=True)},
    }


def sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def cmd_benchmark(_args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = load(REAL)
    models = real.model_names()
    real_dev, real_hold = split_scenarios(real, REAL)
    registry = {
        "split_seed": SPLIT_SEED,
        "real_dev": {"n": len(real_dev), "sha": sha(real_dev), "ids": real_dev},
        "real_holdout": {"n": len(real_hold), "sha": sha(real_hold), "ids": real_hold},
    }
    results = {}
    for cohort in SIM_COHORTS:
        sim = load(cohort)
        sim_dev, sim_hold = split_scenarios(sim, cohort)
        registry[f"{cohort}_dev"] = {"n": len(sim_dev), "sha": sha(sim_dev), "ids": sim_dev}
        registry[f"{cohort}_holdout"] = {
            "n": len(sim_hold), "sha": sha(sim_hold), "ids": sim_hold,
        }
        for split, r_ids, s_ids in [
            ("dev", real_dev, sim_dev),
            ("holdout", real_hold, sim_hold),
        ]:
            r_cells = cell_means(real, REAL, set(r_ids))
            s_cells = cell_means(sim, cohort, set(s_ids))
            domains = sorted({d for _m, d in r_cells})
            metrics = gap_metrics(r_cells, s_cells, models, domains)
            results[f"{cohort}/{split}"] = metrics
            # Disagreement cell ranking (this split).
            cells = []
            for m in models:
                for d in domains:
                    if (m, d) in s_cells and (m, d) in r_cells:
                        cells.append(
                            {
                                "model": m,
                                "domain": d,
                                "sim": round(s_cells[(m, d)][0], 3),
                                "real": round(r_cells[(m, d)][0], 3),
                                "gap": round(s_cells[(m, d)][0] - r_cells[(m, d)][0], 3),
                                "n_sim": s_cells[(m, d)][1],
                                "n_real": r_cells[(m, d)][1],
                            }
                        )
            cells.sort(key=lambda c: -abs(c["gap"]))
            results[f"{cohort}/{split}/cells"] = cells
            logger.info(
                "%s/%s: MAE=%.4f spearman=%.3f kendall=%.3f spread_ratio=%.3f",
                cohort, split, metrics["mae_overall"], metrics["spearman"],
                metrics["kendall"], metrics["spread_ratio"],
            )
    (OUT / "split_registry.json").write_text(json.dumps(registry, indent=1))
    (OUT / "benchmark.json").write_text(json.dumps(results, indent=1))
    logger.info("registry + benchmark -> %s", OUT)


def cmd_cells(args: argparse.Namespace) -> None:
    """Dump top-disagreement DEV cells with episodes for transcript reading."""
    registry = json.loads((OUT / "split_registry.json").read_text())
    bench = json.loads((OUT / "benchmark.json").read_text())
    cohort = args.cohort
    dev_ids = set(registry[f"{cohort}_dev"]["ids"])
    sim = load(cohort)
    domains = scenario_domains(sim, cohort)
    cells = bench[f"{cohort}/dev/cells"][: args.top]
    real = load(REAL)
    real_dev = set(registry["real_dev"]["ids"])
    real_domains = scenario_domains(real, REAL)
    for rank, cell in enumerate(cells):
        m, d = cell["model"], cell["domain"]
        eps = [
            o
            for o in sim.outcomes
            if o.model == m and o.scenario_id in dev_ids and domains[o.scenario_id] == d
        ]
        r_eps = [
            o
            for o in real.outcomes
            if o.model == m
            and o.scenario_id in real_dev
            and real_domains[o.scenario_id] == d
        ]
        payload = {
            "rank": rank,
            "cell": cell,
            "sim_episodes": [
                {
                    "sid": o.scenario_id,
                    "reward": o.reward,
                    "steps": o.steps,
                    "stop": o.stop_reason,
                    "critique": o.critique,
                    "task": o.task[:600],
                    "replies": [r[:1200] for r in o.replies[:6]],
                }
                for o in eps
            ],
            "real_episodes": [
                {
                    "sid": o.scenario_id,
                    "reward": o.reward,
                    "steps": o.steps,
                    "stop": o.stop_reason,
                    "replies": [r[:800] for r in o.replies[:3]],
                }
                for o in r_eps
            ],
        }
        path = OUT / f"cell_{cohort}_{rank:02d}_{m}_{d}.json"
        path.write_text(json.dumps(payload, indent=1))
    logger.info("wrote %d cell dumps -> %s", len(cells), OUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["benchmark", "cells"])
    parser.add_argument("--cohort", default="tau-bench-s80")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    {"benchmark": cmd_benchmark, "cells": cmd_cells}[args.command](args)


if __name__ == "__main__":
    main()
