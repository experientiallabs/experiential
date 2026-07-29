"""Paired-by-seed scoring of the per-arm routed rungs (tau grid-c2 + terminal tb2cost).

The routing lane's complement to the cost corner's held-out replay (theirs: one pinned
14/6 split, paired-by-scenario; identity+loo covers identity only). This script re-splits
each arm's FINAL matrix into fit/held-out at 5 seeds, refits the knn policy through the
PRODUCT path (fit_knn_policy with champion defaults, whose adaptive neighborhood
reproduces the master's rag_num=7/min_pairs=3 at bank 14 by construction; z=0.5 because
findings/guardcal.md does not exist - the A/A-derived z gate has not reported, so the
rung runs at the validated z with the known small-bank hazard stated), replays the five
D-DIAL detents on each seed's held-out band, and scores paired-by-seed against BOTH
baselines (cohort best-single = the bar; fable-5 = the customer-narrative anchor).

Also reported per seed: the DISCOVERED fallback (guard_model=None), because discovery
stability across resplits is a claim the pinned split cannot make.

Conventions: effective cost per completed task via wmo.optimize.scorecard (cache-adjusted;
router-embedding overhead folded at the 3-large list price like the cost corner's records;
compressed-arm EPISODE compressor bill is in the matrix rows' accounting at merge time -
per-seed DELTAS compare rows from the same arm so the bill cancels; absolute compressed
costs are labeled accordingly). Compressed arms fit AND replay through CompressingEmbedder
(C2 representation rule). Embedding spend ~cents under the 2026-07-28 replay ruling,
logged in the output.

Outputs: runs/rung.jsonl (one record per cohort x arm x seed x dial, with per-scenario
detail), findings/r3_sim2real-style fragment rung_numbers_fragment.json for the cost
chat's numbers.json merge (shape coordinated via DECISIONS).
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from wmo.optimize.compression import CompressingEmbedder, CompressionConfig
from wmo.optimize.knn import COST_QUALITY_NAMED_POINTS, apply_cost_quality, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.scorecard import effective_cost_per_completed_task, rows_for_policy
from wmo.providers.base import Embedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rung")

MAIN = Path("~/Desktop/Projects/world-model-harness").expanduser()
DATA = Path("~/Desktop/Projects/wmh-routing-data").expanduser()
RUNS = DATA / "runs" / "rung.jsonl"
SEEDS = [1, 2, 3, 4, 5]
EMBED_PRICE_PER_MTOK = 0.13  # text-embedding-3-large list price (the cost corner's estimate)

COHORTS = {
    "grid-c2": {
        "root": MAIN / ".wmo/jt/grid-c2",
        "dataset": "tau-bench",
        "anchor": "fable-5",
        "best_single": "opus-5",
        "arms": ["identity", "truncate"],  # llmlingua2: blocked on WMO_COMPRESSOR_* creds
    },
    "tb2cost": {
        "root": MAIN / ".wmo/jt/tb2cost",
        "dataset": "terminal-tasks",
        "anchor": "fable-5",
        "best_single": "sonnet-5",
        "arms": ["identity"],  # llmlingua2 blocked on creds; truncate PARTIAL (149/520)
    },
}


class CachingEmbedder:
    """Wraps an embedder with a text->vector cache and a call/token meter."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.chars = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            self.calls += 1
            self.chars += sum(len(t) for t in missing)
            for text, vec in zip(missing, self._inner.embed(missing), strict=True):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    def est_usd(self) -> float:
        return (self.chars / 4) / 1e6 * EMBED_PRICE_PER_MTOK


def split_band(
    sids: list[str], seed: int, fit_fraction: float = 0.7
) -> tuple[list[str], list[str]]:
    """Seeded fit/held-out resplit of the test band (same 70/30 shape as the pinned split)."""
    rng = np.random.default_rng(seed)
    ordered = sorted(sids)
    perm = rng.permutation(len(ordered))
    cut = round(len(ordered) * fit_fraction)
    fit = sorted(ordered[i] for i in perm[:cut])
    held = sorted(ordered[i] for i in perm[cut:])
    return fit, held


def scenario_quality(rows: list, sid: str) -> float | None:
    vals = [r.reward for r in rows if r.scenario_id == sid and r.reward is not None]
    return float(np.mean(vals)) if vals else None


def arm_baseline_rows(matrix: OutcomeMatrix, model: str, ids: list[str]) -> list:
    return [o for o in matrix.outcomes if o.model == model and o.scenario_id in set(ids)]


def score_pair(
    matrix: OutcomeMatrix, routed_rows: list, baseline_rows: list, held: list[str]
) -> dict:
    """Paired per-scenario quality deltas + effective-cost comparison on the held band."""
    deltas, sids_used = [], []
    for sid in held:
        rq = scenario_quality(routed_rows, sid)
        bq = scenario_quality(baseline_rows, sid)
        if rq is not None and bq is not None:
            deltas.append(rq - bq)
            sids_used.append(sid)
    routed_cost = effective_cost_per_completed_task(routed_rows)
    base_cost = effective_cost_per_completed_task(baseline_rows)
    rc = routed_cost.cost_per_completed_task_usd
    bc = base_cost.cost_per_completed_task_usd
    return {
        "n_pairs": len(deltas),
        "quality_delta_points": round(float(np.mean(deltas)) * 100, 2) if deltas else None,
        "sign_wins": int(sum(1 for d in deltas if d > 1e-9)),
        "sign_losses": int(sum(1 for d in deltas if d < -1e-9)),
        "cost_per_completed_task_usd": rc,
        "baseline_cost_per_completed_task_usd": bc,
        "cost_delta_percent": (
            round((rc / bc - 1) * 100, 1) if rc is not None and bc is not None else None
        ),
        "per_scenario_delta": {s: round(d, 4) for s, d in zip(sids_used, deltas, strict=True)},
    }


def run_cohort(name: str, cfg: dict, seeds: list[int]) -> list[dict]:
    records = []
    for arm in cfg["arms"]:
        matrix_path = cfg["root"] / arm / "matrix.json"
        policy_path = cfg["root"] / arm / "policy.json"
        if not matrix_path.exists():
            logger.info("%s/%s: no matrix, skipping", name, arm)
            continue
        matrix = OutcomeMatrix.load(matrix_path)
        landed = (
            RoutingPolicy.model_validate_json(policy_path.read_text())
            if policy_path.exists()
            else None
        )
        spec = landed.embedder if landed else EmbedderSpec(kind="azure", dim=3072)
        compression = landed.fit_compression if landed else None
        inner = spec.build()
        if compression is not None:
            inner = CompressingEmbedder(inner, CompressionConfig(**compression.model_dump()))
        embedder = CachingEmbedder(inner)
        sids = matrix.scenario_ids()
        unscored = sum(1 for o in matrix.outcomes if o.reward is None)
        logger.info(
            "%s/%s: %d scenarios, %d outcomes (%d unscored), compression=%s",
            name, arm, len(sids), len(matrix.outcomes), unscored,
            compression.compressor_id if compression else "none",
        )
        for seed in seeds:
            fit_ids, held = split_band(sids, seed)
            with tempfile.TemporaryDirectory() as tmp:
                policy = fit_knn_policy(
                    matrix,
                    bank_path=Path(tmp) / "bank.npz",
                    fit_ids=fit_ids,
                    embedder=spec,
                    embed_with=embedder,
                    guard_model=None,  # discovery: its stability across seeds is a claim
                    floor_q=0.05,
                    fitted_from=f"rung-paired-seed {name}/{arm} seed{seed}",
                )
                discovered = policy.guard_model
                for dial, dial_name in COST_QUALITY_NAMED_POINTS:
                    dialed = apply_cost_quality(policy, dial)
                    routed_rows = rows_for_policy(
                        matrix, dialed, ids=held, embedder=embedder
                    )
                    mix: dict[str, int] = {}
                    for sid in held:
                        models = {r.model for r in routed_rows if r.scenario_id == sid}
                        for m in models:
                            mix[m] = mix.get(m, 0) + 1
                    rec = {
                        "run_id": f"rung-{name}-{arm}-s{seed}-d{dial}-{uuid.uuid4().hex[:6]}",
                        "ts": datetime.now(tz=UTC).isoformat(),
                        "cohort": name,
                        "dataset": cfg["dataset"],
                        "arm": arm,
                        "seed": seed,
                        "dial": dial,
                        "dial_name": dial_name,
                        "n_eval_scenarios": len(held),
                        "fit_scenarios": len(fit_ids),
                        "discovered_fallback": discovered,
                        "routed_mix": mix,
                        "z": 0.5,
                        "guardcal": "absent (findings/guardcal.md missing); z=0.5 with the "
                        "known small-bank hazard",
                        "vs_best_single": score_pair(
                            matrix, routed_rows,
                            arm_baseline_rows(matrix, cfg["best_single"], held), held,
                        ),
                        "vs_anchor": score_pair(
                            matrix, routed_rows,
                            arm_baseline_rows(matrix, cfg["anchor"], held), held,
                        ),
                    }
                    records.append(rec)
                    logger.info(
                        "%s/%s s%d dial=%.2f: fallback=%s mix=%s | vs %s: dq=%s cost%%=%s | "
                        "vs %s: dq=%s cost%%=%s",
                        name, arm, seed, dial, discovered, mix,
                        cfg["best_single"], rec["vs_best_single"]["quality_delta_points"],
                        rec["vs_best_single"]["cost_delta_percent"],
                        cfg["anchor"], rec["vs_anchor"]["quality_delta_points"],
                        rec["vs_anchor"]["cost_delta_percent"],
                    )
        logger.info(
            "%s/%s embedding meter: %d calls, ~%d chars, est $%.4f",
            name, arm, embedder.calls, embedder.chars, embedder.est_usd(),
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="*", default=list(COHORTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    args = parser.parse_args()
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    for name in args.cohorts:
        all_records.extend(run_cohort(name, COHORTS[name], args.seeds))
    with RUNS.open("a", encoding="utf-8") as fh:
        for rec in all_records:
            fh.write(json.dumps(rec) + "\n")
    logger.info("%d records -> %s", len(all_records), RUNS)


if __name__ == "__main__":
    main()
