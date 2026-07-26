"""Validation gate for the cost/quality dial: measure the slider through the production path.

Four claims are checked, all on routerbench-ours9 with R1's cached text-embedding-3-large
vectors and R1's exact 70/30 stratified splits (5 seeds), through
`wmo.optimize.knn.fit_knn_policy` -> `apply_cost_quality` -> `evaluate_policy`, which is the
same `knn_decision` call serving makes:

0. The cost knob reproduces R1's own `r1-knn3-asym*` rows knob for knob (same family, same
   guard, same lam, novelty floor off). Different implementation, same numbers: that is what
   makes the port faithful rather than merely plausible.
1. Every anchor in `COST_QUALITY_ANCHORS` still delivers the quality and cost it advertises.
   The docstring an operator reads is a set of measurements, so it needs a gate.
2. The dial's quality end also matches R1's independently measured ablation rows: dial 0.0
   against `r1-knn-adapt-floor-q0.5`, dial 0.25 against `r1-knn-adapt-floor-q0.05` (the shipped
   default). Same knobs, different implementation, so agreement means the port is faithful.
3. Cost falls monotonically as the dial rises, and the guard still reverts under the cost knob:
   every guarded or floor-abstained request must land on the pinned baseline, never elsewhere.

`pick_lam` and the asymmetric guard are ports of `RetrievalParams.pick_lam` and `guard='stat_asym'`
from R1's `r1_retrieval_ablations.py`. Two research sweeps are printed for context but not gated:
R1's own `r1-knn3-asym-lam*` rows (same family, no novelty floor) and the R3 `r3-knn-frontier`
sweep that motivated the lam range (a DIFFERENT family: kNN-P probabilities under a fixed-margin
guard over hashing embeddings, which is why its numbers do not transfer).

Split identity, the embedding cache, the recorded-run loaders, and the routing-corpus location
are reused from `validate_knn_promotion.py` rather than copied, so the two gates cannot drift
apart. That corpus is untracked research data: see `$WMO_ROUTING_DATA` in the sibling gate.

Usage:
    uv run python .agents/scripts/validate_cost_quality.py
"""

from __future__ import annotations

import functools
import importlib.util
import json
import logging
import statistics
import sys
import tempfile
from pathlib import Path

from wmo.optimize.knn import (
    COST_QUALITY_ANCHORS,
    apply_cost_quality,
    best_single_on_fit,
    cost_quality_knobs,
    fit_knn_policy,
)
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import KNN_BANK_FILENAME, EmbedderSpec, select_model
from wmo.optimize.routing import evaluate_policy

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("validate-cost-quality")

PROMOTION_GATE = Path(__file__).with_name("validate_knn_promotion.py")
SEEDS = [0, 1, 2, 3, 4]
SHIPPED_FLOOR_Q = 0.05  # the `wmo optimize route fit` default, i.e. dial 0.25
# Every advertised anchor, plus one position on each leg whose only claim is that it sits
# between its neighbors (0.125 on the coverage leg, 0.875 on the price leg).
DIAL = sorted({anchor.cost_quality for anchor in COST_QUALITY_ANCHORS} | {0.125, 0.875})
# The R1 ablation variants the two quality-end anchors reproduce, with the numbers this script
# expects to still find in the records (a moved record is itself a failure worth seeing).
R1_ROWS = {
    0.0: ("r1-knn-adapt-floor-q0.5", 1.14, -13.9),
    0.25: ("r1-knn-adapt-floor-q0.05", 0.99, -24.7),
}
DELTA_TOLERANCE = 0.3  # accuracy points; one flipped scenario on one seed moves ~0.06
COST_TOLERANCE = 2.0  # percentage points of the cost ratio

FAILURES: list[str] = []


@functools.cache
def _promotion_gate() -> object:
    """Load the sibling gate as a module (its split, cache reader, and record loaders).

    Cached: the module is loaded once per process, so the per-variant record reads below do
    not re-execute it.
    """
    spec = importlib.util.spec_from_file_location("validate_knn_promotion", PROMOTION_GATE)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import the promotion gate from {PROMOTION_GATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded_iid(variant: str, params_filter: str | None = None) -> tuple[float, float, int]:
    """Mean (accuracy delta, cost percent, seeds) of an iid-split variant in the R1/R3 records.

    The hardening and head-to-head rounds reran the same variant names on ood splits and tagged
    the records in `notes`; those rows are a different population and are excluded here.
    """
    deltas: list[float] = []
    costs: list[float] = []
    runs = _promotion_gate().routing_data() / "runs"  # ty: ignore[unresolved-attribute]
    for name in ("r1.jsonl", "r3.jsonl"):
        path = runs / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                notes = record.get("notes", "")
                if record["variant"] != variant or record["matrix"] != "routerbench-ours9":
                    continue
                if "H2H" in notes or "split=ood" in notes:
                    continue
                if params_filter is not None and params_filter not in json.dumps(
                    record.get("params", {}), sort_keys=True
                ):
                    continue
                base = record["baselines"]["best_single"]
                deltas.append(record["result"]["accuracy"] - base["accuracy"])
                costs.append(record["result"]["cost_per_call"] / base["cost_per_call"] - 1.0)
    if not deltas:
        raise AssertionError(f"no iid rows recorded for variant {variant} under {runs}")
    return statistics.mean(deltas) * 100, statistics.mean(costs) * 100, len(deltas)


def measure() -> dict[float, tuple[float, float, float]]:
    """Sweep the dial on ours9: dial -> (mean delta points, mean cost percent, routed away)."""
    gate = _promotion_gate()
    data = gate.routing_data()  # ty: ignore[unresolved-attribute]
    matrix = OutcomeMatrix.load(data / "matrices" / "routerbench-ours9_matrix.json")
    embedder = gate.CachedEmbedder(matrix, data / "cache" / "routerbench-ours9-oai3l-tasks.npy")  # ty: ignore[unresolved-attribute]
    spec = EmbedderSpec(
        kind="azure", dim=embedder.dim, deployment="text-embedding-3-large", endpoint="https://x"
    )
    deltas: dict[float, list[float]] = {dial: [] for dial in DIAL}
    costs: dict[float, list[float]] = {dial: [] for dial in DIAL}
    routed: dict[float, list[float]] = {dial: [] for dial in DIAL}

    for seed in SEEDS:
        fit_ids, test_ids = gate.stratified_split(matrix, seed=seed)  # ty: ignore[unresolved-attribute]
        baseline = best_single_on_fit(matrix, fit_ids)
        base_accuracy, base_cost = gate.baseline_on_test(matrix, baseline, test_ids)  # ty: ignore[unresolved-attribute]
        tasks = {outcome.scenario_id: outcome.task for outcome in matrix.outcomes}
        with tempfile.TemporaryDirectory() as directory:
            # Fit ONCE per seed and slide the dial: the property the endpoint depends on (a
            # slider that needed a refit could not be a live control).
            fitted = fit_knn_policy(
                matrix,
                bank_path=Path(directory) / KNN_BANK_FILENAME,
                fit_ids=fit_ids,
                embedder=spec,
                embed_with=embedder,
                floor_q=SHIPPED_FLOOR_Q,
                fitted_from=f"routerbench-ours9 seed={seed}",
            )
            for dial in DIAL:
                policy = apply_cost_quality(fitted, dial)
                result = evaluate_policy(policy, matrix, test_ids, embedder=embedder)
                decisions = [
                    select_model(policy, tasks[sid], embedder=embedder) for sid in sorted(test_ids)
                ]
                stray = sorted(
                    {
                        d.model
                        for d in decisions
                        if ("reverted to" in d.reason or "abstain" in d.reason)
                        and d.model != baseline
                    }
                )
                if stray:
                    FAILURES.append(f"dial {dial}: guard/floor reverted to {stray}, not {baseline}")
                deltas[dial].append((result.accuracy - base_accuracy) * 100)
                costs[dial].append((result.cost_per_scenario / base_cost - 1.0) * 100)
                routed[dial].append(1.0 - result.model_mix.get(baseline, 0.0))
    return {
        dial: (
            statistics.mean(deltas[dial]),
            statistics.mean(costs[dial]),
            statistics.mean(routed[dial]),
        )
        for dial in DIAL
    }


def reference_rows() -> None:
    """Reproduce R1's cost-knob ablation rows through the production knobs, row by row.

    The strongest faithfulness check available for the port: `r1-knn3-asym*` is the same policy
    family under the same guard, measured by the research script, so the production knobs set to
    that configuration (floor off, asymmetric guard, the same lam) must land on the same numbers.
    This is the knob-level check; `measure()` covers the operator-facing dial built on top.
    """
    gate = _promotion_gate()
    data = gate.routing_data()  # ty: ignore[unresolved-attribute]
    matrix = OutcomeMatrix.load(data / "matrices" / "routerbench-ours9_matrix.json")
    embedder = gate.CachedEmbedder(matrix, data / "cache" / "routerbench-ours9-oai3l-tasks.npy")  # ty: ignore[unresolved-attribute]
    spec = EmbedderSpec(
        kind="azure", dim=embedder.dim, deployment="text-embedding-3-large", endpoint="https://x"
    )
    rows = {"r1-knn3-asym": 0.0, "r1-knn3-asym-lam002": 0.02, "r1-knn3-asym-lam005": 0.05}
    deltas: dict[str, list[float]] = {variant: [] for variant in rows}
    costs: dict[str, list[float]] = {variant: [] for variant in rows}
    for seed in SEEDS:
        fit_ids, test_ids = gate.stratified_split(matrix, seed=seed)  # ty: ignore[unresolved-attribute]
        baseline = best_single_on_fit(matrix, fit_ids)
        base_accuracy, base_cost = gate.baseline_on_test(matrix, baseline, test_ids)  # ty: ignore[unresolved-attribute]
        with tempfile.TemporaryDirectory() as directory:
            fitted = fit_knn_policy(
                matrix,
                bank_path=Path(directory) / KNN_BANK_FILENAME,
                fit_ids=fit_ids,
                embedder=spec,
                embed_with=embedder,
                fitted_from=f"routerbench-ours9 seed={seed}",
            )
            for variant, lam in rows.items():
                # R1's configuration exactly: no novelty floor, asymmetric guard, that lam.
                policy = fitted.model_copy(
                    update={"guard_mode": "asymmetric", "pick_lam": lam, "floor_sim": None}
                )
                policy.attach_bank(fitted.knn_bank())
                result = evaluate_policy(policy, matrix, test_ids, embedder=embedder)
                deltas[variant].append((result.accuracy - base_accuracy) * 100)
                costs[variant].append((result.cost_per_scenario / base_cost - 1.0) * 100)

    logger.info("=== the cost knob reproduces R1's own ablation rows, knob for knob ===")
    for variant, lam in rows.items():
        recorded_delta, recorded_cost, seeds = recorded_iid(variant)
        delta = statistics.mean(deltas[variant])
        cost = statistics.mean(costs[variant])
        ok = (
            abs(delta - recorded_delta) <= DELTA_TOLERANCE
            and abs(cost - recorded_cost) <= COST_TOLERANCE
        )
        logger.info(
            "lam=%-5g vs %-20s recorded %+.2fpt/%+.1f%% (%d seeds), measured %+.2fpt/%+.1f%%: %s",
            lam,
            variant,
            recorded_delta,
            recorded_cost,
            seeds,
            delta,
            cost,
            "PASS" if ok else "FAIL",
        )
        if not ok:
            FAILURES.append(f"the cost knob does not reproduce {variant}")


def main() -> None:
    reference_rows()
    measured = measure()
    logger.info("=== cost/quality dial on routerbench-ours9, 5 seeds, fit once per seed ===")
    logger.info(
        "%-8s %-8s %-6s %-11s %9s %9s %8s",
        "dial",
        "floor_q",
        "lam",
        "guard",
        "quality",
        "cost",
        "routed",
    )
    for dial, (delta, cost, routed) in measured.items():
        knobs = cost_quality_knobs(dial)
        logger.info(
            "%-8g %-8.3f %-6.4f %-11s %+8.2fpt %+8.1f%% %7.1f%%",
            dial,
            knobs.floor_q,
            knobs.pick_lam,
            knobs.guard_mode,
            delta,
            cost,
            routed * 100,
        )

    logger.info("=== every advertised anchor still delivers what it advertises ===")
    for anchor in COST_QUALITY_ANCHORS:
        delta, cost, _ = measured[anchor.cost_quality]
        ok = (
            abs(delta - anchor.quality_delta_points) <= DELTA_TOLERANCE
            and abs(cost - anchor.cost_delta_percent) <= COST_TOLERANCE
        )
        logger.info(
            "dial %-6g advertises %+.2fpt/%+.1f%%, measured %+.2fpt/%+.1f%%: %s",
            anchor.cost_quality,
            anchor.quality_delta_points,
            anchor.cost_delta_percent,
            delta,
            cost,
            "PASS" if ok else "FAIL",
        )
        if not ok:
            FAILURES.append(f"anchor {anchor.cost_quality} no longer matches its measured row")

    logger.info("=== the quality end vs R1's independently measured ablation rows ===")
    for dial, (variant, delta_expected, cost_expected) in R1_ROWS.items():
        recorded_delta, recorded_cost, seeds = recorded_iid(variant)
        delta, cost, _ = measured[dial]
        if abs(recorded_delta - delta_expected) > 1e-2 or abs(recorded_cost - cost_expected) > 1e-1:
            FAILURES.append(f"{variant}: the record moved ({recorded_delta:+.2f}pt)")
        ok = (
            abs(delta - recorded_delta) <= DELTA_TOLERANCE
            and abs(cost - recorded_cost) <= COST_TOLERANCE
        )
        logger.info(
            "dial %-5g vs %-24s recorded %+.2fpt/%+.1f%% (%d seeds), measured %+.2fpt/%+.1f%%: %s",
            dial,
            variant,
            recorded_delta,
            recorded_cost,
            seeds,
            delta,
            cost,
            "PASS" if ok else "FAIL",
        )
        if not ok:
            FAILURES.append(f"dial {dial} does not reproduce {variant}")

    logger.info("=== cost monotonicity in the dial ===")
    series = [measured[dial][1] for dial in DIAL]
    monotone = all(
        later <= earlier + 1e-9 for earlier, later in zip(series[:-1], series[1:], strict=True)
    )
    logger.info(
        "cost percent by dial: %s: %s",
        ", ".join(f"{value:+.1f}" for value in series),
        "PASS (non-increasing)" if monotone else "FAIL (not monotone)",
    )
    if not monotone:
        FAILURES.append("cost is not monotone in the dial")

    logger.info("=== context (not gated): R1's own asymmetric-guard cost-knob rows ===")
    logger.info("same family and guard as the price leg, without the novelty floor")
    for variant in (
        "r1-knn3-asym",
        "r1-knn3-asym-lam002",
        "r1-knn3-asym-lam005",
        "r1-knn3-asym-lam01",
        "r1-knn3-asym-shuffled",
    ):
        delta, cost, seeds = recorded_iid(variant)
        logger.info("  %-24s %+.2fpt %+.1f%% (%d seeds)", variant, delta, cost, seeds)
    logger.info("=== context (not gated): the R3 sweep that motivated the lam range ===")
    logger.info("a DIFFERENT family: kNN-P probabilities + fixed-margin guard, hashing embeddings")
    for lam in (0.0, 0.03, 0.05, 0.08):
        delta, cost, seeds = recorded_iid("r3-knn-frontier", params_filter=f'"lam": {lam}')
        logger.info(
            "  r3-knn-frontier lam=%-6g %+.2fpt %+.1f%% (%d seeds)", lam, delta, cost, seeds
        )


if __name__ == "__main__":
    main()
    if FAILURES:
        logger.error("VALIDATION FAILED: %s", "; ".join(FAILURES))
        sys.exit(1)
    logger.info("VALIDATION PASSED")
