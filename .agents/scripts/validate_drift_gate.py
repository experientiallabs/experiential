"""Pre-registered validation gate for the containment gate (`wmo.optimize.drift_gate`).

Chat R3 measured the mechanism with a research script (`r3_bakeoff.py`, arm `r3b-hybrid-svm`:
the champion proposes, a per-model global win-vs-baseline SVM may veto). This script re-runs that
measurement through the PRODUCTION path only: `wmo.optimize.knn.fit_knn_policy` builds the bank,
fits the gate and writes both sidecars, and `wmo.optimize.routing.evaluate_policy` replays the
policy through the same `knn_decision` serving calls a request would take. If the numbers move,
the port changed the algorithm.

The two pre-registered bars, fixed before the port was written:

- `financebench-s80`, iid 70/30, 5 seeds: the gated router must be at PARITY with the best single
  model (mean |delta| <= 0.002, worst seed >= -0.005). This is the bar that matters. It is an
  80-scenario bank, the regime where neighborhoods stop being independent evidence, and the
  UNGATED champion loses 1.85 points there with a worst seed of -11.15.
- `routerbench-ours9`, all three split families x 5 seeds: the gated router must reproduce R3's
  recorded `r3b-hybrid-svm` rows, mean delta +0.0061 within `DELTA_TOLERANCE`. This is the bar
  that says the port did not merely learn to abstain: 1199 scenarios is the regime the champion
  WINS in, and the gate has to keep that win rather than flatten it to the baseline.

Split identity is proven, not assumed. All three research split families are reimplemented here
(they live on research branches, not on main), and every cell's fit/test sizes and best-single
baseline accuracy are checked against the recorded runs in `<routing-data>/runs/r3.jsonl` before
its delta is compared. A cell whose split does not reproduce is a failure, not a warning: two
different splits agreeing on a mean would prove nothing.

Query embeddings come from R1's cached text-embedding-3-large vectors, so no text is re-embedded.
The routing data (`runs/`, `matrices/`, `cache/`) is a multi-GB research corpus that is not in
git; it defaults to the gitignored `.wmo/routing-data/` under the repo root, and
`$WMO_ROUTING_DATA` points the lookup elsewhere.

Usage:
    uv run python .agents/scripts/validate_drift_gate.py             # both bars
    uv run python .agents/scripts/validate_drift_gate.py --s80-only  # the fast one
"""

from __future__ import annotations

import json
import logging
import os
import random
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from wmo.optimize.drift_gate import drift_gate_path_for
from wmo.optimize.knn import best_single_on_fit, fit_knn_policy
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import EmbedderSpec, knn_bank_path_for
from wmo.optimize.routing import evaluate_policy
from wmo.retrieval.embedders import HashingEmbedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("validate-drift-gate")
# The fit logs one line per gate; at 15 cells x 8 models that buries the results.
logging.getLogger("wmo.optimize.drift_gate").setLevel(logging.WARNING)
logging.getLogger("wmo.optimize.knn").setLevel(logging.WARNING)

FAILURES: list[str] = []

ENV_ROUTING_DATA = "WMO_ROUTING_DATA"
DEFAULT_ROUTING_DATA = Path(__file__).resolve().parents[2] / ".wmo" / "routing-data"
SEEDS = [0, 1, 2, 3, 4]

# R3's recorded `r3b-hybrid-svm` result on routerbench-ours9, averaged over all 15 (split, seed)
# cells: +0.0061 accuracy against the best single model at about -5.7% cost.
OURS9_DELTA = 0.0061
DELTA_TOLERANCE = 0.003

# The financebench-s80 parity bar. Stated as an absolute band rather than "> 0" because parity is
# the claim: the gate should make an 80-scenario bank behave EXACTLY like its baseline, and a
# spurious win there would be as much a reproduction failure as a loss.
S80_MEAN_ABS_TOLERANCE = 0.002
S80_WORST_SEED_FLOOR = -0.005
# What the ungated champion does on the same rows, for the contrast the gate exists to produce.
S80_UNGATED_MEAN = -0.0185
S80_UNGATED_WORST = -0.1115

# The research protocol's fractions (r3_bakeoff.py).
IID_TRAIN_FRACTION = 0.7
OOD_TEST_FRACTION = 0.3
# `split_holdout_clusters` clusters with the HASHING embedder at this width, not with the OpenAI
# vectors the router uses: the split is a fixed property of the corpus, so it must not move when
# the embedder does.
OOD_CLUSTER_DIM = 1024
OOD_KMEANS_SEED = 1234


def routing_data() -> Path:
    """Root of the routing research corpus, holding `runs/`, `matrices/`, and `cache/`."""
    override = os.environ.get(ENV_ROUTING_DATA)
    root = Path(override).expanduser() if override else DEFAULT_ROUTING_DATA
    if not root.is_dir():
        raise SystemExit(
            f"routing corpus not found at {root}. It is multi-GB research data that git does "
            f"not carry: set {ENV_ROUTING_DATA} to the directory holding runs/, matrices/, and "
            "cache/, or place the corpus at that default path."
        )
    return root


def stratified_split(
    matrix: OutcomeMatrix, *, train_fraction: float = IID_TRAIN_FRACTION, seed: int = 0
) -> tuple[list[str], list[str]]:
    """The iid split (`split_scenario_ids`), reimplemented; identity is checked per cell.

    Stratified by the scenario id's dataset prefix so no small eval vanishes from either side;
    ids without a prefix share one stratum.
    """
    by_eval: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        prefix = scenario_id.split(":", 1)[0] if ":" in scenario_id else ""
        by_eval.setdefault(prefix, []).append(scenario_id)
    rng = random.Random(seed)
    fit: list[str] = []
    test: list[str] = []
    for _name, ids in sorted(by_eval.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        if len(shuffled) > 1:
            cut = min(max(1, round(len(shuffled) * train_fraction)), len(shuffled) - 1)
        else:
            cut = 1
        fit.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return sorted(fit), sorted(test)


def _holdout_groups(
    groups: dict[str, list[str]], *, test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Greedily assign WHOLE groups to test until `test_fraction` of scenarios are there."""
    total = sum(len(ids) for ids in groups.values())
    keys = sorted(groups, key=str)
    random.Random(seed).shuffle(keys)
    test: list[str] = []
    taken: list[str] = []
    for key in keys:
        if len(test) >= test_fraction * total or len(taken) == len(keys) - 1:
            break
        test.extend(groups[key])
        taken.append(key)
    fit = [sid for key in keys if key not in set(taken) for sid in groups[key]]
    if not fit or not test:
        raise ValueError("degenerate holdout split; check group sizes vs test_fraction")
    return sorted(fit), sorted(test)


def cluster_split(matrix: OutcomeMatrix, *, seed: int) -> tuple[list[str], list[str]]:
    """`split_holdout_clusters`: hold out whole k-means groups of the task embeddings.

    The clustering is pinned to `OOD_KMEANS_SEED`, so the seed rotates WHICH clusters are held
    out rather than the cluster geometry.
    """
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(tasks)
    count = max(4, min(16, len(scenario_ids) // 8))
    count = min(count, len(scenario_ids))
    vectors = np.asarray(HashingEmbedder(dim=OOD_CLUSTER_DIM).embed([tasks[s] for s in scenario_ids]))
    vectors = Normalizer(norm="l2").fit_transform(vectors)
    labels = KMeans(n_clusters=count, random_state=OOD_KMEANS_SEED, n_init="auto").fit_predict(
        vectors
    )
    groups: dict[str, list[str]] = {}
    for scenario_id, label in zip(scenario_ids, labels, strict=True):
        groups.setdefault(str(int(label)), []).append(scenario_id)
    return _holdout_groups(groups, test_fraction=OOD_TEST_FRACTION, seed=seed)


def task_split(matrix: OutcomeMatrix, *, seed: int) -> tuple[list[str], list[str]]:
    """`split_holdout_tasks`: hold out whole id-prefix task groups (Leave-Task-Out)."""
    groups: dict[str, list[str]] = {}
    for scenario_id in matrix.scenario_ids():
        if ":" not in scenario_id:
            raise ValueError(f"scenario id '{scenario_id}' has no task prefix")
        groups.setdefault(scenario_id.split(":", 1)[0], []).append(scenario_id)
    return _holdout_groups(groups, test_fraction=OOD_TEST_FRACTION, seed=seed)


class CachedEmbedder:
    """Serves R1's cached text-embedding-3-large vectors by task text (no API calls).

    The cache is row-aligned to the matrix's scenarios in first-appearance order, which is how
    the research script wrote it.
    """

    def __init__(self, matrix: OutcomeMatrix, cache_path: Path) -> None:
        order: list[str] = []
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            if outcome.scenario_id not in tasks:
                tasks[outcome.scenario_id] = outcome.task
                order.append(outcome.scenario_id)
        vectors = np.load(cache_path)
        if vectors.shape[0] != len(order):
            raise ValueError(
                f"cache {cache_path} has {vectors.shape[0]} rows but the matrix has "
                f"{len(order)} scenarios; it was built for a different matrix"
            )
        self.dim = int(vectors.shape[1])
        self._by_text = {tasks[sid]: vectors[index] for index, sid in enumerate(order)}

    def embed(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._by_text]
        if missing:
            raise KeyError(f"{len(missing)} task texts are not in the embedding cache")
        return [self._by_text[text].tolist() for text in texts]


def baseline_on_test(matrix: OutcomeMatrix, model: str, test_ids: list[str]) -> tuple[float, float]:
    """(mean reward, mean cost) of one model over the test scenarios it was scored on."""
    by_cell: dict[str, list[tuple[float, float]]] = {}
    wanted = set(test_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in wanted and outcome.model == model and outcome.reward is not None:
            by_cell.setdefault(outcome.scenario_id, []).append((outcome.reward, outcome.cost_usd))
    rewards = [sum(r for r, _ in cells) / len(cells) for cells in by_cell.values()]
    costs = [sum(c for _, c in cells) / len(cells) for cells in by_cell.values()]
    return sum(rewards) / len(rewards), sum(costs) / len(costs)


def recorded_runs(matrix_name: str, split: str) -> dict[int, dict[str, float]]:
    """R3's recorded `r3b-hybrid-svm` headline numbers for one (matrix, split), by seed.

    financebench-s80 carries both a 70/30 and a 50/50 iid family under the same split label; the
    70/30 rows are the pre-registered ones, so a cell is keyed by seed AND matched on fit size by
    the caller.
    """
    out: dict[int, dict[str, float]] = {}
    with (routing_data() / "runs" / "r3.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record["variant"] != "r3b-hybrid-svm"
                or record["matrix"] != matrix_name
                or record["params"].get("split") != split
            ):
                continue
            out.setdefault(record["split_seed"], {})
            out[record["split_seed"]][record["fit_scenarios"]] = {
                "accuracy": record["result"]["accuracy"],
                "baseline_accuracy": record["baselines"]["best_single"]["accuracy"],
                "cost": record["result"]["cost_per_call"],
                "baseline_cost": record["baselines"]["best_single"]["cost_per_call"],
                "test": record["test_scenarios"],
            }
    return out


def run_cell(
    matrix: OutcomeMatrix,
    embedder: CachedEmbedder,
    spec: EmbedderSpec,
    fit_ids: list[str],
    test_ids: list[str],
) -> tuple[float, float, str, float]:
    """Fit a gated policy on `fit_ids` and replay it on `test_ids`, all through production code.

    Returns:
        (routed accuracy, routed cost per scenario, the baseline it guarded against, the fraction
        of test requests that left that baseline).
    """
    baseline = best_single_on_fit(matrix, fit_ids)
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "policy.json"
        policy = fit_knn_policy(
            matrix,
            bank_path=knn_bank_path_for(out),
            drift_gate_path=drift_gate_path_for(out),
            fit_ids=fit_ids,
            embedder=spec,
            embed_with=embedder,
            guard_model=baseline,
            fitted_from="validate_drift_gate",
        )
        result = evaluate_policy(policy, matrix, test_ids, embedder=embedder)
    routed = 1.0 - result.model_mix.get(baseline, 0.0)
    return result.accuracy, result.cost_per_scenario, baseline, routed


def measure(
    matrix_name: str,
    cache_name: str,
    split: str,
    *,
    train_fraction: float = IID_TRAIN_FRACTION,
) -> tuple[list[float], list[float]]:
    """Run one (matrix, split) family over every seed, checking split identity as it goes.

    Returns:
        (per-seed accuracy deltas against the best single model, per-seed cost ratios).
    """
    data = routing_data()
    matrix = OutcomeMatrix.load(data / "matrices" / f"{matrix_name}_matrix.json")
    embedder = CachedEmbedder(matrix, data / "cache" / cache_name)
    spec = EmbedderSpec(
        kind="azure", dim=embedder.dim, deployment="text-embedding-3-large", endpoint="https://x"
    )
    reference = recorded_runs(matrix_name, split)
    deltas: list[float] = []
    cost_ratios: list[float] = []

    for seed in SEEDS:
        if split == "iid":
            fit_ids, test_ids = stratified_split(matrix, train_fraction=train_fraction, seed=seed)
        elif split == "ood-cluster":
            fit_ids, test_ids = cluster_split(matrix, seed=seed)
        else:
            fit_ids, test_ids = task_split(matrix, seed=seed)

        expected = reference.get(seed, {}).get(len(fit_ids))
        if expected is None:
            available = sorted(reference.get(seed, {}))
            raise AssertionError(
                f"{matrix_name}/{split} seed {seed}: this split is {len(fit_ids)}/{len(test_ids)}, "
                f"R3 recorded fit sizes {available}: not the same split"
            )
        if len(test_ids) != expected["test"]:
            raise AssertionError(
                f"{matrix_name}/{split} seed {seed}: test side is {len(test_ids)}, R3 recorded "
                f"{expected['test']}: not the same split"
            )
        baseline_name = best_single_on_fit(matrix, fit_ids)
        base_accuracy, base_cost = baseline_on_test(matrix, baseline_name, test_ids)
        if abs(base_accuracy - expected["baseline_accuracy"]) > 1e-4:
            raise AssertionError(
                f"{matrix_name}/{split} seed {seed}: best-single ({baseline_name}) scores "
                f"{base_accuracy:.6f} on test, R3 recorded {expected['baseline_accuracy']:.6f}: "
                "not the same split or baseline"
            )

        accuracy, cost, baseline_name, routed = run_cell(matrix, embedder, spec, fit_ids, test_ids)
        delta = accuracy - base_accuracy
        deltas.append(delta)
        cost_ratios.append(cost / base_cost - 1.0 if base_cost > 0 else 0.0)
        recorded_delta = expected["accuracy"] - expected["baseline_accuracy"]
        logger.info(
            "  %-11s seed %d: acc %.4f vs %s %.4f (delta %+.4f) | cost %+0.1f%% | routed away "
            "%4.1f%% | R3 recorded delta %+.4f",
            split,
            seed,
            accuracy,
            baseline_name,
            base_accuracy,
            delta,
            cost_ratios[-1] * 100,
            routed * 100,
            recorded_delta,
        )
    return deltas, cost_ratios


def s80_gate() -> None:
    """The parity bar: an 80-scenario bank must come out behaving exactly like its baseline."""
    logger.info("=== financebench-s80: containment parity (the bar that matters) ===")
    deltas, cost_ratios = measure("financebench-s80", "financebench-s80-oai3l-tasks.npy", "iid")
    mean_delta = statistics.mean(deltas)
    worst = min(deltas)
    passed = abs(mean_delta) <= S80_MEAN_ABS_TOLERANCE and worst >= S80_WORST_SEED_FLOOR
    logger.info(
        "GATE financebench-s80: mean %+.4f, worst seed %+.4f, mean cost %+0.1f%% "
        "(ungated champion: mean %+.4f, worst %+.4f)",
        mean_delta,
        worst,
        statistics.mean(cost_ratios) * 100,
        S80_UNGATED_MEAN,
        S80_UNGATED_WORST,
    )
    logger.info(
        "GATE financebench-s80: %s (needs |mean| <= %.3f and worst >= %.3f)",
        "PASS" if passed else "FAIL",
        S80_MEAN_ABS_TOLERANCE,
        S80_WORST_SEED_FLOOR,
    )
    if not passed:
        FAILURES.append("financebench-s80 parity")


def ours9_gate() -> None:
    """The keep-the-win bar: 1199 scenarios, all three split families, R3's recorded rows."""
    logger.info("=== routerbench-ours9: the gate must not flatten the champion's win ===")
    deltas: list[float] = []
    cost_ratios: list[float] = []
    for split in ("iid", "ood-cluster", "ood-task"):
        cell_deltas, cell_costs = measure(
            "routerbench-ours9", "routerbench-ours9-oai3l-tasks.npy", split
        )
        deltas.extend(cell_deltas)
        cost_ratios.extend(cell_costs)
    mean_delta = statistics.mean(deltas)
    passed = abs(mean_delta - OURS9_DELTA) <= DELTA_TOLERANCE
    logger.info(
        "GATE routerbench-ours9: mean %+.4f (stdev %.4f over %d cells), mean cost %+0.1f%% "
        "vs R3 recorded %+.4f",
        mean_delta,
        statistics.stdev(deltas),
        len(deltas),
        statistics.mean(cost_ratios) * 100,
        OURS9_DELTA,
    )
    logger.info(
        "GATE routerbench-ours9: %s (needs |mean - %.4f| <= %.3f)",
        "PASS" if passed else "FAIL",
        OURS9_DELTA,
        DELTA_TOLERANCE,
    )
    if not passed:
        FAILURES.append("routerbench-ours9 reproduction")


def main() -> None:
    s80_gate()
    if "--s80-only" not in sys.argv:
        ours9_gate()


if __name__ == "__main__":
    main()
    if FAILURES:
        logger.error("VALIDATION FAILED: %s", ", ".join(FAILURES))
        sys.exit(1)
    logger.info("VALIDATION PASSED")
