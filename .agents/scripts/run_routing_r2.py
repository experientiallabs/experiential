"""r2 experiment driver: faithful ProxRouter vs rank router, IID and OOD splits.

Runs every r2 variant through the shared `evaluate_choices` evaluator on identical splits per
(matrix, split kind, seed) cell and appends RunRecords to the shared runs/r2.jsonl. Variants:

- r2-best-single: the fit-chosen baseline (also every guarded variant's floor).
- r2-rank: the guarded Avengers champion config (K=64, hashing-1024, min_support=4,
  margin=0.03), refit per split; the incumbent to beat.
- r2-rank-tilt: our ADAPTED support-tilt (gamma=0.5), the strawman the faithful method must
  outperform for the adaptation to be retired.
- r2-km-prox / r2-knn-prox: faithful ProxRouter (2510.09852), guarded per protocol; the
  -unguarded twins are diagnostics for how much the guard gives up or saves.
- r2-km-prox-shuffled / r2-knn-prox-shuffled: leak control on ours9 (rewards permuted within
  model): a real method must collapse to ~best-single.

Split kinds: iid (split_scenario_ids), ood-cluster (split_holdout_clusters), ood-task
(split_holdout_tasks, matrices with id prefixes only). Seeds 0-4 each.

Usage: uv run python .agents/scripts/run_routing_r2.py [matrix ...] [--splits iid,ood-cluster]
       [--seeds 0,1,2,3,4] [--quick]
"""

from __future__ import annotations

import logging
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import Normalizer

from wmh.optimize.outcomes import OutcomeMatrix
from wmh.optimize.policy import EmbedderSpec, rank_decision
from wmh.optimize.proxrouter import ProxScorer, fit_km_prox, fit_knn_prox
from wmh.optimize.routing import fit_rank_policy
from wmh.research.routerbench import best_single_model, oracle, split_scenario_ids
from wmh.research.routing_ood import split_holdout_clusters, split_holdout_tasks
from wmh.research.routing_runs import RunRecord, append_run, evaluate_choices
from wmh.retrieval.embedders import HashingEmbedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("r2")

DATA = Path("~/Desktop/Projects/wmh-routing-data").expanduser()
RUNS = DATA / "runs" / "r2.jsonl"
DIM = 1024
RANK_K = 64
PROX_K = 32  # the paper's KM-Prox cluster count
KNN_K = 100  # the paper's kNN-Prox neighbor count
TAU_INV = 20.0  # the paper's 1/tau
GUARD_MARGIN = 0.03  # shared protocol margin (doubled when pricier, inside the deciders)
SPLIT_SEEDS = [0, 1, 2, 3, 4]
MIN_SCENARIOS = 20  # tau-telecom (5 scenarios) cannot be split meaningfully


def _matrices() -> dict[str, OutcomeMatrix]:
    out: dict[str, OutcomeMatrix] = {}
    wm_parts: list[tuple[str, OutcomeMatrix]] = []
    for path in sorted((DATA / "matrices").glob("*_matrix.json")):
        name = path.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(path)
        if len(matrix.scenario_ids()) < MIN_SCENARIOS:
            logger.info("skipping %s: only %d scenarios", name, len(matrix.scenario_ids()))
            continue
        out[name] = matrix
        if name != "routerbench-ours9":
            wm_parts.append((name, matrix))
    if len(wm_parts) >= 2:
        combined = [
            outcome.model_copy(update={"scenario_id": f"{corpus}:{outcome.scenario_id}"})
            for corpus, matrix in wm_parts
            for outcome in matrix.outcomes
        ]
        out["wm-all"] = OutcomeMatrix(pool=wm_parts[0][1].pool, outcomes=combined)
    return out


def _shuffled(matrix: OutcomeMatrix, seed: int = 0) -> OutcomeMatrix:
    """Leak control: permute (reward, success) across scenarios WITHIN each model.

    Marginals per model survive, but any query->model signal is destroyed; a sound router
    collapses to ~best-single here. Costs stay with their original rows so the guard's
    pricier test still sees realistic costs.
    """
    rng = random.Random(seed)
    by_model: dict[str, list[int]] = {}
    for index, outcome in enumerate(matrix.outcomes):
        if outcome.reward is not None:
            by_model.setdefault(outcome.model, []).append(index)
    outcomes = [o.model_copy() for o in matrix.outcomes]
    for rows in by_model.values():
        source = rows[:]
        rng.shuffle(source)
        rewards = [(matrix.outcomes[s].reward, matrix.outcomes[s].success) for s in source]
        for row, (reward, success) in zip(rows, rewards, strict=True):
            outcomes[row].reward = reward
            outcomes[row].success = success
    return OutcomeMatrix(pool=matrix.pool, outcomes=outcomes)


def _dup_text_count(matrix: OutcomeMatrix, fit_ids: list[str], test_ids: list[str]) -> int:
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    fit_texts = {tasks[sid].strip() for sid in fit_ids}
    return sum(1 for sid in test_ids if tasks[sid].strip() in fit_texts)


def run_cell(
    name: str,
    matrix: OutcomeMatrix,
    split_kind: str,
    seed: int,
    *,
    tau_sweep: bool = False,
    control: bool = False,
) -> None:
    """One (matrix, split kind, seed) cell: identical split and embeddings for every variant."""
    spec = EmbedderSpec(dim=DIM)
    if split_kind == "iid":
        fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
    elif split_kind == "ood-cluster":
        fit_ids, test_ids = split_holdout_clusters(
            matrix, embedder=spec, test_fraction=0.3, seed=seed
        )
    elif split_kind == "ood-task":
        fit_ids, test_ids = split_holdout_tasks(matrix, test_fraction=0.3, seed=seed)
    else:
        raise ValueError(f"unknown split kind {split_kind}")
    assert not set(fit_ids) & set(test_ids), "split leaked: fit/test overlap"
    dups = _dup_text_count(matrix, fit_ids, test_ids)

    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    embedder = HashingEmbedder(dim=DIM)
    test_vecs = Normalizer(norm="l2").transform(
        np.asarray(embedder.embed([tasks[sid] for sid in test_ids]))
    )
    ts = datetime.now(tz=UTC).isoformat()

    best_name, _acc, _cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)
    notes = (
        f"best_single={best_name}; oracle acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; "
        f"embedder=hashing-{DIM}; split={split_kind}; dup_test_texts_in_fit={dups}"
    )

    def record(variant: str, params: dict, result, baseline=None) -> None:  # noqa: ANN001
        base = baseline or best_eval
        append_run(
            RunRecord(
                run_id=f"r2-{name}-{split_kind}-s{seed}-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix=name,
                variant=f"r2-{variant}",
                params={**params, "split": split_kind},
                split_seed=seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(test_ids),
                result=result,
                baselines={"best_single": base},
                notes=notes,
            ),
            RUNS,
        )
        logger.info(
            "%s/%s/s%d %s: acc=%.4f cost=$%.5f (best-single %.4f/$%.5f)",
            name,
            split_kind,
            seed,
            f"r2-{variant}",
            result.accuracy,
            result.cost_per_call,
            base.accuracy,
            base.cost_per_call,
        )

    record("best-single", {"model": best_name}, best_eval)

    # Incumbent: guarded Avengers rank router, champion config.
    rank_policy = fit_rank_policy(
        matrix,
        fit_ids=fit_ids,
        embedder=spec,
        n_clusters=RANK_K,
        seed=42,
        guard_model=best_name,
        min_support=4,
        guard_margin=GUARD_MARGIN,
        fitted_from=f"{name} {split_kind} s{seed}",
    )
    decisions = {
        sid: rank_decision(rank_policy, test_vecs[row]).model for row, sid in enumerate(test_ids)
    }
    record(
        "rank",
        {"k": RANK_K, "min_support": 4, "margin": GUARD_MARGIN},
        evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
    )

    # Strawman: the adapted support tilt.
    tilted = rank_policy.model_copy(update={"support_tilt_gamma": 0.5})
    decisions = {
        sid: rank_decision(tilted, test_vecs[row]).model for row, sid in enumerate(test_ids)
    }
    record(
        "rank-tilt",
        {"k": RANK_K, "gamma": 0.5},
        evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
    )

    # Faithful ProxRouter, both reference sets, guarded + unguarded.
    tau_values = [TAU_INV, 5.0, 50.0] if tau_sweep else [TAU_INV]
    for tau_inv in tau_values:
        suffix = "" if tau_inv == TAU_INV else f"-t{tau_inv:g}"
        km = fit_km_prox(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            n_clusters=PROX_K,
            seed=42,
            tau_inv=tau_inv,
            fitted_from=f"{name} {split_kind} s{seed}",
        )
        knn = fit_knn_prox(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            knn_k=KNN_K,
            tau_inv=tau_inv,
            fitted_from=f"{name} {split_kind} s{seed}",
        )
        for kind, policy, params in (
            ("km-prox", km, {"k": PROX_K, "tau_inv": tau_inv}),
            ("knn-prox", knn, {"knn_k": KNN_K, "tau_inv": tau_inv}),
        ):
            scorer = ProxScorer(policy)
            for guarded in (True, False):
                if not guarded and tau_inv != TAU_INV:
                    continue  # unguarded diagnostics only at the paper's tau
                picks = {
                    sid: scorer.decide(
                        test_vecs[row],
                        guard_model=best_name if guarded else None,
                        guard_margin=GUARD_MARGIN if guarded else 0.0,
                    ).model
                    for row, sid in enumerate(test_ids)
                }
                record(
                    f"{kind}{suffix}" if guarded else f"{kind}-unguarded",
                    {**params, "guard": guarded, "margin": GUARD_MARGIN if guarded else 0.0},
                    evaluate_choices(matrix, test_ids, lambda sid, p=picks: p[sid]),
                )

    # Leak control (requested cells only): shuffled labels must collapse to ~best-single.
    if control:
        shuffled = _shuffled(matrix, seed=0)
        s_best, _a, _c = best_single_model(shuffled, fit_ids=fit_ids, eval_ids=test_ids)
        s_best_eval = evaluate_choices(shuffled, test_ids, lambda _sid: s_best)
        for kind, fitter in (("km-prox", fit_km_prox), ("knn-prox", fit_knn_prox)):
            kwargs = {"n_clusters": PROX_K, "seed": 42} if kind == "km-prox" else {"knn_k": KNN_K}
            policy = fitter(shuffled, fit_ids=fit_ids, embedder=spec, tau_inv=TAU_INV, **kwargs)
            scorer = ProxScorer(policy)
            picks = {
                sid: scorer.decide(
                    test_vecs[row], guard_model=s_best, guard_margin=GUARD_MARGIN
                ).model
                for row, sid in enumerate(test_ids)
            }
            record(
                f"{kind}-shuffled",
                {"control": "labels shuffled within model", "guard": True},
                evaluate_choices(shuffled, test_ids, lambda sid, p=picks: p[sid]),
                baseline=s_best_eval,
            )


def main() -> None:
    args = sys.argv[1:]
    wanted = [a for a in args if not a.startswith("--")]
    splits = ["iid", "ood-cluster", "ood-task"]
    for arg in args:
        if arg.startswith("--splits="):
            splits = arg.split("=", 1)[1].split(",")
    seeds = SPLIT_SEEDS
    for arg in args:
        if arg.startswith("--seeds="):
            seeds = [int(s) for s in arg.split("=", 1)[1].split(",")]
    quick = "--quick" in args

    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        has_prefixes = all(":" in sid for sid in matrix.scenario_ids())
        for split_kind in splits:
            if split_kind == "ood-task" and not has_prefixes:
                continue
            for seed in seeds:
                run_cell(
                    name,
                    matrix,
                    split_kind,
                    seed,
                    tau_sweep=(name == "routerbench-ours9" and not quick),
                    control=(name == "routerbench-ours9" and split_kind == "iid" and seed == 0),
                )
    logger.info("runs -> %s", RUNS)


if __name__ == "__main__":
    main()
