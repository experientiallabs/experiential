"""r2 experiment driver: faithful ProxRouter vs rank router, IID and OOD splits.

Runs every r2 variant through the shared `evaluate_choices` evaluator on identical splits per
(matrix, split kind, seed) cell and appends RunRecords to the shared runs/r2.jsonl.

Experiment 1 variants (`run_cell`):
- r2-best-single: the fit-chosen baseline (also every guarded variant's floor).
- r2-rank: the guarded Avengers champion config (K=64, hashing-1024, min_support=4,
  margin=0.03), refit per split; the incumbent to beat.
- r2-rank-tilt: our ADAPTED support-tilt (gamma=0.5), the strawman the faithful method must
  outperform for the adaptation to be retired.
- r2-km-prox / r2-knn-prox: faithful ProxRouter (2510.09852), guarded per protocol; the
  -unguarded twins are diagnostics for how much the guard gives up or saves.
- r2-km-prox-shuffled / r2-knn-prox-shuffled: leak control on ours9 (rewards permuted within
  model): a real method must collapse to ~best-single.

Experiment 2 variants (`run_cell_exp2`, --exp2):
- r2-prox-eb4: knn-prox with empirical-Bayes shrinkage m=4 (fixed), isolating the shrinkage
  effect at the paper's tau.
- r2-prox-val: prox with (kind, tau_inv, shrink_m) chosen on INNER fit-side splits only;
  the legitimate version of experiment 1's post-hoc tau sweep.
- r2-rank-valk: the rank router with its cluster count chosen on inner validation (the
  brief's "k chosen on validation per corpus instead of fixed 64").

Split kinds: iid (split_scenario_ids), ood-cluster (split_holdout_clusters), ood-task
(split_holdout_tasks, matrices with id prefixes only). Seeds 0-4 each.

Usage: uv run python .agents/scripts/run_routing_r2.py [matrix ...]
       [--splits=iid,ood-cluster] [--seeds=0,1,2,3,4] [--quick] [--exp2]
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
from wmh.optimize.proxrouter import ProxPolicy, ProxScorer, fit_km_prox, fit_knn_prox
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

INNER_SEEDS = [10, 11, 12]  # fit-side model-selection splits; outer test never consulted
PROX_GRID_TAU = [5.0, 20.0, 50.0]
PROX_GRID_M = [0.0, 1.0, 4.0, 16.0]
RANK_GRID_K = [8, 16, 32, 64, 128]


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


def _split(
    matrix: OutcomeMatrix, split_kind: str, seed: int, spec: EmbedderSpec
) -> tuple[list[str], list[str]]:
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
    return fit_ids, test_ids


class _Cell:
    """Shared per-cell plumbing: split, embeddings, baseline, and the record helper."""

    def __init__(self, name: str, matrix: OutcomeMatrix, split_kind: str, seed: int) -> None:
        self.name, self.matrix, self.split_kind, self.seed = name, matrix, split_kind, seed
        self.spec = EmbedderSpec(dim=DIM)
        self.fit_ids, self.test_ids = _split(matrix, split_kind, seed, self.spec)
        tasks = {o.scenario_id: o.task for o in matrix.outcomes}
        embedder = HashingEmbedder(dim=DIM)
        self.test_vecs = Normalizer(norm="l2").transform(
            np.asarray(embedder.embed([tasks[sid] for sid in self.test_ids]))
        )
        self.ts = datetime.now(tz=UTC).isoformat()
        self.best_name, _a, _c = best_single_model(
            matrix, fit_ids=self.fit_ids, eval_ids=self.test_ids
        )
        self.best_eval = evaluate_choices(matrix, self.test_ids, lambda _sid: self.best_name)
        oracle_acc, oracle_cost = oracle(matrix, self.test_ids)
        dups = _dup_text_count(matrix, self.fit_ids, self.test_ids)
        self.notes = (
            f"best_single={self.best_name}; oracle acc={oracle_acc:.4f} "
            f"cost=${oracle_cost:.5f}; embedder=hashing-{DIM}; split={split_kind}; "
            f"dup_test_texts_in_fit={dups}"
        )

    def record(self, variant: str, params: dict, result, baseline=None) -> None:  # noqa: ANN001
        base = baseline or self.best_eval
        append_run(
            RunRecord(
                run_id=(
                    f"r2-{self.name}-{self.split_kind}-s{self.seed}-{variant}-"
                    f"{uuid.uuid4().hex[:8]}"
                ),
                ts=self.ts,
                matrix=self.name,
                variant=f"r2-{variant}",
                params={**params, "split": self.split_kind},
                split_seed=self.seed,
                fit_scenarios=len(self.fit_ids),
                test_scenarios=len(self.test_ids),
                result=result,
                baselines={"best_single": base},
                notes=self.notes,
            ),
            RUNS,
        )
        logger.info(
            "%s/%s/s%d %s: acc=%.4f cost=$%.5f (best-single %.4f/$%.5f)",
            self.name,
            self.split_kind,
            self.seed,
            f"r2-{variant}",
            result.accuracy,
            result.cost_per_call,
            base.accuracy,
            base.cost_per_call,
        )

    def prox_picks(self, policy: ProxPolicy, guard: str | None) -> dict[str, str]:
        scorer = ProxScorer(policy)
        return {
            sid: scorer.decide(
                self.test_vecs[row],
                guard_model=guard,
                guard_margin=GUARD_MARGIN if guard else 0.0,
            ).model
            for row, sid in enumerate(self.test_ids)
        }


def run_cell(
    name: str,
    matrix: OutcomeMatrix,
    split_kind: str,
    seed: int,
    *,
    tau_sweep: bool = False,
    control: bool = False,
) -> None:
    """Experiment 1: one (matrix, split kind, seed) cell, every exp-1 variant."""
    cell = _Cell(name, matrix, split_kind, seed)
    fit_ids, test_ids, spec = cell.fit_ids, cell.test_ids, cell.spec
    best_name = cell.best_name

    cell.record("best-single", {"model": best_name}, cell.best_eval)

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
        sid: rank_decision(rank_policy, cell.test_vecs[row]).model
        for row, sid in enumerate(test_ids)
    }
    cell.record(
        "rank",
        {"k": RANK_K, "min_support": 4, "margin": GUARD_MARGIN},
        evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
    )

    # Strawman: the adapted support tilt.
    tilted = rank_policy.model_copy(update={"support_tilt_gamma": 0.5})
    decisions = {
        sid: rank_decision(tilted, cell.test_vecs[row]).model for row, sid in enumerate(test_ids)
    }
    cell.record(
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
            for guarded in (True, False):
                if not guarded and tau_inv != TAU_INV:
                    continue  # unguarded diagnostics only at the paper's tau
                picks = cell.prox_picks(policy, best_name if guarded else None)
                cell.record(
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
                    cell.test_vecs[row], guard_model=s_best, guard_margin=GUARD_MARGIN
                ).model
                for row, sid in enumerate(test_ids)
            }
            cell.record(
                f"{kind}-shuffled",
                {"control": "labels shuffled within model", "guard": True},
                evaluate_choices(shuffled, test_ids, lambda sid, p=picks: p[sid]),
                baseline=s_best_eval,
            )


def run_cell_exp2(name: str, matrix: OutcomeMatrix, split_kind: str, seed: int) -> None:
    """Experiment 2: EB shrinkage, validated prox knobs, validated rank K.

    Knobs are chosen on INNER splits of the fit side only (3 inner seeds), so the outer test
    stays untouched by selection.
    """
    cell = _Cell(name, matrix, split_kind, seed)
    fit_ids, test_ids, spec = cell.fit_ids, cell.test_ids, cell.spec

    # (a) Shrinkage-only ablation: knn-prox, fixed m=4, paper tau.
    knn = fit_knn_prox(
        matrix,
        fit_ids=fit_ids,
        embedder=spec,
        knn_k=KNN_K,
        tau_inv=TAU_INV,
        fitted_from=f"{name} {split_kind} s{seed}",
    )
    eb4 = knn.model_copy(update={"shrink_m": 4.0})
    picks = cell.prox_picks(eb4, cell.best_name)
    cell.record(
        "prox-eb4",
        {"kind": "knn", "knn_k": KNN_K, "tau_inv": TAU_INV, "shrink_m": 4.0, "guard": True},
        evaluate_choices(matrix, test_ids, lambda sid, p=picks: p[sid]),
    )

    # (b) Validated prox: kind x tau_inv x shrink_m chosen on inner splits of fit only.
    kind, tau_inv, shrink_m, inner_acc = _select_prox_knobs(matrix, fit_ids, spec)
    if kind == "km":
        chosen = fit_km_prox(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            n_clusters=PROX_K,
            seed=42,
            tau_inv=tau_inv,
            fitted_from=f"{name} {split_kind} s{seed}",
        )
    else:
        chosen = fit_knn_prox(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            knn_k=KNN_K,
            tau_inv=tau_inv,
            fitted_from=f"{name} {split_kind} s{seed}",
        )
    chosen = chosen.model_copy(update={"shrink_m": shrink_m})
    picks = cell.prox_picks(chosen, cell.best_name)
    cell.record(
        "prox-val",
        {
            "kind": kind,
            "tau_inv": tau_inv,
            "shrink_m": shrink_m,
            "inner_acc": round(inner_acc, 4),
            "guard": True,
        },
        evaluate_choices(matrix, test_ids, lambda sid, p=picks: p[sid]),
    )

    # (c) Rank with validation-chosen cluster count.
    rank_k, rank_inner_acc = _select_rank_k(matrix, fit_ids, spec)
    policy = fit_rank_policy(
        matrix,
        fit_ids=fit_ids,
        embedder=spec,
        n_clusters=rank_k,
        seed=42,
        guard_model=cell.best_name,
        min_support=4,
        guard_margin=GUARD_MARGIN,
        fitted_from=f"{name} {split_kind} s{seed}",
    )
    decisions = {
        sid: rank_decision(policy, cell.test_vecs[row]).model for row, sid in enumerate(test_ids)
    }
    cell.record(
        "rank-valk",
        {
            "k": rank_k,
            "inner_acc": round(rank_inner_acc, 4),
            "min_support": 4,
            "margin": GUARD_MARGIN,
        },
        evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
    )


def _sub_matrix(matrix: OutcomeMatrix, ids: list[str]) -> OutcomeMatrix:
    wanted = set(ids)
    return OutcomeMatrix(
        pool=matrix.pool, outcomes=[o for o in matrix.outcomes if o.scenario_id in wanted]
    )


def _select_prox_knobs(
    matrix: OutcomeMatrix, fit_ids: list[str], spec: EmbedderSpec
) -> tuple[str, float, float, float]:
    """Choose (kind, tau_inv, shrink_m) on inner fit-side splits; test never consulted.

    tau_inv and shrink_m are decision-time knobs, so each inner seed needs only two fits
    (one km, one knn); the grid is swept via model_copy. Ties prefer larger shrink_m then
    smaller tau_inv (the more conservative policy) then cheaper cost.
    """
    sub = _sub_matrix(matrix, fit_ids)
    embedder = HashingEmbedder(dim=DIM)
    tasks = {o.scenario_id: o.task for o in sub.outcomes}
    totals: dict[tuple[str, float, float], list[tuple[float, float]]] = {}
    for inner_seed in INNER_SEEDS:
        ifit, ival = split_scenario_ids(sub, train_fraction=0.7, seed=inner_seed)
        ibest, _a, _c = best_single_model(sub, fit_ids=ifit, eval_ids=ival)
        ival_vecs = Normalizer(norm="l2").transform(
            np.asarray(embedder.embed([tasks[sid] for sid in ival]))
        )
        base = {
            "km": fit_km_prox(sub, fit_ids=ifit, embedder=spec, n_clusters=PROX_K, seed=42),
            "knn": fit_knn_prox(sub, fit_ids=ifit, embedder=spec, knn_k=KNN_K),
        }
        for kind, policy in base.items():
            for tau_inv in PROX_GRID_TAU:
                for m in PROX_GRID_M:
                    tuned = policy.model_copy(update={"tau_inv": tau_inv, "shrink_m": m})
                    scorer = ProxScorer(tuned)
                    picks = {
                        sid: scorer.decide(
                            ival_vecs[row], guard_model=ibest, guard_margin=GUARD_MARGIN
                        ).model
                        for row, sid in enumerate(ival)
                    }
                    result = evaluate_choices(sub, ival, lambda sid, p=picks: p[sid])
                    totals.setdefault((kind, tau_inv, m), []).append(
                        (result.accuracy, result.cost_per_call)
                    )

    def sort_key(item):  # noqa: ANN001, ANN202
        (kind, tau_inv, m), vals = item
        acc = sum(a for a, _c in vals) / len(vals)
        cost = sum(c for _a, c in vals) / len(vals)
        return (-round(acc, 6), -m, tau_inv, round(cost, 8), kind)

    (kind, tau_inv, m), vals = min(totals.items(), key=sort_key)
    return kind, tau_inv, m, sum(a for a, _c in vals) / len(vals)


def _select_rank_k(
    matrix: OutcomeMatrix, fit_ids: list[str], spec: EmbedderSpec
) -> tuple[int, float]:
    """Choose the rank router's cluster count on inner fit-side splits."""
    sub = _sub_matrix(matrix, fit_ids)
    embedder = HashingEmbedder(dim=DIM)
    tasks = {o.scenario_id: o.task for o in sub.outcomes}
    totals: dict[int, list[float]] = {}
    for inner_seed in INNER_SEEDS:
        ifit, ival = split_scenario_ids(sub, train_fraction=0.7, seed=inner_seed)
        ibest, _a, _c = best_single_model(sub, fit_ids=ifit, eval_ids=ival)
        ival_vecs = Normalizer(norm="l2").transform(
            np.asarray(embedder.embed([tasks[sid] for sid in ival]))
        )
        # Tiny fit sides (OOD splits of 25-scenario corpora) can undercut the whole grid;
        # fall back to K=len(ifit) (all-singleton clustering, the guard's safest shape).
        grid = [k for k in RANK_GRID_K if k <= len(ifit)] or [len(ifit)]
        for k in grid:
            policy = fit_rank_policy(
                sub,
                fit_ids=ifit,
                embedder=spec,
                n_clusters=k,
                seed=42,
                guard_model=ibest,
                min_support=4,
                guard_margin=GUARD_MARGIN,
            )
            picks = {
                sid: rank_decision(policy, ival_vecs[row]).model for row, sid in enumerate(ival)
            }
            result = evaluate_choices(sub, ival, lambda sid, p=picks: p[sid])
            totals.setdefault(k, []).append(result.accuracy)
    best_k = min(totals.items(), key=lambda kv: (-sum(kv[1]) / len(kv[1]), kv[0]))[0]
    return best_k, sum(totals[best_k]) / len(totals[best_k])


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
    exp2 = "--exp2" in args

    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        has_prefixes = all(":" in sid for sid in matrix.scenario_ids())
        for split_kind in splits:
            if split_kind == "ood-task" and not has_prefixes:
                continue
            for seed in seeds:
                if exp2:
                    run_cell_exp2(name, matrix, split_kind, seed)
                else:
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
