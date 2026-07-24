"""Controlled routing ablations: every variant x every available matrix -> runs.jsonl + reports.

Variants (identical splits, embeddings, and evaluator per matrix): best-single (fit-chosen),
rank router (Avengers replication, cost-knob sweep), IRT head (cost-knob sweep). Each run
persists a RunRecord with the explain block; per-run markdown reports land in
.wmh/evals/reports/. The dashboard (build_dashboard.py) renders runs.jsonl.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.preprocessing import Normalizer

from wmh.optimize.irt import fit_irt_head
from wmh.optimize.outcomes import OutcomeMatrix
from wmh.optimize.policy import EmbedderSpec, rank_decision
from wmh.optimize.routing import fit_rank_policy, rerank_policy
from wmh.research.routing_runs import RunRecord, append_run, evaluate_choices, run_report
from wmh.retrieval.embedders import HashingEmbedder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ablations")

RUNS = Path(".wmh/evals/runs.jsonl")
REPORTS = Path(".wmh/evals/reports")
DIM = 1024
K = 64
LAMS = [0.0, 0.02, 0.1]
# Multiple disjoint-ish splits: every scenario reaches test across seeds, so per-matrix
# signal comes from mean +- spread over seeds, not one cherry-pickable 70/30 draw.
SPLIT_SEEDS = [0, 1, 2, 3, 4]


def _matrices() -> dict[str, OutcomeMatrix]:
    out: dict[str, OutcomeMatrix] = {}
    rb = Path("/Users/silen/Desktop/Projects/router-refs/routerbench_0shot.pkl")
    if rb.exists():
        from wmh.research.routerbench import load_routerbench

        out["routerbench"] = load_routerbench(rb)
    lrb = Path("/Users/silen/Desktop/Projects/router-refs/LLMRouterBench/results/bench-release")
    if lrb.is_dir():
        from wmh.research.llmrouterbench import load_llmrouterbench

        out["llmrouterbench-flagship"] = load_llmrouterbench(lrb)
    ours = Path(".wmh/evals/routerbench/ours_matrix.json")
    if ours.exists():
        out["routerbench-ours9"] = OutcomeMatrix.load(ours)
    wm_matrices = []
    for wm in sorted(Path(".wmh/evals/wm").glob("*_matrix.json")):
        corpus = wm.stem.removesuffix("_matrix")
        matrix = OutcomeMatrix.load(wm)
        out[f"wm-{corpus}"] = matrix
        wm_matrices.append((corpus, matrix))
    if len(wm_matrices) >= 2:
        # The pooled cross-corpus aggregate: per-corpus test sides are tiny, so THIS is where
        # the statistically real wm signal lives. Scenario ids get a corpus prefix, which also
        # makes the stratified split per-corpus (each corpus contributes to fit AND test).
        combined = []
        for corpus, matrix in wm_matrices:
            for outcome in matrix.outcomes:
                clone = outcome.model_copy(
                    update={"scenario_id": f"{corpus}:{outcome.scenario_id}"}
                )
                combined.append(clone)
        out["wm-all"] = OutcomeMatrix(pool=wm_matrices[0][1].pool, outcomes=combined)
    return out


def _emit(record: RunRecord) -> None:
    append_run(record, RUNS)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{record.run_id}.md").write_text(run_report(record), encoding="utf-8")
    result = record.result
    logger.info(
        "%s/%s %s: acc=%.4f cost=$%.5f p50=%s",
        record.matrix,
        record.variant,
        record.params,
        result.accuracy,
        result.cost_per_call,
        f"{result.latency_p50_s:.2f}s" if result.latency_p50_s else "-",
    )


def run_matrix(name: str, matrix: OutcomeMatrix, split_seed: int = 0) -> None:
    from wmh.research.routerbench import best_single_model, oracle, split_scenario_ids

    fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=split_seed)
    tasks = {o.scenario_id: o.task for o in matrix.outcomes}
    embedder = HashingEmbedder(dim=DIM)
    fit_vecs = np.asarray(embedder.embed([tasks[s] for s in fit_ids]))
    test_vecs = Normalizer(norm="l2").transform(
        np.asarray(embedder.embed([tasks[s] for s in test_ids]))
    )
    ts = datetime.now(tz=UTC).isoformat()

    best_name, _acc, _cost = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
    best_eval = evaluate_choices(matrix, test_ids, lambda _sid: best_name)
    oracle_acc, oracle_cost = oracle(matrix, test_ids)

    def record(variant: str, params: dict, result) -> None:  # noqa: ANN001
        _emit(
            RunRecord(
                run_id=f"{name}-{variant}-{uuid.uuid4().hex[:8]}",
                ts=ts,
                matrix=name,
                variant=variant,
                params=params,
                split_seed=split_seed,
                fit_scenarios=len(fit_ids),
                test_scenarios=len(test_ids),
                result=result,
                baselines={"best_single": best_eval},
                notes=f"best_single={best_name}; oracle acc={oracle_acc:.4f} "
                f"cost=${oracle_cost:.5f}; embedder=hashing-{DIM}",
            )
        )

    record("best-single", {"model": best_name}, best_eval)

    started = time.monotonic()
    policy = fit_rank_policy(
        matrix, fit_ids=fit_ids, embedder=EmbedderSpec(dim=DIM), n_clusters=K, seed=42,
        fitted_from=f"{name} split{split_seed}",
    )
    logger.info("%s: rank fit in %.0fs", name, time.monotonic() - started)
    for lam in LAMS:
        swept = rerank_policy(policy, cost_weight=lam) if lam else policy
        decisions = {
            sid: rank_decision(swept, test_vecs[index]).model
            for index, sid in enumerate(test_ids)
        }
        record(
            "rank", {"k": K, "lam": lam},
            evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
        )

    started = time.monotonic()
    head = fit_irt_head(
        matrix, scenario_ids=fit_ids, embeddings=fit_vecs, seed=42, epochs=300,
        hidden=256, dim=64,
    )
    logger.info("%s: irt fit in %.0fs (pairs=%d)", name, time.monotonic() - started, head.pairs_trained)
    costs_by_model: dict[str, list[float]] = {}
    fit_set = set(fit_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id in fit_set and outcome.reward is not None:
            costs_by_model.setdefault(outcome.model, []).append(outcome.cost_usd)
    mean_cost = {m: sum(v) / len(v) for m, v in costs_by_model.items() if v}
    cost_scale = sum(mean_cost.values()) / len(mean_cost)
    probs = np.stack([head.predict(vec) for vec in test_vecs])  # [T, M]
    penalties = np.asarray(
        [mean_cost.get(m, cost_scale) / cost_scale for m in head.models]
    )
    for lam in LAMS:
        scores = probs - lam * penalties
        picks = [head.models[int(index)] for index in np.argmax(scores, axis=1)]
        decisions = dict(zip(test_ids, picks, strict=True))
        record(
            "irt", {"hidden": 256, "dim": 64, "epochs": 300, "lam": lam},
            evaluate_choices(matrix, test_ids, lambda sid: decisions[sid]),
        )


def main() -> None:
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = SPLIT_SEEDS if "--seeds" in sys.argv else [0]
    for name, matrix in _matrices().items():
        if wanted and name not in wanted:
            continue
        for seed in seeds:
            run_matrix(name, matrix, split_seed=seed)
    logger.info("runs -> %s, reports -> %s", RUNS, REPORTS)


if __name__ == "__main__":
    main()
