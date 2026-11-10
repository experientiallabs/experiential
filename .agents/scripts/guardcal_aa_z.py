"""A/A-derived z for the knn guard: execute the 2026-07-28 pre-registration, no re-tuning.

Hypothesis (DECISIONS "routing master" 2026-07-28): the served guard's z should come from a
fit-time A/A control instead of a global constant. Construction, following the #328
aa_report pattern (wmo/optimize/compaction_fit.py: episode-parity pseudo-arms measuring
nothing) pushed through the KNN GUARD's own paired test:

- Split the fit bank's episodes by parity into even/odd cell means (same (scenario, model)
  grid, two independent samples of the same distribution).
- For every fit scenario as a query (self excluded, mirroring serve time where the query is
  never its own neighbor), select neighbors exactly as `knn_decision` does (adaptive
  rag_num/min_pairs as `fit_knn_policy` derives them, rag_thres 0.95).
- For every model, run the guard's paired test on d_j = odd_mean - even_mean over the
  neighbors scored on both sides: certify iff pairs >= min_pairs and |mean_d| clears
  z * se_eff in EITHER direction (se_eff floored at the binomial max SE below
  SE_FLOOR_MAX_PAIRS pairs, the production small-sample rule; no pricier doubling: both
  sides are the same model). Any certification is a noise-certification by construction.
- aa_z = the smallest z in the registered grid {0.5, 0.75, 1.0, 1.5, 2.0} with ZERO
  certifications, floored at 0.5; if even 2.0 is unclean, serve 2.0 and flag.

Registered bars (one read, no tuning after):
- financebench-s80: champion with z=aa_z holds worst-seed dacc >= -0.005, 5 paired seeds.
- routerbench-ours9: picks BIT-IDENTICAL to the z=0.5 champion (single-episode cells make
  the episode A/A vacuous there, so aa_z = 0.5 by construction; stated, not hidden).
- tau-bench-s80: descriptive.

Champion evaluation runs through the PRODUCTION path (fit_knn_policy + knn_decision,
se_floor default on, fit-discovered baseline, floor_q 0 = the validated champion), cached
oai3l vectors only, $0. Runs -> runs/guardcal.jsonl (aa evidence in notes, never params).

Usage: uv run python .agents/scripts/guardcal_aa_z.py [matrix ...]
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import tempfile
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _load(name: str, filename: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # pydantic resolves deferred annotations via sys.modules
    spec.loader.exec_module(module)
    return module


_rb = _load("gc_routerbench", "_gc_routerbench.py")
_rr = _load("gc_routing_runs", "_gc_routing_runs.py")
best_single_model, oracle, split_scenario_ids = (
    _rb.best_single_model,
    _rb.oracle,
    _rb.split_scenario_ids,
)
RunRecord, append_run, evaluate_choices = (
    _rr.RunRecord,
    _rr.append_run,
    _rr.evaluate_choices,
)

from wmo.optimize.knn import fit_knn_policy  # noqa: E402  (vendored loads must precede)
from wmo.optimize.outcomes import OutcomeMatrix  # noqa: E402
from wmo.optimize.policy import (  # noqa: E402
    DEFAULT_RAG_THRES,
    SE_FLOOR_MAX_PAIRS,
    EmbedderSpec,
    knn_decision,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("guardcal")

DATA = Path("~/Desktop/Projects/wmh-routing-data").expanduser()
RUNS = DATA / "runs" / "guardcal.jsonl"
Z_GRID = [0.5, 0.75, 1.0, 1.5, 2.0]
MATRICES = ["financebench-s80", "routerbench-ours9", "tau-bench-s80"]
SEEDS = range(5)


def load_vectors(stem: str, matrix: OutcomeMatrix) -> dict[str, np.ndarray]:
    """Cached oai3l task vectors, row order = matrix.scenario_ids() (the cache's writer)."""
    path = DATA / "cache" / f"{stem}-oai3l-tasks.npy"
    vecs = np.load(path)
    sids = matrix.scenario_ids()
    if vecs.shape[0] != len(sids):
        raise ValueError(f"{path} has {vecs.shape[0]} rows, matrix has {len(sids)} scenarios")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    return {sid: vecs[i] for i, sid in enumerate(sids)}


def adaptive(bank_size: int) -> tuple[int, int]:
    """fit_knn_policy's adaptive neighborhood rule (rag_num, min_pairs)."""
    rag = min(50, max(4, math.ceil(bank_size / 2)))
    return rag, min(8, max(3, rag // 2))


def parity_cells(
    matrix: OutcomeMatrix, fit_ids: set[str]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Even/odd episode-parity cell means over the fit split (the #328 pseudo-arm split)."""
    sums: dict[int, dict[tuple[str, str], list[float]]] = {
        0: defaultdict(list),
        1: defaultdict(list),
    }
    for outcome in matrix.outcomes:
        if outcome.scenario_id in fit_ids and outcome.reward is not None:
            sums[outcome.episode % 2][(outcome.scenario_id, outcome.model)].append(outcome.reward)
    even = {key: float(np.mean(v)) for key, v in sums[0].items()}
    odd = {key: float(np.mean(v)) for key, v in sums[1].items()}
    return even, odd


def aa_certifications(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    vectors: dict[str, np.ndarray],
    z: float,
) -> int:
    """Count (query, model) noise-certifications of the guard's paired test at z."""
    even, odd = parity_cells(matrix, set(fit_ids))
    if not odd:
        return 0  # single-episode bank: the episode A/A is vacuous (stated in findings)
    rag, min_pairs = adaptive(len(fit_ids))
    bank = np.stack([vectors[sid] for sid in fit_ids])
    models = [entry.name for entry in matrix.pool]
    certifications = 0
    for index, sid in enumerate(fit_ids):
        sims = bank @ vectors[sid]
        sims[index] = -np.inf  # serve-time queries are never their own neighbor
        budget = min(rag, sims.shape[0] - 1)
        kth = float(np.sort(sims)[-budget])
        rows = np.flatnonzero(sims > DEFAULT_RAG_THRES * kth)
        if rows.size == 0:
            rows = np.asarray([int(np.argmax(sims))])
        neighbor_ids = [fit_ids[int(j)] for j in rows]
        for model in models:
            diffs = [
                odd[(n, model)] - even[(n, model)]
                for n in neighbor_ids
                if (n, model) in odd and (n, model) in even
            ]
            pairs = len(diffs)
            if pairs < min_pairs:
                continue
            mean_d = float(np.mean(diffs))
            se = float(np.std(diffs, ddof=1)) / pairs**0.5 if pairs > 1 else 0.0
            if 0 < pairs < SE_FLOOR_MAX_PAIRS:
                se = max(se, (0.25 / pairs) ** 0.5)
            if abs(mean_d) > z * se:
                certifications += 1
    return certifications


def aa_z_for_bank(
    matrix: OutcomeMatrix, fit_ids: list[str], vectors: dict[str, np.ndarray]
) -> tuple[float, dict[float, int]]:
    counts = {z: aa_certifications(matrix, fit_ids, vectors, z) for z in Z_GRID}
    for z in Z_GRID:
        if counts[z] == 0:
            return z, counts
    return Z_GRID[-1], counts  # unclean even at 2.0: serve the cap, flagged in notes


class _CachedEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self._mapping = mapping

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, self._mapping[text])) for text in texts]


def _decide(
    matrix: OutcomeMatrix,
    bank_ids: list[str],
    query_sid: str,
    vectors: dict[str, np.ndarray],
    embedder: _CachedEmbedder,
    baseline: str,
    z: float,
) -> str:
    """One production guard decision for one query against one bank (helper for aa2)."""
    with tempfile.TemporaryDirectory() as tmp:
        policy = fit_knn_policy(
            matrix,
            bank_path=Path(tmp) / "bank.npz",
            fit_ids=bank_ids,
            embedder=EmbedderSpec(dim=3072),
            embed_with=embedder,
            guard_model=baseline,
            z=z,
        )
    return knn_decision(policy, vectors[query_sid]).model


def aa2_scenario_parity(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    vectors: dict[str, np.ndarray],
    embedder: _CachedEmbedder,
    baseline: str,
) -> tuple[float, dict[float, int]]:
    """Scenario-parity A/A (follow-up registration): served-decision disagreement count.

    Split fit scenarios by index parity into two exchangeable half-banks; a
    noise-certification at z = a fit query whose served outcome DIFFERS between the halves
    (exactly one routes away, or both route to different models). aa2_z = smallest grid z
    with zero disagreements, floored 0.5, capped 2.0. Policies are fitted once per
    (half, z); queries exclude their own scenario by construction (each query sits in
    exactly one half, and is dropped from that half's bank when it is its own member).
    """
    even_ids = [sid for i, sid in enumerate(fit_ids) if i % 2 == 0]
    odd_ids = [sid for i, sid in enumerate(fit_ids) if i % 2 == 1]
    counts: dict[float, int] = {}
    for z in Z_GRID:
        policies = {}
        for tag, ids in (("even", even_ids), ("odd", odd_ids)):
            with tempfile.TemporaryDirectory() as tmp:
                policies[tag] = fit_knn_policy(
                    matrix,
                    bank_path=Path(tmp) / "bank.npz",
                    fit_ids=ids,
                    embedder=EmbedderSpec(dim=3072),
                    embed_with=embedder,
                    guard_model=baseline,
                    z=z,
                )
        disagreements = 0
        for index, sid in enumerate(fit_ids):
            own, other = ("even", "odd") if index % 2 == 0 else ("odd", "even")
            own_ids = [s for s in (even_ids if own == "even" else odd_ids) if s != sid]
            pick_other = knn_decision(policies[other], vectors[sid]).model
            pick_own = _decide(matrix, own_ids, sid, vectors, embedder, baseline, z)
            if pick_own != pick_other:
                disagreements += 1
        counts[z] = disagreements
        if disagreements == 0:
            break
    aa2_z = next((z for z in Z_GRID if counts.get(z) == 0), Z_GRID[-1])
    return aa2_z, counts


def cmd_aa2(wanted: list[str]) -> None:
    """Follow-up registration: scenario-parity A/A, measured ONCE per cohort."""
    for stem in wanted:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{stem}_matrix.json")
        vectors = load_vectors(stem, matrix)
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            tasks.setdefault(outcome.scenario_id, outcome.task)
        embedder = _CachedEmbedder({tasks[sid]: vectors[sid] for sid in tasks})
        for seed in SEEDS:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            best_eval = evaluate_choices(matrix, test_ids, lambda _s, b=best_name: b)
            aa2_z, counts = aa2_scenario_parity(matrix, fit_ids, vectors, embedder, best_name)
            with tempfile.TemporaryDirectory() as tmp:
                policy = fit_knn_policy(
                    matrix,
                    bank_path=Path(tmp) / "bank.npz",
                    fit_ids=fit_ids,
                    embedder=EmbedderSpec(dim=3072),
                    embed_with=embedder,
                    guard_model=best_name,
                    z=aa2_z,
                )
            picks = {sid: knn_decision(policy, vectors[sid]).model for sid in test_ids}
            result = evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s])
            routed = sum(1 for s in test_ids if picks[s] != best_name)
            append_run(
                RunRecord(
                    run_id=f"{stem}-guardcal-aa2-{uuid.uuid4().hex[:8]}",
                    ts=datetime.now(tz=UTC).isoformat(),
                    matrix=stem,
                    variant="guardcal-aa2-z",
                    params={"z_mode": "aa2-scenario-parity", "guard": "production-knn"},
                    split_seed=seed,
                    fit_scenarios=len(fit_ids),
                    test_scenarios=len(test_ids),
                    result=result,
                    baselines={"best_single": best_eval},
                    notes=(
                        f"GUARDCAL follow-up registration (scenario-parity A/A); "
                        f"best_single={best_name}; aa2_z={aa2_z} aa2_counts={counts} "
                        f"routed_away={routed}/{len(test_ids)}"
                    ),
                ),
                RUNS,
            )
            logger.info(
                "%s seed%d aa2_z=%g counts=%s | champion@aa2_z dacc=%+.4f routed=%d/%d",
                stem,
                seed,
                aa2_z,
                counts,
                result.accuracy - best_eval.accuracy,
                routed,
                len(test_ids),
            )


def main() -> None:
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or MATRICES
    if "--aa2" in sys.argv:
        cmd_aa2(wanted)
        return
    for stem in wanted:
        matrix = OutcomeMatrix.load(DATA / "matrices" / f"{stem}_matrix.json")
        vectors = load_vectors(stem, matrix)
        tasks: dict[str, str] = {}
        for outcome in matrix.outcomes:
            tasks.setdefault(outcome.scenario_id, outcome.task)
        embedder = _CachedEmbedder({tasks[sid]: vectors[sid] for sid in tasks})
        for seed in SEEDS:
            fit_ids, test_ids = split_scenario_ids(matrix, train_fraction=0.7, seed=seed)
            best_name, _, _ = best_single_model(matrix, fit_ids=fit_ids, eval_ids=test_ids)
            best_eval = evaluate_choices(matrix, test_ids, lambda _s, b=best_name: b)
            oracle_acc, oracle_cost = oracle(matrix, test_ids)
            aa_z, counts = aa_z_for_bank(matrix, fit_ids, vectors)

            picks_by_mode: dict[str, dict[str, str]] = {}
            for mode, z in (("champion-z05", 0.5), ("aa-z", aa_z)):
                with tempfile.TemporaryDirectory() as tmp:
                    policy = fit_knn_policy(
                        matrix,
                        bank_path=Path(tmp) / "bank.npz",
                        fit_ids=fit_ids,
                        embedder=EmbedderSpec(dim=3072),
                        embed_with=embedder,
                        guard_model=best_name,
                        z=z,
                    )
                picks = {sid: knn_decision(policy, vectors[sid]).model for sid in test_ids}
                picks_by_mode[mode] = picks
                result = evaluate_choices(matrix, test_ids, lambda s, p=picks: p[s])
                routed = sum(1 for s in test_ids if picks[s] != best_name)
                append_run(
                    RunRecord(
                        run_id=f"{stem}-guardcal-{mode}-{uuid.uuid4().hex[:8]}",
                        ts=datetime.now(tz=UTC).isoformat(),
                        matrix=stem,
                        variant=f"guardcal-{mode}",
                        params={"z_mode": mode, "guard": "production-knn"},
                        split_seed=seed,
                        fit_scenarios=len(fit_ids),
                        test_scenarios=len(test_ids),
                        result=result,
                        baselines={"best_single": best_eval},
                        notes=(
                            f"GUARDCAL pre-registered A/A-z; best_single={best_name}; "
                            f"oracle acc={oracle_acc:.4f} cost=${oracle_cost:.5f}; "
                            f"embedder=openai; aa_z={aa_z} aa_counts={counts} "
                            f"z_used={z} routed_away={routed}/{len(test_ids)}"
                        ),
                    ),
                    RUNS,
                )
                logger.info(
                    "%s seed%d %s(z=%g): acc=%.4f dacc=%+.4f cost%+.1f%% routed=%d/%d aa=%s",
                    stem,
                    seed,
                    mode,
                    z,
                    result.accuracy,
                    result.accuracy - best_eval.accuracy,
                    (result.cost_per_call / best_eval.cost_per_call - 1) * 100,
                    routed,
                    len(test_ids),
                    counts,
                )
            identical = picks_by_mode["champion-z05"] == picks_by_mode["aa-z"]
            logger.info(
                "%s seed%d: aa_z=%g, picks bit-identical to z=0.5 champion: %s",
                stem,
                seed,
                aa_z,
                identical,
            )


if __name__ == "__main__":
    main()
