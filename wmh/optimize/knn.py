"""The kNN routing fitter: an OutcomeMatrix in, a guarded nearest-neighbor policy out.

This is the validated champion of the routing hill-climb. It is nonparametric: the fit does not
learn weights, it packs the fit split's measured evidence into a bank (L2-normalized task
embeddings plus the per-scenario, per-model reward and cost cells) that serve time retrieves
against. `wmh.optimize.policy.knn_decision` is the algorithm; this module builds what it reads.

Measured on `routerbench-ours9` (1199 scenarios, 9 models, 70/30 stratified splits, queries
embedded with text-embedding-3-large): +1.04 accuracy points over the best single model at -27%
cost per call, winning on 5 of 5 split seeds. The lift comes from the guard, not the retrieval:
unguarded, the same profile router loses to the best single model, because a confident pick off
three lucky neighbors is worse than no pick at all.

Why a separate module from `wmh.optimize.routing`: that module is a faithful replication of one
published router (Avengers cluster-rank) and its docstring is a contract about staying faithful
to it. This family shares nothing with it but the artifact type: no clustering, no parametric
fit, evidence retrieved per request rather than compressed into centroids. Keeping the two fits
apart keeps each one's provenance auditable. They meet at `RoutingPolicy` and at
`evaluate_policy`, which replays any policy kind through the serve-time selection code.

The production contract: `guard_model` is the baseline, pinned by the caller (`--fallback`), and
requests leave it only on evidence. `z` is the confidence knob; raise it to route away less.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wmh.optimize.policy import (
    DEFAULT_KNN_MIN_PAIRS,
    DEFAULT_KNN_Z,
    DEFAULT_RAG_NUM,
    DEFAULT_RAG_THRES,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
)

if TYPE_CHECKING:
    from wmh.optimize.outcomes import OutcomeMatrix
    from wmh.providers.base import Embedder

logger = logging.getLogger(__name__)


def best_single_on_fit(matrix: OutcomeMatrix, fit_ids: list[str]) -> str:
    """The strongest single pool model on `fit_ids`, ties broken toward the cheaper one.

    The default baseline when the caller pins none: routing has to beat the model a user would
    otherwise have picked by hand, so that model is what the guard compares against. A tie on
    quality goes to the cheaper model, the same convention the routing research protocol uses.
    """
    sums: dict[str, tuple[float, float, int]] = {}
    wanted = set(fit_ids)
    for outcome in matrix.outcomes:
        if outcome.scenario_id not in wanted or outcome.reward is None:
            continue
        reward_sum, cost_sum, count = sums.get(outcome.model, (0.0, 0.0, 0))
        sums[outcome.model] = (reward_sum + outcome.reward, cost_sum + outcome.cost_usd, count + 1)
    if not sums:
        raise ValueError(
            "no scored outcomes on the fit scenarios, so there is no baseline to guard against; "
            "check that the matrix carries rewards for the fit split"
        )
    return min(sums, key=lambda model: (-sums[model][0] / sums[model][2], sums[model][1]))


def build_knn_bank(
    matrix: OutcomeMatrix,
    fit_ids: list[str],
    *,
    embedder: Embedder,
) -> KnnBank:
    """Pack the fit split's embeddings and reward/cost cells into a routable bank.

    Cells hold the MEAN over a scenario/model pair's scored episodes, and NaN when that pair was
    never scored: the router weighs a model only over the neighbors it actually ran.
    """
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    missing = [sid for sid in fit_ids if sid not in tasks]
    if missing:
        raise ValueError(f"fit_ids not in the matrix: {sorted(missing)[:5]}")
    if not fit_ids:
        raise ValueError("no fit scenarios; a knn policy has nothing to retrieve against")

    models = [entry.name for entry in matrix.pool]
    row_of = {sid: index for index, sid in enumerate(fit_ids)}
    column_of = {model: index for index, model in enumerate(models)}
    shape = (len(fit_ids), len(models))
    reward_sums = np.zeros(shape)
    cost_sums = np.zeros(shape)
    counts = np.zeros(shape)
    for outcome in matrix.outcomes:
        row = row_of.get(outcome.scenario_id)
        column = column_of.get(outcome.model)
        if row is None or column is None or outcome.reward is None:
            continue
        reward_sums[row, column] += outcome.reward
        cost_sums[row, column] += outcome.cost_usd
        counts[row, column] += 1
    scored = counts > 0
    rewards = np.where(scored, reward_sums / np.maximum(counts, 1), np.nan)
    costs = np.where(scored, cost_sums / np.maximum(counts, 1), np.nan)

    embeddings = np.asarray(embedder.embed([tasks[sid] for sid in fit_ids]), dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0.0, norms, 1.0)
    return KnnBank(
        embeddings=embeddings.astype(np.float32),
        rewards=rewards.astype(np.float32),
        costs=costs.astype(np.float32),
        models=models,
        scenario_ids=list(fit_ids),
    )


def fit_knn_policy(
    matrix: OutcomeMatrix,
    *,
    bank_path: Path,
    fit_ids: list[str] | None = None,
    embedder: EmbedderSpec | None = None,
    embed_with: Embedder | None = None,
    guard_model: str | None = None,
    rag_num: int = DEFAULT_RAG_NUM,
    rag_thres: float = DEFAULT_RAG_THRES,
    z: float = DEFAULT_KNN_Z,
    min_pairs: int = DEFAULT_KNN_MIN_PAIRS,
    se_floor: bool = True,
    floor_q: float = 0.0,
    fitted_from: str | None = None,
) -> RoutingPolicy:
    """Fit a kNN policy on `matrix` (restricted to `fit_ids` when given) and write its sidecar.

    Defaults are the validated champion (rag_num=50, rag_thres=0.95, z=0.5, min_pairs=8,
    se_floor on). `bank_path` receives the evidence bank; write it into the directory the
    policy.json will live in, because that is where serving resolves it from:

        policy = fit_knn_policy(matrix, bank_path=out_dir / KNN_BANK_FILENAME,
                                guard_model="fable-5")
        policy.save(out_dir / POLICY_FILENAME)

    `guard_model` is the pinned baseline: given, it is used verbatim (and must be a pool model
    the matrix scored); None discovers the best single model on the fit split. `embedder` is the
    spec persisted for serve-time query embedding; `embed_with` overrides the function used to
    embed the fit tasks (pass one to reuse an HTTP client, or to serve cached vectors in
    research code) and must be the function that spec describes.
    """
    spec = embedder or EmbedderSpec()
    scenario_ids = fit_ids if fit_ids is not None else matrix.scenario_ids()
    names = {entry.name for entry in matrix.pool}
    if guard_model is not None and guard_model not in names:
        raise ValueError(
            f"guard_model '{guard_model}' is not in the matrix pool (available: {sorted(names)}); "
            "the baseline must be a model the fit measured"
        )
    baseline = guard_model or best_single_on_fit(matrix, scenario_ids)

    bank = build_knn_bank(matrix, scenario_ids, embedder=embed_with or spec.build())
    bank.save(bank_path)
    # Adaptive neighborhood (R1 promotion-hardening H1): scale the neighbor budget and the
    # evidence bar to the bank instead of letting a 50-neighbor budget swallow a 20-row bank
    # (which turns the profile into a global mean and routing to 0%). min() against the
    # caller's values keeps banks >= 2x the budget BIT-IDENTICAL to the validated champion.
    n_bank = len(bank.scenario_ids)
    rag_num = min(rag_num, max(4, -(-n_bank // 2)))
    min_pairs = min(min_pairs, max(3, rag_num // 2))
    # Novelty floor (R1 promotion-hardening H2): abstain to the baseline when a query's best
    # similarity falls below the floor_q quantile of the bank's own nearest-neighbor
    # similarities. iid wins hold at every q; under task drift coverage degrades gracefully
    # toward all-baseline. 0.0 = off (the exact validated champion).
    floor_sim: float | None = None
    if floor_q > 0.0:
        gram = bank.embeddings @ bank.embeddings.T
        np.fill_diagonal(gram, -np.inf)
        self_nn = gram.max(axis=1)
        floor_sim = float(np.quantile(self_nn, floor_q))
    policy = RoutingPolicy(
        kind="knn",
        default_model=baseline,
        guard_model=baseline,
        pool=matrix.pool,
        embedder=spec,
        knn_bank_path=bank_path.name,
        rag_num=rag_num,
        rag_thres=rag_thres,
        knn_z=z,
        knn_min_pairs=min_pairs,
        se_floor=se_floor,
        floor_sim=floor_sim,
        fitted_from=fitted_from,
    )
    policy.attach_bank(bank)
    logger.info(
        "knn fit: %d fit scenarios x %d models, baseline %s%s, z=%g -> %s",
        len(bank.scenario_ids),
        len(bank.models),
        baseline,
        " (pinned)" if guard_model else " (best single on fit)",
        z,
        bank_path,
    )
    return policy
