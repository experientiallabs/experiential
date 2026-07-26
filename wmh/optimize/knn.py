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

Operators do not set those knobs. They set `cost_quality`, one number in [0, 1] that
`apply_cost_quality` maps onto them along the measured frontier: 0 = spend for quality,
1 = spend as little as possible.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

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


def bank_cost_scale(bank: KnnBank) -> float:
    """The bank's cost unit: the mean of the per-model mean cell costs (0 when unpriced).

    What `pick_lam` divides by, so the knob reads as "reward points paid per average call"
    rather than per dollar. Averaging the per-model means (not the cells) keeps the unit
    independent of coverage: a model measured on a handful of scenarios cannot pull the unit
    toward its own price. This is the reference implementation's `cost_scale`.
    """
    per_model = bank.mean_costs()
    priced = per_model[~np.isnan(per_model)]
    return float(priced.mean()) if priced.size else 0.0


def bank_floor_sim(bank: KnnBank, floor_q: float) -> float | None:
    """The novelty floor: the `floor_q` quantile of the bank's own nearest-neighbor similarities.

    A query whose best bank similarity falls below this is out of the fit set's coverage, and
    `knn_decision` abstains to the baseline rather than routing off strangers. None when the
    floor is off (`floor_q <= 0`) or the bank is a single row (no self-NN distribution exists).

    Computed row-block by row-block: the full gram matrix of a large bank is quadratic in
    memory, and only its off-diagonal row maxima are wanted.
    """
    if floor_q <= 0.0 or len(bank.scenario_ids) < 2:
        return None
    rows = bank.embeddings
    block = max(1, 4_000_000 // max(rows.shape[0], 1))
    self_nn = np.empty(rows.shape[0], dtype=np.float64)
    for start in range(0, rows.shape[0], block):
        stop = min(start + block, rows.shape[0])
        gram = (rows[start:stop] @ rows.T).astype(np.float64)
        # Blank each row's own column so a row cannot be its own nearest neighbor.
        gram[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        self_nn[start:stop] = gram.max(axis=1)
    return float(np.quantile(self_nn, floor_q))


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
    pick_lam: float = 0.0,
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
    research code) and must be the function that spec describes. `pick_lam` is the cost knob
    (0 = the champion); operators slide it through `apply_cost_quality` rather than refitting,
    since it changes nothing the fit measured.
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
    rag_cap = max(4, -(-n_bank // 2))
    if rag_num > rag_cap:
        # Only a bank-imposed cap adjusts min_pairs: an operator's explicit tightening on a
        # large bank passes through untouched (the relative rule can return more neighbors
        # than rag_num, so min_pairs > rag_num is a legitimate setting).
        rag_num = rag_cap
        min_pairs = min(min_pairs, max(3, rag_num // 2))
    # Novelty floor (R1 promotion-hardening H2): abstain to the baseline when a query's best
    # similarity falls below the floor_q quantile of the bank's own nearest-neighbor
    # similarities. iid wins hold at every q; under task drift coverage degrades gracefully
    # toward all-baseline. 0.0 = off (the exact validated champion).
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
        floor_sim=bank_floor_sim(bank, floor_q),
        pick_lam=pick_lam,
        # Persisted at fit time so serving never recomputes it: the cost knob's denominator is
        # a property of the fit evidence, and a serve-time recompute would make the same dial
        # setting mean different things on different processes.
        cost_scale=bank_cost_scale(bank),
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


# --------------------------------------------------------------------------- the operator dial

# The dial's fixed points. `COST_QUALITY_BALANCED` is where the shipped default sits, so an
# operator who never touches the slider and one who sets 0.25 serve the same policy.
COST_QUALITY_BALANCED = 0.25
COST_QUALITY_Z = DEFAULT_KNN_Z  # every measured anchor was taken at this confidence bar
COST_QUALITY_MAX_FLOOR_Q = 0.5  # the quality end: abstain on the least-covered half of queries
COST_QUALITY_DEFAULT_FLOOR_Q = 0.05  # the shipped novelty floor
# The savings end. NOT the largest lam that exists, the largest one that still buys savings:
# past it the guard reverts the too-cheap picks back to the pricier baseline, so measured cost
# turns around and RISES while quality keeps falling (ours9, asymmetric guard, floor_q=0.05:
# lam=0.03 -> -46.2%, lam=0.05 -> -45.1%, lam=0.08 -> -38.3%, lam=0.12 -> -31.2%). A dial that
# ran past the turn would sell an operator a worse policy on both axes.
COST_QUALITY_MAX_LAM = 0.03

CostQualityPointName = str


class CostQualityKnobs(BaseModel):
    """The policy knobs one dial position maps to (see `apply_cost_quality`)."""

    knn_z: float
    floor_q: float
    pick_lam: float
    guard_mode: Literal["symmetric", "asymmetric"]


class CostQualityAnchor(BaseModel):
    """One MEASURED point on the cost/quality frontier, with the numbers that were observed.

    Quality and cost are both stated against the best single pool model on the same held-out
    split, which is the bar an operator would otherwise be paying for.

    The serialized field names (`s`, `label`, `quality_delta_pt`, `cost_delta_pct`) are the
    platform's contract for the endpoint config response: these rows are the ONLY deltas that
    surface, and a client interpolates between them itself rather than asking for an
    interpolated delta at an arbitrary dial position, which would read as a measurement.
    """

    cost_quality: float = Field(serialization_alias="s")
    named_point: CostQualityPointName = Field(serialization_alias="label")
    quality_delta_points: float = Field(serialization_alias="quality_delta_pt")
    cost_delta_percent: float = Field(serialization_alias="cost_delta_pct")


# Every anchor was measured on routerbench-ours9 (1199 scenarios, 9 models, 70/30 stratified
# splits, 5 seeds, text-embedding-3-large queries) THROUGH THIS CODE by
# `.agents/scripts/validate_cost_quality.py`, which is also the gate that keeps them true.
_MEASURED = (
    # dial, quality points vs best single, cost percent vs best single
    (0.0, 1.14, -13.9),  # matches R1's r1-knn-adapt-floor-q0.5 to the digit
    (0.25, 0.99, -24.7),  # matches R1's r1-knn-adapt-floor-q0.05; the shipped default
    (0.5, 0.87, -40.8),
    (0.75, 0.20, -43.6),
    (1.0, -0.54, -46.2),
)


def cost_quality_knobs(cost_quality: float) -> CostQualityKnobs:
    """Map the dial to policy knobs. See `apply_cost_quality` for what the numbers mean."""
    if not 0.0 <= cost_quality <= 1.0:
        raise ValueError(
            f"cost_quality must be between 0.0 (max quality) and 1.0 (max savings), "
            f"got {cost_quality}"
        )
    quality_leg = min(cost_quality, COST_QUALITY_BALANCED) / COST_QUALITY_BALANCED
    savings_leg = max(0.0, cost_quality - COST_QUALITY_BALANCED) / (1.0 - COST_QUALITY_BALANCED)
    return CostQualityKnobs(
        knn_z=COST_QUALITY_Z,
        # Weighted-endpoint form, not `start + leg * (end - start)`: it returns each endpoint
        # EXACTLY, so dialing to an anchor reproduces that anchor's policy bit for bit rather
        # than landing a float epsilon away from it.
        floor_q=(1.0 - quality_leg) * COST_QUALITY_MAX_FLOOR_Q
        + quality_leg * COST_QUALITY_DEFAULT_FLOOR_Q,
        pick_lam=savings_leg * COST_QUALITY_MAX_LAM,
        # The balanced point keeps the champion's strict bar; asking for savings past it buys
        # the economic bar, which is what lets the cost knob act (see `apply_cost_quality`).
        guard_mode="symmetric" if cost_quality <= COST_QUALITY_BALANCED else "asymmetric",
    )


# Every MEASURED dial position and its name. One list, so an anchor row's label and the label
# reported for an endpoint sitting on that position are the same string by construction, and
# adding a measured position cannot leave a detent unnamed.
COST_QUALITY_NAMED_POINTS: tuple[tuple[float, str], ...] = (
    (0.0, "Quality max"),
    (COST_QUALITY_BALANCED, "Balanced (default)"),
    (0.5, "Cost saver"),
    (0.75, "Deep saver"),
    (1.0, "Max savings"),
)


def cost_quality_named_point(cost_quality: float) -> CostQualityPointName:
    """The label for a dial position: an anchor's name only when it sits EXACTLY on that anchor.

    Every other position is "Custom". A position between two measured anchors must never borrow
    the nearer one's name: the label is what a UI shows next to that anchor's measured quality
    and cost numbers, so claiming it for an unmeasured position claims the numbers too.
    """
    for position, name in COST_QUALITY_NAMED_POINTS:
        if math.isclose(cost_quality, position, abs_tol=1e-9):
            return name
    return "Custom"


# Built through the mapping, so the table an operator reads and the knobs the endpoint serves
# can never be two different sliders, and sorted by dial position because a frontier read out of
# order is a frontier nobody trusts. `.agents/scripts/validate_cost_quality.py` regates the
# measured columns.
COST_QUALITY_ANCHORS: tuple[CostQualityAnchor, ...] = tuple(
    CostQualityAnchor(
        cost_quality=dial,
        named_point=cost_quality_named_point(dial),
        quality_delta_points=quality,
        cost_delta_percent=cost,
    )
    for dial, quality, cost in sorted(_MEASURED)
)


def apply_cost_quality(policy: RoutingPolicy, cost_quality: float) -> RoutingPolicy:
    """Return a copy of `policy` set to one dial position; the original is untouched.

    ONE number an operator sets per endpoint: 0.0 spends for quality, 1.0 spends as little as
    this policy family can, and the mapping walks the measured frontier between them. It has two
    legs, and the balanced point (0.25) between them is exactly the shipped default:

    - Coverage, over [0.0, 0.25]: `floor_q` (the novelty floor) opens from 0.5 to 0.05, so the
      router is allowed to act on thinner and thinner neighbor coverage. The guard keeps its
      strict bar the whole way, which is why this leg costs almost no quality.
    - Price, over (0.25, 1.0]: `pick_lam` (the cost knob) ramps 0 -> 0.03, pricing every
      candidate at `pick_lam * mean_cost / cost_scale` before the argmax, AND the guard moves to
      its economic bar (`guard_mode="asymmetric"`: a cheaper pick is accepted unless the paired
      evidence says it is significantly worse, instead of having to prove it is better).

    The second half of that is not decoration, it is the whole reason the leg works. Under the
    strict bar a cost-tilted cheap pick is usually reverted straight back to the pricier
    baseline, so turning the knob up SPENDS MORE: measured on ours9 at floor_q=0.05, the strict
    bar bottoms out at -27.7% cost (lam=0.01) and is back to -15.1% by lam=0.08, having given up
    1.5 accuracy points on the way. The economic bar is what R1's ablation found makes a cost
    knob act, and it is where every savings-end anchor below comes from. It is also the real
    price of asking for savings: with a lenient bar on the cheaper side, the protection the
    champion ships with covers pricier picks only, and R1's shuffled-label control for that
    configuration (`r1-knn3-asym-shuffled`: -0.15 pt at -38% cost) shows its savings do not
    depend on the neighbor evidence being informative. Treat dial > 0.25 as a decision to spend
    less, not as a free lunch.

    `knn_z` is pinned to 0.5 because every anchor was measured there: an operator who fitted a
    different confidence bar and then slides the dial gets 0.5, deliberately, so one dial
    setting means one policy. Everything else (bank, pool, baseline, neighbor budget) is carried
    through untouched, so no refit is involved and the slide is instant.

    Measured anchors: routerbench-ours9, 5 split seeds, held out, against the best single pool
    model on the same split, all produced by this code
    (`.agents/scripts/validate_cost_quality.py`, which regates them):

        dial   floor_q  lam    guard        quality     cost
        0.00   0.50     0.00   symmetric    +1.14 pt    -13.9%
        0.25   0.05     0.00   symmetric    +0.99 pt    -24.7%   <- the shipped default
        0.50   0.05     0.01   asymmetric   +0.87 pt    -40.8%
        0.75   0.05     0.02   asymmetric   +0.20 pt    -43.6%
        1.00   0.05     0.03   asymmetric   -0.54 pt    -46.2%

    Those five rows are measurements; every other dial position INTERPOLATES THE KNOBS, not the
    outcome. Cost falls monotonically across the anchors and is expected to between them (both
    knobs only ever make the router cheaper), but the curve is not a straight line, quality is
    not monotone (the guard switch at 0.25 measured cheaper AND better on this iid split), and an
    intermediate position is not a measured promise. Re-measure on your own matrix before quoting
    a number for one: these are RouterBench-derived tasks where the pool's models score within a
    few points of each other, which is the regime most favorable to a cost knob.
    """
    if policy.kind != "knn":
        raise ValueError(
            f"the cost/quality dial adjusts knn policies; this one is kind='{policy.kind}'. "
            "Fit with `wmh optimize route fit --kind knn` to get a dialable endpoint."
        )
    knobs = cost_quality_knobs(cost_quality)
    if knobs.pick_lam > 0.0 and policy.cost_scale <= 0.0:
        raise ValueError(
            f"cost_quality={cost_quality:g} turns on the cost knob, but this policy has no "
            "cost_scale (its fit matrix carried no per-episode costs), so there is no price to "
            f"trade against; refit with costs, or stay at or below {COST_QUALITY_BALANCED:g}"
        )
    bank = policy.knn_bank()
    # Absolute, not relative: the knobs come from the dial alone, so re-applying any setting to
    # an already-slid policy lands on the same artifact instead of compounding.
    provenance = (policy.fitted_from or "unknown").split(" | cost_quality=")[0]
    adjusted = policy.model_copy(
        update={
            "knn_z": knobs.knn_z,
            "floor_sim": bank_floor_sim(bank, knobs.floor_q),
            "pick_lam": knobs.pick_lam,
            "guard_mode": knobs.guard_mode,
            "cost_quality": cost_quality,
            "fitted_from": (
                f"{provenance} | cost_quality={cost_quality:g} "
                f"(floor_q={knobs.floor_q:g}, lam={knobs.pick_lam:g}, guard={knobs.guard_mode})"
            ),
        }
    )
    # Hand the copy the bank it was just measured against: a serving swap must not re-read a
    # 10MB sidecar, and a research caller may have attached a bank that is not on disk at all.
    adjusted.attach_bank(bank)
    return adjusted
