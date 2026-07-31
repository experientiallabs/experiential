"""The kNN routing fitter: an OutcomeMatrix in, a guarded nearest-neighbor policy out.

This is the validated champion of the routing hill-climb. It is nonparametric: the fit does not
learn weights, it packs the fit split's measured evidence into a bank (L2-normalized task
embeddings plus the per-scenario, per-model reward and cost cells) that serve time retrieves
against. `wmo.optimize.policy.knn_decision` is the algorithm; this module builds what it reads.

Measured on `routerbench-ours9` (1199 scenarios, 9 models, 70/30 stratified splits, queries
embedded with text-embedding-3-large): +1.04 accuracy points over the best single model at -27%
cost per call, winning on 5 of 5 split seeds. The lift comes from the guard, not the retrieval:
unguarded, the same profile router loses to the best single model, because a confident pick off
three lucky neighbors is worse than no pick at all.

Why a separate module from `wmo.optimize.routing`: that module is a faithful replication of one
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

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wmo.optimize.compression import CompressingEmbedder, CompressionConfig
from wmo.optimize.policy import (
    DEFAULT_KNN_MIN_PAIRS,
    DEFAULT_KNN_Z,
    DEFAULT_RAG_NUM,
    DEFAULT_RAG_THRES,
    EmbedderSpec,
    KnnBank,
    RoutingPolicy,
    embedder_provenance,
    knn_bank_path_for,
    write_artifact_atomically,
)
from wmo.optimize.routing import evaluate_policy

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix
    from wmo.providers.base import Embedder

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
        floor_q=floor_q,  # kept beside the threshold: see RoutingPolicy.floor_q
        pick_lam=pick_lam,
        # Persisted at fit time so serving never recomputes it: the cost knob's denominator is
        # a property of the fit evidence, and a serve-time recompute would make the same dial
        # setting mean different things on different processes.
        cost_scale=bank_cost_scale(bank),
        fitted_from=fitted_from,
        fit_scenario_ids=list(scenario_ids),
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


class KnnFitOutcome(BaseModel):
    """A written knn policy plus what replaying it over its own fit matrix scored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: RoutingPolicy
    bank_path: Path
    scenarios: int
    # IN-SAMPLE: every request retrieves its own row. Held-out numbers come from the report.
    fit_accuracy: float
    cost_per_scenario: float
    routed_share: float  # fraction of fit scenarios that left the fallback


def fit_knn_artifact(
    matrix: OutcomeMatrix,
    *,
    out_path: Path,
    matrix_source: str,
    embedder: EmbedderSpec,
    fit_ids: list[str] | None = None,
    fallback: str | None = None,
    z: float = DEFAULT_KNN_Z,
    rag_num: int = DEFAULT_RAG_NUM,
    rag_thres: float = DEFAULT_RAG_THRES,
    min_pairs: int = DEFAULT_KNN_MIN_PAIRS,
    se_floor: bool = True,
    floor_q: float = 0.05,
    compression: CompressionConfig | None = None,
    built: Embedder | None = None,
) -> KnnFitOutcome:
    """Fit a knn policy, write it and its bank sidecar, and replay it over the fit matrix.

    The whole knn `fit` step in one call, so `wmo optimize route fit --kind knn` and the fit stage
    of `wmo optimize model` produce byte-identical artifacts from identical inputs. Three details
    are the reason this is shared rather than reimplemented per caller:

    - ONE embedder is built and used for both the fit and the replay; an azure spec would
      otherwise open a second client and embed every scenario twice.
    - The bank sidecar is named after `out_path` and sits beside it, and the fit records that
      name in the policy JSON, so a second knn fit into the same directory gets its own bank
      instead of overwriting this one's.
    - `fitted_from` records the matrix digest, every knob, and the embedder identity, which is
      what lets `tune_policy_dial` tell a fit from a superseded one.

    Raises:
        ValueError: `rag_thres` is not positive, or the fit itself refuses the matrix.
    """
    if rag_thres <= 0.0:
        raise ValueError("rag_thres must be greater than 0")
    selected_ids = fit_ids if fit_ids is not None else matrix.scenario_ids()
    fit_digest = hashlib.sha256("\0".join(selected_ids).encode("utf-8")).hexdigest()[:16]
    # `built` lets a caller substitute the embedding COMPUTATION (a reproduction run serving
    # recorded vectors from a cache) while `embedder` stays the artifact's recorded identity:
    # the vectors are that spec's vectors wherever they were computed, and serving still
    # rebuilds from the spec. The substitute must be geometry-identical to the spec or the
    # bank it fits is a lie; `wmo reproduce` is the only shipped caller.
    if built is None:
        built = embedder.build()
    if compression is not None:
        # Representation consistency, the C2 rule that makes or breaks a compressed arm: the
        # bank rows and the novelty-floor quantile have to live in the geometry of the text
        # SERVING will embed, which is the compressed text. One wrapped embedder covers the fit
        # and the replay below. Serving does not wrap, because its compression stage runs ahead
        # of the router. Fitting the bank uncompressed while serving compresses is the failure
        # this exists to prevent: the floor abstains, and routing silently stops happening.
        built = CompressingEmbedder(built, compression)
    embed_tag = embedder_provenance(embedder)
    policy = fit_knn_policy(
        matrix,
        bank_path=knn_bank_path_for(out_path),
        fit_ids=selected_ids,
        embedder=embedder,
        embed_with=built,
        guard_model=fallback,
        rag_num=rag_num,
        rag_thres=rag_thres,
        z=z,
        min_pairs=min_pairs,
        se_floor=se_floor,
        floor_q=floor_q,
        fitted_from=(
            f"{matrix_source} fit_ids_sha256={fit_digest} "
            f"knn fallback={fallback or 'auto'} z={z:g} k={rag_num} "
            f"thres={rag_thres:g} pairs={min_pairs} se_floor={se_floor} q={floor_q:g} "
            f"{embed_tag}"
        ),
    )
    if compression is not None:
        # Stamped BEFORE the save, because the artifact on disk is what serving mounts and what
        # the mount gate compares. Both halves: what this endpoint serves, and what its evidence
        # was fitted under. They are the same config here by construction, since the fit just
        # embedded through it, and that identity is what the mount gate re-checks.
        # `model_copy` skips validators, so the caller has already resolved the compressor id
        # (an unknown one must fail before anything is written).
        policy = policy.model_copy(
            update={"compression": compression, "fit_compression": compression}
        )
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, selected_ids, embedder=built)
    return KnnFitOutcome(
        policy=policy,
        bank_path=policy.bank_path(),
        scenarios=result.scenarios,
        fit_accuracy=result.accuracy,
        cost_per_scenario=result.cost_per_scenario,
        routed_share=1.0 - result.model_mix.get(policy.default_model, 0.0),
    )


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

# What `apply_cost_quality` appends to `fitted_from` to record the dial position. Sliding is
# absolute, so the suffix is rewritten (never stacked) and `fit_provenance` strips it back off.
COST_QUALITY_PROVENANCE_MARK = " | cost_quality="

# The WHOLE trailer that mark introduces, anchored to the end of the string. `fit_provenance`
# matches this shape rather than searching for the mark alone, because the mark can also occur
# inside the fit half: `fitted_from` opens with an operator-supplied matrix path, and splitting
# on the first occurrence of a substring that path contains would throw away the digest and
# every fit flag after it, collapsing two unrelated fits onto one identity. The fit flags always
# follow the path, so an occurrence inside the path is never at the end and never matches here.
_DIAL_SUFFIX = re.compile(
    re.escape(COST_QUALITY_PROVENANCE_MARK) + r"[\d.eE+-]+ "
    r"\(floor_q=[\d.eE+-]+, lam=[\d.eE+-]+, guard=(?:symmetric|asymmetric)\)$"
)

CostQualityPointName = str


def fit_provenance(policy: RoutingPolicy) -> str:
    """The fit a policy came from, with any dial suffix stripped off.

    `fitted_from` records the outcome matrix and fit parameters, and `apply_cost_quality`
    appends the dial position to it. This returns the fit half alone, which is the identity two
    artifacts must share to be the same fit at different dial positions: a tuned policy and the
    as-fitted snapshot it was dialed from agree here, a refit does not.
    """
    return _DIAL_SUFFIX.sub("", policy.fitted_from or "unknown")


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
# splits, 5 seeds, text-embedding-3-large queries) THROUGH THIS CODE by an offline validation
# script kept outside the repo. Nothing in CI regates them: re-measure before changing them.
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
# order is a frontier nobody trusts. The measured columns are re-derived offline, not in CI.
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
    model on the same split, all produced by this code (re-measured offline, not in CI):

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
            "Fit with `wmo optimize route fit --kind knn` to get a dialable endpoint."
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
    provenance = fit_provenance(policy)
    adjusted = policy.model_copy(
        update={
            "knn_z": knobs.knn_z,
            "floor_sim": bank_floor_sim(bank, knobs.floor_q),
            "floor_q": knobs.floor_q,
            "pick_lam": knobs.pick_lam,
            "guard_mode": knobs.guard_mode,
            "cost_quality": cost_quality,
            "fitted_from": (
                f"{provenance}{COST_QUALITY_PROVENANCE_MARK}{cost_quality:g} "
                f"(floor_q={knobs.floor_q:g}, lam={knobs.pick_lam:g}, guard={knobs.guard_mode})"
            ),
        }
    )
    # Hand the copy the bank it was just measured against: a serving swap must not re-read a
    # 10MB sidecar, and a research caller may have attached a bank that is not on disk at all.
    adjusted.attach_bank(bank)
    return adjusted


class DialResult(BaseModel):
    """What one applied dial position produced, for the caller to report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cost_quality: float
    named_point: CostQualityPointName
    knobs: CostQualityKnobs
    policy_path: Path
    base_path: Path  # the as-fitted snapshot every later slide is re-applied from


def tune_policy_dial(policy_path: Path, cost_quality: float) -> DialResult:
    """Set a fitted policy's cost/quality dial on disk, without refitting anything.

    The first successful slide copies the un-tuned artifact to `<stem>.base<suffix>` and every
    later one re-reads THAT, so the dial is always applied to the policy as fitted and sliding
    twice never compounds.

    That snapshot is only a valid baseline for the fit it came from, so a snapshot describing a
    different fit is refused rather than silently dialed back over the new one. A refused slide
    writes nothing at all, and every write it does make is atomic.

    Raises:
        ValueError: There is no policy at `policy_path`, the snapshot beside it belongs to a
            superseded fit, or the policy has no dial (see `apply_cost_quality`).
    """
    if not policy_path.is_file():
        # Name the producer, as the not-dialable branch of `apply_cost_quality` does: the default
        # argument is a bare `policy.json`, so this is the first thing a new user hits.
        raise ValueError(
            f"no policy file at {policy_path}; fit one first with "
            "`wmo optimize route fit <matrix.json> --kind knn`, or pass the path of an "
            "installed policy (`wmo optimize route pin` and `wmo optimize model` write it to "
            "<root>/models/<name>/policy.json)"
        )
    policy = RoutingPolicy.load(policy_path)
    base_path = policy_path.with_name(f"{policy_path.stem}.base{policy_path.suffix}")
    as_fitted = RoutingPolicy.load(base_path) if base_path.is_file() else None
    if as_fitted is not None and (as_fitted.kind, fit_provenance(as_fitted)) != (
        policy.kind,
        fit_provenance(policy),
    ):
        # A fit overwrites the policy without touching this snapshot, so a refit leaves one
        # behind that describes a policy that no longer exists. Dialing it would report success
        # while replacing the new fit with a slid copy of the superseded one.
        raise ValueError(
            f"{base_path} is the as-fitted snapshot of a different fit than {policy_path}: the "
            f"snapshot holds kind='{as_fitted.kind}' from '{fit_provenance(as_fitted)}', the "
            f"policy holds kind='{policy.kind}' from '{fit_provenance(policy)}'. Tuning it "
            f"would overwrite the current fit with a dialed copy of the old one. Delete "
            f"{base_path} to re-baseline the dial on the fit that is on disk now."
        )
    tuned = apply_cost_quality(policy if as_fitted is None else as_fitted, cost_quality)
    if as_fitted is None:
        # Preserve the artifact as fitted, so the dial is always re-appliable from the fit and
        # never from an already-slid copy of itself. Written only now that the dial has applied:
        # a snapshot left behind by a REJECTED slide would poison the path for the next fit.
        # Both writes are atomic, so the only interruption that leaves a snapshot behind is one
        # where the policy is still the as-fitted bytes the snapshot copied, which is consistent
        # rather than stale.
        write_artifact_atomically(base_path, policy_path.read_bytes())
    tuned.save(policy_path)
    return DialResult(
        cost_quality=cost_quality,
        named_point=cost_quality_named_point(cost_quality),
        knobs=cost_quality_knobs(cost_quality),
        policy_path=policy_path,
        base_path=base_path,
    )
