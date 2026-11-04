"""Data-chosen clusters of the tag space, named by a language model, and admission by cluster.

Every tagged trace is a row of probabilities over one or more tag matrices (:func:`tag_space`
joins them by trace, prefixing each column with its view). The rows are modelled as a mixture of
independent Bernoulli components, with each probability read as a fractional observation, and
fitted by expectation maximization from several seeded k-means++ starts per cluster count.

The number of clusters is chosen by the data, not by a rule. For every count in the registered
range the seeded fits give a stability score (the mean pairwise adjusted Rand index of their hard
assignments), a log likelihood with its Bayesian information criterion, and a cross-validated log
likelihood per trace over folds assigned by a hash of each trace identity. By default the chosen
count is the fewest occupied clusters whose held-out likelihood is within one standard error of
the best, among counts at least ``min_stability`` stable: a soft answer carries less information
than a bit, which a parameter-count penalty overstates, while held-out likelihood measures it
directly. ``criterion="bic"`` selects by BIC instead. A count the data supports only from lucky
starts is never chosen, one cluster is always a candidate, and every per-count score is saved.

Clusters are named by a language model in an agentic loop
(:mod:`exp.simulation.mining.tag_cluster_names`). Clusterings are saved as content-addressed
``tag-clusters`` artifacts and replay from them; the fit itself is deterministic per seed.

:func:`admit_by_cluster` decides which clusters, and how many traces of each, enter a downstream
corpus. It reads only the traces a caller declares eligible (world learning, for a world model),
refuses a clustering that holds a held-out trace, and records why each trace was admitted.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.mining.tag_selection import DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix

logger = logging.getLogger(__name__)

TAG_CLUSTERS_PROTOCOL = "tag-clusters-v1"
"""Version of the mixture fit, count selection, naming loop and admission; bump when any changes."""

CLUSTERS_TYPE = "tag-clusters"
"""Artifact type of a saved :class:`TagClustering` (``clustering.json``)."""

Matrix = npt.NDArray[np.float64]


class ClusteringSpec(ContractModel):
    """How the tag space is clustered and how the cluster count is chosen.

    Attributes:
        min_clusters: Smallest count considered; one lets a structureless space stay whole.
        max_clusters: Largest count considered, capped at the number of traces and at the
            smallest cross-validation training fold.
        seeds: Seeded starts per count; the best fit is kept, all of them measure stability.
        max_iterations: EM iterations per start.
        tolerance: Relative log-likelihood improvement below which a start has converged.
        smoothing: Beta pseudo-count added to each proposition's yes and no per component.
        min_stability: Mean pairwise adjusted Rand index a count needs to be chosen.
        threshold: Component probability strictly above which a proposition fires.
        exemplars: Most typical members kept per cluster for naming.
        uncertain_below: Assignment probability below which a member is uncertain.
        criterion: ``held_out`` chooses among stable counts the fewest clusters whose
            cross-validated log likelihood per trace is within one standard error of the best
            (soft answers carry less information than a bit, which a parameter-count penalty
            overstates); ``bic`` chooses the lowest BIC among stable counts.
        folds: Cross-validation folds, assigned by a hash of each trace identity.
        fold_seeds: Seeded starts per fold fit; the best on the training rows is scored.
    """

    min_clusters: int = Field(default=1, ge=1, le=64)
    max_clusters: int = Field(default=12, ge=1, le=64)
    seeds: int = Field(default=8, ge=2, le=64)
    max_iterations: int = Field(default=200, ge=1, le=10_000)
    tolerance: float = Field(default=1e-7, gt=0.0)
    smoothing: float = Field(default=0.5, gt=0.0)
    min_stability: float = Field(default=0.8, ge=-1.0, le=1.0)
    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)
    exemplars: int = Field(default=3, ge=1, le=16)
    uncertain_below: float = Field(default=0.6, gt=0.0, le=1.0)
    criterion: Literal["held_out", "bic"] = "held_out"
    folds: int = Field(default=5, ge=2, le=20)
    fold_seeds: int = Field(default=2, ge=1, le=16)

    @model_validator(mode="after")
    def _ordered(self) -> ClusteringSpec:
        """Require a non-empty count range."""
        if self.min_clusters > self.max_clusters:
            raise ValueError("min_clusters must not exceed max_clusters")
        return self


class TagSource(ContractModel):
    """One matrix joined into the tag space.

    Attributes:
        bank_id: Bank it answered.
        bank_version: Bank revision.
        view: Rendered view; its columns are prefixed ``<view>:``.
        matrix_sha256: Digest of the matrix.
    """

    bank_id: str
    bank_version: int
    view: str
    matrix_sha256: str


class ClusterCountScore(ContractModel):
    """The evidence for one candidate cluster count.

    Attributes:
        clusters: The count.
        log_likelihood: Best log likelihood over the seeded starts.
        parameters: Free parameters, ``k * columns + k - 1``.
        bic: ``parameters * ln(traces) - 2 * log_likelihood``; lower is better.
        occupied: Components of the best fit that hold at least one trace.
        held_out: Cross-validated log likelihood per trace; higher is better.
        held_out_gap: How far ``held_out`` falls below the best count's, per trace.
        held_out_error: Standard error of that gap over traces.
        stability: Mean pairwise adjusted Rand index of the starts' hard assignments.
        stable: Whether the stability reaches ``min_stability``.
        selected: Whether this count was chosen.
    """

    clusters: int = Field(ge=1)
    log_likelihood: float
    parameters: int = Field(ge=0)
    bic: float
    occupied: int = Field(ge=1)
    held_out: float
    held_out_gap: float = 0.0
    held_out_error: float = 0.0
    stability: float
    stable: bool
    selected: bool = False


class TagCluster(ContractModel):
    """One cluster of the chosen fit.

    Attributes:
        cluster_id: ``c00``, ``c01``, ... by size, largest first.
        size: Members (hard assignment).
        weight: Mixture weight.
        propositions: Each column's probability in this component.
        fired: Columns strictly above the threshold, most probable first.
        distinctive: Columns most above their corpus mean (by at least 0.2), largest lift first.
        members: Member traces, most typical first.
        exemplars: The most typical members.
        uncertain_members: Members whose assignment probability is below ``uncertain_below``.
    """

    cluster_id: str
    size: int = Field(ge=0)
    weight: float = Field(ge=0.0, le=1.0)
    propositions: dict[str, float]
    fired: tuple[str, ...] = ()
    distinctive: tuple[str, ...] = ()
    members: tuple[str, ...] = ()
    exemplars: tuple[str, ...] = ()
    uncertain_members: tuple[str, ...] = ()


class TagClustering(ContractModel):
    """The ``tag-clusters`` record: the chosen mixture over one tag space and how it was chosen.

    Attributes:
        protocol: Tag-clusters protocol.
        spec: The registration.
        sources: Matrices joined into the space.
        columns: Column identities, ``<view>:<question_id>``.
        trace_ids: Clustered traces in row order.
        dropped: Traces missing from some joined matrix, left out of the space.
        scores: One score per candidate count.
        clusters: The chosen fit's clusters.
        assignments: Each trace's cluster.
        confidence: Each trace's assignment probability.
        reason: Why this count was chosen.
    """

    protocol: str = TAG_CLUSTERS_PROTOCOL
    spec: ClusteringSpec
    sources: tuple[TagSource, ...]
    columns: tuple[str, ...]
    trace_ids: tuple[str, ...]
    dropped: tuple[str, ...] = ()
    scores: tuple[ClusterCountScore, ...]
    clusters: tuple[TagCluster, ...]
    assignments: dict[str, str]
    confidence: dict[str, float]
    reason: str

    def identity(self) -> str:
        """Return the content-addressed artifact identity of this clustering."""
        return stable_id(CLUSTERS_TYPE, {"protocol": self.protocol, "record": sha256_json(self)})

    def cluster(self, cluster_id: str) -> TagCluster:
        """Return one cluster.

        Raises:
            KeyError: No cluster has that identity.
        """
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        raise KeyError(f"clustering has no cluster {cluster_id}")


@dataclass(frozen=True)
class _Fit:
    """One EM fit.

    Attributes:
        log_likelihood: Final data log likelihood.
        weights: Mixture weights, shape ``(k,)``.
        theta: Component probabilities, shape ``(k, columns)``.
        unsmoothed: Component probabilities without the smoothing prior, which shrinks small
            components more than large ones; held-out scoring reads these so it compares counts
            on the data rather than on the prior.
        responsibilities: Assignment probabilities, shape ``(traces, k)``.
    """

    log_likelihood: float
    weights: Matrix
    theta: Matrix
    unsmoothed: Matrix
    responsibilities: Matrix


def tag_space(
    matrices: Sequence[TagMatrix], *, trace_ids: Collection[str] | None = None
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], Matrix]:
    """Join matrices by trace into one probability matrix.

    Args:
        matrices: Matrices over distinct views; each column is prefixed with its view.
        trace_ids: Keep only these traces; every trace of any matrix when None.

    Returns:
        Traces in sorted order, traces dropped because some matrix lacks them, the columns and
        the ``(traces, columns)`` matrix.

    Raises:
        ValueError: No matrix is given, two share a view, or no trace is in every matrix.
    """
    if not matrices:
        raise ValueError("the tag space needs at least one matrix")
    views = [matrix.view for matrix in matrices]
    if len(set(views)) != len(views):
        raise ValueError(f"tag space matrices must cover distinct views, got {views}")
    rows = [{row.trace_id: row.values for row in matrix.rows} for matrix in matrices]
    wanted = set(trace_ids) if trace_ids is not None else {t for r in rows for t in r}
    kept = tuple(sorted(t for t in wanted if all(t in r for r in rows)))
    dropped = tuple(sorted(wanted - set(kept)))
    if not kept:
        raise ValueError("no trace is tagged in every joined matrix; nothing can be clustered")
    columns = tuple(f"{m.view}:{q}" for m in matrices for q in m.question_ids)
    values = np.array([[v for r in rows for v in r[t]] for t in kept], dtype=np.float64)
    return kept, dropped, columns, values


def cluster_tags(
    matrices: Sequence[TagMatrix],
    spec: ClusteringSpec | None = None,
    *,
    trace_ids: Collection[str] | None = None,
) -> TagClustering:
    """Fit Bernoulli mixtures over a range of counts and keep the data-chosen one.

    Args:
        matrices: Matrices joined into the space (see :func:`tag_space`).
        spec: The registration; defaults apply when omitted.
        trace_ids: Cluster only these traces, such as the world-learning role.

    Raises:
        ValueError: The space is empty.
    """
    spec = spec or ClusteringSpec()
    traces, dropped, columns, values = tag_space(matrices, trace_ids=trace_ids)
    count = len(traces)
    folds = np.array([int(stable_id("fold", t)[-8:], 16) % spec.folds for t in traces])
    training = count - int(np.bincount(folds).max())
    top = max(1, min(spec.max_clusters, count, training))
    low = min(spec.min_clusters, top)
    scores: list[ClusterCountScore] = []
    best: dict[int, _Fit] = {}
    held: dict[int, Matrix] = {}
    for k in range(low, top + 1):
        fits = [_em(values, k, seed, spec) for seed in range(spec.seeds)]
        chosen = max(fits, key=lambda fit: fit.log_likelihood)
        best[k] = chosen
        labels = [fit.responsibilities.argmax(axis=1) for fit in fits]
        pairs = [
            _adjusted_rand(labels[i], labels[j])
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ]
        stability = 1.0 if k == 1 else float(np.mean(pairs))
        parameters = k * len(columns) + k - 1
        held[k] = (
            _held_out(values, folds, k, spec)
            if training >= k
            else _row_likelihood(values, chosen.unsmoothed, chosen.weights)
        )
        scores.append(
            ClusterCountScore(
                clusters=k,
                log_likelihood=round(chosen.log_likelihood, 6),
                parameters=parameters,
                bic=round(parameters * math.log(count) - 2.0 * chosen.log_likelihood, 6),
                occupied=len(set(chosen.responsibilities.argmax(axis=1).tolist())),
                held_out=round(float(held[k].mean()), 6),
                stability=round(stability, 6),
                stable=stability >= spec.min_stability,
            )
        )
    stable = [score for score in scores if score.stable]
    top_k = max(stable or scores, key=lambda score: (score.held_out, -score.clusters)).clusters
    for position, score in enumerate(scores):
        gap = held[top_k] - held[score.clusters]
        error = float(gap.std(ddof=1) / math.sqrt(count)) if count > 1 else 0.0
        scores[position] = score.model_copy(
            update={"held_out_gap": round(float(gap.mean()), 6), "held_out_error": round(error, 6)}
        )
    stable = [score for score in scores if score.stable]
    if stable and spec.criterion == "held_out":
        pick = min(
            (s for s in stable if s.held_out_gap <= s.held_out_error + 1e-6),
            key=lambda score: (score.occupied, score.clusters),
        )
        reason = (
            f"{pick.occupied} clusters: the fewest occupied within one standard error of the best "
            f"cross-validated log likelihood ({top_k} clusters) among the {len(stable)} of "
            f"{len(scores)} counts with stability at least {spec.min_stability}"
        )
    elif stable:
        pick = min(stable, key=lambda score: (score.bic, score.clusters))
        reason = (
            f"{pick.occupied} clusters: lowest BIC ({pick.bic:.2f}) among the {len(stable)} of "
            f"{len(scores)} counts with stability at least {spec.min_stability}"
        )
    else:
        pick = max(scores, key=lambda score: (score.stability, -score.bic))
        reason = (
            f"{pick.occupied} clusters: no count reached stability {spec.min_stability}, so the "
            f"most stable ({pick.stability:.2f}) was kept"
        )
    scores = [s.model_copy(update={"selected": s.clusters == pick.clusters}) for s in scores]
    clusters, assignments, confidence = _clusters(
        best[pick.clusters], traces, columns, values, spec
    )
    return TagClustering(
        spec=spec,
        sources=tuple(
            TagSource(
                bank_id=m.bank_id,
                bank_version=m.bank_version,
                view=m.view,
                matrix_sha256=sha256_json(m),
            )
            for m in matrices
        ),
        columns=columns,
        trace_ids=traces,
        dropped=dropped,
        scores=tuple(scores),
        clusters=clusters,
        assignments=assignments,
        confidence=confidence,
        reason=reason,
    )


def _held_out(values: Matrix, folds: npt.NDArray[np.int64], k: int, spec: ClusteringSpec) -> Matrix:
    """Return each trace's log likelihood under a mixture fitted without its fold."""
    scores = np.zeros(values.shape[0])
    for fold in range(spec.folds):
        test = folds == fold
        train = ~test
        if not test.any():
            continue
        fit = max(
            (_em(values[train], k, seed, spec) for seed in range(spec.fold_seeds)),
            key=lambda item: item.log_likelihood,
        )
        scores[test] = _row_likelihood(values[test], fit.unsmoothed, fit.weights)
    return scores


def _row_likelihood(values: Matrix, theta: Matrix, weights: Matrix) -> Matrix:
    """Return each row's mixture log likelihood."""
    joint = values @ np.log(theta).T + (1.0 - values) @ np.log1p(-theta).T + np.log(weights)
    peak = joint.max(axis=1, keepdims=True)
    return (peak + np.log(np.exp(joint - peak).sum(axis=1, keepdims=True)))[:, 0]


def _em(values: Matrix, k: int, seed: int, spec: ClusteringSpec) -> _Fit:
    """Fit one Bernoulli mixture from a seeded k-means++ start."""
    count = values.shape[0]
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(count))]
    for _ in range(1, k):
        distance = np.min(
            ((values[:, None, :] - values[centers][None, :, :]) ** 2).sum(axis=2), axis=1
        )
        total = float(distance.sum())
        if total <= 0.0:
            centers.append(int(rng.integers(count)))
            continue
        centers.append(int(rng.choice(count, p=distance / total)))
    theta = np.clip(values[centers] * 0.8 + 0.1, 1e-6, 1 - 1e-6)
    weights = np.full(k, 1.0 / k)
    previous = -math.inf
    responsibilities = np.full((count, k), 1.0 / k)
    likelihood = previous
    for _ in range(spec.max_iterations):
        joint = values @ np.log(theta).T + (1.0 - values) @ np.log1p(-theta).T + np.log(weights)
        peak = joint.max(axis=1, keepdims=True)
        normalizer = peak + np.log(np.exp(joint - peak).sum(axis=1, keepdims=True))
        likelihood = float(normalizer.sum())
        responsibilities = np.exp(joint - normalizer)
        mass = responsibilities.sum(axis=0)
        weights = np.clip(mass / count, 1e-12, None)
        weights = weights / weights.sum()
        theta = (responsibilities.T @ values + spec.smoothing) / (
            mass[:, None] + 2.0 * spec.smoothing
        )
        theta = np.clip(theta, 1e-6, 1 - 1e-6)
        if likelihood - previous <= spec.tolerance * abs(likelihood):
            break
        previous = likelihood
    mass = responsibilities.sum(axis=0)
    unsmoothed = (responsibilities.T @ values) / np.clip(mass, 1e-12, None)[:, None]
    unsmoothed = np.clip(unsmoothed, 1e-4, 1 - 1e-4)
    return _Fit(likelihood, weights, theta, unsmoothed, responsibilities)


def _adjusted_rand(left: npt.NDArray[np.int64], right: npt.NDArray[np.int64]) -> float:
    """Return the adjusted Rand index of two hard assignments."""
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)

    def pairs(x: Matrix) -> float:
        """Count unordered pairs."""
        return float((x * (x - 1) / 2).sum())

    index, rows, cols = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    total = pairs(np.array([float(len(left))]))
    expected = rows * cols / total if total else 0.0
    spread = (rows + cols) / 2 - expected
    if spread == 0:
        return 1.0
    return (index - expected) / spread


def _clusters(
    fit: _Fit,
    traces: tuple[str, ...],
    columns: tuple[str, ...],
    values: Matrix,
    spec: ClusteringSpec,
) -> tuple[tuple[TagCluster, ...], dict[str, str], dict[str, float]]:
    """Read clusters from a fit, ordered by size and then by first member."""
    hard = fit.responsibilities.argmax(axis=1)
    members: dict[int, list[int]] = {}
    for row, component in enumerate(hard):
        members.setdefault(int(component), []).append(row)
    order = sorted(members, key=lambda c: (-len(members[c]), traces[members[c][0]]))
    mean = values.mean(axis=0)
    clusters: list[TagCluster] = []
    assignments: dict[str, str] = {}
    confidence: dict[str, float] = {}
    for position, component in enumerate(order):
        cluster_id = f"c{position:02d}"
        rows = sorted(
            members[component], key=lambda r: (-fit.responsibilities[r, component], traces[r])
        )
        theta = fit.theta[component]
        for row in rows:
            assignments[traces[row]] = cluster_id
            confidence[traces[row]] = round(float(fit.responsibilities[row, component]), 6)
        lift = theta - mean
        clusters.append(
            TagCluster(
                cluster_id=cluster_id,
                size=len(rows),
                weight=round(float(fit.weights[component]), 6),
                propositions={c: round(float(p), 6) for c, p in zip(columns, theta, strict=True)},
                fired=tuple(
                    columns[i]
                    for i in np.argsort(-theta, kind="stable")
                    if theta[i] > spec.threshold
                ),
                distinctive=tuple(
                    columns[i] for i in np.argsort(-lift, kind="stable") if lift[i] >= 0.2
                ),
                members=tuple(traces[r] for r in rows),
                exemplars=tuple(traces[r] for r in rows[: spec.exemplars]),
                uncertain_members=tuple(
                    traces[r]
                    for r in rows
                    if fit.responsibilities[r, component] < spec.uncertain_below
                ),
            )
        )
    return tuple(clusters), assignments, confidence


def assign_clusters(
    clustering: TagClustering, matrices: Sequence[TagMatrix], trace_ids: Collection[str]
) -> dict[str, tuple[str, float]]:
    """Place traces outside the fit (such as mining probes) under the fitted mixture.

    Each trace's posterior over the clusters is computed from the columns its matrices answer;
    a column no given matrix answers for it is marginalized out, which the independent
    Bernoulli components allow exactly. Nothing is refitted and no trace joins a cluster.

    Args:
        clustering: The fitted clustering.
        matrices: Matrices holding the traces' rows, by view as in :func:`tag_space`.
        trace_ids: Traces to place.

    Returns:
        Each placeable trace's most probable cluster and its posterior probability; a trace no
        matrix answers any clustered column for is left out.
    """
    index = {column: position for position, column in enumerate(clustering.columns)}
    theta = np.clip(
        np.array(
            [[c.propositions[col] for col in clustering.columns] for c in clustering.clusters]
        ),
        1e-6,
        1.0 - 1e-6,
    )
    weights = np.array([max(c.weight, 1e-12) for c in clustering.clusters])
    prior = np.log(weights / weights.sum())
    placed: dict[str, tuple[str, float]] = {}
    for trace in sorted(set(trace_ids)):
        columns: list[int] = []
        values: list[float] = []
        for matrix in matrices:
            row = next((r for r in matrix.rows if r.trace_id == trace), None)
            if row is None:
                continue
            for question, value in zip(matrix.question_ids, row.values, strict=True):
                position = index.get(f"{matrix.view}:{question}")
                if position is not None:
                    columns.append(position)
                    values.append(float(value))
        if not columns:
            continue
        x, t = np.array(values), theta[:, columns]
        joint = prior + (np.log(t) * x + np.log1p(-t) * (1.0 - x)).sum(axis=1)
        posterior = np.exp(joint - joint.max())
        posterior /= posterior.sum()
        best = int(posterior.argmax())
        placed[trace] = (clustering.clusters[best].cluster_id, round(float(posterior[best]), 6))
    return placed


class ClusterAdmissionSpec(ContractModel):
    """Which clusters, and how many traces of each, enter a corpus.

    Attributes:
        total: Most traces admitted overall, shared out evenly: every cluster with traces left
            gets one more in turn, so a rare cluster is admitted whole before a large one is
            capped. None admits every eligible member of every admitted cluster.
        per_cluster: Most traces any one cluster contributes, or None.
        exclude_clusters: Clusters admitted none, by identity.
        minimum_cluster_size: Clusters smaller than this are admitted none.
        include_uncertain: Admit members whose assignment is uncertain.
    """

    total: int | None = Field(default=None, ge=1)
    per_cluster: int | None = Field(default=None, ge=1)
    exclude_clusters: tuple[str, ...] = ()
    minimum_cluster_size: int = Field(default=1, ge=1)
    include_uncertain: bool = True


class AdmittedTrace(ContractModel):
    """One admitted trace and why.

    Attributes:
        trace_id: The trace.
        cluster_id: Its cluster.
        cluster_name: The cluster's name, when it was named.
        confidence: Its assignment probability.
        rank: One-based position among its cluster's eligible members, most typical first.
        reason: One sentence a reviewer can read.
    """

    trace_id: str
    cluster_id: str
    cluster_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    reason: str


class ClusterQuota(ContractModel):
    """How one cluster was admitted.

    Attributes:
        cluster_id: The cluster.
        cluster_name: Its name, when named.
        size: Members.
        eligible: Members the caller allowed.
        admitted: Members admitted.
        reason: Why that many.
    """

    cluster_id: str
    cluster_name: str | None = None
    size: int = Field(ge=0)
    eligible: int = Field(ge=0)
    admitted: int = Field(ge=0)
    reason: str


class ClusterAdmission(ContractModel):
    """Traces admitted from clusters, with every cluster's quota and every refusal.

    Attributes:
        spec: The admission registration.
        clustering: Identity of the clustering read.
        named: Whether cluster names were supplied.
        admitted: Admitted traces, by cluster order then rank.
        quotas: One per cluster.
        refused: Eligible traces not admitted, mapped to why.
    """

    spec: ClusterAdmissionSpec
    clustering: str
    named: bool = False
    admitted: tuple[AdmittedTrace, ...]
    quotas: tuple[ClusterQuota, ...]
    refused: dict[str, str] = Field(default_factory=dict)

    def trace_ids(self) -> tuple[str, ...]:
        """Return the admitted traces, sorted."""
        return tuple(sorted(row.trace_id for row in self.admitted))


def admit_by_cluster(
    clustering: TagClustering,
    spec: ClusterAdmissionSpec | None = None,
    *,
    eligible: Collection[str],
    held_out: Collection[str],
    names: Mapping[str, str] | None = None,
) -> ClusterAdmission:
    """Admit eligible traces cluster by cluster, recording why each was admitted.

    Args:
        clustering: The clustering.
        spec: Admission registration; defaults admit every eligible member.
        eligible: Traces the consuming corpus may hold, such as the world-learning role.
        held_out: Final-evaluation traces; a clustering holding any of them is refused.
        names: Cluster names by cluster, when the clustering was named (see
            :meth:`~exp.simulation.mining.tag_cluster_names.ClusterNames.mapping`).

    Raises:
        ValueError: The clustering holds a held-out trace, a name names an unknown cluster, or
            nothing is admitted.
    """
    spec = spec or ClusterAdmissionSpec()
    leaked = sorted(set(clustering.trace_ids) & set(held_out))
    if leaked:
        raise ValueError(
            f"{len(leaked)} clustered traces are held out, for example {leaked[0]}; cluster only "
            "development traces"
        )
    names = dict(names or {})
    unknown = sorted(set(names) - {cluster.cluster_id for cluster in clustering.clusters})
    if unknown:
        raise ValueError(f"cluster names name unknown clusters {unknown}; name this clustering")
    allowed = set(eligible)
    threshold = clustering.spec.uncertain_below
    pools: dict[str, list[str]] = {}
    refused: dict[str, str] = {}
    blocked: dict[str, str] = {}
    for cluster in clustering.clusters:
        members = [t for t in cluster.members if t in allowed]
        if cluster.cluster_id in spec.exclude_clusters:
            blocked[cluster.cluster_id] = "excluded by the admission spec"
        elif cluster.size < spec.minimum_cluster_size:
            blocked[cluster.cluster_id] = (
                f"has {cluster.size} members, below the minimum {spec.minimum_cluster_size}"
            )
        if cluster.cluster_id in blocked:
            refused.update(
                {t: f"cluster {cluster.cluster_id} {blocked[cluster.cluster_id]}" for t in members}
            )
            members = []
        if not spec.include_uncertain:
            unsure = [t for t in members if clustering.confidence[t] < threshold]
            refused.update(
                {t: f"assignment to {cluster.cluster_id} below {threshold}" for t in unsure}
            )
            members = [t for t in members if t not in unsure]
        cap = len(members) if spec.per_cluster is None else min(len(members), spec.per_cluster)
        pools[cluster.cluster_id] = members[:cap]
        refused.update({t: f"beyond the per-cluster cap {spec.per_cluster}" for t in members[cap:]})
    quota = _share(pools, spec.total)
    admitted: list[AdmittedTrace] = []
    quotas: list[ClusterQuota] = []
    for cluster in clustering.clusters:
        name = names.get(cluster.cluster_id)
        label = cluster.cluster_id if name is None else f"{cluster.cluster_id} ({name})"
        pool, take = pools[cluster.cluster_id], quota[cluster.cluster_id]
        how = (
            "every eligible member"
            if take == len(pool) and spec.total is None
            else f"an even share of the total {spec.total}"
        )
        refused.update(
            {
                t: f"beyond cluster {cluster.cluster_id}'s share of the total {spec.total}"
                for t in pool[take:]
            }
        )
        for rank, trace in enumerate(pool[:take], start=1):
            admitted.append(
                AdmittedTrace(
                    trace_id=trace,
                    cluster_id=cluster.cluster_id,
                    cluster_name=name,
                    confidence=clustering.confidence[trace],
                    rank=rank,
                    reason=(
                        f"cluster {label}: member {rank} of {take} admitted ({how}), assignment "
                        f"probability {clustering.confidence[trace]:.2f}"
                    ),
                )
            )
        eligible_count = sum(1 for t in cluster.members if t in allowed)
        quotas.append(
            ClusterQuota(
                cluster_id=cluster.cluster_id,
                cluster_name=name,
                size=cluster.size,
                eligible=eligible_count,
                admitted=take,
                reason=blocked.get(cluster.cluster_id)
                or ("no eligible member" if not eligible_count else how),
            )
        )
    if not admitted:
        raise ValueError(
            "cluster admission admitted no trace; widen the admission spec or check eligibility"
        )
    return ClusterAdmission(
        spec=spec,
        clustering=clustering.identity(),
        named=bool(names),
        admitted=tuple(admitted),
        quotas=tuple(quotas),
        refused=dict(sorted(refused.items())),
    )


def _share(pools: Mapping[str, list[str]], total: int | None) -> dict[str, int]:
    """Share a total evenly over pools in turn, never above a pool's size."""
    if total is None:
        return {cluster: len(pool) for cluster, pool in pools.items()}
    quota = dict.fromkeys(pools, 0)
    left = total
    while left > 0:
        open_pools = [c for c, pool in pools.items() if quota[c] < len(pool)]
        if not open_pools:
            break
        for cluster in open_pools:
            if left == 0:
                break
            quota[cluster] += 1
            left -= 1
    return quota


def persist_tag_clustering(
    store: ArtifactStore,
    clustering: TagClustering,
    *,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> ArtifactInput:
    """Save a clustering, or adopt the identical saved one.

    Raises:
        ValueError: A saved artifact under this identity holds a different clustering.
    """
    identity = clustering.identity()
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        if load_tag_clustering(store, reference) != clustering:
            raise ValueError(f"saved clustering {identity} holds a different record")
        return reference
    return write_record(
        store, identity, CLUSTERS_TYPE, "clustering.json", clustering, code_revision, inputs
    )


def load_tag_clustering(store: ArtifactStore, reference: ArtifactInput) -> TagClustering:
    """Read a saved clustering, refusing a changed or mistyped artifact."""
    payload = read_record(store, reference, CLUSTERS_TYPE, "clustering.json")
    return TagClustering.model_validate_json(payload)


def write_record(
    store: ArtifactStore,
    identity: str,
    kind: str,
    name: str,
    record: BaseModel,
    code_revision: str,
    inputs: Sequence[ArtifactInput],
) -> ArtifactInput:
    """Write one record as a new artifact."""
    manifest = store.write_json(
        artifact_id=identity,
        artifact_type=kind,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
        ),
        files={name: record},
    )
    return artifact_input(manifest)


def read_record(store: ArtifactStore, reference: ArtifactInput, kind: str, name: str) -> bytes:
    """Return one pinned artifact's file after checking its pin and type.

    Raises:
        ValueError: The artifact changed since it was cited or has another type.
    """
    stored = store.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference or stored.manifest.artifact_type != kind:
        raise ValueError(f"artifact {reference.artifact_id} is not the pinned {kind} record")
    return stored.payloads[name]
