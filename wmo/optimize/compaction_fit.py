"""Per-cluster compaction fitter: populate `ClusterRanking.compression` from measured arms.

#265 merged the artifact field ("this cluster's compression choice, fitted jointly with the
model ranking") with nothing fitting it. This module is the fitter. Its inputs are the grid's
per-arm outcome matrices: one OFF arm (uncompressed) plus one matrix per measured compression
config, all on the SAME scenario cohort, each self-describing via
`OutcomeMatrix.measured_compression()`. Its output is a per-cluster
`CompressionConfig | None` map plus the paired evidence behind every decision.

The choice rule is conservative by design, mirroring the routing guard's shape: UNCOMPRESSED
IS THE FALLBACK, and a cluster receives a compressed config only on statistical evidence it
does not lose quality AND a measured effective-cost win. Concretely, an arm passes in a
cluster iff

- quality: the paired per-cell reward delta (arm minus off, cells = (scenario, model) means
  over scored episodes) satisfies mean - z * SE >= 0 on at least `min_pairs` paired cells,
  with the same small-sample SE floor the knn guard uses (`SE_FLOOR_MAX_PAIRS`), and
- cost: the arm's effective cost per COMPLETED task on those cells, compressor bill folded
  as a `RowOverhead`, beats the off arm's on the same cells. Input-token accounting alone is
  banned here: the grid measured dumb deletion RAISING effective cost by lengthening
  episodes, so only the completed-task metric can gate.

Among passing arms the tie-break is lowest effective cost, then lower aggressiveness. Arms
are MEASURED points only: the fitter never interpolates an aggressiveness the grid did not
run, matching the D-DIAL v2 ruling that aggressiveness is a step function over mounted
artifacts.

Two mandatory validations ride along. The A/A bar (`aa_report`) splits the off arm by episode
parity and runs the same gates: a guard loose enough to compress on noise fails there before
it can ship. Control arms (`control=True`) are evaluated and reported but never chosen; a
control that WOULD have won a cluster is an investigation flag, not a result.

Representation rule (proposed in DECISIONS 2026-07-27, C2 round 2): a policy carrying
per-cluster configs routes on RAW text (policy-level `compression` and `fit_compression`
both None) and compresses after cluster assignment. `apply_compaction` enforces that
exclusivity; the serving-side delta lands separately behind the co-signed contract.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from wmo.optimize.compression import (
    CompressionConfig,
    compression_signature,
    servable_compressor,
)
from wmo.optimize.policy import SE_FLOOR_MAX_PAIRS, ClusterRanking, RoutingPolicy
from wmo.optimize.scorecard import (
    DEFAULT_COMPLETION,
    CompletionRule,
    RowOverhead,
    effective_cost_per_completed_task,
)

if TYPE_CHECKING:
    from wmo.optimize.outcomes import OutcomeMatrix, ScenarioOutcome
    from wmo.providers.base import Embedder

logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_Z = 0.5  # the routing guard's confidence bar, reused deliberately
DEFAULT_COMPACTION_MIN_PAIRS = 8  # below this a cluster never deviates from uncompressed

# The A/A pseudo-arm's placeholder identity. It exists only inside `aa_report` evidence rows;
# it is marked control, is never eligible, and never reaches an artifact.
AA_SIGNATURE = "aa-episode-split"


class ClusterArmEvidence(BaseModel):
    """The paired statistics behind one (cluster, arm) decision, kept auditable.

    `would_win` is true when the arm passed both gates and led the tie-break regardless of
    eligibility, so a control or unservable arm that would have taken the cluster is visible
    instead of silently skipped. `chosen` additionally requires eligible and not control.
    """

    cluster_id: int
    signature: str  # compression_signature of the arm
    config: CompressionConfig | None = None  # None only for the A/A pseudo-arm
    control: bool = False
    eligible: bool = True  # servable_compressor accepted the config
    n_pairs: int = 0
    mean_diff: float = 0.0
    se: float = 0.0
    quality_pass: bool = False
    off_cost_per_completed: float | None = None
    arm_cost_per_completed: float | None = None
    cost_pass: bool = False
    would_win: bool = False
    chosen: bool = False


class CompactionFit(BaseModel):
    """The fitter's output: the per-cluster map plus everything needed to audit it."""

    per_cluster: dict[int, CompressionConfig | None]
    evidence: list[ClusterArmEvidence]
    z: float
    min_pairs: int
    coverage: list[str] = Field(default_factory=list)  # human-readable coverage notes
    controls_would_win: list[str] = Field(default_factory=list)  # "cluster=3 truncate/1/0.2"

    def compressed_clusters(self) -> int:
        return sum(1 for config in self.per_cluster.values() if config is not None)


class ArmMatrices(BaseModel):
    """One measured arm: its matrix and how the fitter should treat it."""

    model_config = {"arbitrary_types_allowed": True}

    matrix: object  # OutcomeMatrix; typed loosely to avoid a runtime import cycle
    config: CompressionConfig
    control: bool = False


def _cells(matrix: OutcomeMatrix) -> dict[tuple[str, str], list[ScenarioOutcome]]:
    """Scored rows grouped by (scenario_id, model): the unit quality is paired on."""
    cells: dict[tuple[str, str], list[ScenarioOutcome]] = defaultdict(list)
    for row in matrix.outcomes:
        if row.scored:
            cells[(row.scenario_id, row.model)].append(row)
    return cells


def _cell_reward(rows: list[ScenarioOutcome]) -> float:
    return float(np.mean([row.reward for row in rows]))


def _compressor_overheads(rows: list[ScenarioOutcome]) -> list[RowOverhead]:
    """The rows' own compressor bill as scorecard overheads (the #294 convention)."""
    return [
        RowOverhead(
            scenario_id=row.scenario_id,
            model=row.model,
            episode=row.episode,
            component="compressor",
            cost_usd=row.compressor_cost_usd,
            latency_s=row.compressor_latency_s,
        )
        for row in rows
        if row.compressor_cost_usd > 0.0 or row.compressor_latency_s > 0.0
    ]


def _effective_cost(rows: list[ScenarioOutcome], completion: CompletionRule) -> float | None:
    if not rows:
        return None
    result = effective_cost_per_completed_task(
        rows, overheads=_compressor_overheads(rows), completion=completion
    )
    return result.cost_per_completed_task_usd


def check_cohort(
    off: OutcomeMatrix, arms: list[ArmMatrices], *, allow_uneven: bool = False
) -> list[str]:
    """Enforce that every arm was measured on the off arm's cohort, both directions.

    Strict mode (the default) requires identical scored (scenario, model) cell sets: a grid arm
    measured on a subset ranks its clusters on DIFFERENT evidence, which is the bias the route
    sweep's coverage gate exists to block. `allow_uneven` downgrades the mismatch to printed
    notes; pairing then happens on the intersection per cluster, and the caller labels the fit
    accordingly (the grid mid-flight is the intended use).
    """
    notes: list[str] = []
    off_cells = set(_cells(off))
    for arm in arms:
        arm_cells = set(_cells(arm.matrix))
        missing = len(off_cells - arm_cells)
        extra = len(arm_cells - off_cells)
        if missing or extra:
            notes.append(
                f"{compression_signature(arm.config)}: {missing} off cells unmeasured on this "
                f"arm, {extra} arm cells absent from off ({len(arm_cells & off_cells)} shared)"
            )
    if notes and not allow_uneven:
        raise ValueError(
            "arm matrices do not cover the off arm's cohort; pairing on a subset biases the "
            "per-cluster ranking toward whichever cells each arm happened to keep:\n  "
            + "\n  ".join(notes)
            + "\nRe-run the missing cells, or pass allow_uneven for an interim, "
            "candidate-labeled fit."
        )
    return notes


def overlay_clusters(
    matrix: OutcomeMatrix,
    *,
    embed_with: Embedder,
    n_clusters: int = 8,
    seed: int = 42,
    default_model: str,
) -> tuple[list[ClusterRanking], dict[str, int]]:
    """K-means compression-overlay clusters on the off arm's raw task embeddings.

    For `kind="rank"` policies the fitted clusters already exist and the caller uses
    `assign_to_clusters` against them instead. This overlay is for `kind="knn"` policies, which
    route without clusters: the overlay rides on the artifact solely so the compress stage has
    an assignment surface, fitted on the SAME raw geometry the router embeds queries in (the
    proposed serving delta reuses the decision's own query embedding, so overlay centroids in
    any other space would be meaningless).

    The `ranking` field is stamped `[default_model]`: it satisfies the model's shape and makes
    plain that routing never reads these clusters. Conventions mirror `fit_rank_policy`
    (L2-normalized embeddings, k-means++ with the reference seed).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import Normalizer

    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(tasks)
    if not scenario_ids:
        raise ValueError("no scenarios to cluster")
    embeddings = np.asarray(embed_with.embed([tasks[sid] for sid in scenario_ids]))
    embeddings = Normalizer(norm="l2").fit_transform(embeddings)
    k = min(n_clusters, len(scenario_ids))
    kmeans = KMeans(
        n_clusters=k,
        random_state=seed,
        init="k-means++",
        n_init="auto",
        max_iter=1000,
        algorithm="elkan",
    )
    labels = kmeans.fit_predict(embeddings)
    assignment = {sid: int(label) for sid, label in zip(scenario_ids, labels, strict=True)}
    counts = defaultdict(int)
    for cluster in assignment.values():
        counts[cluster] += 1
    clusters = [
        ClusterRanking(
            cluster_id=cluster_id,
            label=f"compaction-{cluster_id}",
            centroid=kmeans.cluster_centers_[cluster_id].tolist(),
            ranking=[default_model],
            total=counts[cluster_id],
        )
        for cluster_id in range(k)
    ]
    return clusters, assignment


def assign_to_clusters(
    clusters: list[ClusterRanking], matrix: OutcomeMatrix, *, embed_with: Embedder
) -> dict[str, int]:
    """Nearest-centroid assignment of the matrix's scenarios to existing policy clusters.

    Used on `kind="rank"` policies so the compaction gates run on THE clusters the policy
    routes with, not a parallel clustering that would drift from it.
    """
    tasks: dict[str, str] = {}
    for outcome in matrix.outcomes:
        tasks.setdefault(outcome.scenario_id, outcome.task)
    scenario_ids = list(tasks)
    embeddings = np.asarray(embed_with.embed([tasks[sid] for sid in scenario_ids]))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0.0, 1.0, norms)
    centroids = np.asarray([c.centroid for c in clusters])
    cluster_ids = [c.cluster_id for c in clusters]
    sims = embeddings @ centroids.T
    return {
        sid: cluster_ids[int(np.argmax(sims[index]))]
        for index, sid in enumerate(scenario_ids)
    }


def fit_compaction(
    assignment: dict[str, int],
    off: OutcomeMatrix,
    arms: list[ArmMatrices],
    *,
    z: float = DEFAULT_COMPACTION_Z,
    min_pairs: int = DEFAULT_COMPACTION_MIN_PAIRS,
    completion: CompletionRule = DEFAULT_COMPLETION,
    allow_uneven: bool = False,
) -> CompactionFit:
    """The gates: per cluster, choose among measured arms or stay uncompressed.

    `assignment` maps scenario_id to cluster_id (from the policy's own clusters via
    `assign_to_clusters`, or from `overlay_clusters`). See the module docstring for the rule;
    every decision's paired statistics are returned as evidence rows.
    """
    if off.measured_compression() is not None:
        raise ValueError("the off arm must be uncompressed; got a matrix with a compression arm")
    signatures = [compression_signature(arm.config) for arm in arms]
    if len(set(signatures)) != len(signatures):
        raise ValueError(f"duplicate arm signatures: {sorted(signatures)}")
    coverage = check_cohort(off, arms, allow_uneven=allow_uneven)

    off_cells = _cells(off)
    cluster_ids = sorted(set(assignment.values()))
    evidence: list[ClusterArmEvidence] = []
    per_cluster: dict[int, CompressionConfig | None] = {}
    controls_would_win: list[str] = []

    for cluster_id in cluster_ids:
        member_cells = [key for key in off_cells if assignment.get(key[0]) == cluster_id]
        candidates: list[ClusterArmEvidence] = []
        for arm in arms:
            signature = compression_signature(arm.config)
            arm_cells = _cells(arm.matrix)
            paired = [key for key in member_cells if key in arm_cells]
            diffs = np.asarray(
                [_cell_reward(arm_cells[key]) - _cell_reward(off_cells[key]) for key in paired]
            )
            n_pairs = int(diffs.size)
            mean_diff = float(diffs.mean()) if n_pairs else 0.0
            se = float(diffs.std(ddof=1)) / n_pairs**0.5 if n_pairs > 1 else 0.0
            if 0 < n_pairs < SE_FLOOR_MAX_PAIRS:
                se = max(se, (0.25 / n_pairs) ** 0.5)
            quality_pass = n_pairs >= min_pairs and mean_diff - z * se >= 0.0

            arm_rows = [row for key in paired for row in arm_cells[key]]
            off_rows = [row for key in paired for row in off_cells[key]]
            arm_cost = _effective_cost(arm_rows, completion)
            off_cost = _effective_cost(off_rows, completion)
            cost_pass = arm_cost is not None and off_cost is not None and arm_cost < off_cost

            try:
                servable_compressor(arm.config)
                eligible = True
            except ValueError:
                eligible = False
            candidates.append(
                ClusterArmEvidence(
                    cluster_id=cluster_id,
                    signature=signature,
                    config=arm.config,
                    control=arm.control,
                    eligible=eligible,
                    n_pairs=n_pairs,
                    mean_diff=mean_diff,
                    se=se,
                    quality_pass=quality_pass,
                    off_cost_per_completed=off_cost,
                    arm_cost_per_completed=arm_cost,
                    cost_pass=cost_pass,
                )
            )

        passing = [c for c in candidates if c.quality_pass and c.cost_pass]
        if passing:
            # Tie-break lowest effective cost, then lower aggressiveness (conservative).
            ranked = sorted(
                passing,
                key=lambda c: (c.arm_cost_per_completed, c.config.aggressiveness),
            )
            ranked[0].would_win = True
            winner = next((c for c in ranked if c.eligible and not c.control), None)
            if winner is not None:
                winner.chosen = True
                per_cluster[cluster_id] = winner.config
            else:
                per_cluster[cluster_id] = None
            for candidate in ranked:
                if candidate.control and candidate.would_win:
                    controls_would_win.append(f"cluster={cluster_id} {candidate.signature}")
        else:
            per_cluster[cluster_id] = None
        evidence.extend(candidates)

    fit = CompactionFit(
        per_cluster=per_cluster,
        evidence=evidence,
        z=z,
        min_pairs=min_pairs,
        coverage=coverage,
        controls_would_win=controls_would_win,
    )
    logger.info(
        "compaction fit: %d/%d clusters compressed (z=%g, min_pairs=%d)%s",
        fit.compressed_clusters(),
        len(cluster_ids),
        z,
        min_pairs,
        f"; CONTROLS WOULD WIN: {controls_would_win}" if controls_would_win else "",
    )
    return fit


def aa_pseudo_arms(off: OutcomeMatrix) -> tuple[OutcomeMatrix, OutcomeMatrix]:
    """Split the off arm by episode parity into two pseudo-arms measuring NOTHING.

    Any per-cluster difference between the two sides is noise by construction, which is what
    makes them the fitter's A/A control: gates loose enough to deviate here would compress on
    noise in production. Episode indices are renumbered per side so each pseudo-matrix is
    well-formed on its own.
    """
    from wmo.optimize.outcomes import OutcomeMatrix

    even: list[ScenarioOutcome] = []
    odd: list[ScenarioOutcome] = []
    counters: dict[tuple[str, str, int], int] = defaultdict(int)
    for row in off.outcomes:
        side = even if row.episode % 2 == 0 else odd
        key = (row.scenario_id, row.model, row.episode % 2)
        side.append(row.model_copy(update={"episode": counters[key]}))
        counters[key] += 1
    return (
        OutcomeMatrix(pool=off.pool, outcomes=even),
        OutcomeMatrix(pool=off.pool, outcomes=odd),
    )


def aa_report(
    assignment: dict[str, int],
    off: OutcomeMatrix,
    *,
    z: float = DEFAULT_COMPACTION_Z,
    min_pairs: int = DEFAULT_COMPACTION_MIN_PAIRS,
    completion: CompletionRule = DEFAULT_COMPLETION,
) -> list[ClusterArmEvidence]:
    """The A/A kill bar: run the gates on the episode-parity pseudo-arms.

    Returns the evidence rows whose gates BOTH passed (the deviations); an empty list is the
    bar passing. The pseudo-arm is marked control and carries a placeholder config, so nothing
    from this report can reach an artifact even if a caller mishandles it.
    """
    pseudo_off, pseudo_arm = aa_pseudo_arms(off)
    placeholder = CompressionConfig(compressor_id=AA_SIGNATURE, compressor_version="0")
    rows: list[ClusterArmEvidence] = []
    fit = fit_compaction(
        assignment,
        pseudo_off,
        [ArmMatrices(matrix=pseudo_arm, config=placeholder, control=True)],
        z=z,
        min_pairs=min_pairs,
        completion=completion,
        allow_uneven=True,  # single-episode cells exist on one side only; pairing drops them
    )
    for row in fit.evidence:
        row.signature = AA_SIGNATURE
        row.config = None
        if row.quality_pass and row.cost_pass:
            rows.append(row)
    return rows


class CompactionArtifact(BaseModel):
    """The per-cluster compaction map as a policy SIDECAR (knn policies, v1).

    The merged `RoutingPolicy` validator forbids clusters on a knn policy, so the overlay
    cannot ride on the policy model without the co-signed contract delta. Until that lands,
    the map ships beside the policy the way the knn bank itself does (`knn_bank_path`): the
    clusters here carry centroids in the SAME raw embedding geometry the policy routes in,
    plus their fitted compression configs. Nothing serves this yet; the serving delta is
    gated on the DECISIONS co-sign.
    """

    clusters: list[ClusterRanking]
    z: float
    min_pairs: int
    fitted_from: str | None = None


COMPACTION_SIDECAR_FILENAME = "policy_compaction.json"


def save_compaction_sidecar(
    clusters: list[ClusterRanking],
    fit: CompactionFit,
    path,  # noqa: ANN001 - Path-like
    *,
    fitted_from: str | None = None,
) -> CompactionArtifact:
    """Write the stamped overlay next to a knn policy (see `CompactionArtifact`)."""
    from pathlib import Path

    stamped = [
        cluster.model_copy(update={"compression": fit.per_cluster.get(cluster.cluster_id)})
        for cluster in clusters
    ]
    for cluster in stamped:
        if cluster.compression is not None:
            servable_compressor(cluster.compression)
    artifact = CompactionArtifact(
        clusters=stamped, z=fit.z, min_pairs=fit.min_pairs, fitted_from=fitted_from
    )
    Path(path).write_text(artifact.model_dump_json(indent=2))
    return artifact


def apply_compaction(policy: RoutingPolicy, fit: CompactionFit) -> RoutingPolicy:
    """Stamp the fitted per-cluster map onto `policy` under the exclusivity rule.

    The proposed representation rule (DECISIONS 2026-07-27): a policy carrying per-cluster
    configs routes on RAW text, so policy-level `compression` and `fit_compression` must both
    be None; mixing modes in one artifact is refused here, not discovered at mount. Every
    stamped config is checked servable, so a control or churny arm that leaked into the map
    fails closed at stamp time.

    Cluster-carrying kinds only (`rank`): a knn policy cannot carry clusters under the merged
    validator, so its map goes through `save_compaction_sidecar` instead.
    """
    if policy.compression is not None or policy.fit_compression is not None:
        raise ValueError(
            "per-cluster compaction requires a raw-routed policy: policy-level compression "
            f"is {compression_signature(policy.compression)} and fit_compression is "
            f"{compression_signature(policy.fit_compression)}. Fit the base policy without "
            "--compressor; per-cluster mode and endpoint-level mode never mix."
        )
    if not policy.clusters:
        raise ValueError(
            "policy has no clusters to stamp; fit a rank policy or attach overlay_clusters first"
        )
    known = {cluster.cluster_id for cluster in policy.clusters}
    unknown = sorted(set(fit.per_cluster) - known)
    if unknown:
        raise ValueError(f"fit names cluster ids absent from the policy: {unknown[:5]}")
    for cluster_id, config in fit.per_cluster.items():
        if config is not None:
            servable_compressor(config)
        del cluster_id
    clusters = [
        cluster.model_copy(update={"compression": fit.per_cluster.get(cluster.cluster_id)})
        for cluster in policy.clusters
    ]
    return policy.model_copy(update={"clusters": clusters})
