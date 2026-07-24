"""The routing policy artifact: what the routing optimizer emits and the endpoint serves.

An endpoint = {world model, policy, evidence, URL}; this module is the policy leg. Two kinds:

- `static`: every request goes to `default_model`. Valid without any optimizer run, so an
  endpoint serves from day one and the improvement report has an honest "before" state.
- `rank`: the Avengers cluster-rank router (arXiv 2505.19797), replicated faithfully from the
  reference implementation (ZhangYiqun018/Avengers, core/routing/rank_router.py): embed the
  request, softmax the distances to the `top_k_clusters` nearest k-means centres
  (`-beta * (1 - centre . query)` logits), score every pool model by the probability-weighted
  reciprocal of its per-cluster accuracy rank (`1 / (rank + 0.1)`; models absent from a
  cluster's ranking score `1 / default_rank`), and route to the argmax.

The FIT that produces rank policies lives in `wmh.optimize.routing`; this module pins the
artifact schema and the serve-time selection so serving, reports, and the platform stay stable
across fitter iterations.

Serve-time stickiness: `select_model` keeps a conversation's incumbent model whenever the policy
is sticky (the default). Provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads and pays cold writes; until the fitter learns a real switching rule
(expected gain vs switch cost), pure affinity is the honest default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from wmh.providers.base import Embedder, ProviderConfig, ProviderKind
from wmh.providers.pool import PoolEntry
from wmh.providers.registry import get_provider
from wmh.retrieval.embedders import BatchedEmbedder, HashingEmbedder

POLICY_VERSION = 2
POLICY_FILENAME = "policy.json"

# Avengers reference defaults (config/experts_template.yaml: top_k=2, beta=6.0,
# default_rank=999). Kept on the policy so serving needs no side-channel config.
DEFAULT_TOP_K_CLUSTERS = 2
DEFAULT_BETA = 6.0
DEFAULT_RANK = 999


class EmbedderSpec(BaseModel):
    """How to reproduce the policy's embedding function at serve time.

    `hashing` is deterministic, offline, and credential-free, so a policy file is fully
    self-contained. `azure` uses an Azure embedding deployment (per-entry credential
    conventions matching the model pool: `api_key_env` names the env var); the fitter records
    whichever the fit actually used, and serving reconstructs the identical function.
    """

    kind: Literal["hashing", "azure"] = "hashing"
    dim: int = 512
    deployment: str | None = None  # azure embedding deployment name
    endpoint: str | None = None
    api_key_env: str | None = None
    batch: int = 256  # provider embeds are chunked to this many texts per request

    @model_validator(mode="after")
    def _validate_backend(self) -> EmbedderSpec:
        if self.kind == "azure" and not (self.deployment and self.endpoint):
            raise ValueError("an azure embedder spec needs deployment and endpoint")
        return self

    def build(self) -> Embedder:
        if self.kind == "hashing":
            return HashingEmbedder(dim=self.dim)
        api_key = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(
                    f"embedder spec: environment variable {self.api_key_env} is unset or "
                    "empty; export that account's API key"
                )
        provider = get_provider(
            ProviderConfig(
                kind=ProviderKind.AZURE_OPENAI,
                model=self.deployment or "",
                embed_model=self.deployment,
                embed_dim=self.dim,
                endpoint=self.endpoint,
                api_version="2024-10-21",
            ),
            api_key=api_key,
        )
        return BatchedEmbedder(provider, batch=self.batch)


class ClusterRanking(BaseModel):
    """One fitted cluster: its centroid and the pool models ranked by in-cluster accuracy."""

    cluster_id: int
    label: str = ""  # human-readable, surfaces as cluster_label in the request log
    centroid: list[float]
    ranking: list[str] = Field(min_length=1)  # pool entry names, best first
    scores: dict[str, float] = Field(default_factory=dict)  # per-model mean reward (evidence)
    costs: dict[str, float] = Field(default_factory=dict)  # per-model mean cost (evidence)
    total: int = 0  # fit scenarios that landed in this cluster


class RoutingDecision(BaseModel):
    """Where one request goes and why (the request log's model/cluster/routing_reason)."""

    model: str
    cluster_id: int | None = None
    cluster_label: str = ""
    reason: str


class RoutingPolicy(BaseModel):
    """The persisted policy artifact (see module docstring)."""

    version: int = POLICY_VERSION
    kind: Literal["static", "rank"]
    default_model: str  # the static answer; also the fallback for degenerate rank inputs
    pool: list[PoolEntry]  # snapshot of the roster this policy was defined over
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    clusters: list[ClusterRanking] = Field(default_factory=list)
    top_k_clusters: int = Field(default=DEFAULT_TOP_K_CLUSTERS, ge=1)
    beta: float = Field(default=DEFAULT_BETA, gt=0.0)
    default_rank: int = Field(default=DEFAULT_RANK, ge=1)
    sticky: bool = True  # keep a conversation's incumbent model (see module docstring)
    # ProxRouter-inspired support tilt (2510.09852, ADAPTED to clusters: their exponential
    # tilt reweights nonparametric scores by a prior; ours multiplies cluster probabilities
    # by support^gamma so thin outlier clusters lose routing weight). 0 = off (reference).
    support_tilt_gamma: float = 0.0
    # Fit-set mean cost per scored episode (all models): the unit the cost knob trades against
    # one reward point. 0 when the fit carried no usable costs.
    cost_scale: float = 0.0
    fitted_from: str | None = None  # provenance: the outcome matrix the fitter used

    @model_validator(mode="after")
    def _validate(self) -> RoutingPolicy:
        names = {entry.name for entry in self.pool}
        if self.default_model not in names:
            raise ValueError(
                f"default_model '{self.default_model}' is not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.kind == "static" and self.clusters:
            raise ValueError("a static policy carries no clusters; use kind='rank'")
        if self.kind == "rank":
            if not self.clusters:
                raise ValueError("a rank policy needs at least one fitted cluster")
            for cluster in self.clusters:
                unknown = [name for name in cluster.ranking if name not in names]
                if unknown:
                    raise ValueError(
                        f"cluster {cluster.cluster_id} ranks {unknown}, "
                        f"not in the policy pool (available: {sorted(names)})"
                    )
                if len(cluster.centroid) != self.embedder.dim:
                    raise ValueError(
                        f"cluster {cluster.cluster_id} centroid has dim "
                        f"{len(cluster.centroid)}, embedder dim is {self.embedder.dim}"
                    )
        return self

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> RoutingPolicy:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def select_model(
    policy: RoutingPolicy, text: str, *, incumbent: str | None = None
) -> RoutingDecision:
    """Pick the pool model for one request.

    `text` is whatever the caller deems the routable content (serving passes the latest user
    message). `incumbent` is the model already serving this conversation, if any: a sticky
    policy keeps it (per-model prompt caches make switching expensive), unless it has been
    retired from the pool, in which case the request re-routes as if fresh.
    """
    names = {entry.name for entry in policy.pool}
    if incumbent is not None and incumbent in names and policy.sticky:
        return RoutingDecision(model=incumbent, reason="sticky: conversation affinity")
    if policy.kind == "static":
        return RoutingDecision(model=policy.default_model, reason="static policy")

    query = np.asarray(policy.embedder.build().embed([text])[0])
    return rank_decision(policy, query)


def rank_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """The Avengers rank-routing core, on an already-embedded query.

    Shared by `select_model` (one live request) and batch evaluation (the benchmark embeds
    all scenarios once) so the served path and the measured path can never diverge. Faithful
    to the reference implementation: the embedder output is L2-normalized (their Normalizer
    step); centres are used exactly as fitted, NOT re-normalized (k-means centres of unit
    vectors are near-unit; the reference dots them raw, so we do too).
    """
    names = {entry.name for entry in policy.pool}
    centres = np.asarray([cluster.centroid for cluster in policy.clusters])
    dists = 1.0 - centres @ query
    top = np.argsort(dists)[: policy.top_k_clusters]
    logits = -policy.beta * dists[top]
    probs = np.exp(logits - logits.max())
    if policy.support_tilt_gamma > 0.0:
        support = np.asarray(
            [max(policy.clusters[int(index)].total, 1) for index in top], dtype=np.float64
        )
        probs = probs * support**policy.support_tilt_gamma
    probs /= probs.sum()

    scores: dict[str, float] = {}
    for cluster_index, prob in zip(top, probs, strict=True):
        ranking = policy.clusters[int(cluster_index)].ranking
        for name in names:
            if name in ranking:
                rank_score = 1.0 / (ranking.index(name) + 0.1)
                scores[name] = scores.get(name, 0.0) + float(prob) * rank_score
    for name in names:
        scores.setdefault(name, 1.0 / policy.default_rank)

    # Argmax; ties break by pool order (deterministic; the reference relies on dict order).
    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    winner = max(scores.items(), key=lambda kv: (kv[1], -pool_order[kv[0]]))[0]
    nearest = policy.clusters[int(top[0])]
    label = f" ({nearest.label})" if nearest.label else ""
    return RoutingDecision(
        model=winner,
        cluster_id=nearest.cluster_id,
        cluster_label=nearest.label,
        reason=f"rank router: nearest cluster {nearest.cluster_id}{label}",
    )
