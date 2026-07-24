"""The routing policy artifact: what a routing optimizer emits and the endpoint serves.

An endpoint = {world model, policy, evidence, URL}; this module is the policy leg. Two kinds:

- `static`: every request goes to `default_model`. Valid without any optimizer run, so an
  endpoint serves from day one and the improvement report has an honest "before" state.
- `cluster`: embed the request, route to the nearest cluster's assigned model (the
  cluster-then-assign family the routing literature supports at our data scale; see
  .agents/docs/research/routing-lit-review.md).

The FITTING that produces cluster policies is deliberately not here: the benchmark and the
implementation approach get agreed first (DECISIONS.md 2026-07-24, Silen direction). This module
pins the artifact schema and the serve-time selection rule so serving, the report, and the
platform integrate now, and the fitter drops in later without touching consumers.

Serve-time stickiness: `select_model` keeps a conversation's incumbent model whenever the policy
is sticky (the default). Provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads and pays cold writes; until the fitter learns a real switching rule
(expected gain vs switch cost), pure affinity is the honest default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from wmh.providers.pool import PoolEntry
from wmh.retrieval.embedders import HashingEmbedder

POLICY_VERSION = 1
POLICY_FILENAME = "policy.json"


class EmbedderSpec(BaseModel):
    """How to reproduce the policy's embedding function at serve time.

    Hashing only for now: it is deterministic, offline, and credential-free, so a policy file
    is self-contained. Provider-backed embedders join alongside the fitter if the benchmark
    shows hashing's locality is the bottleneck.
    """

    kind: Literal["hashing"] = "hashing"
    dim: int = 512

    def build(self) -> HashingEmbedder:
        return HashingEmbedder(dim=self.dim)


class ClusterAssignment(BaseModel):
    """One cluster: its centroid in embedding space and the pool model that serves it."""

    cluster_id: int
    label: str = ""  # human-readable, surfaces as cluster_label in the request log
    centroid: list[float]
    model: str  # pool entry name


class RoutingDecision(BaseModel):
    """Where one request goes and why (the request log's model/cluster/routing_reason)."""

    model: str
    cluster_id: int | None = None
    cluster_label: str = ""
    reason: str


class RoutingPolicy(BaseModel):
    """The persisted policy artifact (see module docstring)."""

    version: int = POLICY_VERSION
    kind: Literal["static", "cluster"]
    default_model: str  # the static answer; also the fallback if a cluster model retires
    pool: list[PoolEntry]  # snapshot of the roster this policy was defined over
    embedder: EmbedderSpec = Field(default_factory=EmbedderSpec)
    clusters: list[ClusterAssignment] = Field(default_factory=list)
    sticky: bool = True  # keep a conversation's incumbent model (see module docstring)
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
            raise ValueError("a static policy carries no clusters; use kind='cluster'")
        if self.kind == "cluster":
            if not self.clusters:
                raise ValueError("a cluster policy needs at least one cluster assignment")
            for cluster in self.clusters:
                if cluster.model not in names:
                    raise ValueError(
                        f"cluster {cluster.cluster_id} routes to '{cluster.model}', "
                        f"which is not in the policy pool (available: {sorted(names)})"
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
    centroids = np.asarray([cluster.centroid for cluster in policy.clusters])
    # HashingEmbedder output is L2-normalized; fitted centroids may not be, so normalize here.
    norms = np.linalg.norm(centroids, axis=1)
    norms[norms == 0.0] = 1.0
    best = int(np.argmax(centroids @ query / norms))
    cluster = policy.clusters[best]
    return RoutingDecision(
        model=cluster.model,
        cluster_id=cluster.cluster_id,
        cluster_label=cluster.label,
        reason=f"cluster {cluster.cluster_id}" + (f" ({cluster.label})" if cluster.label else ""),
    )
