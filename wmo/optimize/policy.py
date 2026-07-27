"""The routing policy artifact: what the routing optimizer emits and the endpoint serves.

An endpoint = {world model, policy, evidence, URL}; this module is the policy leg. Three kinds:

- `static`: every request goes to `default_model`. Valid without any optimizer run, so an
  endpoint serves from day one and the improvement report has an honest "before" state.
- `knn`: the validated champion (see `wmo.optimize.knn` for the fit and the measured result).
  Nonparametric: retrieve the request's neighbors among the fit scenarios, read each pool
  model's sim-weighted mean reward over exactly those neighbors, and route to the argmax ONLY
  when a paired statistical guard says the evidence supports leaving the baseline. Otherwise the
  baseline (`guard_model`) serves. The neighbor evidence is heavy (an L2-normalized fit
  embedding matrix plus the per-scenario reward and cost cells), so it lives in a `.npz`
  sidecar next to `policy.json` (`knn_bank_path`) and loads lazily on first use.
- `rank`: the Avengers cluster-rank router (arXiv 2505.19797), replicated faithfully from the
  reference implementation (ZhangYiqun018/Avengers, core/routing/rank_router.py): embed the
  request, softmax the distances to the `top_k_clusters` nearest k-means centres
  (`-beta * (1 - centre . query)` logits), score every pool model by the probability-weighted
  reciprocal of its per-cluster accuracy rank (`1 / (rank + 0.1)`, summed over the mixed
  clusters the model appears in), and route to the argmax. `default_rank` is the floor for a
  model absent from ALL of the mixed clusters (`1 / default_rank`); a model ranked in one and
  absent from another simply collects nothing from the latter, it is not charged a default rank
  there. This matches the reference.

The FIT that produces rank policies lives in `wmo.optimize.routing`; this module pins the
artifact schema and the serve-time selection so serving, reports, and the platform stay stable
across fitter iterations.

Serve-time stickiness: `select_model` keeps a conversation's incumbent model whenever the policy
is sticky (the default). Provider prompt caches are per-model, so switching mid-conversation
forfeits warm cache reads and pays cold writes; until the fitter learns a real switching rule
(expected gain vs switch cost), pure affinity is the honest default.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from wmo.providers.base import Embedder, ProviderConfig, ProviderKind
from wmo.providers.pool import PoolEntry
from wmo.providers.registry import get_provider
from wmo.retrieval.embedders import BatchedEmbedder, HashingEmbedder

POLICY_VERSION = 2
POLICY_FILENAME = "policy.json"

# Avengers reference defaults (config/experts_template.yaml: top_k=2, beta=6.0,
# default_rank=999). Kept on the policy so serving needs no side-channel config.
DEFAULT_TOP_K_CLUSTERS = 2
DEFAULT_BETA = 6.0
DEFAULT_RANK = 999

# kNN champion defaults: the exact configuration validated on routerbench-ours9
# (+1.0 accuracy point over the best single model at -27% cost, 5/5 split seeds).
KNN_BANK_FILENAME = "policy_knn_bank.npz"
# The sidecar suffix a fit DERIVES from its policy path, so two policies fitted into one
# directory own two banks instead of racing for one shared name. APPENDED rather than
# substituted for the policy's extension (`support.json` -> `support.json.bank.npz`): appending
# is injective on filenames, while replacing the extension would map `support.json` and
# `support.yaml` onto one bank and reintroduce the collision. `KNN_BANK_FILENAME` above stays
# the resolution fallback for artifacts that record no path of their own; see
# `RoutingPolicy.knn_bank_path`.
KNN_BANK_SUFFIX = ".bank.npz"
DEFAULT_RAG_NUM = 50
DEFAULT_RAG_THRES = 0.95
DEFAULT_KNN_Z = 0.5
DEFAULT_KNN_MIN_PAIRS = 8
# Below this many paired neighbors the guard floors its standard error at the maximal binomial
# SE, sqrt(0.25/n). A SMALL-SAMPLE correction only: on a thin bank a lucky zero-variance
# neighborhood otherwise makes any positive mean difference look significant, which is how a
# small-bank router talks itself into pricier-and-worse picks. Large-n empirical SEs are
# reliable, and flooring them there would just tax real wins.
SE_FLOOR_MAX_PAIRS = 30

# Banks are read once per policy instance; the lock is module level so the cache costs the
# policy no per-instance state (a lock stored on the model would make two otherwise identical
# policies compare unequal).
_BANK_LOAD_LOCK = threading.Lock()


def write_artifact_atomically(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` through a staging file, replacing it in one step.

    Serving reads an artifact directory while the optimizer writes it, and a half-written
    policy.json is a mount failure rather than a slightly stale endpoint. It is also what lets
    a command that writes several artifacts promise that a failure leaves the old ones intact.
    `KnnBank.save` stages the same way for the sidecar it streams through numpy.

    The staging name is unique PER CALL. `replace` is atomic, but a staging path shared between
    two concurrent writers is not: they would interleave on one file, so one could publish the
    other's bytes under its own name and the loser would fail on a file already renamed away.
    It is also hidden and cleaned up on failure, so an interrupted write cannot leave litter in
    an artifact directory that serving and the fitter both scan.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        staging.write_bytes(payload)
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def knn_bank_path_for(policy_path: Path) -> Path:
    """The evidence sidecar that belongs to one policy file.

    `models/support.json` -> `models/support.json.bank.npz`. Distinct policy filenames always
    give distinct bank filenames (see `KNN_BANK_SUFFIX`), and one owner for the derivation means
    the fitter, the CLI's console line, and any tooling that cleans an artifact directory cannot
    disagree about which `.npz` belongs to which policy.
    """
    return policy_path.with_name(f"{policy_path.name}{KNN_BANK_SUFFIX}")


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


def embedder_provenance(spec: EmbedderSpec) -> str:
    """The embedding-function half of `fitted_from`, taken from the spec serving rebuilds.

    Derived from `EmbedderSpec` rather than from the caller's flags, so it cannot describe an
    embedder different from the one recorded in the artifact. The azure RESOURCE is part of the
    identity and not just the deployment name: two accounts routinely hold a deployment of the
    same name and dimension, their embeddings are not interchangeable, and a refit that only
    repointed the endpoint therefore has to read as a different fit. The credential variable is
    left out on purpose: renaming it does not move a single vector.
    """
    tag = f"{spec.kind}-{spec.dim}"
    return tag if spec.kind == "hashing" else f"{tag}/{spec.deployment}@{spec.endpoint}"


@dataclass(frozen=True)
class KnnBank:
    """The fit-split evidence a `knn` policy routes against: the `.npz` sidecar's contents.

    Arrays are aligned: row `i` is `scenario_ids[i]`, column `j` is `models[j]`. A reward or
    cost cell is NaN when that model has no scored episode for that scenario, which is how a
    ragged outcome matrix stays a dense array; the router weighs only the scored cells.

    Embeddings are float32 and L2-normalized at fit time, so serve-time neighbor search is one
    matrix product. float32 halves a bank that is already ~10MB at 3072 dimensions, and costs
    nothing measurable: cosine similarities agree with float64 to ~1e-7, far below the reward
    differences the guard tests.
    """

    embeddings: np.ndarray  # (scenarios, dim) float32, L2-normalized
    rewards: np.ndarray  # (scenarios, models) float32, NaN where unscored
    costs: np.ndarray  # (scenarios, models) float32 USD, NaN where unscored
    models: list[str]
    scenario_ids: list[str]

    def __post_init__(self) -> None:
        scenarios, models = len(self.scenario_ids), len(self.models)
        if not scenarios or not models:
            raise ValueError("a knn bank needs at least one scenario and one model")
        for name, array in (("rewards", self.rewards), ("costs", self.costs)):
            if array.shape != (scenarios, models):
                raise ValueError(
                    f"{name} has shape {array.shape}, expected "
                    f"({scenarios}, {models}) from the scenario and model lists"
                )
        if self.embeddings.shape[0] != scenarios or self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings has shape {self.embeddings.shape}, expected "
                f"({scenarios}, dim) from the scenario list"
            )

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def mean_costs(self) -> np.ndarray:
        """Per-model mean cost over the bank's scored cells (NaN for a model with none).

        The unit the guard's pricier-than-baseline test compares. Cells, not episodes: a model
        that happened to be rerun more often on some scenario does not get extra weight.
        """
        scored = ~np.isnan(self.costs)
        counts = scored.sum(axis=0)
        totals = np.where(scored, np.nan_to_num(self.costs), 0.0).sum(axis=0)
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)

    def save(self, path: Path) -> None:
        """Write the sidecar atomically (a half-written 10MB bank must not be loadable).

        Staged under a name unique to this call, for the reason spelled out in
        `write_artifact_atomically`: two fits racing on one `--out` derive the same bank path,
        and a shared staging file would let one publish the OTHER's evidence under its own name.
        A bank is streamed through numpy rather than buffered into bytes, so it stages itself
        instead of going through that helper.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with staging.open("wb") as handle:
                np.savez(
                    handle,
                    embeddings=self.embeddings.astype(np.float32),
                    rewards=self.rewards.astype(np.float32),
                    costs=self.costs.astype(np.float32),
                    models=np.asarray(self.models),
                    scenario_ids=np.asarray(self.scenario_ids),
                )
            staging.replace(path)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> KnnBank:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                embeddings=np.asarray(data["embeddings"], dtype=np.float32),
                rewards=np.asarray(data["rewards"], dtype=np.float32),
                costs=np.asarray(data["costs"], dtype=np.float32),
                models=[str(name) for name in data["models"]],
                scenario_ids=[str(sid) for sid in data["scenario_ids"]],
            )


class ClusterRanking(BaseModel):
    """One fitted cluster: its centroid and the pool models ranked by in-cluster accuracy."""

    cluster_id: int
    label: str = ""  # human-readable, surfaces as cluster_label in the request log
    centroid: list[float]
    ranking: list[str] = Field(min_length=1)  # pool entry names, best first
    scores: dict[str, float] = Field(default_factory=dict)  # per-model mean reward (evidence)
    costs: dict[str, float] = Field(default_factory=dict)  # per-model mean cost (evidence)
    # Per-model scored EPISODES behind those means: the support the fit-time guard weighs, kept
    # so `rerank_policy` can re-apply the identical check without the outcome matrix.
    support: dict[str, int] = Field(default_factory=dict)
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
    kind: Literal["static", "rank", "knn"]
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
    support_tilt_gamma: float = Field(default=0.0, ge=0.0)
    # The unit a cost knob trades against one reward point: "one average call". Derived per kind,
    # each matching its own knob's reference implementation. rank: the fit set's mean cost per
    # scored EPISODE. knn: the mean of the per-model mean cell costs, which is what
    # `pick_lam`'s reference divides by (coverage-robust, since a model measured on few
    # scenarios cannot drag the unit toward its own price). 0 when the fit carried no costs.
    cost_scale: float = 0.0
    # The fit-time only-replace-if-better guard, recorded so any later transform of this policy
    # (today `rerank_policy`) re-applies the SAME floor instead of quietly dropping it. None
    # means the fit ran unguarded.
    guard_model: str | None = None
    min_support: int | None = None  # scored episodes a challenger needs to lead its cluster
    guard_margin: float | None = None  # reward the challenger must beat the guard by
    fitted_from: str | None = None  # provenance: the outcome matrix the fitter used

    # kNN policies only (see module docstring and `wmo.optimize.knn`). The fitter records the
    # bank it actually wrote (`knn_bank_path_for(<policy path>)`), so serving resolves the
    # sidecar EXPLICITLY instead of by convention and two policies can share a directory. The
    # value is a bare FILENAME resolved next to policy.json, so a model directory stays
    # portable; an absolute path is honored as given (research code and tests point at banks
    # elsewhere). The default is the legacy conventional name, which is what an artifact
    # written before the fitter recorded a derived name resolves to.
    knn_bank_path: str = KNN_BANK_FILENAME
    rag_num: int = Field(default=DEFAULT_RAG_NUM, ge=1)  # neighbor budget
    # A neighbor is a fit scenario with similarity above `rag_thres` times the `rag_num`-th best
    # similarity: a RELATIVE rule, so a query in a dense region keeps more evidence than one out
    # on its own, instead of every query being handed exactly k neighbors of any quality.
    rag_thres: float = Field(default=DEFAULT_RAG_THRES, gt=0.0, le=1.0)
    # The confidence knob: standard errors of paired evidence a non-baseline pick must clear
    # (doubled when the pick is pricier than the baseline). Higher = stricter = fewer requests
    # leave the baseline. 0 routes on any positive mean difference.
    knn_z: float = Field(default=DEFAULT_KNN_Z, ge=0.0)
    # How the guard treats the two sides of the price comparison (R1's `stat`/`stat_asym`).
    # "symmetric" (the champion): a pricier pick needs 2z standard errors, a cheaper one needs z,
    # so BOTH must be positively supported. "asymmetric": a pricier pick needs z, and a cheaper
    # one only has to clear -z, i.e. it is accepted unless the evidence says it is significantly
    # worse. The asymmetric bar is what makes `pick_lam` able to act at all: under the symmetric
    # bar a cost-tilted cheap pick is usually reverted straight back to the pricier baseline,
    # which spends MORE, not less (measured; see `wmo.optimize.knn.apply_cost_quality`).
    guard_mode: Literal["symmetric", "asymmetric"] = "symmetric"
    knn_min_pairs: int = Field(default=DEFAULT_KNN_MIN_PAIRS, ge=0)  # neighbors scored on both
    se_floor: bool = True  # small-sample variance floor (see SE_FLOOR_MAX_PAIRS)
    # Novelty floor: queries whose best bank similarity is below this abstain to the baseline
    # (None = off). Set at fit time from the floor_q quantile of bank self-NN similarities;
    # the serving-side coverage/robustness knob for task drift.
    floor_sim: float | None = None
    # The QUANTILE `floor_sim` came from. Kept alongside it because the threshold alone cannot be
    # read back: 0.4 similarity is a strict floor on one bank and a loose one on another, so
    # nothing downstream can tell which coverage setting a policy is on (which dial position it
    # matches, what to report on the config endpoint) from `floor_sim`. None means unknown: a
    # policy written before this field existed, or one whose threshold was set by hand.
    floor_q: float | None = Field(default=None, ge=0.0, le=1.0)
    # The cost knob (R1 `pick_lam`): the raw pick maximizes
    # `profile[m] - pick_lam * mean_cost[m] / cost_scale` instead of the profile alone, so
    # pick_lam is "reward points paid per average-call-cost unit". The guard runs AFTER, on the
    # untilted paired evidence, so cost pressure can only ever demote a pick, never promote one
    # the evidence rejects. 0 = off (the validated champion). Slide it with
    # `wmo.optimize.knn.apply_cost_quality`, which maps one operator-facing dial onto this.
    pick_lam: float = Field(default=0.0, ge=0.0)
    # The operator dial that produced the knobs above, when one did: 0 = max quality,
    # 1 = max savings (`wmo.optimize.knn.apply_cost_quality`). None means "as fitted", never
    # slid. Provenance, not an input: serving reads the knobs, and the mapping is absolute, so
    # re-applying any dial setting to any policy of this kind lands on the same knobs.
    cost_quality: float | None = Field(default=None, ge=0.0, le=1.0)

    # Set by `save`/`load` so a relative `knn_bank_path` resolves against the policy file, and
    # the lazily loaded bank. Private: not part of the artifact.
    _source_dir: Path | None = PrivateAttr(default=None)
    _bank: KnnBank | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate(self) -> RoutingPolicy:
        names = {entry.name for entry in self.pool}
        if self.default_model not in names:
            raise ValueError(
                f"default_model '{self.default_model}' is not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.guard_model is not None and self.guard_model not in names:
            raise ValueError(
                f"guard_model '{self.guard_model}' is not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.kind != "rank" and self.clusters:
            raise ValueError(f"a {self.kind} policy carries no clusters; use kind='rank'")
        if self.kind == "knn":
            # The whole algorithm is "leave the baseline only on evidence", so a knn policy
            # without a baseline is not a weaker policy, it is an undefined one. default_model
            # and guard_model are the same thing here, and being able to set them apart would
            # only invite two disagreeing fallbacks.
            if self.guard_model is None:
                raise ValueError(
                    "a knn policy needs guard_model set to its baseline model (the fallback "
                    "every unsupported request reverts to); fit with --fallback"
                )
            if self.guard_model != self.default_model:
                raise ValueError(
                    f"knn guard_model '{self.guard_model}' must equal default_model "
                    f"'{self.default_model}': the baseline and the fallback are one model"
                )
            if self.pick_lam > 0.0 and self.cost_scale <= 0.0:
                # Caught here rather than per request: a policy whose cost knob has no cost
                # unit to divide by is an unservable artifact, and finding that out inside a
                # request would turn a config mistake into a 502 per call.
                raise ValueError(
                    f"pick_lam={self.pick_lam:g} needs a positive cost_scale, but this policy "
                    "has none; refit on a matrix that carries per-episode costs "
                    "(`wmo optimize route fit --kind knn`) before turning the cost knob"
                )
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
        """Write the policy artifact atomically (a torn policy.json must not be loadable)."""
        write_artifact_atomically(path, self.model_dump_json(indent=2).encode("utf-8"))
        self._source_dir = path.parent

    @classmethod
    def load(cls, path: Path) -> RoutingPolicy:
        policy = cls.model_validate_json(path.read_text(encoding="utf-8"))
        policy._source_dir = path.parent
        return policy

    def bank_path(self) -> Path:
        """Where this policy's kNN sidecar lives (see `knn_bank_path`)."""
        if self.kind != "knn":
            raise ValueError(f"a {self.kind} policy has no knn bank")
        candidate = Path(self.knn_bank_path)
        if candidate.is_absolute():
            return candidate
        return (self._source_dir or Path()) / candidate

    def attach_bank(self, bank: KnnBank) -> None:
        """Use `bank` as this policy's evidence instead of reading the sidecar.

        The fitter primes a freshly fitted policy with the bank it just built, so fit-then-
        evaluate does not round trip through disk; tests use it to route against a hand-built
        bank. Validated exactly like a loaded one.
        """
        self._validate_bank(bank, source="the attached bank")
        self._bank = bank

    def knn_bank(self) -> KnnBank:
        """Load (once, then cached) the neighbor evidence this kNN policy routes against."""
        if self.kind != "knn":
            raise ValueError(f"a {self.kind} policy has no knn bank")
        with _BANK_LOAD_LOCK:
            if self._bank is None:
                path = self.bank_path()
                if not path.is_file():
                    raise FileNotFoundError(
                        f"knn policy bank not found at {path}: a knn policy.json is served "
                        f"together with its '{self.knn_bank_path}' sidecar. Copy the sidecar "
                        f"next to the policy file, or refit with "
                        f"`wmo optimize route fit --kind knn`."
                    )
                try:
                    bank = KnnBank.load(path)
                    self._validate_bank(bank, source=str(path))
                except (ValueError, KeyError) as error:
                    raise ValueError(f"invalid knn bank at {path}: {error}") from error
                self._bank = bank
            return self._bank

    def _validate_bank(self, bank: KnnBank, *, source: str) -> None:
        """Check a bank against this policy: same embedder dimension, known models, a baseline."""
        if bank.dim != self.embedder.dim:
            raise ValueError(
                f"{source} has {bank.dim}-dimensional embeddings but the policy's embedder "
                f"spec is {self.embedder.dim}-dimensional; the bank was fitted with a "
                "different embedder than this policy would embed requests with"
            )
        names = {entry.name for entry in self.pool}
        unknown = [model for model in bank.models if model not in names]
        if unknown:
            raise ValueError(
                f"{source} carries models {unknown} that are not in the policy pool "
                f"(available: {sorted(names)})"
            )
        if self.guard_model not in bank.models:
            raise ValueError(
                f"{source} does not cover baseline '{self.guard_model}' (it covers "
                f"{sorted(bank.models)}); the bank was fitted over a different pool"
            )
        column = bank.rewards[:, bank.models.index(self.guard_model)]
        if bool(np.isnan(column).all()):
            raise ValueError(
                f"{source} has no scored reward for baseline '{self.guard_model}' on any fit "
                "scenario, so the guard could never compare a pick against it and every "
                "request would revert unmeasured; fit on a matrix that measured the baseline, "
                "or pin a baseline it did measure"
            )


def select_model(
    policy: RoutingPolicy,
    text: str,
    *,
    incumbent: str | None = None,
    embedder: Embedder | None = None,
) -> RoutingDecision:
    """Pick the pool model for one request.

    `text` is whatever the caller deems the routable content (serving passes the latest user
    message). `incumbent` is the model already serving this conversation, if any: a sticky
    policy keeps it (per-model prompt caches make switching expensive), unless it has been
    retired from the pool, in which case the request re-routes as if fresh.

    `embedder` lets a caller that routes many requests reuse ONE embedder built from
    `policy.embedder` (an azure spec otherwise constructs a fresh HTTP client per call); it must
    be the function this policy's spec describes, or the fitted centroids (rank) and neighbor
    bank (knn) are meaningless. Default None builds it from the spec per call.
    """
    names = {entry.name for entry in policy.pool}
    if incumbent is not None and incumbent in names and policy.sticky:
        return RoutingDecision(model=incumbent, reason="sticky: conversation affinity")
    if policy.kind == "static":
        return RoutingDecision(model=policy.default_model, reason="static policy")

    query = np.asarray((embedder or policy.embedder.build()).embed([text])[0])
    if policy.kind == "knn":
        return knn_decision(policy, query)
    return rank_decision(policy, query)


def rank_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """The Avengers rank-routing core, on an already-embedded query.

    Shared by `select_model` (one live request) and batch evaluation (the benchmark embeds
    all scenarios once) so the served path and the measured path can never diverge. Faithful
    to the reference implementation: the query is L2-normalized HERE, so no caller can skip the
    step and silently change the softmax temperature (`beta * distance` is scale-sensitive even
    though the nearest-cluster order is not); centres are used exactly as fitted, NOT
    re-normalized (k-means centres of unit vectors are near-unit; the reference dots them raw,
    so we do too).
    """
    names = {entry.name for entry in policy.pool}
    centres = np.asarray([cluster.centroid for cluster in policy.clusters])
    vector = np.asarray(query, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        # A zero query keeps its zeros, exactly like sklearn's Normalizer.
        vector = vector / norm
    dists = 1.0 - centres @ vector
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


def knn_decision(policy: RoutingPolicy, query: np.ndarray) -> RoutingDecision:
    """The kNN reward-profile core with its paired guard, on an already-embedded query.

    Shared by `select_model` (one live request) and batch evaluation, so the served path and the
    measured path cannot diverge. Three stages, each of which the returned `reason` accounts for:

    1. Neighbors: the fit scenarios whose similarity beats `rag_thres` times the `rag_num`-th
       best similarity. The query is L2-normalized HERE (bank rows already are), so no caller
       can skip the step and turn the similarity threshold into a magnitude test.
    2. Profile: each model's similarity-weighted mean reward over the neighbors it was scored
       on. The raw pick maximizes that profile minus the cost knob's price term,
       `pick_lam * mean_cost / cost_scale` (zero at the default pick_lam=0; ties break by pool
       order, deterministically).
    3. Guard: a non-baseline pick must beat the baseline on PAIRED per-neighbor reward
       differences, mean > `knn_z` standard errors (doubled when the pick is also pricier, the
       asymmetry that kills confidently-wrong pricier-and-worse picks; see `guard_mode` for the
       economic variant that only asks a cheaper pick not to be significantly worse), on at
       least `knn_min_pairs` neighbors scored on both sides. Otherwise the baseline serves.

    The guard is what makes the policy safe to deploy: absent evidence, the answer is the
    baseline, so the worst case is the baseline's behavior rather than a confident stranger's.
    It runs on the UNTILTED evidence and after the cost knob, exactly as in the research code:
    the knob reorders candidates, and the guard then re-vetoes whatever it cannot support, so
    turning the knob up can never talk the router into a pick the evidence rejects.
    """
    if policy.kind != "knn":
        raise ValueError(f"knn_decision needs a knn policy, got kind='{policy.kind}'")
    bank = policy.knn_bank()
    baseline = policy.default_model

    vector = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    sims = bank.embeddings @ vector
    if policy.floor_sim is not None and float(np.max(sims)) < policy.floor_sim:
        return RoutingDecision(
            model=baseline,
            reason=(
                f"knn novelty abstain: best similarity {float(np.max(sims)):.3f} below the "
                f"fit-bank floor {policy.floor_sim:.3f}, serving {baseline}"
            ),
        )

    budget = min(policy.rag_num, sims.shape[0])
    kth = float(np.sort(sims)[-budget])
    rows = np.flatnonzero(sims > policy.rag_thres * kth)
    if rows.size == 0:
        # Every similarity at or below the cut (possible when the kth is negative, so the
        # threshold moves the wrong way): fall back to the single nearest fit scenario.
        rows = np.asarray([int(np.argmax(sims))])

    weights = np.clip(sims[rows], 0.0, None).astype(np.float64)[:, None]
    rewards = bank.rewards[rows].astype(np.float64)
    scored = ~np.isnan(rewards)
    weight_totals = (scored * weights).sum(axis=0)
    reward_totals = (np.where(scored, np.nan_to_num(rewards), 0.0) * weights).sum(axis=0)
    profile = np.full(weight_totals.shape, np.nan)
    np.divide(reward_totals, weight_totals, out=profile, where=weight_totals > 0.0)

    pool_order = {entry.name: index for index, entry in enumerate(policy.pool)}
    candidates = [
        (index, name) for index, name in enumerate(bank.models) if not np.isnan(profile[index])
    ]
    if not candidates:
        return RoutingDecision(
            model=baseline,
            reason=f"knn: {rows.size} neighbors carry no scored reward, serving {baseline}",
        )
    mean_cost = bank.mean_costs()
    # The cost knob prices each candidate in average-call units before the argmax; at
    # pick_lam=0 the key is the bare profile, bit-identical to the validated champion. A model
    # the bank never priced pays exactly one unit, the reference's default.
    tilt = np.zeros(profile.shape)
    if policy.pick_lam > 0.0:
        priced = np.where(np.isnan(mean_cost), policy.cost_scale, mean_cost)
        tilt = policy.pick_lam * priced / policy.cost_scale
    pick_index, pick = max(
        candidates, key=lambda item: (profile[item[0]] - tilt[item[0]], -pool_order[item[1]])
    )
    if pick == baseline:
        # Two different decisions land here, and the log has to tell them apart: the baseline
        # genuinely led on reward, or the cost knob demoted a leader that priced badly. The
        # second is the dominant case on the savings leg of the dial, so naming it (and the model
        # it outbid) is what makes that leg debuggable from the request log at all.
        leader_index, leader = max(
            candidates, key=lambda item: (profile[item[0]], -pool_order[item[1]])
        )
        if leader != baseline:
            return RoutingDecision(
                model=baseline,
                reason=(
                    f"knn cost knob (lam={policy.pick_lam:g}): {baseline} serves; {leader} led "
                    f"{rows.size} neighbors on evidence (profile {profile[leader_index]:.3f} vs "
                    f"{profile[pick_index]:.3f}) but not on price"
                ),
            )
        return RoutingDecision(
            model=baseline,
            reason=f"knn: baseline {baseline} leads {rows.size} neighbors "
            f"(profile {profile[pick_index]:.3f})",
        )

    base_index = bank.models.index(baseline)
    paired = scored[:, pick_index] & scored[:, base_index]
    diffs = rewards[paired, pick_index] - rewards[paired, base_index]
    pairs = int(diffs.size)
    mean_diff = float(diffs.mean()) if pairs else 0.0
    error = float(diffs.std(ddof=1)) / pairs**0.5 if pairs > 1 else 0.0
    if policy.se_floor and 0 < pairs < SE_FLOOR_MAX_PAIRS:
        error = max(error, (0.25 / pairs) ** 0.5)
    pricier = bool(mean_cost[pick_index] > mean_cost[base_index])
    if policy.guard_mode == "asymmetric":
        z_effective = policy.knn_z if pricier else -policy.knn_z
    else:
        z_effective = 2 * policy.knn_z if pricier else policy.knn_z
    needed = z_effective * error

    if pairs < policy.knn_min_pairs or not mean_diff > needed:
        detail = (
            f"{pairs} paired neighbors, delta={mean_diff:+.3f}, needs "
            f"> {z_effective:g}xSE={needed:.3f}"
        )
        if pairs < policy.knn_min_pairs:
            detail = f"{pairs} paired neighbors < {policy.knn_min_pairs} required"
        return RoutingDecision(
            model=baseline,
            reason=f"knn guard: reverted to {baseline}, evidence insufficient ({detail})",
        )
    knob = f", cost knob lam={policy.pick_lam:g}" if policy.pick_lam > 0.0 else ""
    # Only the symmetric bar doubles z for a pricier pick; the asymmetric one holds it at z.
    price_note = ""
    if pricier:
        price_note = " (pricier)" if policy.guard_mode == "asymmetric" else " (pricier, doubled z)"
    return RoutingDecision(
        model=pick,
        reason=f"knn: {rows.size} neighbors, delta={mean_diff:+.3f} > {z_effective:g}xSE"
        f"={needed:.3f}{price_note}{knob}",
    )
