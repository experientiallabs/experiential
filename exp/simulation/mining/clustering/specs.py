"""Which clustering a mining run uses, named explicitly.

Every backend answers one question, "which cases belong together", and they disagree about what
"together" means. Naming the backend in a spec rather than inferring it keeps that disagreement
visible: the choice is persisted with the task set, so a corpus's grouping can be reproduced and
argued with later.

- **descriptor** is the default and reproduces the provider-free path exactly. Seedless, real
  centres, group count tied to the budget.
- **kmeans** is the same shape of answer with the usual k-means++ restarts, for when the number of
  groups should be set directly rather than derived from the budget.
- **bernoulli_mixture** is the only one that reads tag probabilities as probabilities. It treats a
  0.4 as the decision model's uncertainty rather than as a coordinate four tenths of the way along
  an axis, and it can choose the group count itself by BIC, which matters on a corpus nobody has
  labelled. It needs the untransformed answers, so it is available only behind a decision-tag
  selection geometry.
- **atoms_hierarchy** is the only one that produces a tree. Fine atoms stay fixed while the tree
  above them is cut at any number of groups, so changing the group count costs nothing and
  stratified sampling has coarse strata to work with.

Three further backends were considered and left out. Non-negative matrix factorization and a
threshold tree both need machinery of their own before they answer anything, and nothing here asks
for either yet. A density backend would mean hundreds of delicate lines of dimensionality reduction
and density clustering to reach what ``bernoulli_mixture`` already offers: a group count nobody had
to choose, plus a pool of cases the model explains poorly.

Randomness is confined to the seed on each spec. Selection itself has no random source at all, so
recording the seed beside the backend is what makes a mined task set reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.clustering.descriptor import default_cluster_count, descriptor_clusters
from exp.simulation.mining.clustering.hierarchy import Hierarchy, atoms_hierarchy_fit
from exp.simulation.mining.clustering.kmeans import kmeans_fit
from exp.simulation.mining.clustering.mixture import bernoulli_mixture_fit
from exp.simulation.mining.geometry import CoreSetGeometry


class ClusteringError(ValueError):
    """A clustering spec cannot be applied to the evidence it was given."""


class DescriptorClusterSpec(ContractModel):
    """Seedless Lloyd iteration over the selection geometry, with real central traces.

    Attributes:
        kind: Fixed descriptor discriminator, defaulting to ``descriptor``.
    """

    kind: Literal["descriptor"] = "descriptor"


class KMeansClusterSpec(ContractModel):
    """k-means++ seeding with Lloyd iteration and restarts.

    Attributes:
        kind: Fixed k-means discriminator, defaulting to ``kmeans``.
        k: Groups to form, or None to use the provider-free path's ``round(sqrt(n))`` bounded by
            the task budget.
        n_init: Independent seedings tried, keeping the lowest inertia.
        max_iter: Lloyd iterations per restart.
        tol: Summed squared centroid shift at which a restart stops.
        seed: Base seed. k-means++ is genuinely random, so this is recorded rather than implied.
    """

    kind: Literal["kmeans"] = "kmeans"
    k: int | None = Field(default=None, ge=1)
    n_init: int = Field(default=10, ge=1)
    max_iter: int = Field(default=300, ge=1)
    tol: float = Field(default=1e-4, ge=0.0)
    seed: int = 42


class BernoulliMixtureClusterSpec(ContractModel):
    """Latent-class mixture over per-tag Bernoulli profiles, fitted by EM.

    It consumes the untransformed decision-model answers, so it requires a decision-tag selection
    geometry; a signed hashing vector has no reading as a per-tag probability.

    Attributes:
        kind: Fixed mixture discriminator, defaulting to ``bernoulli_mixture``.
        k: Groups to form, or None to fit every count in ``k_grid`` and keep the lowest BIC.
        k_grid: Candidate counts tried when ``k`` is None.
        max_iter: EM iterations per fit.
        tol: Relative change in total log-likelihood at which EM stops.
        prior: Symmetric Beta smoothing, which must be positive so a tag that never fires still has
            a finite log profile.
        novelty_quantile: Upper quantile of training novelty moved into one extra group, or None to
            label every trace by its best group. The pool is an ordinary nonnegative group identity
            because a cluster identity is persisted coverage evidence.
        seed: Seed for the k-means partition EM starts from.
    """

    kind: Literal["bernoulli_mixture"] = "bernoulli_mixture"
    k: int | None = Field(default=None, ge=1)
    k_grid: tuple[int, ...] = (5, 10, 25, 50)
    max_iter: int = Field(default=300, ge=1)
    tol: float = Field(default=1e-6, ge=0.0)
    prior: float = Field(default=1.0, gt=0.0)
    novelty_quantile: float | None = Field(default=None, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _grid(self) -> BernoulliMixtureClusterSpec:
        """Require a usable grid when the group count is left to BIC."""
        if self.k is None and not self.k_grid:
            raise ValueError(
                "a Bernoulli mixture with no k chooses one from k_grid, so k_grid must list at "
                "least one candidate group count"
            )
        return self


class AtomsHierarchyClusterSpec(ContractModel):
    """Fine k-means atoms with a size-weighted Ward tree above them.

    Attributes:
        kind: Fixed hierarchy discriminator, defaulting to ``atoms_hierarchy``.
        atom_count: Atoms to fit, clamped to the candidate count.
        groups: Groups the tree is cut into for cluster identities.
        n_init: k-means restarts for the atom fit.
        max_iter: k-means iterations per restart.
        tol: k-means convergence bound.
        seed: Seed for the atom fit.
    """

    kind: Literal["atoms_hierarchy"] = "atoms_hierarchy"
    atom_count: int = Field(default=64, ge=1)
    groups: int = Field(default=16, ge=1)
    n_init: int = Field(default=4, ge=1)
    max_iter: int = Field(default=300, ge=1)
    tol: float = Field(default=1e-4, ge=0.0)
    seed: int = 42

    @model_validator(mode="after")
    def _cut_fits(self) -> AtomsHierarchyClusterSpec:
        """Reject a cut finer than the atoms it would have to cut."""
        if self.groups > self.atom_count:
            raise ValueError(
                f"an atoms hierarchy cut into {self.groups} groups needs at least that many "
                f"atoms, but atom_count is {self.atom_count}"
            )
        return self


ClusterSpec = Annotated[
    DescriptorClusterSpec
    | KMeansClusterSpec
    | BernoulliMixtureClusterSpec
    | AtomsHierarchyClusterSpec,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class ClusterAssignment:
    """One partition's grouping, and the coarse strata a sampler may fill evenly.

    Attributes:
        labels: Cluster identity per candidate identity. Every candidate appears and every
            identity is nonnegative, because a cluster identity is persisted coverage evidence.
        strata: Coarser groups for stratified sampling, or None when the backend publishes no
            hierarchy and the flat clusters are the only grouping available.
        hierarchy: The tree over assignment units, when the backend built one.
    """

    labels: Mapping[str, int]
    strata: Mapping[str, int] | None = None
    hierarchy: Hierarchy | None = None


def cluster_candidates(
    spec: ClusterSpec,
    *,
    geometry: CoreSetGeometry,
    target: int,
    probabilities: Mapping[str, tuple[float, ...]] | None = None,
) -> ClusterAssignment:
    """Group one partition's candidates under the named backend.

    Args:
        spec: Which backend to use.
        geometry: Core-set positions and workload for this partition.
        target: Representative slots this partition may fill, which bounds the default group count.
        probabilities: Untransformed decision-model answers per candidate, required by the
            Bernoulli mixture and unused by the others.

    Returns:
        The grouping, plus strata and a tree when the backend produces them.

    Raises:
        ClusteringError: The backend cannot read the evidence it was given.
    """
    if not geometry.ids:
        return ClusterAssignment(labels={})
    if isinstance(spec, DescriptorClusterSpec):
        return ClusterAssignment(
            labels=descriptor_clusters(geometry, default_cluster_count(len(geometry.ids), target))
        )
    if isinstance(spec, KMeansClusterSpec):
        requested = spec.k or default_cluster_count(len(geometry.ids), target)
        fit = kmeans_fit(
            _matrix(geometry, geometry.vectors),
            min(requested, len(geometry.ids)),
            n_init=spec.n_init,
            max_iter=spec.max_iter,
            tol=spec.tol,
            seed=spec.seed,
        )
        return ClusterAssignment(labels=_labels(geometry, fit.labels))
    if isinstance(spec, BernoulliMixtureClusterSpec):
        return ClusterAssignment(
            labels=_labels(geometry, _mixture_labels(spec, geometry, probabilities))
        )
    if not isinstance(spec, AtomsHierarchyClusterSpec):  # pragma: no cover - union is exhaustive
        raise ClusteringError(
            f"no clustering backend is wired for spec kind {spec.kind!r}; a spec added to the "
            "union needs a branch here rather than falling through to another backend"
        )
    fit = atoms_hierarchy_fit(
        _matrix(geometry, geometry.vectors),
        atom_count=spec.atom_count,
        groups=spec.groups,
        n_init=spec.n_init,
        max_iter=spec.max_iter,
        tol=spec.tol,
        seed=spec.seed,
    )
    return ClusterAssignment(
        labels=_labels(geometry, fit.atoms),
        strata=_labels(geometry, fit.labels),
        hierarchy=fit.hierarchy,
    )


def _mixture_labels(
    spec: BernoulliMixtureClusterSpec,
    geometry: CoreSetGeometry,
    probabilities: Mapping[str, tuple[float, ...]] | None,
) -> np.ndarray:
    """Fit the mixture on untransformed answers, refusing to guess them from a geometry."""
    if probabilities is None:
        raise ClusteringError(
            "the bernoulli_mixture backend reads untransformed decision-model answers, which only "
            "a decision_tag selection geometry produces; choose that geometry and tag the traces, "
            "or cluster with descriptor, kmeans or atoms_hierarchy"
        )
    missing = sorted(set(geometry.ids) - set(probabilities))
    if missing:
        raise ClusteringError(
            f"the bernoulli_mixture backend has no answers for {len(missing)} candidates, "
            f"starting with {missing[0]}; tag every candidate the selection covers"
        )
    try:
        fit = bernoulli_mixture_fit(
            _matrix(geometry, probabilities),
            k=spec.k,
            k_grid=spec.k_grid,
            max_iter=spec.max_iter,
            tol=spec.tol,
            prior=spec.prior,
            novelty_quantile=spec.novelty_quantile,
            seed=spec.seed,
        )
    except ValueError as exc:
        raise ClusteringError(str(exc)) from exc
    return fit.labels


def _matrix(geometry: CoreSetGeometry, rows: Mapping[str, tuple[float, ...]]) -> np.ndarray:
    """Stack one row per candidate in ascending identity order.

    Sorting by identity rather than input order is what makes every numpy backend produce the same
    grouping however the corpus arrived, which the input-order independence of the whole pipeline
    depends on.
    """
    return np.asarray([rows[candidate_id] for candidate_id in geometry.ids], dtype=np.float64)


def _labels(geometry: CoreSetGeometry, labels: np.ndarray) -> dict[str, int]:
    """Key one backend's positional labels by candidate identity."""
    return {
        candidate_id: int(label)
        for candidate_id, label in zip(geometry.ids, labels.tolist(), strict=True)
    }
