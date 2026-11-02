"""How leftover representative budget is spent, once the guarantees are already paid for.

Selection has two halves. The first half is not negotiable: reserved tail slots, one anchor per
lineage, the budget ceiling and conservation of workload mass carry the coverage and leakage
guarantees the whole pipeline rests on. The second half is taste, and this module owns it. By the
time a strategy here runs, every lineage is already represented and every reserved slot has been
offered its pick, so a strategy can only decide which of several defensible fills to use. That is
why a strategy is a spec rather than a flag: it is a recorded choice about emphasis, and it goes
into the artifact identity so a task set states how its remaining slots were spent.

Each strategy answers the same question differently:

- **medoid** spends the rest on being *representative*: one real centre per cluster, then
  farthest-first for whatever is left. A reviewer reading the task set sees the typical case of
  each group.
- **proportional** spends it on *fidelity to traffic*: a cluster carrying a third of the workload
  gets about a third of the slots, so the task set's shape matches the corpus's shape.
- **diversity** spends it on *spread*, ignoring groups entirely. Useful when the clustering is
  uncertain and what matters is covering the space.
- **stratified** spends it *evenly across groups*, so a small but distinct group is not outvoted by
  a large one. This is the one to use when rare behaviour matters more than common behaviour.
- **representative** spends it on the cases *most of the corpus looks like*: each cluster's share
  goes to its most typical members rather than its most distinctive ones. This is what a set of
  evaluation seeds wants, because an eval built from a group's outliers measures behaviour the
  group mostly does not exhibit.
- **balanced** is ``representative`` with equal shares instead of mass-proportional ones, for when
  every kind of work has to be evaluated whether or not it is common.

The four older strategies all end in farthest-first, so they all spend leftover budget on the cases
least like anything chosen. That is deliberate for ``diversity`` and incidental for the rest, and it
is the one emphasis an eval seed set does not want, which is why the two newer strategies refill
from their own clusters instead of falling through to it.

Every strategy is bound by the same rules, and the tests check all of them rather than trusting the
prose: never exceed the target, never select one identity twice, always terminate, and break every
tie by ascending trace identity so the result never depends on input order.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from typing import Annotated, Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.simulation.mining.geometry import CoreSetGeometry


class MedoidSamplingSpec(ContractModel):
    """Fill with one real cluster medoid each, then farthest-first coverage.

    This is the default, and it is what a provider-free build has always done.

    Attributes:
        kind: Fixed medoid discriminator, defaulting to ``medoid``.
    """

    kind: Literal["medoid"] = "medoid"


class ProportionalSamplingSpec(ContractModel):
    """Split the remaining budget across clusters by the workload mass each carries.

    Largest-remainder apportionment, so the slots sum to exactly the remaining budget rather than
    drifting by a rounding error. Each cluster spends its share on its medoid first and then on
    within-cluster farthest-first points, so a large cluster contributes variety rather than one
    repeated centre.

    Attributes:
        kind: Fixed proportional discriminator, defaulting to ``proportional``.
    """

    kind: Literal["proportional"] = "proportional"


class DiversitySamplingSpec(ContractModel):
    """Fill by global farthest-first alone, ignoring cluster membership.

    Clusters still appear in the coverage report, because they are audit evidence; they just do not
    steer the fill. Worth choosing when the clustering is a weak hypothesis and coverage of the
    space matters more than coverage of the groups.

    Attributes:
        kind: Fixed diversity discriminator, defaulting to ``diversity``.
    """

    kind: Literal["diversity"] = "diversity"


class StratifiedSamplingSpec(ContractModel):
    """Give every stratum an equal share, round-robin, mass-descending.

    Strata are the clusterer's coarse groups when it publishes a hierarchy cut, and the flat
    clusters otherwise. Equal shares are the point: a stratum holding 2% of the workload gets the
    same budget as one holding 40%, which is what makes rare behaviour survive selection.

    Attributes:
        kind: Fixed stratified discriminator, defaulting to ``stratified``.
    """

    kind: Literal["stratified"] = "stratified"


class RepresentativeSamplingSpec(ContractModel):
    """Split the remaining budget by cluster mass, then spend each share on typical members.

    Shares are the same largest-remainder apportionment ``proportional`` uses; what differs is how
    a cluster spends its share. Instead of one centre and then the points furthest from everything
    chosen, a cluster hands back the members that most of it looks like, in
    :meth:`~exp.simulation.mining.geometry.CoreSetGeometry.representatives` order. A cluster that
    runs out of members returns its slots to the apportionment rather than leaking them into
    farthest-first, so the whole budget stays inside the clusters.

    Attributes:
        kind: Fixed representative discriminator, defaulting to ``representative``.
    """

    kind: Literal["representative"] = "representative"


class BalancedSamplingSpec(ContractModel):
    """Give every cluster the same share, and spend each share on typical members.

    The picker is ``representative``'s; the allocation is ``stratified``'s. Use it when every kind
    of work has to be evaluated at comparable depth whatever share of traffic it carries, and use
    ``representative`` when the seed set should look like the traffic.

    Attributes:
        kind: Fixed balanced discriminator, defaulting to ``balanced``.
    """

    kind: Literal["balanced"] = "balanced"


SamplingSpec = Annotated[
    MedoidSamplingSpec
    | ProportionalSamplingSpec
    | DiversitySamplingSpec
    | StratifiedSamplingSpec
    | RepresentativeSamplingSpec
    | BalancedSamplingSpec,
    Field(discriminator="kind"),
]


def record_selection(
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    candidate_id: str,
    reason: str,
) -> None:
    """Add one selected source trace or append another auditable reason to it.

    Args:
        selected_ids: Selection order being built. Order is load-bearing: it fixes task order in
            the persisted task set and therefore the artifact digest.
        reasons: Audit reasons keyed by selected identity.
        candidate_id: Identity to retain.
        reason: Why this phase wanted it.
    """
    if candidate_id not in selected_ids:
        selected_ids.append(candidate_id)
    reasons[candidate_id].append(reason)


def members_by_group(labels: Mapping[str, int]) -> dict[int, list[str]]:
    """Group identities by their label, each group's members in ascending identity order.

    Args:
        labels: One group label per identity.

    Returns:
        Members keyed by label. Only non-empty groups appear, so a clusterer that lost every
        member of a group never reports an empty one.
    """
    grouped: dict[int, list[str]] = defaultdict(list)
    for identity in sorted(labels):
        grouped[labels[identity]].append(identity)
    return dict(grouped)


def fill_budget(
    spec: SamplingSpec,
    *,
    geometry: CoreSetGeometry,
    labels: Mapping[str, int],
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
    strata: Mapping[str, int] | None = None,
) -> None:
    """Spend the remaining representative budget under one sampling strategy.

    Args:
        spec: Which strategy to apply.
        geometry: Core-set positions and workload for this partition.
        labels: Cluster label per candidate identity.
        selected_ids: Selection order so far, extended in place.
        reasons: Audit reasons so far, extended in place.
        target: Total selection size this partition may reach.
        strata: Coarse groups for stratified sampling, or None to stratify on ``labels``.

    Raises:
        ValueError: The strategy could not reach the target because candidates ran out, which
            means the target was computed against a different candidate set.
    """
    if isinstance(spec, MedoidSamplingSpec):
        _fill_medoid(geometry, labels, selected_ids, reasons, target)
    elif isinstance(spec, ProportionalSamplingSpec):
        _fill_proportional(geometry, labels, selected_ids, reasons, target)
    elif isinstance(spec, StratifiedSamplingSpec):
        _fill_stratified(geometry, strata or labels, selected_ids, reasons, target)
    elif isinstance(spec, RepresentativeSamplingSpec | BalancedSamplingSpec):
        _fill_representative(
            geometry,
            labels,
            selected_ids,
            reasons,
            target,
            even=isinstance(spec, BalancedSamplingSpec),
        )
    elif not isinstance(spec, DiversitySamplingSpec):  # pragma: no cover - union is exhaustive
        raise ValueError(
            f"no sampling strategy is wired for spec kind {spec.kind!r}; a spec added to the "
            "union needs a branch here rather than silently falling through to farthest-first"
        )
    # Every strategy ends here, and diversity is nothing but this: whatever budget a strategy left
    # is spent on the points furthest from everything already selected.
    _fill_farthest_first(geometry, selected_ids, reasons, target)


def _fill_medoid(
    geometry: CoreSetGeometry,
    labels: Mapping[str, int],
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
) -> None:
    """Retain one real medoid per cluster, in ascending cluster order.

    An already-selected medoid records the extra reason without consuming budget, and it does so
    before the budget check, so a cluster whose centre a reserved slot already picked is still
    reported as medoid-covered even once the budget is spent.
    """
    for cluster_id, member_ids in sorted(members_by_group(labels).items()):
        medoid_id = geometry.medoid(member_ids)
        if medoid_id in selected_ids:
            reasons[medoid_id].append(f"cluster-medoid:{cluster_id}")
            continue
        if len(selected_ids) >= target:
            break
        record_selection(selected_ids, reasons, medoid_id, f"cluster-medoid:{cluster_id}")


def _fill_proportional(
    geometry: CoreSetGeometry,
    labels: Mapping[str, int],
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
) -> None:
    """Apportion the remaining budget by cluster workload mass, then fill each cluster."""
    groups = members_by_group(labels)
    remaining = target - len(selected_ids)
    if remaining <= 0:
        return
    shares = _largest_remainder(
        {
            cluster_id: sum(geometry.masses[member] for member in members)
            for cluster_id, members in groups.items()
        },
        remaining,
    )
    for cluster_id, share in sorted(shares.items()):
        member_ids = groups[cluster_id]
        medoid_id = geometry.medoid(member_ids)
        if medoid_id in selected_ids:
            # An earlier phase already took this cluster's centre. Record that the cluster is
            # medoid-covered, exactly as the medoid strategy does, without spending a slot on it.
            reasons[medoid_id].append(f"cluster-medoid:{cluster_id}")
        for position in range(share):
            if len(selected_ids) >= target:
                return
            if all(member in selected_ids for member in member_ids):
                break
            if position == 0 and medoid_id not in selected_ids:
                record_selection(selected_ids, reasons, medoid_id, f"cluster-medoid:{cluster_id}")
                continue
            record_selection(
                selected_ids,
                reasons,
                geometry.farthest_first(selected_ids, within=member_ids),
                f"cluster-fill:{cluster_id}",
            )


def _fill_representative(
    geometry: CoreSetGeometry,
    labels: Mapping[str, int],
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
    *,
    even: bool,
) -> None:
    """Spend the budget on each cluster's most typical members, by mass or evenly.

    Apportionment is recomputed each pass over the clusters, and a cluster with nothing left to
    offer is dropped from it, so slots a small cluster cannot use are handed to clusters that can
    rather than falling through to the farthest-first tail. Every pass either selects something or
    returns, which is what makes the loop terminate on a partition the target overshoots.

    Every pick records ``cluster-representative``. The medoid is the first of them whenever no
    earlier phase took it, but it is not labelled as one: once a reserved slot has covered a
    cluster's centre the first pick here is deliberately the next most typical case, and a
    ``cluster-medoid`` reason on it would describe a centre this phase did not choose.
    """
    groups = members_by_group(labels)
    while len(selected_ids) < target:
        live = {
            cluster_id: member_ids
            for cluster_id, member_ids in groups.items()
            if any(member_id not in selected_ids for member_id in member_ids)
        }
        if not live:
            return
        weights = (
            dict.fromkeys(live, 1)
            if even
            else {
                cluster_id: sum(geometry.masses[member_id] for member_id in member_ids)
                for cluster_id, member_ids in live.items()
            }
        )
        progressed = False
        shares = _largest_remainder(weights, target - len(selected_ids))
        for cluster_id, share in sorted(shares.items()):
            for member_id in geometry.representatives(
                live[cluster_id], share, covered_ids=selected_ids
            ):
                if len(selected_ids) >= target:
                    return
                record_selection(
                    selected_ids, reasons, member_id, f"cluster-representative:{cluster_id}"
                )
                progressed = True
        if not progressed:
            return


def _fill_stratified(
    geometry: CoreSetGeometry,
    strata: Mapping[str, int],
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
) -> None:
    """Give every stratum an equal share, round-robin over mass-descending strata."""
    groups = members_by_group(strata)
    ordered = sorted(
        groups,
        key=lambda stratum: (-sum(geometry.masses[member] for member in groups[stratum]), stratum),
    )
    while len(selected_ids) < target:
        progressed = False
        for stratum in ordered:
            if len(selected_ids) >= target:
                return
            member_ids = groups[stratum]
            if all(member in selected_ids for member in member_ids):
                continue
            medoid_id = geometry.medoid(member_ids)
            chosen = (
                medoid_id
                if medoid_id not in selected_ids
                else geometry.farthest_first(selected_ids, within=member_ids)
            )
            record_selection(selected_ids, reasons, chosen, f"stratum:{stratum}")
            progressed = True
        if not progressed:
            return


def _fill_farthest_first(
    geometry: CoreSetGeometry,
    selected_ids: list[str],
    reasons: MutableMapping[str, list[str]],
    target: int,
) -> None:
    """Spend whatever budget a strategy left on globally farthest-first coverage."""
    while len(selected_ids) < target:
        record_selection(
            selected_ids, reasons, geometry.farthest_first(selected_ids), "farthest-first"
        )


def _largest_remainder(weights: Mapping[int, int], total: int) -> dict[int, int]:
    """Apportion ``total`` whole slots across weighted groups without losing one to rounding.

    Args:
        weights: Nonnegative weight per group, such as workload mass.
        total: Exact number of slots to hand out.

    Returns:
        Slots per group, summing to ``total``. Remainders go to the largest fractional part first,
        with ties broken by descending weight and then ascending group id, so the apportionment
        never depends on mapping order.
    """
    mass = sum(weights.values())
    if mass <= 0 or not weights:
        return dict.fromkeys(weights, 0)
    exact = {group: weight * total / mass for group, weight in weights.items()}
    shares = {group: int(value) for group, value in exact.items()}
    ranked = sorted(
        weights,
        key=lambda group: (-(exact[group] - shares[group]), -weights[group], group),
    )
    for group in ranked[: total - sum(shares.values())]:
        shares[group] += 1
    return shares
