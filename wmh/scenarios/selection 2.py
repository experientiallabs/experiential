"""Representative selection: SemDeDup + hybrid-allocation medoid picking with failure pinning.

Given clustered facet embeddings and a scenario budget K, pick which real traces become scenarios:

1. SemDeDup (arXiv 2303.09540): drop near-duplicate facets within a cluster before selection, so
   thirty rewordings of the same request can't claim thirty slots.
2. Hybrid allocation: ~70% of the budget goes to clusters proportionally to their corpus mass (the
   eval mirrors traffic), the rest round-robin across clusters (the long tail keeps coverage).
3. Within a cluster the first pick is the medoid (the real trace nearest everything else); extra
   slots go farthest-first for intra-cluster diversity.
4. Failure pinning: every failure category present in the corpus keeps at least one exemplar,
   regardless of frequency — rare-but-critical traces are exactly the ones proportional sampling
   silently drops.

Each selection carries `weight`: the fraction of the (deduped) corpus it stands for, so downstream
scoring can report a traffic-weighted number. Weights sum to 1 over the selection.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from wmh.scenarios.clustering import normalize_rows
from wmh.scenarios.facets import Outcome, TraceFacet

DEDUP_THRESHOLD = 0.95
PROPORTIONAL_FRACTION = 0.7


class SelectedTrace(BaseModel):
    """One trace chosen to become a scenario, with the corpus mass it represents."""

    trace_id: str
    cluster_id: int
    weight: float  # fraction of the deduped corpus this selection stands for
    pinned_failure: str | None = None  # failure category this pick was retained for, if any


def semdedup_keep(
    embeddings: np.ndarray, labels: np.ndarray, *, threshold: float = DEDUP_THRESHOLD
) -> list[int]:
    """Indices that survive within-cluster near-duplicate removal (first occurrence wins).

    Compares cosine similarity only within a cluster (the SemDeDup trick: k-means already grouped
    near-duplicates, so the quadratic pass stays per-cluster).
    """
    matrix = normalize_rows(embeddings)
    kept: list[int] = []
    for cluster_id in sorted(set(labels.tolist())):
        member_indices = np.flatnonzero(labels == cluster_id)
        cluster_kept: list[int] = []
        for index in member_indices.tolist():
            duplicate = any(
                float(matrix[index] @ matrix[other]) > threshold for other in cluster_kept
            )
            if not duplicate:
                cluster_kept.append(index)
        kept.extend(cluster_kept)
    return sorted(kept)


def hybrid_select(
    facets: list[TraceFacet],
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget: int,
    *,
    proportional_fraction: float = PROPORTIONAL_FRACTION,
    dedup_threshold: float = DEDUP_THRESHOLD,
) -> list[SelectedTrace]:
    """Pick `budget` representative traces from a clustered facet corpus.

    See the module docstring for the algorithm. Cluster mass (and thus weights) is measured on the
    deduped corpus. Raises when the budget is not positive; a budget larger than the deduped corpus
    returns everything.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if len(facets) != len(embeddings) or len(facets) != len(labels):
        raise ValueError("facets, embeddings, and labels must be parallel")
    if not facets:
        return []

    matrix = normalize_rows(embeddings)
    kept = semdedup_keep(embeddings, labels, threshold=dedup_threshold)
    by_cluster: dict[int, list[int]] = {}
    for index in kept:
        by_cluster.setdefault(int(labels[index]), []).append(index)
    total_kept = len(kept)
    if budget >= total_kept:
        selections = [_selection(facets[i], int(labels[i]), 1.0 / total_kept) for i in sorted(kept)]
        return _pin_failures(selections, facets, matrix, labels, by_cluster)

    slots = _allocate_slots(by_cluster, budget, proportional_fraction)
    selections: list[SelectedTrace] = []
    for cluster_id, cluster_slots in slots.items():
        member_indices = by_cluster[cluster_id]
        chosen = _pick_representatives(matrix, member_indices, cluster_slots)
        cluster_weight = len(member_indices) / total_kept
        for index in chosen:
            selections.append(_selection(facets[index], cluster_id, cluster_weight / len(chosen)))
    selections = _pin_failures(selections, facets, matrix, labels, by_cluster)
    # Clusters allocated zero slots contribute no selection, so their mass would silently vanish
    # from the weights; renormalize so weights always sum to 1 over the returned selection.
    total_weight = sum(s.weight for s in selections)
    return [s.model_copy(update={"weight": s.weight / total_weight}) for s in selections]


def _selection(facet: TraceFacet, cluster_id: int, weight: float) -> SelectedTrace:
    return SelectedTrace(trace_id=facet.trace_id, cluster_id=cluster_id, weight=weight)


def _allocate_slots(
    by_cluster: dict[int, list[int]], budget: int, proportional_fraction: float
) -> dict[int, int]:
    """Split `budget` slots across clusters: proportional share + round-robin coverage share.

    Proportional slots follow cluster mass (largest-remainder rounding); the remaining coverage
    slots go one per cluster in descending-mass order, cycling. No cluster is allocated more slots
    than it has (deduped) members; leftover slots spill to the largest clusters with capacity.
    """
    cluster_ids = sorted(by_cluster, key=lambda c: len(by_cluster[c]), reverse=True)
    capacity = {c: len(by_cluster[c]) for c in cluster_ids}
    total = sum(capacity.values())
    proportional_budget = min(budget, round(budget * proportional_fraction))

    # Largest-remainder proportional allocation, capped by capacity.
    quotas = {c: proportional_budget * capacity[c] / total for c in cluster_ids}
    slots = {c: min(int(quotas[c]), capacity[c]) for c in cluster_ids}
    remainders = sorted(cluster_ids, key=lambda c: quotas[c] - int(quotas[c]), reverse=True)
    leftover = proportional_budget - sum(slots.values())
    for cluster_id in remainders:
        if leftover <= 0:
            break
        if slots[cluster_id] < capacity[cluster_id]:
            slots[cluster_id] += 1
            leftover -= 1

    # Coverage slots: uncovered clusters first (the long tail is the whole point of this share),
    # then cycle clusters by descending mass, one slot each, skipping full clusters.
    remaining = budget - sum(slots.values())
    for cluster_id in cluster_ids:
        if remaining <= 0:
            break
        if slots[cluster_id] == 0 and capacity[cluster_id] > 0:
            slots[cluster_id] = 1
            remaining -= 1
    while remaining > 0:
        progressed = False
        for cluster_id in cluster_ids:
            if remaining <= 0:
                break
            if slots[cluster_id] < capacity[cluster_id]:
                slots[cluster_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # every cluster saturated; budget > corpus, handled by caller
            break
    return {c: s for c, s in slots.items() if s > 0}


def _pick_representatives(matrix: np.ndarray, member_indices: list[int], slots: int) -> list[int]:
    """Medoid first, then farthest-first: real, central exemplars with intra-cluster diversity."""
    members = np.asarray(member_indices)
    if slots >= len(members):
        return members.tolist()
    similarities = matrix[members] @ matrix[members].T
    chosen: list[int] = [int(members[similarities.mean(axis=1).argmax()])]  # the medoid
    while len(chosen) < slots:
        chosen_rows = matrix[np.asarray(chosen)]
        best_similarity = (matrix[members] @ chosen_rows.T).max(axis=1)
        best_similarity[np.isin(members, chosen)] = np.inf  # never re-pick
        chosen.append(int(members[best_similarity.argmin()]))
    return chosen


def _pin_failures(
    selections: list[SelectedTrace],
    facets: list[TraceFacet],
    matrix: np.ndarray,
    labels: np.ndarray,
    by_cluster: dict[int, list[int]],
) -> list[SelectedTrace]:
    """Ensure every failure category in the (deduped) corpus keeps at least one exemplar.

    A missing category's medoid replaces the currently lowest-weight unpinned selection, so the
    budget holds. The replaced selection's weight transfers, keeping weights summing to 1.
    """
    facet_by_id = {facet.trace_id: facet for facet in facets}
    kept_indices = [i for members in by_cluster.values() for i in members]
    categories: dict[str, list[int]] = {}
    for index in kept_indices:
        facet = facets[index]
        if facet.outcome is Outcome.FAILURE and facet.failure_category:
            categories.setdefault(facet.failure_category, []).append(index)

    covered = {
        facet_by_id[s.trace_id].failure_category
        for s in selections
        if facet_by_id[s.trace_id].outcome is Outcome.FAILURE
    }
    for category, member_indices in sorted(categories.items()):
        if category in covered:
            continue
        members = np.asarray(member_indices)
        similarities = matrix[members] @ matrix[members].T
        exemplar = int(members[similarities.mean(axis=1).argmax()])
        replaceable = [s for s in selections if s.pinned_failure is None]
        if not replaceable:
            break
        victim = min(replaceable, key=lambda s: s.weight)
        selections[selections.index(victim)] = SelectedTrace(
            trace_id=facets[exemplar].trace_id,
            cluster_id=int(labels[exemplar]),
            weight=victim.weight,
            pinned_failure=category,
        )
        covered.add(category)
    return selections
