"""Average-linkage agglomeration over the distinct sets of propositions a corpus fired.

This is the one backend that reads a tagged corpus as sets rather than as coordinates, and the
reason is that the coordinates have nothing in them. Continuous tag probabilities carry no density
structure: at every threshold the whole corpus is one connected mass, so a clusterer measuring
distance between probability rows reports groups that are artifacts of where the cut landed.
Rounding each answer to fired or not fired recovers the structure. A tagged corpus resolves into
far fewer distinct fired-proposition sets than it has traces, and the overlap between those sets is
a real signal about which cases are the same kind of work.

Three choices here are load-bearing and each one was settled against a real corpus:

- **Jaccard distance**, ``1 - |A and B| / |A or B|``, because what matters is which propositions two
  cases share relative to how many they mention between them. Euclidean distance on the same
  indicator rows would call two rich, mostly-overlapping cases far apart simply because they fired
  many propositions.
- **Average linkage.** Single linkage fails outright on this data: it chains, running straight from
  singletons into one enormous component, so there is no height at which the tree says anything.
  Average linkage produces a smooth gradient of merge heights instead, which is what makes the
  dendrogram worth re-cutting at a different height.
- **Agglomerating over distinct patterns, weighted by multiplicity**, rather than over every
  candidate. Identical rows sit at distance zero, so the raw agglomeration's first merges collapse
  each pattern at height zero, and from there the weighted update measures what the raw run would
  have measured, at a size smaller by the corpus's duplication factor. The two runs agree up to
  tie-breaking rather than exactly: equal distances are common on indicator data, and the two break
  a tie in different index spaces, a candidate index there and a canonical pattern index here, so a
  corpus carrying ties can come out as two different trees. This run's tie-break is the one worth
  having, because a pattern index is fixed by which bits the pattern names while a candidate index
  is fixed by the order the corpus arrived in.

The all-zero pattern is held out of the tree rather than merged into it. Jaccard between two empty
sets is ``1 - 0/0``, which is not a distance, and there is no defensible value to substitute: zero
would make every case that fired nothing identical to every other such case *and* let them pull in
their neighbours, and one would scatter them as maximally distinct from everything. Holding them out
and giving them one shared cluster is the only treatment that matches what they are. See
:func:`pattern_hierarchy_fit`.

The completed agglomeration is also publishable as a re-cuttable tree, which is what lets a sampler
spend its budget by descending the dendrogram instead of over the flat partition one cut produced.
Holding the all-zero pattern out of the merges leaves it parentless, so the tree a fit describes is
a forest of two until the two are joined. See :func:`pattern_tree`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exp.simulation.mining.clustering.hierarchy import Hierarchy

_CUT_TOLERANCE = 1e-12
"""How far above a cut a merge height may measure and still be read as at that cut.

A Jaccard height is a distance in ``[0, 1]``, so this is a tolerance on that scale: far above the
last-bit rounding of a weighted-mean linkage update, and far below the gap between two heights that
genuinely differ, since those are rationals over the corpus's set sizes and multiplicities. Without
it the published contract, that every merge at or below the cut is applied, fails on exactly the
merges where it matters most, the ones whose exact height is the cut: a pair agreeing on seven of
the ten propositions the two of them name sits at exactly 3/10 and has to merge at the default cut
of 0.3.
"""


@dataclass(frozen=True)
class PatternHierarchyFit:
    """One completed agglomeration over distinct fired-proposition sets.

    Attributes:
        labels: ``(N,)`` cluster identity per input row, contiguous from zero and numbered by each
            cluster's lowest input row, so which identity a cluster gets is fixed by its members
            rather than by the order the merges happened in. End-to-end independence from the order
            a corpus arrived in is a separate guarantee and comes from the caller stacking its rows
            in ascending candidate identity, which
            :mod:`exp.simulation.mining.clustering.specs` does, not from this numbering.
        patterns: ``(P, T)`` the distinct patterns as boolean rows, in canonical ascending bit
            order rather than in order of first appearance.
        pattern_of_row: ``(N,)`` index into ``patterns`` for each input row.
        weights: ``(P,)`` multiplicity per pattern, the number of input rows carrying it.
        merges: ``(M, 2)`` the pattern index pair each merge joined, in merge order, lower index
            first. A cluster is always named by the lowest pattern index it holds, so the same
            index identifies it in every later merge.
        heights: ``(M,)`` Jaccard average-linkage height per merge, non-decreasing. Average linkage
            never inverts mathematically, and :func:`average_linkage` clamps what it measured to a
            running maximum so float64 rounding cannot invert it either.
        empty_pattern: Index in ``patterns`` of the all-zero pattern, or None when every row fired
            something. It takes part in no merge and holds its own cluster at every cut.
    """

    labels: np.ndarray
    patterns: np.ndarray
    pattern_of_row: np.ndarray
    weights: np.ndarray
    merges: np.ndarray
    heights: np.ndarray
    empty_pattern: int | None


def jaccard_distances(patterns: np.ndarray) -> np.ndarray:
    """Return the pairwise Jaccard distance between non-empty binary rows.

    Args:
        patterns: ``(P, T)`` rows read as set membership. Every row must name at least one member.

    Returns:
        ``(P, P)`` symmetric distances in ``[0, 1]``, exactly zero on the diagonal and otherwise
        the nearest double to the exact rational distance.

    Raises:
        ValueError: A row is empty, and Jaccard distance between two empty sets is undefined.
    """
    membership = np.asarray(patterns, dtype=np.float64)
    if membership.ndim != 2:
        raise ValueError("Jaccard distance needs a two-dimensional membership matrix")
    sizes = membership.sum(axis=1)
    if membership.shape[0] and float(sizes.min()) <= 0.0:
        raise ValueError(
            "Jaccard distance between two empty sets is undefined, so an all-zero pattern has to "
            "be held out of the agglomeration rather than measured against the other patterns"
        )
    intersection = membership @ membership.T
    union = sizes[:, None] + sizes[None, :] - intersection
    # ``(union - intersection) / union`` rather than ``1 - intersection / union``. Both counts are
    # exact integers in float64, so the difference is exact and the quotient is the exact rational
    # distance rounded once. The subtracted form rounds twice and lands an ulp away, which is
    # enough to put a merge whose exact height is the cut on the wrong side of it: eight and nine
    # propositions sharing seven is exactly 3/10, while ``1 - 7/10`` is 0.30000000000000004 and so
    # above the default cut of 0.3.
    return (union - intersection) / union


def average_linkage(distances: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerate weighted points by average linkage, recording every merge height.

    The Lance-Williams update is the weighted mean
    ``d(i + j, k) = (w_i d(i, k) + w_j d(j, k)) / (w_i + w_j)``, and weighting it by how many
    candidates each side stands for is what makes a run over distinct patterns identical to a run
    over every candidate.

    Args:
        distances: ``(P, P)`` symmetric distances with a zero diagonal.
        weights: ``(P,)`` positive multiplicity per point.

    Returns:
        ``(merges, heights)`` over the ``P - 1`` merges, in merge order. ``merges[s]`` is the index
        pair joined, lower index first, and that lower index names the merged cluster from then on.
        ``heights[s]`` is that merge's linkage height, clamped to a running maximum so the heights
        are non-decreasing.

    Raises:
        ValueError: The matrix is not square and two-dimensional, it is not symmetric, the weights
            do not match it, or a weight is not positive.
    """
    cost = np.asarray(distances, dtype=np.float64).copy()
    mass = np.asarray(weights, dtype=np.float64).copy()
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] == 0:
        raise ValueError("average linkage needs a non-empty square distance matrix")
    if mass.shape != (cost.shape[0],):
        raise ValueError("average linkage needs one weight per point")
    if float(mass.min()) <= 0.0:
        raise ValueError("average linkage needs a positive weight per point")
    if not np.array_equal(cost, cost.T):
        # The closest-pair search reads one triangle, through the row reductions in _closest_pair,
        # so an asymmetric matrix would quietly merge a pair that is not the lowest one. Refusing
        # it by name costs one pass over a matrix this function then walks a point at a time.
        raise ValueError(
            "average linkage reads a symmetric distance matrix, so d(i, j) and d(j, i) have to "
            "carry the same value; measure the pairs once and mirror the result"
        )
    count = cost.shape[0]
    np.fill_diagonal(cost, np.inf)
    merges = np.zeros((max(count - 1, 0), 2), dtype=np.int64)
    heights = np.zeros(max(count - 1, 0))
    for step in range(count - 1):
        left, right = _closest_pair(cost)
        merges[step] = (left, right)
        heights[step] = cost[left, right]
        updated = (mass[left] * cost[left] + mass[right] * cost[right]) / (mass[left] + mass[right])
        cost[left, :] = cost[:, left] = updated
        cost[left, left] = np.inf
        # The absorbed cluster leaves the search entirely. Its row and column stay infinite for the
        # rest of the run, which is also why the loop can never pick a dead index again.
        cost[right, :] = cost[:, right] = np.inf
        mass[left] += mass[right]
    # Average linkage is non-inverting, so the true heights are non-decreasing. In float64 they
    # invert by about one ulp on a small share of real runs, which is enough to stop a cut's prefix
    # rule in front of a merge that belongs below it. Clamping to a running maximum deletes that
    # noise rather than inventing structure: the only height it can move is one that a merge
    # already made has proved to be no lower.
    return merges, np.maximum.accumulate(heights)


def pattern_hierarchy_fit(indicators: np.ndarray, *, cut: float) -> PatternHierarchyFit:
    """Group indicator rows by average-linkage Jaccard agglomeration over distinct patterns.

    The all-zero rows are the corpus's least distinctive cases: the decision model found none of
    its propositions true of them. Isolating each one as its own cluster would therefore be exactly
    backwards, spending the selection budget on the cases with the least to say, so they are given
    one shared cluster and excluded from the tree.

    Args:
        indicators: ``(N, T)`` rows of 0.0 and 1.0, one per candidate, as the ``indicator`` tag
            transform produces. Row order fixes the cluster numbering and nothing else.
        cut: Jaccard average-linkage height to cut the dendrogram at, in ``[0, 1]``. Every merge at
            or below it is applied and every merge above it is not.

    Returns:
        The distinct patterns, the tree built over them and one cluster identity per input row.

    Raises:
        ValueError: The matrix is empty or not two-dimensional, it carries a value that is neither
            zero nor one, or the cut is outside ``[0, 1]``.
    """
    rows = np.asarray(indicators, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("pattern agglomeration needs a non-empty two-dimensional matrix")
    if not bool(np.isin(rows, (0.0, 1.0)).all()):
        raise ValueError(
            "pattern agglomeration reads which propositions fired, so every coordinate must be "
            "exactly 0.0 or 1.0; select with the 'indicator' tag transform rather than handing it "
            "probabilities or a normalized geometry"
        )
    if not 0.0 <= cut <= 1.0:
        raise ValueError(f"a Jaccard average-linkage cut is a distance in [0, 1], got {cut}")
    # Canonical ascending bit order, not order of first appearance: the cluster a candidate lands
    # in has to be the same whichever order the corpus arrived in.
    patterns, inverse = np.unique(rows > 0.0, axis=0, return_inverse=True)
    pattern_of_row = np.asarray(inverse).reshape(-1)
    weights = np.bincount(pattern_of_row, minlength=patterns.shape[0]).astype(np.float64)
    empty = _empty_pattern(patterns)
    active = np.array(
        [index for index in range(patterns.shape[0]) if index != empty], dtype=np.int64
    )
    merges = np.zeros((0, 2), dtype=np.int64)
    heights = np.zeros(0)
    if active.shape[0] > 1:
        subset, heights = average_linkage(jaccard_distances(patterns[active]), weights[active])
        # Back to pattern indexes. ``active`` is ascending, so the lower-index-first invariant and
        # the "a cluster is named by its lowest pattern" invariant both survive the mapping.
        merges = active[subset]
    labels = _numbered(_cut_at(patterns.shape[0], merges, heights, cut)[pattern_of_row])
    return PatternHierarchyFit(
        labels=labels,
        patterns=patterns,
        pattern_of_row=pattern_of_row,
        weights=weights,
        merges=merges,
        heights=heights,
        empty_pattern=empty,
    )


def pattern_labels_at(fit: PatternHierarchyFit, cut: float) -> np.ndarray:
    """Re-cut one completed agglomeration at another height without refitting it.

    The merges and their heights are the whole of what a cut reads, so a fit answers every height,
    not only the one it was asked for. That is what lets one run publish a fine partition and a
    coarser one over it at the same time.

    Args:
        fit: A completed agglomeration over distinct patterns.
        cut: Jaccard average-linkage height to cut at, in ``[0, 1]``.

    Returns:
        ``(N,)`` cluster identity per input row, numbered by each cluster's lowest input row
        exactly as ``fit.labels`` is, so the two partitions are comparable.

    Raises:
        ValueError: The cut is outside ``[0, 1]``.
    """
    if not 0.0 <= cut <= 1.0:
        raise ValueError(f"a Jaccard average-linkage cut is a distance in [0, 1], got {cut}")
    clusters = _cut_at(fit.patterns.shape[0], fit.merges, fit.heights, cut)
    return _numbered(clusters[fit.pattern_of_row])


def pattern_tree(fit: PatternHierarchyFit) -> Hierarchy:
    """Publish one completed agglomeration as a tree a sampler can descend.

    The leaves are the patterns, so node ``p`` is pattern ``p`` and ``leaf_of_unit[p] == p``: the
    finest assignment unit of this backend is the fired-proposition set, not the candidate. Node
    ``P + s`` is the s-th merge and carries that merge's Jaccard height, while a leaf carries
    ``0.0``, which is the height two identical patterns would have merged at.

    The all-zero pattern takes part in no merge, so the merges alone describe two parentless nodes:
    the agglomeration's root and that held-out leaf. :attr:`Hierarchy.root` refuses a forest, so
    the two are joined under one synthetic root at height ``1.0``. That height is a measurement
    rather than a filler: ``1.0`` is the maximum Jaccard distance, and a case that fired nothing
    shares nothing with a case that fired something, so maximally far apart is the honest reading.
    A cut therefore separates the cases that fired nothing before it separates anything else, which
    is the same statement holding them out of the agglomeration already made.

    Args:
        fit: A completed agglomeration over distinct patterns.

    Returns:
        A single-rooted tree over the patterns, whatever the fit held: one pattern, nothing but
        all-zero rows, an agglomeration with no all-zero pattern beside it, or both.
    """
    count = int(fit.patterns.shape[0])
    merges = fit.merges.tolist()
    # Every merge turns two parentless nodes into one, so what is left parentless afterwards is the
    # agglomeration's root plus the held-out all-zero leaf, when the corpus produced one.
    nodes = count + len(merges)
    loose = count - len(merges)
    parent = np.full(nodes + (1 if loose > 1 else 0), -1, dtype=np.int64)
    height = np.zeros(len(parent))
    node_of_cluster = np.arange(count)
    for step, (left, right) in enumerate(merges):
        merged = count + step
        parent[node_of_cluster[left]] = parent[node_of_cluster[right]] = merged
        height[merged] = float(fit.heights[step])
        # A cluster is named by the lowest pattern index it holds, so ``left`` goes on naming the
        # merged cluster and ``right`` is never read as a cluster name again.
        node_of_cluster[left] = merged
    if loose > 1:
        parent[np.flatnonzero(parent[:nodes] == -1)] = nodes
        height[nodes] = 1.0
    return Hierarchy(parent=parent, leaf_of_unit=np.arange(count), height=height)


def _closest_pair(cost: np.ndarray) -> tuple[int, int]:
    """Return the closest live pair, breaking every tie by the lowest index pair.

    The tie-break is written out rather than inherited from how numpy orders a flat search. Two
    patterns at an identical distance are common on indicator data, and which of them merges first
    decides the whole tree above it, so this is the one place the backend's input-order
    independence can be lost.

    Two reductions find that pair without ever listing the ties. The matrix is symmetric, so the
    first row to attain the matrix minimum holds the lower index of the lowest pair: a column below
    that row attaining the same value would have made its own row attain it earlier. The first
    column attaining the minimum inside that row is therefore the pair's upper index. Listing every
    tied index pair instead costs Python time proportional to the number of ties, and on binary
    data that is routinely most of the matrix.
    """
    left = int(np.argmin(cost.min(axis=1)))
    return left, int(np.argmin(cost[left]))


def _empty_pattern(patterns: np.ndarray) -> int | None:
    """Return the index of the all-zero pattern, or None when no row fired nothing."""
    found = np.flatnonzero(~patterns.any(axis=1))
    return int(found[0]) if found.shape[0] else None


def _cut_at(count: int, merges: np.ndarray, heights: np.ndarray, cut: float) -> np.ndarray:
    """Return each pattern's cluster after applying every merge at or below the cut height.

    The heights :func:`average_linkage` publishes are non-decreasing, so the merges at or below the
    cut are a prefix of the merge order. Taking them as a prefix is what keeps the result a real
    partition: skipping one merge and applying a later one would join clusters that the skipped
    merge was supposed to have formed.

    A height within :data:`_CUT_TOLERANCE` of the cut is read as at the cut, so a merge whose exact
    height is the cut is applied whichever way the arithmetic that produced it rounded.
    """
    representative = np.arange(count)
    limit = cut + _CUT_TOLERANCE
    for step, (left, right) in enumerate(merges.tolist()):
        if float(heights[step]) > limit:
            break
        representative[representative == right] = left
    return representative


def _numbered(keys: np.ndarray) -> np.ndarray:
    """Number clusters by the lowest input row in each, which fixes the published identities."""
    order: dict[int, int] = {}
    for key in keys.tolist():
        order.setdefault(int(key), len(order))
    return np.array([order[int(key)] for key in keys.tolist()], dtype=np.int64)
