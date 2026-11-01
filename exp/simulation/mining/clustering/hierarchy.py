"""Fine atoms plus a re-cuttable tree over them.

Two things make this the only backend that answers "how many groups?" after the fact. The atoms are
the stable unit of assignment, so the tree above them can be cut at any level without reassigning a
single trace, and the merges are Ward merges, so a cut reproduces exactly the partition Ward would
have produced on the traces themselves had every atom been forced to stay whole. A reviewer who
wants eight groups and then sixteen pays for one fit.

Ward over weighted points is what makes that equivalence hold. The merge cost
``n_i n_j / (n_i + n_j) * ||c_i - c_j||^2`` is the increase in within-group sum of squares that
merging two atoms would cause, and the Lance-Williams update keeps that true for every later merge
without revisiting the underlying points.

A tempting extra step is left out on purpose: computing, for every trace, its distance to the
nearest atom in each group, by masking the distance table one group at a time. That table is only
useful for a per-group confidence margin, which nothing here reads, and it is fragile. When the
tree cannot expand to the requested number of groups, some group index has no atoms at all, its
mask is empty, and taking a minimum over an empty axis raises rather than returning an answer. So
the table is never built, and a cut that yields fewer groups than asked for stays an ordinary
outcome: ``labels_at`` renumbers whatever the cut produced into a contiguous range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exp.simulation.mining.clustering.centroids import squared_distances
from exp.simulation.mining.clustering.kmeans import kmeans_fit


def weighted_ward(centroids: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build Ward linkage over weighted points.

    Args:
        centroids: ``(A, D)`` point coordinates, one per atom.
        sizes: ``(A,)`` weight per point, normally the number of traces in that atom.

    Returns:
        ``(parent, height)`` over ``2A - 1`` nodes. Nodes ``0..A-1`` are the points, node ``A + s``
        is the s-th merge, the root's parent is ``-1``, and a merge's height is the increase in
        within-group sum of squares it caused.

    Raises:
        ValueError: There are no points, or the weights do not match the points.
    """
    if centroids.ndim != 2 or len(centroids) == 0:
        raise ValueError("weighted Ward linkage needs at least one point")
    if len(sizes) != len(centroids):
        raise ValueError("weighted Ward linkage needs one weight per point")
    count = len(centroids)
    # A zero-weight atom would divide by zero in the merge cost; it is empty, so a floor that keeps
    # it far from dominating any merge is the honest treatment.
    weights = np.maximum(np.asarray(sizes, dtype=np.float64), 1e-12)
    cost = (
        weights[:, None] * weights[None, :] / (weights[:, None] + weights[None, :])
    ) * squared_distances(centroids, centroids)
    np.fill_diagonal(cost, np.inf)
    node_of = np.arange(count)
    parent = np.full(2 * count - 1, -1)
    height = np.zeros(2 * count - 1)
    for step in range(count - 1):
        left, right = np.unravel_index(np.argmin(cost), cost.shape)
        merged_node = count + step
        parent[node_of[left]] = parent[node_of[right]] = merged_node
        height[merged_node] = cost[left, right]
        with np.errstate(invalid="ignore"):
            updated = (
                (weights[left] + weights) * cost[left]
                + (weights[right] + weights) * cost[right]
                - weights * cost[left, right]
            ) / (weights[left] + weights[right] + weights)
        cost[left, :] = cost[:, left] = updated
        cost[left, left] = np.inf
        cost[right, :] = cost[:, right] = np.inf
        weights[left] += weights[right]
        node_of[left] = merged_node
    return parent, height


@dataclass(frozen=True)
class Hierarchy:
    """A tree over assignment units, cuttable at any number of groups.

    Attributes:
        parent: ``parent[node]`` is the node's parent, or ``-1`` for the root.
        leaf_of_unit: Leaf node of each finest assignment unit, by unit index.
        height: Merge height per node, ordering which node a cut expands first. Without it a cut
            expands the largest subtree instead, which is a reasonable fallback but a different
            tree shape.
    """

    parent: np.ndarray
    leaf_of_unit: np.ndarray
    height: np.ndarray | None = None

    @property
    def root(self) -> int:
        """Return the single root node.

        Raises:
            ValueError: The tree has no root or more than one.
        """
        roots = np.flatnonzero(self.parent == -1)
        if len(roots) != 1:
            raise ValueError(f"a hierarchy needs exactly one root, found {len(roots)}")
        return int(roots[0])

    def children(self, node: int) -> np.ndarray:
        """Return the direct children of one node."""
        return np.flatnonzero(self.parent == node)

    def leaves(self, node: int) -> list[int]:
        """Return every leaf below one node, sorted."""
        stack, found = [node], []
        while stack:
            current = stack.pop()
            kids = self.children(current)
            if len(kids):
                stack.extend(kids.tolist())
            else:
                found.append(current)
        return sorted(found)

    def path(self, node: int) -> list[int]:
        """Return the node identities from the root down to one node."""
        walked = [node]
        while self.parent[walked[-1]] != -1:
            walked.append(int(self.parent[walked[-1]]))
        return walked[::-1]

    def cut(self, groups: int) -> np.ndarray:
        """Map every unit to its ancestor in a cut of the tree into about ``groups`` nodes.

        Args:
            groups: Requested number of groups.

        Returns:
            One ancestor node per unit. Fewer than ``groups`` distinct nodes come back when the
            tree runs out of expandable nodes first, which is a valid partition and not an error.
        """

        def priority(node: int) -> float:
            """Expand the highest merge first, or the largest subtree with no heights."""
            return float(self.height[node]) if self.height is not None else len(self.leaves(node))

        frontier = [self.root]
        while len(frontier) < groups:
            expandable = [node for node in frontier if len(self.children(node))]
            if not expandable:
                break
            node = max(expandable, key=priority)
            frontier.remove(node)
            frontier.extend(self.children(node).tolist())
        group_of_leaf = {leaf: node for node in frontier for leaf in self.leaves(node)}
        return np.array([group_of_leaf[leaf] for leaf in self.leaf_of_unit])

    def labels_at(self, units: np.ndarray, groups: int) -> np.ndarray:
        """Return contiguous group identities for the named units at one cut.

        Args:
            units: Unit indexes to label.
            groups: Requested number of groups.

        Returns:
            Group identities in ``0..G-1``, where ``G`` is the number of groups the cut achieved.
        """
        cut = self.cut(groups)
        _, contiguous = np.unique(cut, return_inverse=True)
        return contiguous[np.asarray(units)]


@dataclass(frozen=True)
class AtomsHierarchyFit:
    """Fine atoms, the tree over them and the labels one cut produces.

    Attributes:
        labels: ``(N,)`` group identity per trace at the requested cut.
        atoms: ``(N,)`` atom index per trace, the stable assignment unit.
        centroids: ``(A, D)`` atom centroids.
        sizes: ``(A,)`` trace count per atom.
        hierarchy: The tree over the atoms.
        group_of_atom: ``(A,)`` group identity per atom at the requested cut.
    """

    labels: np.ndarray
    atoms: np.ndarray
    centroids: np.ndarray
    sizes: np.ndarray
    hierarchy: Hierarchy
    group_of_atom: np.ndarray


def atoms_hierarchy_fit(
    features: np.ndarray,
    *,
    atom_count: int,
    groups: int,
    n_init: int = 4,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
) -> AtomsHierarchyFit:
    """Fit fine k-means atoms and a size-weighted Ward tree over their centroids.

    Args:
        features: ``(N, D)`` point coordinates.
        atom_count: Number of atoms, clamped to the number of points.
        groups: Groups to cut the tree into for the returned labels.
        n_init: k-means restarts for the atom fit.
        max_iter: k-means iterations per restart.
        tol: k-means convergence bound.
        seed: Seed for the atom fit.

    Returns:
        The atoms, the tree and the labels at the requested cut.

    Raises:
        ValueError: The point set is empty or a requested count is not positive.
    """
    points = np.asarray(features, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("an atoms hierarchy needs a non-empty two-dimensional feature matrix")
    if atom_count < 1 or groups < 1:
        raise ValueError("an atoms hierarchy needs at least one atom and at least one group")
    resolved_atoms = min(atom_count, points.shape[0])
    fit = kmeans_fit(points, resolved_atoms, n_init=n_init, max_iter=max_iter, tol=tol, seed=seed)
    sizes = np.bincount(fit.labels, minlength=resolved_atoms).astype(np.float64)
    parent, height = weighted_ward(fit.centroids, sizes)
    hierarchy = Hierarchy(parent, np.arange(resolved_atoms), height)
    group_of_atom = hierarchy.labels_at(np.arange(resolved_atoms), min(groups, resolved_atoms))
    return AtomsHierarchyFit(
        labels=group_of_atom[fit.labels],
        atoms=fit.labels,
        centroids=fit.centroids,
        sizes=sizes,
        hierarchy=hierarchy,
        group_of_atom=group_of_atom,
    )
