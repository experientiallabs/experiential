"""Deterministic disjoint-set union used by mining and SFT leakage grouping."""

from __future__ import annotations

from collections.abc import Iterable


class UnionFind:
    """Lexically stable union-find with path compression."""

    def __init__(self, values: Iterable[str]) -> None:
        """Initialize each identity as its own component.

        Args:
            values: Source identities to place in singleton components.
        """
        self._parent: dict[str, str] = {}
        for value in values:
            self.add(value)

    def add(self, value: str) -> None:
        """Add one identity when it has not been seen before.

        Args:
            value: Identity to insert as its own component when absent.
        """
        self._parent.setdefault(value, value)

    def find(self, value: str) -> str:
        """Return one component root with path compression.

        Args:
            value: Identity whose component root is requested.

        Returns:
            The lexically stable root of the component that owns ``value``.
        """
        self.add(value)
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        """Join two components using the lexically smallest root.

        Args:
            left: First identity to join.
            right: Second identity to join.
        """
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self._parent[right_root] = left_root
        else:
            self._parent[left_root] = right_root
