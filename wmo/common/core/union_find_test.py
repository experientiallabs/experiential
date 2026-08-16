"""Tests for the shared lexically stable union-find."""

from __future__ import annotations

from wmo.common.core.union_find import UnionFind


def test_union_find_keeps_the_lexically_smallest_root() -> None:
    """Joined identities share the smaller root after path compression."""
    groups = UnionFind(("lineage-b", "lineage-a", "lineage-c"))

    groups.union("lineage-c", "lineage-b")
    groups.union("lineage-b", "lineage-a")

    assert groups.find("lineage-a") == "lineage-a"
    assert groups.find("lineage-b") == "lineage-a"
    assert groups.find("lineage-c") == "lineage-a"
