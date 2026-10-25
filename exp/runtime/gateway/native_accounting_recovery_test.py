"""Tenant fairness and lifetime tests for background write recovery."""

from exp.runtime.gateway.native_accounting_recovery import FairRecovery


def test_large_failed_tenant_cannot_starve_a_small_tenant_across_batches() -> None:
    selection = FairRecovery[tuple[str, int]](lambda item: item[0])
    hot = [("hot", index) for index in range(40)]
    quiet = [("quiet", 0)]
    first = selection.select(hot + quiet, 16)
    second = selection.select(hot + quiet, 16)
    assert quiet[0] in first
    assert quiet[0] in second
    assert len(first) == len(set(first)) == 16
    assert len(second) == len(set(second)) == 16
    assert set(second) - set(first)


def test_one_item_batches_rotate_tenants_and_discard_removed_state() -> None:
    selection = FairRecovery[tuple[str, int]](lambda item: item[0])
    items = [("a", 0), ("a", 1), ("b", 0)]
    assert selection.select(items, 1) == [("a", 0)]
    assert selection.select(items, 1) == [("b", 0)]
    assert selection.select(items, 1) == [("a", 1)]
    assert selection.select([], 16) == []
    assert selection.select([("c", 0)], 16) == [("c", 0)]
