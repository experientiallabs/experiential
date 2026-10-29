"""Bounded work rotation for one control plane's accounting recovery."""

import pytest

from exp.runtime.gateway.native_accounting_recovery import RecoveryRotation


def test_retained_failure_cannot_starve_later_work_across_batches() -> None:
    """A persistent first item yields to every later identity before being retried."""
    selection = RecoveryRotation[tuple[str, int]](lambda item: item[0])
    items = [(str(index), index) for index in range(40)]
    first = selection.select(items, 16)
    second = selection.select(items, 16)
    third = selection.select(items, 16)
    assert first == items[:16]
    assert second == items[16:32]
    assert third == items[32:] + items[:8]
    assert set(first + second + third) == set(items)


def test_rotation_discards_removed_work_and_reads_current_payloads() -> None:
    """Replaced payloads do not stay pinned in the cursor, and new work joins once."""
    selection = RecoveryRotation[tuple[str, int]](lambda item: item[0])
    items = [("a", 0), ("b", 0), ("c", 0)]
    assert selection.select(items, 1) == [("a", 0)]
    assert selection.select([("a", 1), ("c", 1), ("d", 0)], 2) == [("c", 1), ("a", 1)]
    assert selection.select([("a", 2), ("c", 2), ("d", 1)], 1) == [("d", 1)]
    assert selection.select([], 16) == []
    assert selection.select([("e", 0)], 16) == [("e", 0)]


def test_one_sweep_never_repeats_work_and_zero_limit_does_not_advance() -> None:
    """A large batch visits each item at most once, independent of prior batch size."""
    selection = RecoveryRotation[str](lambda item: item)
    assert selection.select(["a", "b"], 1) == ["a"]
    assert selection.select(["a", "b"], 0) == []
    assert selection.select(["a", "b"], 16) == ["b", "a"]


def test_invalid_work_or_limit_fails_before_advancing_rotation() -> None:
    """Duplicate identities cannot silently drop a retained write."""
    selection = RecoveryRotation[str](lambda item: item)
    assert selection.select(["a", "b"], 1) == ["a"]
    with pytest.raises(ValueError, match="identities must be unique"):
        selection.select(["a", "a"], 1)
    with pytest.raises(ValueError, match="limit must be nonnegative"):
        selection.select(["a", "b"], -1)
    assert selection.select(["a", "b"], 1) == ["b"]
