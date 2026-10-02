"""Cross-platform snapshot read budgets and real SQLite budget-authoring acceptance."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pytest

from exp.common.config.settings import GatewayResourceSettings
from exp.common.core.artifacts import canonical_json_bytes
from exp.runtime.gateway import snapshot_file
from exp.runtime.gateway.budget_authority import read_budget_snapshot
from exp.runtime.gateway.budgets import BudgetScope, BudgetScopeKind, SQLiteBudgetStore
from exp.runtime.gateway.budgets_test import _activate_chain, _authority, _Clock
from exp.runtime.gateway.snapshot_file import read_snapshot_bytes


def test_real_root_and_descendant_authoring_and_resource_override(tmp_path: Path) -> None:
    """Full private-state authoring admits the same valid file under a raised resource budget."""
    clock = _Clock()
    store, _ledger, budgets, _key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    size = (tmp_path / "chain-snapshot-org").stat().st_size
    small = SQLiteBudgetStore(budgets.database_path, clock=clock, snapshot_max_bytes=size - 1)
    # Root pool controls keep their no-file path even under a one-byte budget.
    SQLiteBudgetStore(budgets.database_path, clock=clock, snapshot_max_bytes=1).set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_nano_usd=1000,
    )
    root = BudgetScope(
        kind=BudgetScopeKind.DEPLOYMENT, alias_id="coding", pool_id="pool", deployment_id="primary"
    )
    with pytest.raises(ValueError, match="budget_snapshot_max_bytes"):
        small.set_limit(organization_id="org", period="2026-08", scope=root, limit_nano_usd=100)
    exact = SQLiteBudgetStore(budgets.database_path, clock=clock, snapshot_max_bytes=size)
    for scope in (
        root,
        BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="child-pool"),
        BudgetScope(
            kind=BudgetScopeKind.DEPLOYMENT,
            alias_id="coding",
            pool_id="child-pool",
            deployment_id="child",
        ),
    ):
        changed, _ = exact.set_limit(
            organization_id="org", period="2026-08", scope=scope, limit_nano_usd=100
        )
        assert changed
    assert (
        read_budget_snapshot(
            budgets.database_path, "chain-snapshot-org", catalog.identity_sha256(), size
        )
        == catalog
    )


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, float("inf"), "123", 2**63])
def test_snapshot_resource_budget_is_a_strict_positive_integer(value: object) -> None:
    """Invalid limits never reach allocation or file reads."""
    with pytest.raises(ValueError):
        GatewayResourceSettings.model_validate({"budget_snapshot_max_bytes": value})


def test_bounded_reader_accepts_limit_and_rejects_growth(tmp_path: Path) -> None:
    """Fixed chunk reads stop at the explicit byte budget rather than trusting stat alone."""
    (tmp_path / "snapshot").write_bytes(b"abcd")
    assert read_snapshot_bytes(tmp_path, "snapshot", 4) == b"abcd"
    with pytest.raises(ValueError, match="4 bytes.*3-byte") as error:
        read_snapshot_bytes(tmp_path, "snapshot", 3)
    assert isinstance(error.value, snapshot_file.SnapshotSizeError)
    assert (error.value.size, error.value.maximum) == (4, 3)
    assert "budget_snapshot_max_bytes" in str(error.value)


def test_growth_after_initial_stat_still_obeys_the_read_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk accounting detects a file growing after its initial size check."""
    path = tmp_path / "snapshot"
    path.write_bytes(b"ab")
    original = snapshot_file.os.fstat
    calls = 0

    def grow(descriptor: int) -> os.stat_result:
        """Return the first stat then grow the test file before the reader loops."""
        nonlocal calls
        info = original(descriptor)
        calls += 1
        if calls == 1:
            with path.open("ab") as writer:
                writer.write(b"cdef")
        return info

    @contextmanager
    def stream_without_platform_lock(root: Path, relative: str) -> Iterator[BinaryIO]:
        """Isolate the shared bounded reader from OS-specific write-denial semantics."""
        with path.open("rb") as stream:
            yield stream

    monkeypatch.setattr(snapshot_file, "snapshot_stream", stream_without_platform_lock)
    monkeypatch.setattr(snapshot_file.os, "fstat", grow)
    with pytest.raises(ValueError, match="resource budget"):
        read_snapshot_bytes(tmp_path, "snapshot", 4)


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor ownership; Windows owns CRT handles")
def test_posix_stream_construction_failure_closes_leaf_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed buffered-stream construction leaves neither file nor directory handles open."""
    (tmp_path / "snapshot").write_bytes(b"data")
    descriptor: int | None = None

    def fail_stream(fd: int, mode: str) -> BinaryIO:
        """Capture the real file descriptor before simulating stream allocation failure."""
        nonlocal descriptor
        descriptor = fd
        raise OSError("stream allocation failed")

    monkeypatch.setattr(snapshot_file.os, "fdopen", fail_stream)
    with pytest.raises(OSError, match="stream allocation failed"):
        read_snapshot_bytes(tmp_path, "snapshot", 4)
    assert descriptor is not None
    try:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        # Close the leaked descriptor in a failing regression run too.
        try:
            os.close(descriptor)
        except OSError:
            pass


@pytest.mark.parametrize("mutation", ["replace", "rewrite", "directory", "appearance"])
def test_prepared_snapshot_rejects_path_and_content_races(tmp_path: Path, mutation: str) -> None:
    """An operation-scoped handle proof never accepts a new path or rewritten leaf."""
    directory = tmp_path / "snapshots"
    directory.mkdir()
    path = directory / "plain.json"
    if mutation != "appearance":
        path.write_bytes(b"{}")
    with snapshot_file.prepare_snapshot_file(tmp_path, "snapshots/plain.json", 1024) as prepared:
        prepared.validate_current()
        if os.name == "nt":
            pytest.skip("Windows retained handles deny mutations; tested by the Windows backend")
        if mutation == "replace":
            path.unlink()
            path.write_bytes(b"{}")
        elif mutation == "rewrite":
            stamp = path.stat()
            path.write_bytes(b"[]")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        elif mutation == "directory":
            directory.rename(tmp_path / "old")
            directory.mkdir()
            path.write_bytes(b"{}")
        else:
            path.write_bytes(b"{}")
        with pytest.raises(ValueError, match="changed"):
            prepared.validate_current()
    with pytest.raises(ValueError, match="closed"):
        prepared.validate_current()


def test_prepared_absent_parent_cannot_be_created_before_fence(tmp_path: Path) -> None:
    """Even a still-absent leaf fails when its previously missing parent appears."""
    with snapshot_file.prepare_snapshot_file(tmp_path, "missing/plain.json", 1024) as prepared:
        assert prepared.take_bytes() is None
        (tmp_path / "missing").mkdir()
        with pytest.raises(ValueError, match="changed"):
            prepared.validate_current()


def test_catalog_fixture_size_is_measured_not_a_universal_limit(tmp_path: Path) -> None:
    """The representative graph fits the default without claiming arbitrary catalog bounds."""
    clock = _Clock()
    store, _ledger, _budgets, _key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    size = len(canonical_json_bytes(catalog.model_dump(mode="json")))
    assert 0 < size < GatewayResourceSettings().budget_snapshot_max_bytes
