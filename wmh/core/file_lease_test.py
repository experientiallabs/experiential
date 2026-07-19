"""Tests for shared POSIX file and directory leases."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

import wmh.core.file_lease as mod
from wmh.core.file_lease import (
    exclusive_posix_directory_lease,
    exclusive_posix_file_lease,
)


def _directory_lease(descriptor: int) -> AbstractContextManager[None]:
    return exclusive_posix_directory_lease(
        descriptor,
        unsupported_error=RuntimeError("directory leases require POSIX"),
        irregular_directory_error=OSError("lease target must be a directory"),
        contention_error=RuntimeError("directory is already leased"),
    )


@pytest.mark.skipif(os.name != "posix", reason="file leases require POSIX")
def test_directory_lease_stays_bound_to_inode_after_path_replacement(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "journal"
    displaced = tmp_path / "displaced-journal"
    directory.mkdir(mode=0o700)
    first = os.open(directory, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    second = os.open(directory, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    try:
        with _directory_lease(first):
            mutable_lock_path = directory / ".legacy-lock"
            mutable_lock_path.write_text("old", encoding="utf-8")
            mutable_lock_path.unlink()
            mutable_lock_path.write_text("replacement", encoding="utf-8")
            directory.rename(displaced)
            directory.mkdir(mode=0o700)

            with pytest.raises(RuntimeError, match="already leased"):
                with _directory_lease(second):
                    pytest.fail("a second descriptor for the leased inode must not enter")
    finally:
        os.close(second)
        os.close(first)


@pytest.mark.skipif(os.name != "posix", reason="file leases require POSIX")
def test_file_lease_cleanup_failures_never_mask_the_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    locking = mod.fcntl
    assert locking is not None
    real_flock = locking.flock
    real_close = os.close
    leased_descriptor: int | None = None

    def fail_unlock(descriptor: int, operation: int) -> None:
        nonlocal leased_descriptor
        if operation & locking.LOCK_EX:
            leased_descriptor = descriptor
        if operation == locking.LOCK_UN:
            raise OSError("forced lease unlock failure")
        real_flock(descriptor, operation)

    def fail_close(descriptor: int) -> None:
        if descriptor == leased_descriptor:
            raise OSError("forced lease descriptor close failure")
        real_close(descriptor)

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(locking, "flock", fail_unlock)
            scoped.setattr(os, "close", fail_close)
            with pytest.raises(ValueError, match="primary operation failed") as captured:
                with exclusive_posix_file_lease(
                    tmp_path / "operation.lock",
                    unsupported_error=RuntimeError("file leases require POSIX"),
                    irregular_file_error=OSError("lease target must be regular"),
                    contention_error=RuntimeError("file is already leased"),
                ):
                    raise ValueError("primary operation failed")
    finally:
        if leased_descriptor is not None:
            real_close(leased_descriptor)

    notes = getattr(captured.value, "__notes__", [])
    assert any("forced lease unlock failure" in note for note in notes)
    assert any("forced lease descriptor close failure" in note for note in notes)
